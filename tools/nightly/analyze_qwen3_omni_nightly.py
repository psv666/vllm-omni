#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze Qwen3-Omni coverage and redundancy in the CUDA nightly pipeline.

The analyzer has two deliberately separate inputs:

* repository truth: ``test-nightly.yml``, pytest collection, and perf JSON;
* Buildkite history: the last N scheduled ``main`` builds with ``NIGHTLY=1``.

Buildkite credentials are read only from environment variables and are never
written to reports or cache files.  A static-only report is still produced when
credentials are absent, but destructive recommendations are suppressed.

Usage::

    export BUILDKITE_API_TOKEN=...
    export BUILDKITE_ORGANIZATION_SLUG=...
    export BUILDKITE_PIPELINE_SLUG=...
    python tools/nightly/analyze_qwen3_omni_nightly.py

For deterministic/offline analysis, pass a normalized history fixture::

    python tools/nightly/analyze_qwen3_omni_nightly.py \
      --history-json /path/to/history.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_YAML = REPO_ROOT / ".buildkite" / "cuda" / "test-nightly.yml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "contributing" / "ci" / "qwen3_omni_nightly_redundancy_analysis.md"
DEFAULT_CSV = REPO_ROOT / "docs" / "contributing" / "ci" / "qwen3_omni_nightly_redundancy_cases.csv"
API_ROOT = "https://api.buildkite.com/v2"
WEB_ROOT = "https://buildkite.com"

FUNCTION_LABEL = "Omni · Function Test with H100"
ACCURACY_LABEL = "Omni · Accuracy Test"
MULTI_REPLICA_LABEL = "Omni · Multi-Replica Startup Test with 4x H100"
QWEN_TOKEN = "qwen3_omni"
TERMINAL_FAILURE_STATES = {"failed", "failing", "timed_out", "canceled", "canceling"}
SUCCESS_STATES = {"passed", "finished"}
CASE_RESULT_RE = re.compile(r"(?m)^(tests/[^\s]+::[^\s]+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)")
NODE_ID_RE = re.compile(r"tests/[^\s,]+::[^\s,]+")
FAILED_NODE_RE = re.compile(r"(?m)(?:^|\s)FAILED\s+(tests/[^\s]+::[^\s]+)")
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
BUILDKITE_ESCAPE_RE = re.compile(r"\x1b_bk;[^\x07]*\x07")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]")


@dataclass(frozen=True)
class NightlyJob:
    label: str
    key: str
    category: str
    gpu_count: int
    timeout_minutes: int | None
    pytest_command: str
    perf_config: str | None = None


@dataclass
class StaticCase:
    case_id: str
    job_key: str
    job_label: str
    category: str
    gpu_count: int
    dimensions: dict[str, Any]
    capabilities: set[str]
    request_count: int | None
    startup_key: str
    source_path: str
    static_signature: str = ""
    overlap_hint: str = ""


@dataclass
class Attempt:
    build_number: int
    build_commit: str
    job_key: str
    job_id: str
    state: str
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    retried: bool
    retry_count: int
    gpu_count: int | None = None
    soft_failed: bool = False
    log: str = ""
    case_results: list[tuple[str, str]] = field(default_factory=list)
    failure_signature: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class JobSummary:
    executions: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    flakes: int = 0
    p50_seconds: float | None = None
    p95_seconds: float | None = None
    gpu_hours: float = 0.0
    retry_gpu_hours: float = 0.0
    soft_failures: int = 0
    unknown_gpu_attempts: int = 0
    failure_signatures: list[str] = field(default_factory=list)


@dataclass
class CaseSummary:
    executions: int = 0
    failures: int = 0
    skips: int = 0
    flakes: int = 0
    independent_failures: int = 0
    failure_builds: set[int] = field(default_factory=set)


class BuildkiteError(RuntimeError):
    """Buildkite API failure that is safe to show without credentials."""


class BuildkiteClient:
    def __init__(self, token: str, organization: str, pipeline: str, *, timeout: int = 30) -> None:
        if not token or not organization or not pipeline:
            raise ValueError("token, organization, and pipeline are required")
        self._token = token
        self.organization = organization
        self.pipeline = pipeline
        self.timeout = timeout

    @property
    def pipeline_path(self) -> str:
        org = urllib.parse.quote(self.organization, safe="")
        pipeline = urllib.parse.quote(self.pipeline, safe="")
        return f"/organizations/{org}/pipelines/{pipeline}"

    def _request(self, path_or_url: str, *, authenticated: bool = True) -> tuple[bytes, Any]:
        url = path_or_url if path_or_url.startswith("https://") else API_ROOT + path_or_url
        headers = {"Accept": "application/json", "User-Agent": "vllm-omni-nightly-analysis/1"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(url, headers=headers)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    self._honor_rate_limit_headers(response.headers)
                    return body, response.headers
            except urllib.error.HTTPError as exc:
                if (exc.code == 429 or 500 <= exc.code < 600) and attempt < 3:
                    time.sleep(self._retry_delay(exc.headers, attempt))
                    continue
                raise BuildkiteError(
                    f"Buildkite API returned HTTP {exc.code} for {urllib.parse.urlsplit(url).path}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 3:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise BuildkiteError(
                    f"Buildkite API request failed for {urllib.parse.urlsplit(url).path}: {exc.reason}"
                ) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _retry_delay(headers: Any, attempt: int) -> float:
        retry_after = headers.get("Retry-After") if headers else None
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except (TypeError, ValueError):
            return float(min(2**attempt, 8))

    @staticmethod
    def _honor_rate_limit_headers(headers: Any) -> None:
        """Pause only when either documented rate-limit bucket is exhausted."""
        remaining_values = []
        for name in ("RateLimit-Remaining", "RateLimit-User-Remaining"):
            try:
                remaining_values.append(int(headers.get(name)))
            except (TypeError, ValueError):
                pass
        if remaining_values and min(remaining_values) <= 0:
            reset_values = []
            for name in ("RateLimit-Reset", "RateLimit-User-Reset"):
                try:
                    reset_values.append(float(headers.get(name)))
                except (TypeError, ValueError):
                    pass
            if reset_values:
                now = time.time()
                waits = [value - now if value > now else value for value in reset_values]
                time.sleep(min(max(max(waits), 0.0), 60.0))

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, Any]:
        if params:
            path = f"{path}?{urllib.parse.urlencode(params, doseq=True)}"
        body, headers = self._request(path)
        try:
            return json.loads(body), headers
        except json.JSONDecodeError as exc:
            raise BuildkiteError(f"Buildkite API returned invalid JSON for {path.split('?', 1)[0]}") from exc

    def get_text(self, path: str) -> str:
        body, _ = self._request(path)
        return body.decode("utf-8", errors="replace")

    def download_public_bytes(self, url: str) -> bytes:
        """Download a signed artifact URL without forwarding the API token."""
        body, _ = self._request(url, authenticated=False)
        return body


class PublicBuildkiteClient:
    """Read Buildkite's anonymous web data endpoints for a public pipeline."""

    def __init__(self, organization: str, pipeline: str, *, timeout: int = 60) -> None:
        if not organization or not pipeline:
            raise ValueError("organization and pipeline are required")
        self.organization = organization
        self.pipeline = pipeline
        self.timeout = timeout

    @property
    def pipeline_path(self) -> str:
        org = urllib.parse.quote(self.organization, safe="")
        pipeline = urllib.parse.quote(self.pipeline, safe="")
        return f"/{org}/{pipeline}"

    def _request(self, path_or_url: str) -> tuple[bytes, Any]:
        url = path_or_url if path_or_url.startswith("https://") else WEB_ROOT + path_or_url
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json, text/html, text/plain;q=0.9",
                "Accept-Encoding": "gzip",
                "User-Agent": "vllm-omni-nightly-analysis/1",
            },
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body, response.headers
            except urllib.error.HTTPError as exc:
                if (exc.code == 429 or 500 <= exc.code < 600) and attempt < 3:
                    time.sleep(BuildkiteClient._retry_delay(exc.headers, attempt))
                    continue
                raise BuildkiteError(
                    f"Buildkite public web returned HTTP {exc.code} for {urllib.parse.urlsplit(url).path}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 3:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise BuildkiteError(
                    f"Buildkite public web request failed for {urllib.parse.urlsplit(url).path}: {exc.reason}"
                ) from exc
        raise AssertionError("unreachable")

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if params:
            separator = "&" if "?" in path else "?"
            path = f"{path}{separator}{urllib.parse.urlencode(params, doseq=True)}"
        body, _ = self._request(path)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise BuildkiteError(f"Buildkite public web returned invalid JSON for {path.partition('?')[0]}") from exc

    def get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        if params:
            separator = "&" if "?" in path else "?"
            path = f"{path}{separator}{urllib.parse.urlencode(params, doseq=True)}"
        body, _ = self._request(path)
        return body.decode("utf-8", errors="replace")


def _flatten_steps(steps: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for raw in steps:
        if not isinstance(raw, dict):
            continue
        nested = raw.get("steps")
        if isinstance(nested, list):
            yield from _flatten_steps(nested)
        else:
            yield raw


def _pytest_command(step: dict[str, Any]) -> str:
    commands = step.get("commands") or []
    if isinstance(commands, str):
        commands = [commands]
    for block in commands:
        for line in str(block).splitlines():
            if "pytest" not in line or line.lstrip().startswith("#"):
                continue
            return line[line.index("pytest") :].strip().rstrip(";")
    return ""


def _slug(text: str) -> str:
    text = re.sub(r":[a-zA-Z0-9_+-]+:", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text


def _gpu_count(step: dict[str, Any]) -> int:
    mirror = str(step.get("mirror_hardwares") or "")
    match = re.fullmatch(r"h100_(\d+)", mirror)
    return int(match.group(1)) if match else 0


def _nightly_jobs_from_data(data: dict[str, Any]) -> list[NightlyJob]:
    selected: list[NightlyJob] = []
    for step in _flatten_steps(data.get("steps") or []):
        label = str(step.get("label") or "")
        command = _pytest_command(step)
        is_function = FUNCTION_LABEL in label or (
            re.search(r"(?:^|\s)tests/e2e/?(?:\s|$)", command) is not None
            and "full_model and H100 and omni" in command
            and "tests/e2e/accuracy" not in command
        )
        is_accuracy = (
            ACCURACY_LABEL in label or "tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py" in command
        ) and QWEN_TOKEN in command
        is_perf = "run_benchmark.py" in command and "tests/dfx/perf/tests/test_qwen3_omni_" in command
        is_multi = "tests/e2e/online_serving/test_qwen3_omni_multi_replicas.py" in command and not is_perf
        if not (is_function or is_accuracy or is_perf or is_multi):
            continue
        if is_function:
            category = "function"
        elif is_accuracy:
            category = "accuracy"
        elif is_perf:
            category = "performance"
        else:
            category = "multi-replica-function"
        config_match = re.search(r"--test-config-file(?:=|\s+)(\S+)", command)
        selected.append(
            NightlyJob(
                label=label,
                key=str(step.get("key") or f"label-{_slug(label)}"),
                category=category,
                gpu_count=_gpu_count(step),
                timeout_minutes=int(step["timeout_in_minutes"]) if step.get("timeout_in_minutes") else None,
                pytest_command=command,
                perf_config=config_match.group(1) if config_match else None,
            )
        )
    return selected


def load_nightly_jobs(yaml_path: Path) -> list[NightlyJob]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return _nightly_jobs_from_data(data)


def load_nightly_jobs_at_commit(repo_root: Path, commit: str) -> list[NightlyJob] | None:
    proc = subprocess.run(
        ["git", "show", f"{commit}:.buildkite/cuda/test-nightly.yml"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    data = yaml.safe_load(proc.stdout) or {}
    return _nightly_jobs_from_data(data)


def _collect_argv(command: str) -> list[str]:
    parts = shlex.split(command)
    try:
        pytest_index = parts.index("pytest")
    except ValueError as exc:
        raise ValueError(f"No pytest executable in command: {command}") from exc
    args = parts[pytest_index + 1 :]
    args = [arg for arg in args if arg not in {"-s", "-v", "-sv", "-vs", "-vv"}]
    return [sys.executable, "-m", "pytest", "--collect-only", "-q", *args]


def _run_collection(repo_root: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=repo_root, text=True, capture_output=True, check=False)


def _qwen_e2e_targets(repo_root: Path, argv: list[str]) -> list[str]:
    ignored: list[str] = []
    for index, arg in enumerate(argv):
        if arg.startswith("--ignore="):
            ignored.append(arg.split("=", 1)[1].rstrip("/"))
        elif arg == "--ignore" and index + 1 < len(argv):
            ignored.append(argv[index + 1].rstrip("/"))
    targets = []
    for path in sorted((repo_root / "tests" / "e2e").rglob("test_*.py")):
        relative = path.relative_to(repo_root).as_posix()
        if any(relative == item or relative.startswith(item + "/") for item in ignored):
            continue
        source = path.read_text(encoding="utf-8")
        if QWEN_TOKEN in relative.lower() or "Qwen/Qwen3-Omni" in source:
            targets.append(relative)
    return targets


def collect_node_ids(repo_root: Path, job: NightlyJob) -> tuple[list[str], str | None]:
    argv = _collect_argv(job.pytest_command)
    proc = _run_collection(repo_root, argv)
    warning = None
    if proc.returncode not in {0, 5} and job.category == "function":
        targets = _qwen_e2e_targets(repo_root, argv)
        narrowed = []
        for arg in argv:
            if arg.rstrip("/") == "tests/e2e":
                narrowed.extend(targets)
            else:
                narrowed.append(arg)
        broad_errors = [
            line.strip() for line in (proc.stdout + "\\n" + proc.stderr).splitlines() if line.startswith("ERROR")
        ]
        reason = broad_errors[-1] if broad_errors else f"exit {proc.returncode}"
        proc = _run_collection(repo_root, narrowed)
        warning = (
            f"Broad tests/e2e collection failed ({reason}); used model-reference fallback with {len(targets)} files"
        )
    if proc.returncode not in {0, 5}:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        tail = detail[-1] if detail else "unknown collection error"
        raise RuntimeError(f"pytest collection failed for {job.label}: {tail}")
    nodes = [line.strip() for line in proc.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    if job.category == "function":
        nodes = [node for node in nodes if QWEN_TOKEN in node.lower()]
    return nodes, warning


def _function_ast(
    repo_root: Path, node_id: str
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, dict[str, Any]]:
    path_text, rest = node_id.split("::", 1)
    function_name = rest.split("[", 1)[0]
    path = repo_root / path_text
    if not path.exists() or path.suffix != ".py":
        return None, {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants: dict[str, Any] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == function_name:
            return statement, constants
    return None, constants


def _eval_request_num(node: ast.AST, constants: dict[str, Any]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and isinstance(constants.get(node.id), int):
        return int(constants[node.id])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_max_batch_size":
        if not node.args:
            return 5
        try:
            size = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            return None
        return {"few": 5, "medium": 100, "large": 256}.get(size)
    return None


def infer_request_count(repo_root: Path, node_id: str) -> int | None:
    function, constants = _function_ast(repo_root, node_id)
    if function is None:
        return None
    counts: list[int] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"send_omni_request", "send_completions_http_request"}:
            continue
        request_num = 1
        for keyword in node.keywords:
            if keyword.arg == "request_num":
                parsed = _eval_request_num(keyword.value, constants)
                if parsed is None:
                    return None
                request_num = parsed
                break
        counts.append(request_num)
    return sum(counts) if counts else None


def _parameter_id(node_id: str) -> str:
    match = re.search(r"\[([^]]+)\]$", node_id)
    return match.group(1) if match else "unparameterized"


def _function_name(node_id: str) -> str:
    return node_id.split("::", 1)[-1].split("[", 1)[0]


def _coverage_for_node(repo_root: Path, node_id: str, job: NightlyJob) -> tuple[dict[str, Any], set[str]]:
    function_name = _function_name(node_id)
    lower = function_name.lower()
    param = _parameter_id(node_id)
    source_path = node_id.split("::", 1)[0]
    function, _ = _function_ast(repo_root, node_id)
    source = ast.get_source_segment((repo_root / source_path).read_text(encoding="utf-8"), function) if function else ""
    source = source or ""

    interface = "offline" if "/offline_inference/" in source_path else "online"
    dimensions: dict[str, Any] = {"category": job.category, "parameter": param, "interface": interface}
    capabilities: set[str] = {job.category, f"{interface}-inference"}
    if job.category == "accuracy":
        capabilities.add("accuracy-gate")
        if "daily_omni" in lower:
            capabilities.update({"video-input", "text-output", "daily-omni"})
        if "seed_tts" in lower:
            capabilities.update({"text-input", "audio-output", "wer"})
    if "multi_replica" in lower or "multi_replicas" in source_path:
        capabilities.add("multi-replica")
    if "prefix_cach" in lower:
        capabilities.add("prefix-cache")
    if any(token in lower for token in ("invalid", "rejected", "bad_values")):
        capabilities.add("error-path")
        dimensions["scenario"] = "invalid-input"
    if any(token in lower for token in ("long", "large", "default_loader_sampling")) or "LONG_" in source:
        capabilities.add("long-or-boundary-input")
    if "autoround" in source_path:
        capabilities.add("quantization:autoround-w4a16")
        dimensions["quantization"] = "autoround-w4a16"
    if "modelopt" in source_path:
        capabilities.add("quantization:modelopt-nvfp4-w4a4")
        dimensions["quantization"] = "modelopt-nvfp4-w4a4"
    if "speaker" in lower:
        capabilities.add("speaker-selection")
        dimensions["scenario"] = "speaker-selection"
        speaker = re.search(r'["\']speaker["\']\s*:\s*["\']([^"\']+)', source)
        if speaker:
            dimensions["speaker"] = speaker.group(1)
    if "language" in lower:
        capabilities.add("multilingual")
        dimensions["scenario"] = "language"
    if "one_word" in lower:
        capabilities.add("pronunciation-smoke")
        dimensions["scenario"] = "one-word-pronunciation"
    if "audio_in_video" in lower:
        capabilities.add("audio-in-video")
        dimensions["scenario"] = lower.removeprefix("test_")

    modalities: set[str] = {"text-input"}
    if "mix_to" in lower:
        modalities.update({"audio-input", "image-input", "video-input"})
    elif "audio_in_video" in lower:
        modalities.update({"audio-input", "video-input"})
    else:
        input_side = lower.removeprefix("test_").split("_to_", 1)[0]
        if "audio" in input_side:
            modalities.add("audio-input")
        if "image" in input_side:
            modalities.add("image-input")
        if "video" in input_side:
            modalities.add("video-input")

    explicit_text = '"modalities": ["text"]' in source or "'modalities': ['text']" in source
    explicit_audio = '"modalities": ["audio"]' in source or "'modalities': ['audio']" in source
    output_side = lower.split("_to_", 1)[1] if "_to_" in lower else ""
    if explicit_text:
        modalities.add("text-output")
    elif explicit_audio:
        modalities.add("audio-output")
    elif output_side:
        if "text" in output_side:
            modalities.add("text-output")
        if "audio" in output_side:
            modalities.add("audio-output")
    elif job.category != "accuracy":
        modalities.update({"text-output", "audio-output"})
    capabilities.update(modalities)

    stream_match = re.search(r"""["']stream["']\s*:\s*(True|False)""", source)
    if stream_match:
        dimensions["stream"] = stream_match.group(1) == "True"
        capabilities.add("streaming" if dimensions["stream"] else "non-streaming")
    if param == "async_chunk" or param.startswith("batch_token"):
        dimensions["async_chunk"] = True
        capabilities.add("async-chunk")
    elif param == "default" and "test_qwen3_omni_expansion.py" in source_path:
        dimensions["async_chunk"] = False
        capabilities.add("no-async-chunk")
    if param.startswith("batch_token"):
        dimensions["token_budget_variant"] = "max_num_batched_tokens=2048"
        capabilities.add("reduced-token-budget")
    dimensions["input_modalities"] = sorted(item for item in modalities if item.endswith("-input"))
    dimensions["output_modalities"] = sorted(item for item in modalities if item.endswith("-output"))
    return dimensions, capabilities


def _perf_request_count(params: dict[str, Any]) -> int | None:
    value = params.get("num_prompts")
    if isinstance(value, int):
        return value
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return sum(value)
    return None


def _perf_static_cases(repo_root: Path, job: NightlyJob, node_ids: list[str]) -> list[StaticCase]:
    if not job.perf_config:
        return []
    config_path = repo_root / job.perf_config
    configs = json.loads(config_path.read_text(encoding="utf-8"))
    flattened: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for config in configs:
        for index, params in enumerate(config.get("benchmark_params") or []):
            if params.get("enabled", True):
                flattened.append((config, params, index))
    cases: list[StaticCase] = []
    for ordinal, (config, params, param_index) in enumerate(flattened):
        case_id = (
            node_ids[ordinal] if ordinal < len(node_ids) else f"{job.perf_config}::{config['test_name']}[{param_index}]"
        )
        server = config.get("server_params") or {}
        extra_body = params.get("extra_body") or {}
        mm_limits = params.get("random_mm_limit_mm_per_prompt") or {}
        mode = "async" if "--async-chunk" in (server.get("extra_cli_args") or []) else "sync"
        if server.get("use_omni") is False:
            mode = "vllm-text"
        capabilities = {"performance-regression", f"backend:{params.get('backend', 'unknown')}", f"mode:{mode}"}
        for modality in mm_limits:
            capabilities.add(f"{modality}-input")
        output_modalities = extra_body.get("modalities") or ["text", "audio"]
        for modality in output_modalities:
            capabilities.add(f"{modality}-output")
        if server.get("stage_overrides"):
            capabilities.add("multi-replica")
        if params.get("max_concurrency"):
            capabilities.add("concurrency-sweep")
        if params.get("request_rate"):
            capabilities.add("request-rate-sweep")
        dimensions = {
            "category": job.category,
            "server_mode": mode,
            "dataset": params.get("dataset_name"),
            "backend": params.get("backend"),
            "endpoint": params.get("endpoint"),
            "benchmark_index": param_index,
            "max_concurrency": params.get("max_concurrency"),
            "request_rate": params.get("request_rate"),
            "input_modalities": sorted(mm_limits),
            "output_modalities": output_modalities,
        }
        cases.append(
            StaticCase(
                case_id=case_id,
                job_key=job.key,
                job_label=job.label,
                category=job.category,
                gpu_count=job.gpu_count,
                dimensions=dimensions,
                capabilities=capabilities,
                request_count=_perf_request_count(params),
                startup_key=f"{job.key}:{config.get('test_name')}",
                source_path=job.perf_config,
            )
        )
    return cases


def build_static_cases(repo_root: Path, jobs: list[NightlyJob]) -> tuple[list[StaticCase], list[str]]:
    cases: list[StaticCase] = []
    warnings: list[str] = []
    for job in jobs:
        try:
            node_ids, collect_warning = collect_node_ids(repo_root, job)
            if collect_warning:
                warnings.append(collect_warning)
        except RuntimeError as exc:
            warnings.append(str(exc))
            node_ids = []
        if job.category == "performance":
            cases.extend(_perf_static_cases(repo_root, job, node_ids))
            continue
        for node_id in node_ids:
            dimensions, capabilities = _coverage_for_node(repo_root, node_id, job)
            parameter = _parameter_id(node_id)
            source_path = node_id.split("::", 1)[0]
            startup_key = f"{job.key}:{source_path}:{parameter}"
            cases.append(
                StaticCase(
                    case_id=node_id,
                    job_key=job.key,
                    job_label=job.label,
                    category=job.category,
                    gpu_count=job.gpu_count,
                    dimensions=dimensions,
                    capabilities=capabilities,
                    request_count=infer_request_count(repo_root, node_id),
                    startup_key=startup_key,
                    source_path=source_path,
                )
            )
    _annotate_static_relationships(cases)
    return cases, warnings


def _static_signature(case: StaticCase) -> str:
    payload = {
        "category": case.category,
        "dimensions": case.dimensions,
        "capabilities": sorted(case.capabilities),
        "requests": case.request_count,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _annotate_static_relationships(cases: list[StaticCase]) -> None:
    by_signature: dict[str, list[StaticCase]] = defaultdict(list)
    for case in cases:
        case.static_signature = _static_signature(case)
        by_signature[case.static_signature].append(case)
    for group in by_signature.values():
        if len(group) > 1:
            ids = [item.case_id for item in group]
            for case in group:
                case.overlap_hint = "; ".join(item for item in ids if item != case.case_id)
    for case in cases:
        if case.overlap_hint:
            continue
        if "multi-replica" in case.capabilities:
            case.overlap_hint = (
                "multi-replica functional/performance jobs overlap in topology, but assert different risks"
            )
        elif "wer" in case.capabilities:
            case.overlap_hint = "one-word pronunciation smoke overlaps in speech quality, but is not a WER replacement"
        elif case.dimensions.get("server_mode") == "vllm-text":
            case.overlap_hint = "async perf has text-output scenarios, but uses the Omni serving backend"


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_seconds(started: str | None, finished: str | None) -> float | None:
    start = _parse_time(started)
    finish = _parse_time(finished)
    if not start or not finish:
        return None
    return max(0.0, (finish - start).total_seconds())


def _normalize_label(text: str) -> str:
    text = re.sub(r":[a-zA-Z0-9_+-]+:", " ", text)
    text = "".join(ch if ch.isascii() else (ch if ch in "·" else " ") for ch in text)
    return re.sub(r"\s+", " ", text).strip().lower()


def match_job(job_data: dict[str, Any], jobs: list[NightlyJob]) -> NightlyJob | None:
    step_key = str(job_data.get("step_key") or "")
    if step_key:
        for definition in jobs:
            if definition.key == step_key:
                return definition
    name = _normalize_label(str(job_data.get("name") or job_data.get("label") or ""))
    for definition in jobs:
        target = _normalize_label(definition.label)
        if target and (target in name or name in target):
            return definition
    return None


def classify_build(build: dict[str, Any], *, branch: str) -> str:
    env = build.get("env") or {}
    if build.get("branch") != branch or build.get("source") != "schedule" or build.get("pull_request"):
        return "excluded"
    if str(env.get("WEEKLY", "")) == "1":
        return "excluded_weekly"
    if str(env.get("NIGHTLY", "")) == "1":
        return "scheduled_nightly"
    return "schedule_unresolved"


def is_scheduled_nightly(build: dict[str, Any], *, branch: str) -> bool:
    return classify_build(build, branch=branch) == "scheduled_nightly"


def _next_link(headers: Any) -> str | None:
    value = headers.get("Link") if headers else None
    if not value:
        return None
    for part in str(value).split(","):
        match = re.match(r'\s*<([^>]+)>;\s*rel="?next"?', part)
        if match:
            return match.group(1)
    return None


def _list_classic_pages(client: BuildkiteClient, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_path: str | None = path
    next_params: dict[str, Any] | None = {**params, "per_page": 100}
    while next_path:
        payload, headers = client.get_json(next_path, next_params)
        if not isinstance(payload, list):
            raise BuildkiteError(f"Expected a list from {urllib.parse.urlsplit(next_path).path}")
        rows.extend(item for item in payload if isinstance(item, dict))
        next_path = _next_link(headers)
        next_params = None
    return rows


def _list_job_attempts(client: BuildkiteClient, build_number: int) -> list[dict[str, Any]]:
    path: str | None = f"{client.pipeline_path}/builds/{build_number}/jobs"
    params: dict[str, Any] | None = {
        "group_key": "nightly-omni-test-group",
        "include_retried_jobs": "true",
        "per_page": 100,
    }
    rows: list[dict[str, Any]] = []
    while path:
        payload, _ = client.get_json(path, params)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise BuildkiteError("Buildkite jobs endpoint returned an unexpected cursor page")
        rows.extend(item for item in payload["items"] if isinstance(item, dict))
        links = payload.get("links") or {}
        path = links.get("next")
        params = None
    return rows


def _artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": artifact.get("id"),
        "job_id": artifact.get("job_id"),
        "state": artifact.get("state"),
        "filename": artifact.get("filename"),
        "path": artifact.get("path"),
        "mime_type": artifact.get("mime_type"),
        "file_size": artifact.get("file_size"),
        "sha1sum": artifact.get("sha1sum"),
        "download_url": artifact.get("download_url"),
    }


def _job_artifacts(client: BuildkiteClient, job_id: str) -> list[dict[str, Any]]:
    org = urllib.parse.quote(client.organization, safe="")
    rows = _list_classic_pages(client, f"/organizations/{org}/jobs/{job_id}/artifacts", {})
    return [_artifact_metadata(row) for row in rows]


def _normalize_log(log: str) -> str:
    normalized = BUILDKITE_ESCAPE_RE.sub("", log)
    normalized = ANSI_ESCAPE_RE.sub("", normalized).replace("\r\n", "\n").replace("\r", "\n")
    return CONTROL_CHARACTER_RE.sub("", normalized)


def _case_results(log: str, *, fallback_status: str | None = None) -> list[tuple[str, str]]:
    normalized = _normalize_log(log)
    results = [tuple(match.groups()) for match in CASE_RESULT_RE.finditer(normalized)]
    explicit_ids = {case_id for case_id, _ in results}
    failed_ids = set(FAILED_NODE_RE.findall(normalized))
    seen: set[str] = set()
    for match in NODE_ID_RE.finditer(normalized):
        case_id = match.group(0)
        if case_id in seen or case_id in explicit_ids:
            continue
        seen.add(case_id)
        if case_id in failed_ids:
            results.append((case_id, "FAILED"))
        elif fallback_status:
            results.append((case_id, fallback_status))
    return results


def _schedule_metadata(client: BuildkiteClient) -> list[dict[str, Any]]:
    schedules = _list_classic_pages(client, f"{client.pipeline_path}/schedules", {})
    out = []
    for schedule in schedules:
        env = schedule.get("env") or {}
        out.append(
            {
                "id": schedule.get("id"),
                "label": schedule.get("label"),
                "branch": schedule.get("branch"),
                "cronline": schedule.get("cronline"),
                "message": schedule.get("message"),
                "enabled": schedule.get("enabled"),
                "nightly": str(env.get("NIGHTLY", "")) == "1",
                "weekly": str(env.get("WEEKLY", "")) == "1",
            }
        )
    return out


def fetch_history(
    client: BuildkiteClient,
    jobs: list[NightlyJob],
    *,
    repo_root: Path,
    branch: str,
    max_builds: int,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    pages_scanned = 0
    page = 1
    schedules = _schedule_metadata(client)
    historical_jobs_cache: dict[str, list[NightlyJob] | None] = {}
    while len(accepted) < max_builds:
        builds, _ = client.get_json(
            f"{client.pipeline_path}/builds",
            {
                "branch": branch,
                "exclude_jobs": "true",
                "exclude_pipeline": "true",
                "per_page": 100,
                "page": page,
            },
        )
        pages_scanned += 1
        if not isinstance(builds, list) or not builds:
            break
        for build in builds:
            classification = classify_build(build, branch=branch)
            if classification == "schedule_unresolved":
                unresolved.append(
                    {
                        "number": build.get("number"),
                        "id": build.get("id"),
                        "commit": build.get("commit"),
                        "reason": "scheduled main build lacks an unambiguous NIGHTLY=1 environment",
                    }
                )
                continue
            if classification != "scheduled_nightly":
                continue
            number = int(build["number"])
            commit = str(build.get("commit") or "")
            if commit not in historical_jobs_cache:
                historical_jobs_cache[commit] = load_nightly_jobs_at_commit(repo_root, commit) if commit else None
            historical_jobs = historical_jobs_cache[commit]
            definitions = historical_jobs or jobs
            gpu_source = f"git:{commit}:.buildkite/cuda/test-nightly.yml" if historical_jobs else "unknown"
            normalized_jobs: list[dict[str, Any]] = []
            for raw_job in _list_job_attempts(client, number):
                definition = match_job(raw_job, definitions)
                if definition is None or raw_job.get("type", "script") != "script":
                    continue
                canonical = next((item for item in jobs if item.key == definition.key), None)
                if canonical is None:
                    canonical = next((item for item in jobs if item.category == definition.category), definition)
                job_id = str(raw_job.get("id") or "")
                log = ""
                artifacts: list[dict[str, Any]] = []
                if job_id:
                    org = urllib.parse.quote(client.organization, safe="")
                    log = _normalize_log(client.get_text(f"/organizations/{org}/jobs/{job_id}/log.txt"))
                    artifacts = _job_artifacts(client, job_id)
                normalized_jobs.append(
                    {
                        "job_key": canonical.key,
                        "job_id": job_id,
                        "state": raw_job.get("state"),
                        "soft_failed": bool(raw_job.get("soft_failed")),
                        "exit_status": raw_job.get("exit_status"),
                        "signal": raw_job.get("signal"),
                        "broken_reason": raw_job.get("broken_reason"),
                        "started_at": raw_job.get("started_at"),
                        "finished_at": raw_job.get("finished_at"),
                        "retried": bool(raw_job.get("retried")),
                        "retried_in_job_id": raw_job.get("retried_in_job_id"),
                        "retry_count": int(raw_job.get("retries_count") or 0),
                        "retry_source": raw_job.get("retry_source"),
                        "gpu_count": definition.gpu_count if historical_jobs else None,
                        "gpu_count_source": gpu_source,
                        "log": log,
                        "case_results": _case_results(log),
                        "failure_signature": _failure_signature(log)
                        if str(raw_job.get("state")) in TERMINAL_FAILURE_STATES
                        else "",
                        "artifacts": artifacts,
                    }
                )
            accepted.append(
                {
                    "number": number,
                    "id": build.get("id"),
                    "commit": commit,
                    "source": build.get("source"),
                    "state": build.get("state"),
                    "created_at": build.get("created_at"),
                    "started_at": build.get("started_at"),
                    "finished_at": build.get("finished_at"),
                    "rebuilt_from": build.get("rebuilt_from"),
                    "classification": classification,
                    "jobs": normalized_jobs,
                }
            )
            if len(accepted) >= max_builds:
                break
        if len(builds) < 100:
            break
        page += 1
        if page > 20:
            break
    return {
        "schema_version": 1,
        "organization": client.organization,
        "pipeline": client.pipeline,
        "branch": branch,
        "selection": "branch=main, source=schedule, env.NIGHTLY=1, env.WEEKLY!=1, no pull request",
        "candidate_pages_scanned": pages_scanned,
        "schedule_definitions": schedules,
        "schedule_unresolved": unresolved,
        "builds": accepted,
    }


def _public_nightly_build_numbers(
    client: PublicBuildkiteClient, *, branch: str, max_builds: int
) -> tuple[list[int], int]:
    numbers: list[int] = []
    seen: set[int] = set()
    pages_scanned = 0
    page = 1
    build_link = re.compile(rf'href="{re.escape(client.pipeline_path)}/builds/(\d+)"')
    while len(numbers) < max_builds:
        html = client.get_text(
            f"{client.pipeline_path}/builds",
            {"branch": branch, "query": "Scheduled nightly build", "page": page},
        )
        pages_scanned += 1
        page_numbers = [int(value) for value in build_link.findall(html)]
        for number in page_numbers:
            if number not in seen:
                seen.add(number)
                numbers.append(number)
                if len(numbers) >= max_builds:
                    break
        if not page_numbers or 'rel="next"' not in html:
            break
        page += 1
        if page > 20:
            break
    return numbers, pages_scanned


def classify_public_build(build: dict[str, Any], *, branch: str) -> str:
    if build.get("branch_name") != branch or build.get("source") != "schedule" or build.get("pull_request"):
        return "excluded"
    if str(build.get("message") or "").strip().lower() == "scheduled nightly build":
        return "public_scheduled_nightly"
    return "schedule_unresolved"


def _public_job_state(job: dict[str, Any]) -> str:
    state = str(job.get("state") or "")
    if job.get("timed_out_at"):
        return "timed_out"
    if job.get("canceled_at"):
        return "canceled"
    if state in {"finished", "broken"}:
        if job.get("passed") is True or job.get("exit_status") == 0:
            return "passed"
        return "failed"
    return state


def _public_artifact_metadata(artifact: dict[str, Any], job_id: str) -> dict[str, Any]:
    return {
        "id": artifact.get("id"),
        "job_id": job_id,
        "state": artifact.get("state"),
        "filename": artifact.get("file_name"),
        "path": artifact.get("path"),
        "mime_type": artifact.get("mime_type"),
        "file_size": artifact.get("file_size"),
        "sha1sum": artifact.get("sha1sum"),
        "sha256sum": artifact.get("sha256sum"),
    }


def _public_job_details(
    client: PublicBuildkiteClient, raw_job: dict[str, Any]
) -> tuple[str, list[dict[str, Any]], list[str]]:
    base_path = str(raw_job.get("base_path") or "")
    job_id = str(raw_job.get("id") or "")
    if not base_path:
        return "", [], [f"job {job_id or 'unknown'} has no public base_path"]
    warnings: list[str] = []
    log = ""
    artifacts: list[dict[str, Any]] = []
    try:
        log = _normalize_log(client.get_text(f"{base_path}/download.txt"))
    except BuildkiteError as exc:
        warnings.append(f"job {job_id}: log unavailable ({exc})")
    try:
        payload = client.get_json(f"{base_path}/artifacts")
        if isinstance(payload, list):
            artifacts = [_public_artifact_metadata(item, job_id) for item in payload if isinstance(item, dict)]
        else:
            warnings.append(f"job {job_id}: artifact endpoint returned an unexpected payload")
    except BuildkiteError as exc:
        warnings.append(f"job {job_id}: artifacts unavailable ({exc})")
    return log, artifacts, warnings


def _historical_job_definitions(
    client: PublicBuildkiteClient,
    repo_root: Path,
    commit: str,
) -> tuple[list[NightlyJob] | None, str]:
    local = load_nightly_jobs_at_commit(repo_root, commit)
    if local is not None:
        return local, f"git:{commit}:.buildkite/cuda/test-nightly.yml"
    raw_url = (
        "https://raw.githubusercontent.com/vllm-project/vllm-omni/"
        f"{urllib.parse.quote(commit, safe='')}/.buildkite/cuda/test-nightly.yml"
    )
    try:
        data = yaml.safe_load(client.get_text(raw_url)) or {}
    except (BuildkiteError, yaml.YAMLError):
        return None, "unknown"
    return _nightly_jobs_from_data(data), f"github:{commit}:.buildkite/cuda/test-nightly.yml"


def fetch_public_history(
    client: PublicBuildkiteClient,
    jobs: list[NightlyJob],
    *,
    repo_root: Path,
    branch: str,
    max_builds: int,
    workers: int = 6,
) -> dict[str, Any]:
    # Public search results can contain UI rebuilds whose message still says
    # "Scheduled nightly build". Over-fetch candidates so those rebuilds (and
    # builds without matching Qwen3-Omni jobs) do not shrink the requested
    # number of valid nightly executions.
    candidate_limit = max_builds + max(20, max_builds)
    numbers, pages_scanned = _public_nightly_build_numbers(client, branch=branch, max_builds=candidate_limit)
    accepted: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    collection_warnings: list[str] = []
    historical_jobs_cache: dict[str, tuple[list[NightlyJob] | None, str]] = {}
    for index, number in enumerate(numbers, start=1):
        build = client.get_json(f"{client.pipeline_path}/builds/{number}.json")
        if not isinstance(build, dict):
            collection_warnings.append(f"build {number}: public JSON returned an unexpected payload")
            continue
        classification = classify_public_build(build, branch=branch)
        if classification != "public_scheduled_nightly":
            unresolved.append(
                {
                    "number": number,
                    "id": build.get("id"),
                    "commit": build.get("commit_id"),
                    "reason": "public build did not retain exact source/branch/nightly-message evidence",
                }
            )
            continue
        commit = str(build.get("commit_id") or "")
        if commit not in historical_jobs_cache:
            historical_jobs_cache[commit] = _historical_job_definitions(client, repo_root, commit)
        historical_jobs, gpu_source = historical_jobs_cache[commit]
        definitions = historical_jobs or jobs
        build_data_path = str(build.get("build_data_base_path") or "")
        jobs_payload = client.get_json(
            f"{build_data_path}/jobs",
            {"include_retried_jobs": "true", "paginate": "false"},
        )
        raw_jobs = jobs_payload.get("records") if isinstance(jobs_payload, dict) else None
        if not isinstance(raw_jobs, list):
            collection_warnings.append(f"build {number}: public jobs endpoint returned an unexpected payload")
            continue
        selected: list[tuple[dict[str, Any], NightlyJob, NightlyJob]] = []
        for raw_job in raw_jobs:
            if not isinstance(raw_job, dict) or raw_job.get("type", "script") != "script":
                continue
            definition = match_job(raw_job, definitions)
            if definition is None:
                continue
            canonical = next((item for item in jobs if item.key == definition.key), None)
            if canonical is None:
                canonical = next((item for item in jobs if item.category == definition.category), definition)
            selected.append((raw_job, definition, canonical))
        if not selected:
            unresolved.append(
                {
                    "number": number,
                    "id": build.get("id"),
                    "commit": commit,
                    "reason": "strict nightly build had no matching Qwen3-Omni jobs",
                }
            )
            continue
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            details = list(executor.map(lambda item: _public_job_details(client, item[0]), selected))
        normalized_jobs: list[dict[str, Any]] = []
        for (raw_job, definition, canonical), (log, artifacts, job_warnings) in zip(selected, details, strict=True):
            collection_warnings.extend(f"build {number}: {warning}" for warning in job_warnings)
            state = _public_job_state(raw_job)
            fallback_status = "PASSED" if state in SUCCESS_STATES and not raw_job.get("soft_failed") else None
            normalized_jobs.append(
                {
                    "job_key": canonical.key,
                    "job_id": str(raw_job.get("id") or ""),
                    "state": state,
                    "soft_failed": bool(raw_job.get("soft_failed")),
                    "exit_status": raw_job.get("exit_status"),
                    "signal": None,
                    "broken_reason": raw_job.get("stack_error_detail"),
                    "started_at": raw_job.get("started_at"),
                    "finished_at": raw_job.get("finished_at"),
                    "retried": bool(raw_job.get("retried_in_job_uuid")),
                    "retried_in_job_id": raw_job.get("retried_in_job_uuid"),
                    "retry_count": 0,
                    "retry_source": raw_job.get("retry_type"),
                    "gpu_count": definition.gpu_count if historical_jobs else None,
                    "gpu_count_source": gpu_source,
                    "log": log,
                    "case_results": _case_results(log, fallback_status=fallback_status),
                    "failure_signature": _failure_signature(log) if state in TERMINAL_FAILURE_STATES else "",
                    "artifacts": artifacts,
                }
            )
        accepted.append(
            {
                "number": number,
                "id": build.get("id"),
                "commit": commit,
                "source": build.get("source"),
                "state": build.get("state"),
                "created_at": build.get("created_at"),
                "started_at": build.get("started_at"),
                "finished_at": build.get("finished_at"),
                "rebuilt_from": None,
                "classification": classification,
                "jobs": normalized_jobs,
            }
        )
        print(
            f"Public Buildkite history: {len(accepted)}/{max_builds} accepted "
            f"(candidate {index}) build #{number}, "
            f"{len(normalized_jobs)} Qwen3-Omni attempts",
            file=sys.stderr,
            flush=True,
        )
        if len(accepted) >= max_builds:
            break
    return {
        "schema_version": 1,
        "collector": "public_web",
        "organization": client.organization,
        "pipeline": client.pipeline,
        "branch": branch,
        "selection": (
            "public page query plus branch=main, source=schedule, exact message='Scheduled nightly build', "
            "no pull request; build env and schedule definitions are not public"
        ),
        "candidate_pages_scanned": pages_scanned,
        "schedule_definitions": [],
        "schedule_unresolved": unresolved,
        "collection_warnings": collection_warnings,
        "builds": accepted,
    }


def load_history(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"schema_version": 1, "builds": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("builds"), list):
        raise ValueError("history JSON must be an object containing a builds list")
    return payload


def attempts_from_history(history: dict[str, Any]) -> list[Attempt]:
    attempts: list[Attempt] = []
    for build in history.get("builds") or []:
        number = int(build["number"])
        for job in build.get("jobs") or []:
            started = job.get("started_at")
            finished = job.get("finished_at")
            attempts.append(
                Attempt(
                    build_number=number,
                    build_commit=str(build.get("commit") or ""),
                    job_key=str(job["job_key"]),
                    job_id=str(job.get("job_id") or ""),
                    state=str(job.get("state") or ""),
                    started_at=started,
                    finished_at=finished,
                    duration_seconds=_duration_seconds(started, finished),
                    retried=bool(job.get("retried")),
                    retry_count=int(job.get("retry_count") or 0),
                    gpu_count=int(job["gpu_count"]) if job.get("gpu_count") is not None else None,
                    soft_failed=bool(job.get("soft_failed")),
                    log=str(job.get("log") or ""),
                    case_results=[tuple(item) for item in job.get("case_results") or []],
                    failure_signature=str(job.get("failure_signature") or ""),
                    artifacts=list(job.get("artifacts") or []),
                )
            )
    return attempts


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _failure_signature(log: str) -> str:
    log = _normalize_log(log)
    candidates = re.findall(
        r"(?m)^(?:E\s+|FAILED\s+|.*(?:CUDA out of memory|TimeoutError|AssertionError).*)[^\n]*", log
    )
    if not candidates:
        return "job failed without a parsed pytest signature"
    value = re.sub(r"\s+", " ", candidates[-1]).strip()
    value = re.sub(r"0x[0-9a-fA-F]+", "0x…", value)
    return value[:240]


def summarize_jobs(attempts: list[Attempt], jobs: list[NightlyJob]) -> dict[str, JobSummary]:
    del jobs  # GPU count comes from each historical commit, never from the current YAML.
    grouped: dict[tuple[int, str], list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[(attempt.build_number, attempt.job_key)].append(attempt)
    by_job: dict[str, list[list[Attempt]]] = defaultdict(list)
    for (_, job_key), group in grouped.items():
        by_job[job_key].append(group)
    summaries: dict[str, JobSummary] = {}
    for key, build_groups in by_job.items():
        summary = JobSummary(executions=len(build_groups))
        durations: list[float] = []
        signatures: set[str] = set()
        for group in build_groups:
            ordered = sorted(group, key=lambda item: item.finished_at or item.started_at or "")
            final = next((item for item in reversed(ordered) if not item.retried), ordered[-1])
            if final.duration_seconds is not None:
                durations.append(final.duration_seconds)
            if final.soft_failed:
                summary.soft_failures += 1
            elif final.state in SUCCESS_STATES:
                summary.successes += 1
            elif final.state in TERMINAL_FAILURE_STATES:
                summary.failures += 1
                signatures.add(final.failure_signature or _failure_signature(final.log))
            retries = max(0, len(group) - 1)
            summary.retries += retries
            if (
                retries
                and final.state in SUCCESS_STATES
                and any(item.state in TERMINAL_FAILURE_STATES for item in group[:-1])
            ):
                summary.flakes += 1
            for item in group:
                if item.gpu_count is None:
                    summary.unknown_gpu_attempts += 1
                    continue
                seconds = item.duration_seconds or 0.0
                cost = seconds * item.gpu_count / 3600
                summary.gpu_hours += cost
                if item is not final:
                    summary.retry_gpu_hours += cost
            if final.gpu_count is not None and final.duration_seconds is None:
                summary.unknown_gpu_attempts += 1
        summary.p50_seconds = statistics.median(durations) if durations else None
        summary.p95_seconds = _percentile(durations, 0.95)
        summary.failure_signatures = sorted(signatures)
        summaries[key] = summary
    return summaries


def summarize_cases(attempts: list[Attempt], static_cases: list[StaticCase]) -> dict[str, CaseSummary]:
    known = {case.case_id for case in static_cases}
    events_by_build: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for attempt in attempts:
        events = attempt.case_results or _case_results(attempt.log)
        for case_id, result in events:
            if case_id in known:
                events_by_build[attempt.build_number].append((case_id, result))
    summaries = {case_id: CaseSummary() for case_id in known}
    for build_number, events in events_by_build.items():
        by_case: dict[str, list[str]] = defaultdict(list)
        for case_id, result in events:
            by_case[case_id].append(result)
        failed_cases: set[str] = set()
        for case_id, results in by_case.items():
            summary = summaries[case_id]
            summary.executions += 1
            if any(result in {"FAILED", "ERROR"} for result in results):
                summary.failures += 1
                summary.failure_builds.add(build_number)
                failed_cases.add(case_id)
            if all(result in {"SKIPPED", "XFAIL"} for result in results):
                summary.skips += 1
            if any(result in {"FAILED", "ERROR"} for result in results[:-1]) and results[-1] == "PASSED":
                summary.flakes += 1
        if len(failed_cases) == 1:
            summaries[next(iter(failed_cases))].independent_failures += 1
    return summaries


def failure_overlap(case: StaticCase, summaries: dict[str, CaseSummary]) -> str:
    own = summaries.get(case.case_id, CaseSummary()).failure_builds
    if not own:
        return case.overlap_hint
    best_id = ""
    best_score = 0.0
    for other_id, other in summaries.items():
        if other_id == case.case_id or not other.failure_builds:
            continue
        union = own | other.failure_builds
        score = len(own & other.failure_builds) / len(union)
        if score > best_score:
            best_id, best_score = other_id, score
    if best_score >= 0.8:
        return f"{best_id} (failure Jaccard={best_score:.2f})"
    return case.overlap_hint


def recommendation(case: StaticCase, summary: CaseSummary, duplicates: list[StaticCase]) -> tuple[str, str]:
    unique_risk = bool(
        case.capabilities
        & {"accuracy-gate", "performance-regression", "multi-replica", "async-chunk", "no-async-chunk", "error-path"}
    )
    if summary.executions < 20:
        if unique_risk:
            return "保留", "中：历史样本不足，但覆盖独立风险路径"
        return "证据不足", "未知：少于 20 次有效历史执行"
    if summary.independent_failures:
        return "保留", "低：历史上发现过独立故障"
    if len(duplicates) > 1:
        canonical = sorted(duplicates, key=lambda item: item.case_id)[0]
        if case.case_id != canonical.case_id:
            return "合并候选", "中：静态覆盖等价且无独立故障；灰度后再删除"
    if unique_risk:
        return "保留", "低：覆盖独立配置或质量维度"
    return "证据不足", "中：无独立故障，但缺少可证明的替代覆盖"


CSV_FIELDS = [
    "case_id",
    "nightly_job",
    "配置维度",
    "覆盖能力",
    "执行次数",
    "P50时长秒_job级",
    "P95时长秒_job级",
    "GPU数",
    "GPU-hours_job累计",
    "失败数",
    "重试数_job级",
    "flake率",
    "独立故障数",
    "重合案例",
    "建议",
    "风险",
    "预计节省GPU-hours",
    "请求数_静态",
    "模型启动组",
]


def build_rows(
    cases: list[StaticCase],
    job_summaries: dict[str, JobSummary],
    case_summaries: dict[str, CaseSummary],
) -> list[dict[str, Any]]:
    duplicate_groups: dict[str, list[StaticCase]] = defaultdict(list)
    for case in cases:
        duplicate_groups[case.static_signature].append(case)
    rows: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: (item.job_label, item.case_id)):
        case_summary = case_summaries.get(case.case_id, CaseSummary())
        job_summary = job_summaries.get(case.job_key, JobSummary())
        advice, risk = recommendation(case, case_summary, duplicate_groups[case.static_signature])
        flake_rate = case_summary.flakes / case_summary.executions if case_summary.executions else None
        rows.append(
            {
                "case_id": case.case_id,
                "nightly_job": case.job_label,
                "配置维度": json.dumps(case.dimensions, sort_keys=True, ensure_ascii=False),
                "覆盖能力": "; ".join(sorted(case.capabilities)),
                "执行次数": case_summary.executions,
                "P50时长秒_job级": _rounded(job_summary.p50_seconds),
                "P95时长秒_job级": _rounded(job_summary.p95_seconds),
                "GPU数": case.gpu_count,
                "GPU-hours_job累计": round(job_summary.gpu_hours, 3)
                if job_summary.executions and not job_summary.unknown_gpu_attempts
                else "",
                "失败数": case_summary.failures,
                "重试数_job级": job_summary.retries,
                "flake率": round(flake_rate, 4) if flake_rate is not None else "",
                "独立故障数": case_summary.independent_failures,
                "重合案例": failure_overlap(case, case_summaries),
                "建议": advice,
                "风险": risk,
                "预计节省GPU-hours": "" if advice not in {"合并候选", "移出 nightly", "删除"} else "需灰度计量",
                "请求数_静态": case.request_count if case.request_count is not None else "",
                "模型启动组": case.startup_key,
            }
        )
    return rows


def _rounded(value: float | None) -> float | str:
    return round(value, 2) if value is not None else ""


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds / 60:.1f} min"


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(
    path: Path,
    *,
    repo_root: Path,
    commit: str,
    jobs: list[NightlyJob],
    cases: list[StaticCase],
    rows: list[dict[str, Any]],
    history: dict[str, Any] | None,
    job_summaries: dict[str, JobSummary],
    warnings: list[str],
) -> None:
    by_job_cases: dict[str, list[StaticCase]] = defaultdict(list)
    for case in cases:
        by_job_cases[case.job_key].append(case)
    total_startups = len({case.startup_key for case in cases})
    total_requests = sum(case.request_count or 0 for case in cases)
    history_builds = len((history or {}).get("builds") or [])
    unresolved = len((history or {}).get("schedule_unresolved") or [])
    pages_scanned = (history or {}).get("candidate_pages_scanned")
    collector = str((history or {}).get("collector") or "rest_api")
    collector_label = "anonymous public web" if collector == "public_web" else "REST API"
    history_status = (
        f"已载入 {history_builds} 次符合条件的 scheduled nightly"
        if history_builds
        else "未载入；所有删除/移出结论均被抑制"
    )
    if history_builds:
        history_status += (
            f"；来源 {collector_label}；扫描 {pages_scanned or '未知'} 页，另有 {unresolved} 次 schedule_unresolved"
        )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if collector == "public_web":
        selection_note = (
            "- 匿名公开样本必须同时满足 `branch=main`、`source=schedule`、精确消息 "
            "`Scheduled nightly build` 且不是 PR；公开 build JSON 不暴露 env 或 schedule 定义。"
        )
        access_note = (
            "- job、重试、日志与 artifact 元数据来自 Buildkite 页面使用的匿名 web data 端点；"
            "不使用 token，端点属于前端实现，未来可能变化。"
        )
    else:
        selection_note = (
            "- REST API 样本必须同时满足 `branch=main`、`source=schedule`、`env.NIGHTLY=1`、"
            "`env.WEEKLY!=1` 且不是 PR；歧义样本记录为 `schedule_unresolved`。"
        )
        access_note = (
            "- API token 只从环境变量读取；规范化历史不保存环境变量、请求头、完整原始日志或短期 artifact 下载 URL。"
        )

    lines = [
        "# Qwen3-Omni Nightly Buildkite 冗余案例分析",
        "",
        f"> 生成时间：{generated}  ",
        f"> 基线：`{repo_root}` @ `{commit}` (`main`)  ",
        f"> 历史证据：{history_status}",
        "",
        "## 结论摘要",
        "",
    ]
    advice_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        advice_counts[str(row["建议"])] += 1
    if not history_builds:
        lines.append(
            "当前只完成了可复现的静态执行清单与覆盖矩阵。由于没有可用的 Buildkite 历史数据，"
            "没有案例满足“至少 20 次历史执行”的删除门槛，因此本报告不会给出删除结论。"
        )
    else:
        lines.append(
            f"分析了 {history_builds} 次 scheduled nightly；逐案建议统计为 "
            + "、".join(f"{key} {value}" for key, value in sorted(advice_counts.items()))
            + "。"
        )
    lines.extend(
        [
            "",
            f"静态展开得到 **{len(cases)} 个 pytest/perf 场景**、**{total_startups} 个模型启动组**，"
            f"并能静态确认至少 **{total_requests} 个请求**（accuracy 等动态数据集请求未估算）。",
            "",
            "## Nightly 执行图",
            "",
            (
                "| Buildkite job | 类型 | H100 | 场景数 | 启动组 | 请求数 | 成功率 | 失败 | 重试 | "
                "flake | P50 | P95 | GPU-hours | 重试 GPU-hours |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for job in jobs:
        job_cases = by_job_cases.get(job.key, [])
        summary = job_summaries.get(job.key, JobSummary())
        gpu_hours = "—"
        if summary.executions and not summary.unknown_gpu_attempts:
            gpu_hours = f"{summary.gpu_hours:.2f}"
        startup_count = len({case.startup_key for case in job_cases})
        request_count = sum(case.request_count or 0 for case in job_cases)
        success_rate = summary.successes / summary.executions if summary.executions else 0.0
        retry_gpu_hours = "—" if summary.unknown_gpu_attempts else f"{summary.retry_gpu_hours:.2f}"
        p50 = _fmt_duration(summary.p50_seconds)
        p95 = _fmt_duration(summary.p95_seconds)
        lines.append(
            f"| {_md_escape(job.label)} | {job.category} | {job.gpu_count} | {len(job_cases)} | "
            f"{startup_count} | {request_count} | {success_rate:.1%} | {summary.failures} | "
            f"{summary.retries} | {summary.flakes} | {p50} | {p95} | {gpu_hours} | {retry_gpu_hours} |"
        )
    lines.extend(
        [
            "",
            "耗时和 GPU-hours 均保持 **job 粒度**；未从日志获得 case 时间戳时不会平均摊派。",
            "",
            "## 冗余与相邻覆盖",
            "",
        ]
    )
    exact_groups: dict[str, list[StaticCase]] = defaultdict(list)
    for case in cases:
        exact_groups[case.static_signature].append(case)
    duplicates = [group for group in exact_groups.values() if len(group) > 1]
    if duplicates:
        lines.append("以下场景具有相同静态覆盖签名；只有达到历史门槛后才可进入合并灰度：")
        lines.append("")
        for group in duplicates:
            lines.append("- " + "; ".join(f"`{case.case_id}`" for case in group))
    else:
        lines.append("没有发现可仅凭静态证据认定的完全重复场景。")
    lines.extend(
        [
            "",
            "需要特别避免的误判：",
            "",
            "- sync 与 async-chunk 请求相似，但覆盖不同调度/流式数据路径。",
            "- vLLM text perf 与 Omni text-output perf 的 backend、endpoint 和启动方式不同。",
            "- multi-replica 功能测试验证路由与输出；multi-replica perf 验证扩展性能，不能互相替代。",
            "- one-word pronunciation 是随机性烟测；Seed-TTS WER 是数据集准确率门禁，两者只存在相邻覆盖。",
            "- `batch_token_64` 参数 ID 实际配置为 `max_num_batched_tokens=2048`，应单独修正命名，但不构成冗余。",
            "",
            "## 逐案建议",
            "",
            "完整字段见同目录 CSV。下表只展示需要关注的案例；`证据不足` 不等于建议删除。",
            "",
            "| Case | Job | 执行 | 独立故障 | 建议 | 风险/理由 |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    interesting = [row for row in rows if row["建议"] != "保留" or row["重合案例"]]
    if not interesting:
        interesting = rows[:20]
    for row in interesting:
        lines.append(
            f"| `{_md_escape(row['case_id'])}` | {_md_escape(row['nightly_job'])} | {row['执行次数']} | "
            f"{row['独立故障数']} | {row['建议']} | {_md_escape(row['风险'])} |"
        )
    lines.extend(
        [
            "",
            "## 优化基线与灰度验收",
            "",
        ]
    )
    removable = [row for row in rows if row["建议"] in {"移出 nightly", "删除"}]
    if removable:
        lines.append(f"有 {len(removable)} 个案例通过自动门槛；实际节省需在 7 次灰度中按 job 时间戳计量。")
    else:
        lines.append("当前没有案例通过自动删除门槛，优化后的安全基线与现状相同，预计节省为 0 GPU-hours。")
    if history_builds:
        total_gpu_hours = sum(summary.gpu_hours for summary in job_summaries.values())
        average_gpu_hours = total_gpu_hours / history_builds
        critical_p95 = max(
            (summary.p95_seconds or 0.0 for summary in job_summaries.values()),
            default=0.0,
        )
        after_value = "待 7 次灰度实测" if removable else None
        lines.extend(
            [
                "",
                "| 指标 | 当前 20 次基线 | 安全优化后 |",
                "|---|---:|---:|",
                f"| nightly job 数 | {len(jobs)} | {after_value or len(jobs)} |",
                f"| 模型启动组 | {total_startups} | {after_value or total_startups} |",
                f"| 累计 GPU-hours | {total_gpu_hours:.2f} | {after_value or f'{total_gpu_hours:.2f}'} |",
                f"| 平均 GPU-hours/nightly | {average_gpu_hours:.2f} | {after_value or f'{average_gpu_hours:.2f}'} |",
                f"| P95 关键路径 | {_fmt_duration(critical_p95)} | {after_value or _fmt_duration(critical_p95)} |",
            ]
        )
    lines.extend(
        [
            "",
            "灰度时保留原集合为对照，连续运行 7 次精简版 nightly。只有同时满足以下条件才落地：",
            "",
            "1. 历史回放仍能捕获全部已知独立故障。",
            "2. 关键覆盖维度至少保留一个 nightly 场景。",
            "3. 没有新增连续 flake。",
            "4. GPU-hours 或关键路径下降至少 15%。",
            "",
            "回滚方式是恢复候选案例原有 pytest target/参数化；本分析阶段不修改 nightly YAML。",
            "",
            "## 方法、限制与来源",
            "",
            "- 仅分析 `.buildkite/cuda/test-nightly.yml` 中指定的 Qwen3-Omni H100 功能、"
            "accuracy、perf 和 multi-replica job。",
            selection_note,
            access_note,
            "- 每次 attempt 的 GPU 数从该 build commit 的 nightly YAML 解析；历史提交不可用时 GPU-hours 留空。",
            "- 静态来源：[nightly YAML](../../../.buildkite/cuda/test-nightly.yml)、"
            "[Qwen3-Omni expansion tests](../../../tests/e2e/online_serving/test_qwen3_omni_expansion.py)、"
            "[perf configs](../../../tests/dfx/perf/tests/) 和 "
            "[分析工具](../../../tools/nightly/analyze_qwen3_omni_nightly.py)。",
            "- Buildkite API 的端点与字段依据见 [API 调研记录](qwen3_omni_nightly_buildkite_api_research.md)。",
        ]
    )
    if warnings:
        lines.extend(["", "### 收集告警", ""])
        lines.extend(f"- {_md_escape(item)}" for item in warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False
    )
    return proc.stdout.strip() or "unknown"


def _safe_history_cache(history: dict[str, Any]) -> dict[str, Any]:
    """Drop logs and signed artifact URLs before persisting normalized history."""
    safe = {key: value for key, value in history.items() if key != "builds"}
    safe_builds: list[dict[str, Any]] = []
    for build in history.get("builds") or []:
        safe_jobs = []
        for job in build.get("jobs") or []:
            clean = {key: value for key, value in job.items() if key not in {"log"}}
            clean["artifacts"] = [
                {key: value for key, value in artifact.items() if key != "download_url"}
                for artifact in job.get("artifacts") or []
            ]
            safe_jobs.append(clean)
        safe_builds.append({**{key: value for key, value in build.items() if key != "jobs"}, "jobs": safe_jobs})
    safe["builds"] = safe_builds
    return safe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--nightly-yaml", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--history-json", type=Path, help="Use normalized Buildkite history instead of the API")
    parser.add_argument("--history-cache-out", type=Path, help="Write sanitized metadata without logs or signed URLs")
    parser.add_argument("--max-builds", type=int, default=20)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--public-web", action="store_true", help="Use anonymous public Buildkite web data")
    parser.add_argument("--organization", default=os.getenv("BUILDKITE_ORGANIZATION_SLUG") or "vllm")
    parser.add_argument("--pipeline", default=os.getenv("BUILDKITE_PIPELINE_SLUG") or "vllm-omni")
    parser.add_argument("--public-workers", type=int, default=6)
    parser.add_argument("--static-only", action="store_true", help="Do not call Buildkite even when credentials exist")
    parser.add_argument("--strict-history", action="store_true", help="Fail instead of emitting a static-only report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    yaml_path = (args.nightly_yaml or repo_root / ".buildkite" / "cuda" / "test-nightly.yml").resolve()
    report_path = (args.report or repo_root / DEFAULT_REPORT.relative_to(REPO_ROOT)).resolve()
    csv_path = (args.csv or repo_root / DEFAULT_CSV.relative_to(REPO_ROOT)).resolve()
    jobs = load_nightly_jobs(yaml_path)
    cases, warnings = build_static_cases(repo_root, jobs)

    history: dict[str, Any] | None = None
    if args.history_json:
        history = load_history(args.history_json)
    elif not args.static_only:
        if args.public_web:
            client = PublicBuildkiteClient(args.organization, args.pipeline)
            history = fetch_public_history(
                client,
                jobs,
                repo_root=repo_root,
                branch=args.branch,
                max_builds=args.max_builds,
                workers=args.public_workers,
            )
        else:
            token = os.getenv("BUILDKITE_API_TOKEN", "")
            organization = os.getenv("BUILDKITE_ORGANIZATION_SLUG", "")
            pipeline = os.getenv("BUILDKITE_PIPELINE_SLUG", "")
            if token and organization and pipeline:
                client = BuildkiteClient(token, organization, pipeline)
                history = fetch_history(
                    client, jobs, repo_root=repo_root, branch=args.branch, max_builds=args.max_builds
                )
            else:
                missing = [
                    name
                    for name, value in (
                        ("BUILDKITE_API_TOKEN", token),
                        ("BUILDKITE_ORGANIZATION_SLUG", organization),
                        ("BUILDKITE_PIPELINE_SLUG", pipeline),
                    )
                    if not value
                ]
                message = "Buildkite history unavailable; missing " + ", ".join(missing)
                if args.strict_history:
                    raise SystemExit(message)
                warnings.append(message)

    if history:
        warnings.extend(str(item) for item in history.get("collection_warnings") or [])

    attempts = attempts_from_history(history or {"builds": []})
    job_summaries = summarize_jobs(attempts, jobs)
    case_summaries = summarize_cases(attempts, cases)
    rows = build_rows(cases, job_summaries, case_summaries)
    write_csv(rows, csv_path)
    write_report(
        report_path,
        repo_root=repo_root,
        commit=_git_commit(repo_root),
        jobs=jobs,
        cases=cases,
        rows=rows,
        history=history,
        job_summaries=job_summaries,
        warnings=warnings,
    )
    if args.history_cache_out and history:
        args.history_cache_out.parent.mkdir(parents=True, exist_ok=True)
        args.history_cache_out.write_text(
            json.dumps(_safe_history_cache(history), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Jobs: {len(jobs)}, cases: {len(cases)}, history builds: {len((history or {}).get('builds') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
