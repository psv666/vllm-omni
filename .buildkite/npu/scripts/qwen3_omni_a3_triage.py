# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run or summarize the two-pass Qwen3-Omni A3 experiment from a source checkout."""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "f602d0295852c430434f8cd2cd87da287db1a96e"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REV = "26291f793822fb6be9555850f06dfe95f2d7e695"
CI_IMAGE = "sha256:fdcede880eeba2b93d0dc3fc8532d0cb88dd1c2586f5f61cff7e923b7f55cd69"
SOURCE_CONFIG = ROOT / "tests/dfx/perf/tests/test_qwen3_omni_no_async_chunk.json"
TRACE_ENV = "VLLM_OMNI_A3_THINKER_TRACE_DIR"
COUNTS = {1: 4, 4: 16, 8: 32, 16: 64}
LIMIT = 4096
ANSI = re.compile(r"\x1b_bk;t=\d+\x07|\x1b\[[0-?]*[ -/]*[@-~]")
MEASUREMENT = re.compile(r"^Maximum request concurrency:\s*(\d+)\s*$")
TIMING = re.compile(r"\[OmniTiming\] req=(\S+) .*?stages=\[0:([\d.]+)s,1:([\d.]+)s,2:([\d.]+)s\]")
RESULT = re.compile(r"_random_(\d+)_(\d+)_in2500_out900_")
METRICS = ("mean_tpot_ms", "median_tpot_ms", "median_itl_ms", "mean_e2el_ms")
RUNTIME_CHECK = """
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path

import torch_npu

npu = torch_npu.npu
devices = [{"name": npu.get_device_name(i)} for i in range(npu.device_count())]
versions = {name: md.version(name) for name in ("vllm", "vllm-ascend", "torch", "torch-npu")}
cann = {}
for path in sorted(Path("/usr/local/Ascend/ascend-toolkit/latest").glob("*/*install.info")):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("version="):
            cann[path.name] = line.split("=", 1)[1].strip()
print("A3_RUNTIME=" + json.dumps({
    "versions": versions, "devices": devices, "cann": cann,
    "omni": importlib.util.find_spec("vllm_omni").origin,
}))
"""


class TriageError(Exception):
    """A missing prerequisite or incomplete experiment."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def checkpoint(model_dir):
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache:
        hf_home = os.environ.get("HF_HOME")
        xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache = (Path(hf_home) if hf_home else xdg_cache / "huggingface") / "hub"
    snapshot = model_dir or Path(cache) / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct" / "snapshots" / MODEL_REV
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file() or not (snapshot / "config.json").is_file():
        raise TriageError(f"cached checkpoint missing for {MODEL} at {MODEL_REV[:12]}")
    shards = set(read_json(index)["weight_map"].values())
    if not shards or any(not (snapshot / shard).is_file() for shard in shards):
        raise TriageError("cached checkpoint has missing weight shards")
    if snapshot.name != MODEL_REV or snapshot.parent.name != "snapshots":
        download_meta = snapshot / ".cache/huggingface/download"
        for name in {"config.json", *shards}:
            info = download_meta / f"{name}.metadata"
            if not info.is_file() or info.read_text(encoding="utf-8").partition("\n")[0].strip() != MODEL_REV:
                raise TriageError(f"local checkpoint lacks download provenance for {MODEL_REV[:12]}")
    return snapshot.resolve()


def preflight(directory, env, base_image_digest, model_dir):
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=False).returncode:
        raise TriageError(f"checkout must descend from {BASE[:12]}")
    model = checkpoint(model_dir)
    completed = subprocess.run(
        [sys.executable, "-c", RUNTIME_CHECK],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (directory / "preflight.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    line = next((s for s in completed.stdout.splitlines() if s.startswith("A3_RUNTIME=")), None)
    if completed.returncode or line is None:
        for name in ("torch_npu", "vllm", "vllm-ascend", "torch"):
            if (
                f"No module named '{name}'" in completed.stderr
                or f"No package metadata was found for {name}" in completed.stderr
            ):
                raise TriageError(f"runtime dependency missing: {name}")
        raise TriageError("runtime or device inspection failed; see preflight.log locally")
    runtime = json.loads(line.split("=", 1)[1])
    if Path(runtime.pop("omni")).resolve() != ROOT / "vllm_omni/__init__.py":
        raise TriageError("Python resolved Omni outside the diagnostic checkout")
    versions = runtime["versions"]
    if versions["vllm"].split("+", 1)[0] != "0.34.0":
        raise TriageError(f"expected vLLM 0.34.0, found {versions['vllm']}")
    if len(runtime["devices"]) < 3:
        raise TriageError("three logical A3 devices required for TP2 plus Talker/Code2Wav")
    runtime.update(
        checkout=git("rev-parse", "HEAD"),
        benchmark_base=BASE,
        model_rev=MODEL_REV,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        base_image_digest=base_image_digest,
        ci_reference_match=(base_image_digest == CI_IMAGE and versions["torch-npu"] == "2.10.0.post7"),
    )
    return runtime, model


def make_config(concurrencies, model):
    (config,) = read_json(SOURCE_CONFIG)
    (workload,) = (row for row in config["benchmark_params"] if row["dataset_name"] == "random")
    payload = workload["extra_body"]["sampling_params_list"]
    if (
        workload["random_input_len"] != 2500
        or workload["random_output_len"] != 900
        or any([p["min_tokens"], p["max_tokens"]] != [n, n] for p, n in zip(payload, (900, 1536)))
        or len(payload) != 2
        or not workload["extra_body"].get("return_stage_metrics")
        or not workload.get("ignore_eos")
    ):
        raise TriageError("pinned CI workload does not have the expected token contract")
    for concurrency in concurrencies:
        index = workload["max_concurrency"].index(concurrency)
        if workload["num_prompts"][index] != COUNTS[concurrency]:
            raise TriageError(f"unexpected CI request count at c={concurrency}")
    workload["max_concurrency"] = concurrencies
    workload["num_prompts"] = [COUNTS[c] for c in concurrencies]
    workload["num_warmups"] = [max(2, c) for c in concurrencies]
    workload.pop("baseline", None)
    config["benchmark_params"] = [workload]
    server = config["server_params"]
    server["model"] = str(model)
    server["extra_cli_args"] += [
        "--deploy-config",
        str(ROOT / "vllm_omni/deploy/qwen3_omni_moe.yaml"),
        "--stage-init-timeout",
        "900",
    ]
    return [config]


def log_metrics(path):
    rows = defaultdict(dict)
    active = None
    starting = False
    graph_roles = set()
    runner_roles = set()
    errors = []
    for line in ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).splitlines():
        line = line.strip()
        match = MEASUREMENT.fullmatch(line)
        if line == "Starting main benchmark run...":
            starting = True
        elif match and starting:
            active = int(match.group(1))
            starting = False
        elif line.startswith("Successful requests:") or "Serving Benchmark Result" in line:
            active = None
        elif active is not None and (timing := TIMING.search(line)):
            key, *values = timing.groups()
            rows[active][key] = tuple(map(float, values))
        if "Worker_TP" in line:
            role = line.split("Worker_TP", 1)[1].split()[0]
            if role in {"0", "1"}:
                if "ACL graph replay is active" in line:
                    graph_roles.add(role)
                if "NPU Model Runner V2 loaded" in line:
                    runner_roles.add(role)
        # Export fixed categories only. Container logs can contain internal paths and credentials.
        for indicator, category in (
            ("hcclGetRootInfo", "HCCL initialization"),
            ("OutOfMemoryError", "out of memory"),
            ("Warmup requests failed", "warmup failure"),
            ("warmup requests failed", "warmup failure"),
            ("Initial test run failed", "initial request failure"),
            ("timed out", "timeout"),
            ("ERROR collecting", "pytest collection"),
            ("RuntimeError:", "worker or runtime exception"),
            ("AssertionError", "benchmark assertion"),
            ("no tests ran", "no benchmark collected"),
        ):
            if indicator in line and category not in errors:
                errors.append(category)
    return rows, {"runner_v2_tp": sorted(runner_roles), "acl_graph_tp": sorted(graph_roles)}, errors


def performance(directory, log, concurrencies):
    timings, graph, errors = log_metrics(log)
    artifacts = defaultdict(list)
    for path in directory.glob("result_*.json"):
        if match := RESULT.search(path.name):
            artifacts[int(match.group(1))].append(path)
    cases = {}
    for c in concurrencies:
        count = COUNTS[c]
        paths = artifacts[c]
        if len(paths) != 1:
            cases[c] = {"error": f"expected one result artifact, found {len(paths)}"}
            continue
        result = read_json(paths[0])
        stages = result.get("request_stage_metrics", [])
        token_ok = len(stages) == count and all(
            stage.get("0", {}).get("num_tokens_out") == 900 and stage.get("1", {}).get("num_tokens_out") == 1536
            for stage in stages
        )
        values = timings[c].values()
        valid_metrics = all(isinstance(result.get(key), (int, float)) and math.isfinite(result[key]) for key in METRICS)
        case = {
            "completed": result.get("completed"),
            "failed": result.get("failed"),
            "timing_count": len(values),
            "tokens_900_1536": token_ok,
        }
        if valid_metrics:
            case.update({key: round(result[key], 2) for key in METRICS})
        if values:
            case["thinker_s"], case["talker_s"], case["code2wav_s"] = (
                round(statistics.median(column), 2) for column in zip(*values)
            )
        if (
            result.get("completed") != count
            or result.get("failed") != 0
            or result.get("num_tpot_samples") != count
            or len(values) != count
            or not token_ok
            or not valid_metrics
        ):
            case["error"] = "request, token, timing, or metric validation failed"
        cases[c] = case
    return cases, graph, errors


def verdict(cases):
    if any("error" in row for row in cases.values()) or not {8, 16} <= cases.keys():
        return "invalid"
    c8, c16 = cases[8], cases[16]
    if c8["median_tpot_ms"] < 40 and c16["median_tpot_ms"] >= 100 and c16["thinker_s"] >= 90:
        return "reproduced"
    if c8["median_tpot_ms"] < 40 and c16["median_tpot_ms"] < 75 and c16["thinker_s"] < 75:
        return "near_normal"
    return "inconclusive"


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def summarize_trace(trace):
    windows_path = trace / "windows.jsonl"
    if not windows_path.is_file():
        return {}, ["missing benchmark windows"]
    bounds = defaultdict(dict)
    for line in windows_path.read_text(encoding="utf-8").splitlines():
        phase, boundary, concurrency, timestamp = json.loads(line)
        bounds[(phase, concurrency)][boundary] = timestamp
    groups = defaultdict(list)
    errors = []
    owners = set()
    for path in sorted(trace.glob("*.json")):
        owner = path.stem.split("-", 1)[0]
        if owner in owners:
            errors.append(f"duplicate trace process {owner}")
        owners.add(owner)
        data = read_json(path)
        if data["dropped"]:
            errors.append(f"{owner} buffer dropped {data['dropped']} records")
        for kind, batch, metric, started, duration in data["rows"]:
            for (phase, concurrency), interval in bounds.items():
                if interval.get("start", math.inf) <= started < interval.get("end", -1):
                    groups[(phase, concurrency, kind, batch, owner, metric)].append(duration)
                    break
    for owner in sorted({"engine", "tp0", "tp1"} - owners):
        errors.append(f"missing trace from {owner}")
    for c in (8, 16):
        present = {
            (owner, metric)
            for phase, concurrency, kind, _, owner, metric in groups
            if phase == "measured" and concurrency == c and kind == "decode"
        }
        required = {("engine", "schedule"), ("engine", "dispatch_to_consume")}
        required.update((tp, name) for tp in ("tp0", "tp1") for name in ("execute_model", "sample_tokens"))
        if required - present:
            errors.append(
                f"c={c} missing measured decode timings for {','.join('/'.join(x) for x in sorted(required - present))}"
            )
    summary = {}
    for key, samples in sorted(groups.items()):
        summary["/".join(map(str, key))] = [
            round(statistics.median(samples) / 1e6, 4),
            round(percentile95(samples) / 1e6, 4),
            len(samples),
        ]
    return summary, errors


def run_cases(directory, env, model, concurrencies, probe):
    directory.mkdir()
    config = directory / "config.json"
    log = directory / "run.log"
    write_json(config, make_config(concurrencies, model))
    process_env = {**env, "BENCHMARK_DIR": str(directory)}
    if probe:
        trace = directory / "trace"
        trace.mkdir()
        process_env[TRACE_ENV] = str(trace)
    print(f"Running c={concurrencies}. Full logs stay in {directory}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-s",
                "-v",
                "-m",
                "npu",
                "tests/dfx/perf/scripts/run_benchmark.py",
                "--test-config-file",
                str(config),
            ],
            cwd=ROOT,
            env=process_env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    cases, graph, errors = performance(directory, log, concurrencies)
    result = {
        "concurrencies": concurrencies,
        "verdict": verdict(cases) if rc == 0 else "invalid",
        "pytest_exit": rc,
        "cases": cases,
        **graph,
        "errors": errors,
    }
    if probe:
        timings, gaps = summarize_trace(trace)
        write_json(trace / "timing-summary.json", timings)
        result["probe_gaps"] = gaps
        compact = {}
        for c in (8, 16):
            for kind in ("decode", "prefill", "mixed", "unknown"):
                prefix = f"measured/{c}/{kind}/"
                selected = [(key, values) for key, values in timings.items() if key.startswith(prefix)]
                if not selected:
                    continue
                counts = {
                    int(key.split("/")[3]): values[-1] for key, values in selected if key.endswith("/engine/schedule")
                }
                if not counts:
                    continue
                batch = c if kind == "decode" and c in counts else max(counts, key=counts.get)
                for key, values in selected:
                    if int(key.split("/")[3]) == batch:
                        compact[key.removeprefix("measured/")] = values
        result["timing_ms_p50_p95_n"] = compact
    return result


def emit(summary):
    compact = json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    if len(compact.encode()) + len("A3_SUMMARY_BEGIN\n\nA3_SUMMARY_END\n") > LIMIT:
        run = summary.get("run", {})
        timings = run.get("timing_ms_p50_p95_n", {})
        run["timing_ms_p50_p95_n"] = {key: value for key, value in timings.items() if "/decode/" in key}
    compact = json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    if len(compact.encode()) + len("A3_SUMMARY_BEGIN\n\nA3_SUMMARY_END\n") > LIMIT:
        compact = json.dumps({"error": "summary exceeds 4 KiB; inspect local summary.json"})
    print(f"A3_SUMMARY_BEGIN\n{compact}\nA3_SUMMARY_END", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "reproduce", "profile", "replay"))
    parser.add_argument("--output", type=Path, help="new directory for local logs and results")
    parser.add_argument("--first-pass", type=Path, help="first-pass summary.json required for profile")
    parser.add_argument("--base-image-digest", default=os.environ.get("A3_BASE_IMAGE_DIGEST"))
    parser.add_argument("--model-dir", type=Path, help="local verified model snapshot instead of the HF cache")
    parser.add_argument("--log", type=Path, help="existing complete job log for replay")
    parser.add_argument("--artifacts", type=Path, help="directory of existing result JSON for replay")
    args = parser.parse_args()
    if args.phase == "replay":
        if args.log is None or args.artifacts is None:
            parser.error("replay requires --log and --artifacts")
        cases, graph, errors = performance(args.artifacts, args.log, [8, 16])
        emit({"phase": args.phase, "run": {"verdict": verdict(cases), "cases": cases, **graph, "errors": errors}})
        return 0 if verdict(cases) != "invalid" else 1
    if args.output is None:
        parser.error("benchmark phases require --output")
    if args.phase == "profile" and args.first_pass is None:
        parser.error("profile requires --first-pass")
    if args.base_image_digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", args.base_image_digest):
        parser.error("--base-image-digest must be sha256 followed by 64 lowercase hex digits")
    directory = args.output.resolve()
    output_created = False
    summary = {"phase": args.phase}
    try:
        directory.mkdir(parents=True)
        output_created = True
        env = os.environ.copy()
        env.pop(TRACE_ENV, None)
        env.update(
            PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH")))),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
            PYTHONUNBUFFERED="1",
        )
        if args.phase == "profile":
            first = read_json(args.first_pass)
            if first.get("phase") != "reproduce" or first.get("run", {}).get("verdict") != "reproduced":
                raise TriageError("profile requires a successful reproduction summary")
        runtime, model = preflight(directory, env, args.base_image_digest, args.model_dir)
        summary["runtime"] = runtime
        if args.phase == "check":
            run = {"verdict": "ready"}
        elif args.phase == "reproduce":
            run = run_cases(directory / "initial", env, model, [8, 16], probe=False)
            if run["verdict"] == "near_normal" and runtime["ci_reference_match"]:
                summary["initial_verdict"] = run["verdict"]
                run = run_cases(directory / "ci-order", env, model, [1, 4, 8, 16], probe=False)
        else:
            if runtime != first["runtime"]:
                raise TriageError("runtime differs from first pass; repeat reproduce in this container")
            if first["run"]["concurrencies"] not in ([8, 16], [1, 4, 8, 16]):
                raise TriageError("invalid first-pass benchmark sequence")
            run = run_cases(directory / "instrumented", env, model, first["run"]["concurrencies"], probe=True)
            if run["verdict"] != "reproduced":
                run["probe_gaps"].append("the regression was not reproduced with the probe enabled")
        summary["run"] = run
    except TriageError as exc:
        summary["error"] = str(exc)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        if output_created:
            (directory / "launcher-error.log").write_text(str(exc) + "\n", encoding="utf-8")
        summary["error"] = f"{type(exc).__name__} during setup or result processing"
    if output_created:
        write_json(directory / "summary.json", summary)
    emit(summary)
    return int("error" in summary or summary["run"]["verdict"] == "invalid" or bool(summary["run"].get("probe_gaps")))


if __name__ == "__main__":
    sys.exit(main())
