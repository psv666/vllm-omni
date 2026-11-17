# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read the local reproduction YAML without evaluating shell expressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml


def mapping(value: object, label: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} setting(s): {', '.join(sorted(unknown))}")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or "\0" in value:
        raise ValueError(f"{label} must be a string without NUL characters")
    return value


def flag(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def local_path(value: object, label: str, base: Path, home: str) -> str:
    value = text(value, label)
    if not value:
        return ""
    if value == "~" or value.startswith("~/"):
        value = home + value[1:]
    path = Path(value)
    return str(path if path.is_absolute() else base / path)


def safe_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme in ("http", "https"):
        return parsed._replace(netloc=parsed.netloc.rsplit("@", 1)[-1], query="", fragment="").geturl()
    return value


def settings(config: object, base: Path, home: str, profile_override: str | None) -> dict[str, str]:
    config = mapping(
        config,
        "top-level",
        {
            "profile",
            "paths",
            "profiles",
            "offline",
            "prefetch",
            "capture_stderr",
            "log_level",
            "source_repo",
            "hf_endpoint",
        },
    )
    profile = profile_override or config.get("profile", "ci")
    if profile not in ("ci", "original"):
        raise ValueError("profile must be ci or original")
    paths = mapping(
        config.get("paths", {}),
        "paths",
        {
            "repro_root",
            "hf_cache_dir",
            "qwen3_omni_model_dir",
            "judge_model_dir",
            "duplex_eval_dataset_dir",
            "server_vad_model_path",
        },
    )
    profiles = mapping(config.get("profiles", {}), "profiles", {"ci", "original"})
    selected = mapping(profiles.get(profile, {}), f"profiles.{profile}", {"gpus", "repro_root"})
    gpus = selected.get("gpus")
    if (
        not isinstance(gpus, list)
        or len(gpus) != 3
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
        or len(set(gpus)) != 3
    ):
        raise ValueError(f"profiles.{profile}.gpus must contain three distinct non-negative integers")
    base_root = local_path(paths.get("repro_root", ""), "paths.repro_root", base, home)
    if not base_root:
        raise ValueError("paths.repro_root is required")
    repro_root = local_path(selected.get("repro_root", base_root), f"profiles.{profile}.repro_root", base, home)
    if not repro_root:
        raise ValueError(f"profiles.{profile}.repro_root must not be empty")
    cache = local_path(paths.get("hf_cache_dir", ""), "paths.hf_cache_dir", base, home)
    cache = cache or str(Path(base_root) / "hf-cache")
    offline = flag(config, "offline", True)
    local_inputs = {}
    for key in ("qwen3_omni_model_dir", "judge_model_dir", "duplex_eval_dataset_dir"):
        local_inputs[key] = local_path(paths.get(key, ""), f"paths.{key}", base, home)
        if offline and not local_inputs[key]:
            raise ValueError(f"paths.{key} is required when offline is true; edit the YAML before running")
    local_inputs["server_vad_model_path"] = local_path(
        paths.get("server_vad_model_path", ""), "paths.server_vad_model_path", base, home
    )
    level = text(config.get("log_level", "INFO"), "log_level").upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("log_level must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
    source_repo = text(config.get("source_repo", "https://github.com/psv666/vllm-omni.git"), "source_repo")
    endpoint = text(config.get("hf_endpoint", "https://huggingface.co"), "hf_endpoint")
    resolved = {
        "profile": profile,
        "gpus": gpus,
        "paths": {"repro_root": repro_root, "hf_cache_dir": cache, **local_inputs},
        "offline": offline,
        "prefetch": flag(config, "prefetch", True),
        "capture_stderr": flag(config, "capture_stderr", True),
        "log_level": level,
        "source_repo": safe_url(source_repo),
        "hf_endpoint": safe_url(endpoint),
    }
    return {
        "REPRO_PROFILE": profile,
        "GPUS": ",".join(str(gpu) for gpu in gpus),
        "REPRO_ROOT": repro_root,
        "HF_CACHE_DIR": cache,
        "QWEN3_OMNI_MODEL_DIR": local_inputs["qwen3_omni_model_dir"],
        "JUDGE_MODEL_DIR": local_inputs["judge_model_dir"],
        "DUPLEX_EVAL_DATASET_DIR": local_inputs["duplex_eval_dataset_dir"],
        "SERVER_VAD_MODEL_PATH": local_inputs["server_vad_model_path"],
        "REPRO_OFFLINE": "1" if offline else "0",
        "HF_HUB_OFFLINE": "1" if offline else "0",
        "HF_DATASETS_OFFLINE": "1" if offline else "0",
        "PREFETCH": "1" if resolved["prefetch"] else "0",
        "CAPTURE_STDERR": "1" if resolved["capture_stderr"] else "0",
        "VLLM_LOGGING_LEVEL": level,
        "REPRO_REPO_URL": source_repo,
        "HF_ENDPOINT": endpoint,
        "REPRO_SETTINGS_JSON": json.dumps(resolved, ensure_ascii=True, indent=2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--home-dir", required=True)
    parser.add_argument("--profile", choices=("ci", "original"))
    args = parser.parse_args()
    try:
        data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        values = settings(data, args.base_dir, args.home_dir, args.profile)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Invalid reproduction config: {error}", file=sys.stderr)
        return 2
    for key, value in values.items():
        sys.stdout.buffer.write(key.encode("utf-8") + b"\0" + value.encode("utf-8") + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
