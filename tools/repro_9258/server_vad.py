# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configure and verify offline Silero VAD without starting a duplex engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def configure_deploy(server_params: dict, model_path: str, destination: Path) -> None:
    # The old test harness ignores update fields for --deploy-config paths.
    # Write an overlay for both profiles so the setting reaches both servers.
    args = list(server_params["extra_cli_args"])
    if args.count("--deploy-config") != 1:
        raise ValueError("Expected one --deploy-config in the pinned benchmark")
    index = args.index("--deploy-config") + 1
    source = Path(args[index]).resolve(strict=True)
    deploy = yaml.safe_load(source.read_text(encoding="utf-8"))
    if base := deploy.get("base_config"):
        deploy["base_config"] = str((source.parent / base).resolve(strict=True))
    deploy.setdefault("duplex_session", {})["server_vad_model_path"] = model_path
    destination.write_text(yaml.safe_dump(deploy, sort_keys=False), encoding="utf-8")
    args[index] = str(destination)
    server_params["extra_cli_args"] = args


def check_backend(model_path: Path, report: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    from vllm_omni.engine.duplex.vad import (
        SILERO_VAD_REPO_ID,
        SILERO_VAD_REVISION,
        SILERO_VAD_SHA256,
        SileroVADBackendProvider,
    )

    # An explicit file makes the production provider verify its pinned hash
    # and load ONNX Runtime, without consulting the Hub or torch fallback.
    backend = SileroVADBackendProvider(model_path=str(model_path)).get()
    probability, _ = backend.infer(np.zeros(backend.frame_samples, dtype=np.float32), backend.new_state())
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"Invalid Silero VAD silence probability: {probability}")
    metadata = {
        "model_path": str(model_path),
        "bytes": model_path.stat().st_size,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "expected_sha256": SILERO_VAD_SHA256,
        "repo": SILERO_VAD_REPO_ID,
        "revision": SILERO_VAD_REVISION,
        "backend": type(backend).__name__,
        "onnxruntime": ort.__version__,
        "frame_samples": backend.frame_samples,
        "silence_probability": probability,
    }
    report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Verified offline server_vad: " + json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    check_backend(arguments.model_path, arguments.report)
