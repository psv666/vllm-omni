# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Importing the neutral session runtime must not initialize a transport."""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runtime_import_does_not_load_openai_or_duplex():
    # Match the existing engine import probes: shared-filesystem imports can
    # exceed 30 seconds, and GPU visibility leaked by other tests is irrelevant.
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from vllm_omni.entrypoints.realtime.runtime import RealtimeRuntime; "
            "assert not any(name.startswith('vllm_omni.entrypoints.openai') "
            "or name.startswith('vllm_omni.engine.duplex') for name in sys.modules)",
        ],
        check=True,
        env=env,
        timeout=180,
    )
