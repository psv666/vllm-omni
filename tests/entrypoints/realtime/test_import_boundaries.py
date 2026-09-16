# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Importing the neutral session runtime must not initialize a transport."""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runtime_import_does_not_load_transports_or_duplex():
    # Match the existing engine import probes: shared-filesystem imports can
    # exceed 30 seconds, and GPU visibility leaked by other tests is irrelevant.
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

from vllm_omni.entrypoints.openai.realtime.runtime import RealtimeRuntime

forbidden = (
    "vllm_omni.entrypoints.openai.api_server",
    "vllm_omni.entrypoints.openai.realtime.codec",
    "vllm_omni.entrypoints.openai.realtime.connection",
    "vllm_omni.entrypoints.openai.realtime.events",
    "vllm_omni.entrypoints.openai.realtime.routing",
    "vllm_omni.entrypoints.openai.realtime.qwen3",
    "vllm_omni.entrypoints.duplex",
    "vllm_omni.engine.duplex",
)
loaded = [name for name in sys.modules if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)]
assert not loaded, loaded

# Existing package-level imports still resolve to the original serving objects.
from vllm_omni.entrypoints import openai
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

for name in ("build_async_omni", "omni_init_app_state", "omni_run_server"):
    assert getattr(openai, name) is getattr(api_server, name)
assert openai.OmniOpenAIServingChat is OmniOpenAIServingChat
""",
        ],
        check=True,
        env=env,
        timeout=180,
    )
