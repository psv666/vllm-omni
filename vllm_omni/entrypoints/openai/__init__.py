# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI-compatible API entrypoints, with lazy public serving exports."""


def __getattr__(name: str):
    # Importing a session/runtime module must not initialize the API server,
    # its WebSocket transports, or the unrelated duplex serving stack.
    if name in {"build_async_omni", "omni_init_app_state", "omni_run_server"}:
        from vllm_omni.entrypoints.openai import api_server

        return getattr(api_server, name)
    if name == "OmniOpenAIServingChat":
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        return OmniOpenAIServingChat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "omni_run_server",
    "build_async_omni",
    "omni_init_app_state",
    "OmniOpenAIServingChat",
]
