# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Legacy output modes still use the shared response lifecycle."""

import asyncio

import numpy as np
import pytest

from tests.entrypoints.openai_api.test_serving_video_stream import MockWebSocket, _audio_result
from vllm_omni.entrypoints.openai.serving_video_stream import QwenOmniStreamingVideoHandler, StreamingVideoSessionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_audio_only_legacy_response_is_not_a_failed_empty_text_response():
    class Engine:
        async def generate(self, **kwargs):
            assert kwargs["output_modalities"] == ["audio"]
            yield _audio_result(np.ones(4800, dtype=np.float32))

    class Handler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt_token_ids": [1]}

    handler = Handler(chat_service=object(), engine_client=Engine())
    ws = MockWebSocket()
    await handler._process_query_engine(
        ws,
        StreamingVideoSessionConfig(modalities=["audio"]),
        [],
        bytearray(),
        [],
        "Say hello.",
        "legacy-audio",
        asyncio.Event(),
        {},
    )
    assert not [event for event in ws.sent if event["type"] == "error"]
    assert [event["type"] for event in ws.sent][-2:] == ["response.text.done", "response.output_audio.done"]
