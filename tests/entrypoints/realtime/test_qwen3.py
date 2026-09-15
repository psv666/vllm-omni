# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen adapter contracts at the real generate boundary, without model weights."""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.sampling_params import RequestOutputKind, SamplingParams

from tests.helpers.serving_chat import build_serving_chat
from vllm_omni.config.stage_config import StageConfig
from vllm_omni.entrypoints.realtime.contracts import Content, Item, SessionConfig, TurnSnapshot, VisualPolicy
from vllm_omni.entrypoints.realtime.qwen3 import Qwen3RealtimeAdapter
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def image_url(color):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def snapshot(parts=None, *, items=None, **config):
    wire_items = items or [{"role": "user", "content": parts or [{"type": "input_text", "text": "Hi"}]}]
    converted = []
    for item in wire_items:
        content = []
        for part in item["content"]:
            kind = part["type"]
            if kind == "input_image":
                content.append(
                    Content(
                        "image",
                        data=base64.b64decode(part["image_url"].split(",", 1)[1]),
                        detail=part.get("detail", "auto"),
                    )
                )
            elif kind == "input_audio":
                content.append(Content("audio", data=base64.b64decode(part["audio"])))
            else:
                content.append(
                    Content("audio" if kind == "audio" else "text", text=part.get("text", part.get("transcript", "")))
                )
        converted.append(Item(item["role"], tuple(content)))
    omni = config.get("omni", {})
    return TurnSnapshot(
        tuple(converted),
        SessionConfig(
            "test",
            config.get("instructions", ""),
            config.get("output_modalities", ["text"])[0],
            config.get("max_output_tokens"),
            VisualPolicy(
                omni.get("input_image_retention", "client"),
                50,
                omni.get("input_image_sample", 4),
                omni.get("input_image_evs", True),
            ),
        ),
    )


def adapter_with_outputs(outputs):
    engine = MagicMock()
    engine.stage_configs = [
        StageConfig(stage_id=index, model_stage=name, is_comprehension=index == 0)
        for index, name in enumerate(("thinker", "talker", "code2wav"))
    ]
    engine.default_sampling_params_list = [
        SamplingParams(temperature=0.4, max_tokens=64),
        SamplingParams(temperature=0.7, max_tokens=96),
        SamplingParams(temperature=0.8, max_tokens=128),
    ]

    async def generate(**kwargs):
        for output in outputs:
            yield output

    engine.generate.side_effect = generate
    engine.preprocess = AsyncMock(return_value={"prompt": "prepared"})
    adapter = Qwen3RealtimeAdapter(build_serving_chat(engine_client=engine), engine, preprocess=engine.preprocess)
    return adapter, engine


def text_output(text, finish_reason=None):
    return OmniRequestOutput(
        final_output_type="text", outputs=[SimpleNamespace(text=text, finish_reason=finish_reason)]
    )


def audio_output(chunks, rate=24000, finish_reason="stop"):
    return OmniRequestOutput(
        final_output_type="audio",
        outputs=[SimpleNamespace(multimodal_output={"audio": chunks, "sr": rate}, finish_reason=finish_reason)],
    )


@pytest.mark.asyncio
async def test_preprocess_preserves_server_default_chat_template_kwargs(monkeypatch):
    engine = MagicMock()
    handler = build_serving_chat(
        engine_client=engine,
        chat_template="server-template",
        default_chat_template_kwargs={"enable_thinking": False},
    )
    prompt = {"prompt_token_ids": [1, 2, 3]}
    preprocess_chat = AsyncMock(return_value=([], [prompt]))
    monkeypatch.setattr(handler, "_preprocess_chat", preprocess_chat)
    adapter = Qwen3RealtimeAdapter(handler, engine)
    messages = [{"role": "user", "content": "Hi"}]

    prepared = await adapter.prepare(messages, {"model": "test", "output_modalities": ["text"]}, has_audio=False)

    request = preprocess_chat.call_args.args[0]
    assert request.messages == messages
    preprocess_chat.assert_awaited_once_with(
        request,
        request.messages,
        default_template="server-template",
        default_template_content_format="auto",
        default_template_kwargs={
            "enable_thinking": False,
            "add_generation_prompt": True,
            "continue_final_message": False,
        },
        renderer=handler.renderer,
        add_generation_prompt=True,
        continue_final_message=False,
        add_special_tokens=False,
    )
    assert prepared.prompt is prompt


@pytest.mark.asyncio
async def test_max_output_tokens_only_changes_comprehension_stage_generate_kwargs():
    adapter, engine = adapter_with_outputs([text_output("Hi", "length")])
    outputs = [out async for out in adapter.generate(snapshot(max_output_tokens=7), "request")]
    request = engine.preprocess.call_args.args[0]
    assert request.max_tokens == 7
    kwargs = engine.generate.call_args.kwargs
    assert kwargs["request_id"] == "request"
    assert kwargs["prompt"] == {"prompt": "prepared"}
    assert kwargs["output_modalities"] == ["text"]
    params = kwargs["sampling_params_list"]
    assert [p.max_tokens for p in params] == [7, 96, 128]
    assert params[0].output_kind == RequestOutputKind.DELTA
    assert [p.max_tokens for p in engine.default_sampling_params_list] == [64, 96, 128]
    assert outputs[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_audio_mode_requests_thinker_text_and_does_not_overwrite_length():
    adapter, engine = adapter_with_outputs([text_output("Hello", "length"), audio_output([torch.ones(16)])])
    outputs = [out async for out in adapter.generate(snapshot(output_modalities=["audio"]), "request")]
    assert engine.generate.call_args.kwargs["output_modalities"] == ["text", "audio"]
    assert outputs[0].text == "Hello"
    assert outputs[-1].finish_reason == "length"


def test_canonical_history_keeps_all_messages_and_transcript_in_order():
    adapter, _ = adapter_with_outputs([])
    items = [
        {"role": "user", "content": [{"type": "input_text", "text": "First question"}]},
        {"role": "assistant", "content": [{"type": "audio", "transcript": "First answer"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "Second question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Second answer"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "Now recall both"}]},
    ]
    messages, has_audio = adapter.build_messages(snapshot(items=items, instructions="Be concise"))
    assert messages[0] == {"role": "system", "content": "Be concise"}
    assert [message["content"][0]["text"] for message in messages[1:]] == [
        "First question",
        "First answer",
        "Second question",
        "Second answer",
        "Now recall both",
    ]
    assert not has_audio


@pytest.mark.parametrize("retention,indices", [("client", [0, 1, 2, 3, 4]), ("rolling", [0, 1, 4])])
def test_image_selection_preserves_order_and_client_mode_keeps_every_image(retention, indices):
    adapter, _ = adapter_with_outputs([])
    urls = [image_url((value * 40, 0, 0)) for value in range(5)]
    parts = [{"type": "input_text", "text": "Before"}]
    parts.extend({"type": "input_image", "image_url": url, "detail": "high"} for url in urls)
    parts.append({"type": "input_text", "text": "After"})
    turn = snapshot(parts, omni={"input_image_retention": retention, "input_image_sample": 3, "input_image_evs": False})
    messages, _ = adapter.build_messages(turn)
    content = messages[0]["content"]
    assert content[0]["text"] == "Before" and content[-1]["text"] == "After"
    assert [p["image_url"] for p in content[1:-1]] == [{"url": urls[i], "detail": "high"} for i in indices]
    assert len(turn.items[0].content) == 7


def test_rolling_evs_only_filters_model_input_not_canonical_items():
    adapter, _ = adapter_with_outputs([])
    red, blue = image_url((255, 0, 0)), image_url((0, 0, 255))
    turn = snapshot(
        [{"type": "input_image", "image_url": url} for url in (red, red, blue)],
        omni={"input_image_retention": "rolling", "input_image_sample": 4},
    )
    messages, _ = adapter.build_messages(turn)
    assert [p["image_url"]["url"] for p in messages[0]["content"]] == [red, blue]
    assert len(turn.items[0].content) == 3


def test_committed_pcm24_is_resampled_and_wrapped_for_existing_pcm16_pipeline():
    adapter, _ = adapter_with_outputs([])
    raw = np.zeros(4800, dtype="<i2").tobytes()
    messages, has_audio = adapter.build_messages(
        snapshot([{"type": "input_audio", "audio": base64.b64encode(raw).decode()}])
    )
    assert has_audio
    payload = messages[0]["content"][0]["input_audio"]
    assert payload["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(payload["data"]))) as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert audio.getnframes() == 3200


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fast", "slow"])
async def test_cumulative_audio_drains_only_new_chunks_and_trims_first_once(mode):
    first, second, third = torch.arange(5000, dtype=torch.float32), torch.arange(10.0), torch.arange(20.0)
    adapter, _ = adapter_with_outputs([audio_output([first], 22050), audio_output([first, second, third], 22050)])
    adapter._engine_client.stage_configs[-1].sampling_constraints = {"output_kind": RequestOutputKind.CUMULATIVE}
    outputs = [
        out
        async for out in adapter.generate_prepared(
            await adapter.prepare(
                [{"role": "user", "content": "Hi"}],
                {"output_modalities": ["audio"], "omni": {"_audio_delta_mode": mode}},
                has_audio=False,
            ),
            "request",
        )
    ]
    audio = [out for out in outputs if out.audio is not None]
    assert [out.sample_rate for out in audio] == [22050, 22050]
    np.testing.assert_array_equal(audio[0].audio, first.numpy()[1920:])
    np.testing.assert_array_equal(audio[1].audio, np.concatenate([second.numpy(), third.numpy()]))


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["output_payload", "audio_data"])
async def test_audio_output_level_fallback_and_actual_rate(location):
    result = OmniRequestOutput(final_output_type="audio")
    if location == "output_payload":
        result._multimodal_output = {"audio": [torch.ones(8)], "sample_rate": 48000}
    else:
        result.audio_data = [torch.ones(8)]
        result.sample_rate = 48000
    adapter, _ = adapter_with_outputs([result])
    outputs = [out async for out in adapter.generate(snapshot(output_modalities=["audio"]), "request")]
    assert outputs[0].sample_rate == 48000
    np.testing.assert_array_equal(outputs[0].audio, np.ones(8))


@pytest.mark.asyncio
async def test_engine_error_output_is_not_reported_as_successful_empty_response():
    adapter, _ = adapter_with_outputs([OmniRequestOutput.from_error("request", "stage failed")])
    with pytest.raises(RuntimeError, match="stage failed"):
        _ = [out async for out in adapter.generate(snapshot(), "request")]


@pytest.mark.asyncio
async def test_delta_output_keeps_identical_text_and_each_fresh_audio_payload():
    adapter, _ = adapter_with_outputs(
        [
            text_output("ha"),
            text_output("ha"),
            audio_output([torch.ones(8)]),
            audio_output([torch.full((8,), 2.0)]),
        ]
    )
    outputs = [
        out async for out in adapter.generate(snapshot(output_modalities=["audio"], max_output_tokens=16), "request")
    ]
    assert [out.text for out in outputs if out.text] == ["ha", "ha"]
    audio = [out.audio for out in outputs if out.audio is not None]
    assert len(audio) == 2
    np.testing.assert_array_equal(audio[0], np.ones(8))
    np.testing.assert_array_equal(audio[1], np.full(8, 2.0))


@pytest.mark.asyncio
async def test_cumulative_single_tensor_emits_new_samples_and_cumulative_text_is_deduplicated():
    first, second = torch.ones(8), torch.full((5,), 2.0)
    adapter, engine = adapter_with_outputs(
        [
            text_output("ha"),
            text_output("haha"),
            audio_output(first),
            audio_output(torch.cat([first, second])),
        ]
    )
    for stage in engine.stage_configs:
        stage.sampling_constraints = {"output_kind": RequestOutputKind.CUMULATIVE}
    outputs = [out async for out in adapter.generate(snapshot(output_modalities=["audio"]), "request")]
    assert [out.text for out in outputs if out.text] == ["ha", "ha"]
    audio = [out.audio for out in outputs if out.audio is not None]
    assert len(audio) == 2
    np.testing.assert_array_equal(audio[0], first.numpy())
    np.testing.assert_array_equal(audio[1], second.numpy())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [RequestOutputKind.DELTA, RequestOutputKind.CUMULATIVE])
async def test_buffered_audio_keeps_all_chunks_across_empty_terminal(kind):
    first, second = torch.ones(8), torch.full((5,), 2.0)
    later = second if kind == RequestOutputKind.DELTA else torch.cat([first, second])
    adapter, engine = adapter_with_outputs([audio_output(first), audio_output(later), audio_output(None)])
    engine.stage_configs[-1].sampling_constraints = {"output_kind": kind}
    outputs = [
        out
        async for out in adapter.generate_prepared(
            await adapter.prepare(
                [{"role": "user", "content": "Hi"}],
                {"output_modalities": ["audio"], "omni": {"_audio_chunk_mode": "off"}},
                has_audio=False,
            ),
            "request",
        )
    ]
    audio = [out.audio for out in outputs if out.audio is not None]
    assert len(audio) == 1
    np.testing.assert_array_equal(audio[0], torch.cat([first, second]).numpy())
