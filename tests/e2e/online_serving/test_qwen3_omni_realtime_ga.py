# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official OpenAI SDK against one Qwen3 GA server for the whole module.

Set VLLM_OMNI_GA_TEST_BASE_URL to reuse an already reserved/running server.
Otherwise the fixture starts one full Qwen3 thinker/talker/code2wav pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pytest
import websockets
import yaml
from openai import AsyncOpenAI

from tests.entrypoints.openai_api.conftest_video import make_jpeg
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.core_model, pytest.mark.advanced_model, pytest.mark.omni]

MODEL = os.environ.get("VLLM_OMNI_TEST_QWEN3_OMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
TIMEOUT = float(os.environ.get("VLLM_OMNI_GA_RESPONSE_TIMEOUT", "180"))


@pytest.fixture(scope="module", params=os.environ.get("VLLM_OMNI_GA_ASYNC_MODES", "on,off").split(","))
def ga_server(request, tmp_path_factory):
    mode = request.param
    config = yaml.safe_load(Path("vllm_omni/deploy/qwen3_omni_moe.yaml").read_text())
    config["async_chunk"] = mode == "on"
    deploy = tmp_path_factory.mktemp("ga-deploy") / "qwen3.yaml"
    deploy.write_text(yaml.safe_dump(config))

    external_url = os.environ.get("VLLM_OMNI_GA_TEST_BASE_URL")
    if external_url:
        yield SimpleNamespace(base_url=external_url.rstrip("/") + "/", model=MODEL, mode=mode)
        return
    with OmniServer(
        MODEL,
        [
            "--deploy-config",
            str(deploy),
            "--realtime-profile",
            "openai-realtime",
            "--stage-init-timeout",
            "600",
            "--init-timeout",
            "900",
        ],
        env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    ) as server:
        yield SimpleNamespace(base_url=f"http://{server.host}:{server.port}/v1/", model=server.model, mode=mode)


async def receive_until(connection, event_type, *, allow_error=False):
    async def collect():
        events = []
        while True:
            event = await connection.recv()
            payload = event.model_dump()
            events.append(payload)
            if event.type == "error" and not allow_error:
                pytest.fail(f"GA server error: {payload}")
            if event.type == event_type:
                return events

    return await asyncio.wait_for(collect(), timeout=TIMEOUT)


async def configure(connection, output, *, instructions=""):
    first = await asyncio.wait_for(connection.recv(), timeout=20)
    assert first.type == "session.created"
    assert first.session.type == "realtime"
    await connection.session.update(
        session={
            "type": "realtime",
            "output_modalities": [output],
            "instructions": instructions,
            "max_output_tokens": 128,
            "audio": {"input": {"turn_detection": None}},
        }
    )
    await receive_until(connection, "session.updated")


async def create_item(connection, content):
    await connection.conversation.item.create(item={"type": "message", "role": "user", "content": content})
    await receive_until(connection, "conversation.item.done")


async def respond(connection):
    await connection.response.create()
    events = await receive_until(connection, "response.done")
    done = events[-1]["response"]
    assert done["status"] == "completed", done
    assert sum(event["type"] == "response.done" for event in events) == 1
    assert all(event.get("event_id") for event in events)
    assert any(event["type"] == "response.output_item.done" for event in events)
    return events


def response_text(events):
    return "".join(
        event["delta"]
        for event in events
        if event["type"]
        in {"response.output_text.delta", "response.output_audio_transcript.delta", "response.text.delta"}
    )


def pcm_wav(pcm, sample_rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(pcm)
    return buffer.getvalue()


def save_artifacts(name, events, *, wav_bytes=None):
    directory = os.environ.get("VLLM_OMNI_GA_ARTIFACT_DIR")
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if wav_bytes is not None:
        (root / f"{name}.wav").write_bytes(wav_bytes)
    (root / f"{name}.txt").write_text(response_text(events))
    # Preserve identities and event order without embedding large PCM payloads.
    summary = []
    for event in events:
        event = dict(event)
        if event["type"] == "response.output_audio.delta":
            raw = base64.b64decode(event.pop("delta", event.pop("data", "")))
            event["audio_payload_bytes"] = len(raw)
        summary.append(event)
    (root / f"{name}.events.json").write_text(json.dumps(summary, indent=2))


@pytest.fixture(scope="module")
def spoken_beijing_pcm():
    audio = generate_synthetic_audio(5, 1, sample_rate=24000, phrase_text="Please say the word Beijing.")
    with wave.open(io.BytesIO(base64.b64decode(audio["base64"])), "rb") as source:
        assert source.getframerate() == 24000
        assert source.getnchannels() == 1 and source.getsampwidth() == 2
        pcm = source.readframes(source.getnframes())
        save_artifacts("input_audio", [], wav_bytes=pcm_wav(pcm))
        return pcm


@pytest.fixture(scope="module")
def spoken_color_pcm():
    audio = generate_synthetic_audio(5, 1, sample_rate=24000, phrase_text="What color is the picture?")
    with wave.open(io.BytesIO(base64.b64decode(audio["base64"])), "rb") as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (24000, 1, 2)
        pcm = source.readframes(source.getnframes())
        save_artifacts("input_color_audio", [], wav_bytes=pcm_wav(pcm))
        return pcm


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("output", ["text", "audio"])
@pytest.mark.parametrize("input_kind", ["text", "image", "audio", "mixed", "mixed_audio"])
async def test_ga_sdk_input_modalities(ga_server, spoken_beijing_pcm, spoken_color_pcm, input_kind, output):
    async with AsyncOpenAI(base_url=ga_server.base_url, api_key="EMPTY") as client:
        # No duplex/profile query extension: model is the SDK's standard argument.
        async with client.realtime.connect(model=ga_server.model) as connection:
            instructions = "Answer briefly in English."
            if input_kind == "audio":
                instructions += " Follow the instruction spoken in the audio."
            elif input_kind in {"image", "mixed", "mixed_audio"}:
                instructions += " Name the image's dominant color."
            await configure(connection, output, instructions=instructions)
            image_url = "data:image/jpeg;base64," + base64.b64encode(make_jpeg(255, 0, 0, size=224)).decode()
            if input_kind == "audio":
                # GA PCM is 24 kHz; these bytes contain no WAV header.
                chunk_bytes = 4800
                for offset in range(0, len(spoken_beijing_pcm), chunk_bytes):
                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(spoken_beijing_pcm[offset : offset + chunk_bytes]).decode()
                    )
                await connection.input_audio_buffer.commit()
                committed = await receive_until(connection, "conversation.item.done")
                assert any(event["type"] == "input_audio_buffer.committed" for event in committed)
                assert not any(event["type"].startswith("response.") for event in committed)
                expected = "beijing"
            else:
                content = []
                if input_kind in {"image", "mixed", "mixed_audio"}:
                    content.append({"type": "input_image", "image_url": image_url})
                if input_kind == "mixed_audio":
                    content.append({"type": "input_audio", "audio": base64.b64encode(spoken_color_pcm).decode()})
                if input_kind == "text":
                    content.append({"type": "input_text", "text": "What is the capital of China?"})
                elif input_kind in {"mixed", "mixed_audio"}:
                    content.append({"type": "input_text", "text": "What color is this image?"})
                await create_item(connection, content)
                expected = "beijing" if input_kind == "text" else "red"
            events = await respond(connection)
            raw_audio = b"".join(
                base64.b64decode(event["delta"], validate=True)
                for event in events
                if event["type"] == "response.output_audio.delta"
            )
            save_artifacts(f"{input_kind}-{output}", events, wav_bytes=pcm_wav(raw_audio) if raw_audio else None)
            assert expected in response_text(events).lower(), response_text(events)
            types = [event["type"] for event in events]
            if output == "audio":
                chunks = [
                    base64.b64decode(e["delta"], validate=True)
                    for e in events
                    if e["type"] == "response.output_audio.delta"
                ]
                pcm = b"".join(chunks)
                assert len(pcm) > 2400 and len(pcm) % 2 == 0
                assert not pcm.startswith(b"RIFF")
                assert np.max(np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.int32))) > 100
                assert "response.output_audio.done" in types
                assert "response.output_audio_transcript.done" in types
                assert "response.output_text.delta" not in types
            else:
                assert "response.output_text.done" in types
                assert "response.output_audio.delta" not in types


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
async def test_ga_sdk_history_and_explicit_item_delete(ga_server):
    async with AsyncOpenAI(base_url=ga_server.base_url, api_key="EMPTY") as client:
        async with client.realtime.connect(model=ga_server.model) as connection:
            await configure(connection, "text")
            await create_item(
                connection,
                [
                    {
                        "type": "input_text",
                        "text": "In our story, the pet rabbit is named Clover. Remember the name and reply OK.",
                    }
                ],
            )
            await respond(connection)
            await create_item(
                connection,
                [{"type": "input_text", "text": "What is the rabbit's name in our story? Reply with its name only."}],
            )
            assert "clover" in response_text(await respond(connection)).lower()
            await connection.conversation.item.create(
                item={
                    "id": "delete_me",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Unused"}],
                }
            )
            await receive_until(connection, "conversation.item.done")
            await connection.conversation.item.delete(item_id="delete_me")
            deleted = await receive_until(connection, "conversation.item.deleted")
            assert deleted[-1]["item_id"] == "delete_me"


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("cancel_after", ["created", "text_delta", "audio_delta"])
async def test_ga_sdk_cancel_and_restart_same_connection(ga_server, cancel_after):
    async with AsyncOpenAI(base_url=ga_server.base_url, api_key="EMPTY") as client:
        async with client.realtime.connect(model=ga_server.model) as connection:
            output = "audio" if cancel_after == "audio_delta" else "text"
            await configure(connection, output)
            await create_item(
                connection,
                [
                    {
                        "type": "input_text",
                        "text": (
                            "Tell a long story about exploring a castle. "
                            "Describe every room in detail, using at least 500 words."
                        ),
                    }
                ],
            )
            await connection.response.create(response={"max_output_tokens": 1024})
            target = {
                "created": "response.created",
                "text_delta": "response.output_text.delta",
                "audio_delta": "response.output_audio.delta",
            }[cancel_after]
            started = await receive_until(connection, target)
            response_id = next(event["response"]["id"] for event in started if event["type"] == "response.created")
            if cancel_after != "created":
                assert started[-1]["delta"]
            await connection.response.cancel(response_id=response_id)
            cancelled = await receive_until(connection, "response.done", allow_error=True)
            final = cancelled[-1]["response"]
            assert final["id"] == response_id
            if ga_server.mode == "off" and cancel_after == "audio_delta" and final["status"] != "cancelled":
                # The engine has finished synthesis before releasing its single
                # waveform. Splitting it for delivery cannot cancel past work.
                assert final["status"] in {"completed", "incomplete"}
                errors = [event for event in cancelled if event["type"] == "error"]
                if not errors:
                    errors = await receive_until(connection, "error", allow_error=True)
                assert errors[-1]["error"]["code"] == "response_cancel_not_active"
            else:
                assert final["status"] == "cancelled"
                assert not [event for event in cancelled if event["type"] == "error"]
            assert sum(event["type"] == "response.done" for event in cancelled) == 1
            await create_item(connection, [{"type": "input_text", "text": "Now say hello in English."}])
            restarted = await respond(connection)
            restarted_id = restarted[-1]["response"]["id"]
            assert restarted_id != response_id
            assert all(event.get("response_id", restarted_id) == restarted_id for event in restarted)
            assert "hello" in response_text(restarted).lower()


async def receive_legacy_until(connection, event_type):
    async def collect():
        events = []
        while True:
            event = json.loads(await connection.recv())
            events.append(event)
            assert event["type"] != "error", event
            if event["type"] == event_type:
                return events

    return await asyncio.wait_for(collect(), timeout=TIMEOUT)


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("with_audio_input", [False, True])
async def test_legacy_video_frame_ack_and_wav_output_on_ga_server(ga_server, spoken_color_pcm, with_audio_input):
    base = urlsplit(ga_server.base_url)
    url = urlunsplit(("wss" if base.scheme == "https" else "ws", base.netloc, "/v1/video/chat/stream", "", ""))
    frame = base64.b64encode(make_jpeg(255, 0, 0, size=224)).decode()
    events = []
    chunks = []
    async with websockets.connect(url, max_size=32 * 1024 * 1024) as connection:
        await connection.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": ga_server.model,
                    "modalities": ["text", "audio"],
                    "enable_frame_filter": False,
                    "sampling_params_list": [{"temperature": 0, "max_tokens": 64}],
                }
            )
        )
        await connection.send(
            json.dumps({"type": "video.frame", "data": frame, "frame_id": "legacy-red", "pts_ms": 100})
        )
        acknowledgements = await receive_legacy_until(connection, "video.frame.ack")
        events.extend(acknowledgements)
        ack = acknowledgements[-1]
        assert ack["accepted"] and ack["frame_id"] == "legacy-red"
        assert ack["pts_ms"] == 100
        if with_audio_input:
            from scipy.signal import resample_poly

            source = np.frombuffer(spoken_color_pcm, dtype="<i2").astype(np.float32)
            pcm16 = np.clip(np.rint(resample_poly(source, 2, 3)), -32768, 32767).astype("<i2").tobytes()
            assert len(pcm16) == len(spoken_color_pcm) * 2 // 3
            assert not pcm16.startswith(b"RIFF")
            for offset in range(0, len(pcm16), 3200):
                await connection.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "data": base64.b64encode(pcm16[offset : offset + 3200]).decode(),
                        }
                    )
                )
        await connection.send(
            json.dumps({"type": "video.query", "text": "What color is this image? Answer in one word."})
        )
        response_events = await receive_legacy_until(connection, "response.output_audio.done")
        events.extend(response_events)
        for event in response_events:
            if event["type"] == "response.output_audio.delta":
                assert event["format"] == "wav"
                payload = base64.b64decode(event["data"], validate=True)
                assert payload.startswith(b"RIFF")
                with wave.open(io.BytesIO(payload)) as audio:
                    assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
                    chunks.append(audio.readframes(audio.getnframes()))
        consumed = [event for event in events if event["type"] == "video.frames.consumed"]
        assert len(consumed) == 1 and consumed[0]["frame_ids"] == ["legacy-red"]
        assert "red" in response_text(events).lower(), response_text(events)
        assert len(b"".join(chunks)) > 2400
        save_artifacts(f"legacy-video-audio-input-{with_audio_input}", events, wav_bytes=pcm_wav(b"".join(chunks)))
        await connection.send(json.dumps({"type": "video.done"}))
        await receive_legacy_until(connection, "session.done")


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
async def test_ga_sdk_rolling_camera_frames_preserve_newest_image(ga_server):
    async with AsyncOpenAI(base_url=ga_server.base_url, api_key="EMPTY") as client:
        async with client.realtime.connect(model=ga_server.model) as connection:
            await configure(
                connection, "text", instructions="Answer questions about the latest camera image in English."
            )
            await connection.send(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "omni": {
                            "input_image_retention": "rolling",
                            "input_image_max_items": 2,
                            "input_image_evs": False,
                        },
                    },
                }
            )
            updated = await receive_until(connection, "session.updated")
            assert updated[-1]["session"]["omni"]["input_image_max_items"] == 2
            assert updated[-1]["session"]["omni"]["input_image_evs"] is False
            deletion_events: list[dict] = []
            for name, color in [("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255))]:
                image = "data:image/jpeg;base64," + base64.b64encode(make_jpeg(*color, size=224)).decode()
                await connection.conversation.item.create(
                    item={
                        "id": f"camera_{name}",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": image}],
                    }
                )
                events = await receive_until(connection, "conversation.item.done")
                assert events[-1]["item"]["id"] == f"camera_{name}"
                deletion_events.extend(event for event in events if event["type"] == "conversation.item.deleted")
            assert [event["item_id"] for event in deletion_events] == ["camera_red"]
            await create_item(
                connection,
                [
                    {
                        "type": "input_text",
                        "text": "What color is the most recent camera image? Answer with one color word.",
                    }
                ],
            )
            response = await respond(connection)
            assert "blue" in response_text(response).lower(), response_text(response)
            save_artifacts("rolling-camera", response)


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
async def test_ga_sdk_repeated_audio_and_concurrent_session_isolation(ga_server):
    repeats = int(os.environ.get("VLLM_OMNI_GA_REPEATS", "3"))
    records = []

    async def one(color, rgb):
        started_at = time.monotonic()
        async with AsyncOpenAI(base_url=ga_server.base_url, api_key="EMPTY") as client:
            async with client.realtime.connect(model=ga_server.model) as connection:
                await configure(
                    connection, "audio", instructions="Name only the dominant color of the image in English."
                )
                url = "data:image/jpeg;base64," + base64.b64encode(make_jpeg(*rgb, size=224)).decode()
                await create_item(connection, [{"type": "input_image", "image_url": url}])
                events = await respond(connection)
                text = response_text(events)
                assert color in text.lower(), text
                pcm = b"".join(
                    base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta"
                )
                assert len(pcm) > 2400 and len(pcm) % 2 == 0 and not pcm.startswith(b"RIFF")
                final = events[-1]["response"]["output"][0]
                assert next(e["item"] for e in events if e["type"] == "conversation.item.done") == final
                records.append(
                    {
                        "response_id": events[-1]["response"]["id"],
                        "text": text,
                        "audio_bytes": len(pcm),
                        "seconds": time.monotonic() - started_at,
                    }
                )

    for _ in range(repeats):
        await one("red", (255, 0, 0))
    await asyncio.gather(one("red", (255, 0, 0)), one("blue", (0, 0, 255)))
    assert len({record["response_id"] for record in records}) == repeats + 2
    directory = os.environ.get("VLLM_OMNI_GA_ARTIFACT_DIR")
    if directory:
        (Path(directory) / f"repeat-{ga_server.mode}.json").write_text(json.dumps(records, indent=2))


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
async def test_chat_completion_still_works_on_ga_server(ga_server):
    async with AsyncOpenAI(base_url=ga_server.base_url, api_key="EMPTY") as client:
        result = await client.chat.completions.create(
            model=ga_server.model,
            messages=[{"role": "user", "content": "What is the capital of China?"}],
            max_completion_tokens=64,
            extra_body={"modalities": ["text"]},
        )
        assert "beijing" in result.choices[0].message.content.lower()


@pytest.mark.asyncio
@hardware_test(res={"cuda": "H100"}, num_cards=2)
async def test_legacy_realtime_profile_override_preserves_pcm16_contract(ga_server, spoken_beijing_pcm):
    from scipy.signal import resample_poly

    samples = np.frombuffer(spoken_beijing_pcm, dtype="<i2").astype(np.float32)
    pcm16 = np.clip(np.rint(resample_poly(samples, 2, 3)), -32768, 32767).astype("<i2").tobytes()
    base = urlsplit(ga_server.base_url)
    uri = urlunsplit(("ws", base.netloc, "/v1/realtime", "profile=qwen3-legacy", ""))
    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": ga_server.model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
        for offset in range(0, len(pcm16), 3200):
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16[offset : offset + 3200]).decode(),
                    }
                )
            )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        events = await receive_legacy_until(ws, "response.output_audio.done")
        chunks = [base64.b64decode(e["audio"]) for e in events if e["type"] == "response.output_audio.delta"]
        assert chunks and sum(map(len, chunks)) > 2400
        assert any(e["type"] == "transcription.done" and e["text"] for e in events)
