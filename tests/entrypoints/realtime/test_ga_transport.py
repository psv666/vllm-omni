# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GA lifecycle and selector contracts, without model weights."""

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
from openai.types.realtime.realtime_server_event import RealtimeServerEvent
from pydantic import TypeAdapter
from starlette.websockets import WebSocketDisconnect

from tests.entrypoints.realtime.test_runtime import Adapter
from vllm_omni.entrypoints.openai.realtime.codec import decode_event, encode_item
from vllm_omni.entrypoints.openai.realtime.connection import RealtimeGAConnection
from vllm_omni.entrypoints.openai.realtime.contracts import RealtimeError, SessionConfig
from vllm_omni.entrypoints.openai.realtime.events import GAEventEncoder
from vllm_omni.entrypoints.openai.realtime.routing import select_realtime_route

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Socket:
    def __init__(self):
        self.input: asyncio.Queue[dict[str, Any] | Exception] = asyncio.Queue()
        self.output: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent = []

    async def accept(self):
        pass

    async def close(self):
        pass

    async def receive_json(self):
        item = await self.input.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def send_json(self, item):
        self.sent.append(item)
        await self.output.put(item)

    async def until(self, kind):
        while True:
            item = await asyncio.wait_for(self.output.get(), timeout=3)
            if item["type"] == kind:
                return item


@asynccontextmanager
async def connection(adapter=None, socket=None):
    ws = socket or Socket()
    connection = RealtimeGAConnection(ws, adapter or Adapter(), "test")
    task = asyncio.create_task(connection.handle_connection())
    try:
        yield ws, connection
    finally:
        ws.input.put_nowait(WebSocketDisconnect())
        await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "audio"])
@pytest.mark.parametrize("finish", ["completed", "failed", "cancelled"])
async def test_complete_lifecycle_and_schema(mode, finish):
    adapter = Adapter(hold=finish == "cancelled", fail=finish == "failed")
    async with connection(adapter) as (ws, conn):
        await ws.until("session.created")
        ws.input.put_nowait({"type": "session.update", "session": {"type": "realtime", "output_modalities": [mode]}})
        await ws.until("session.updated")
        ws.input.put_nowait(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "user",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
            }
        )
        await ws.until("conversation.item.done")
        first = len(ws.sent)
        ws.input.put_nowait({"type": "response.create"})
        await ws.until("response.content_part.added")
        if finish == "cancelled":
            await ws.until(
                "response.output_audio_transcript.delta" if mode == "audio" else "response.output_text.delta"
            )
            ws.input.put_nowait({"type": "response.cancel"})
        done = await ws.until("response.done")
        assert done["response"]["status"] == finish
        events = ws.sent[first:]
        kinds = [event["type"] for event in events]
        assert kinds[:4] == [
            "response.created",
            "response.output_item.added",
            "conversation.item.added",
            "response.content_part.added",
        ]
        assert kinds[-4:] == [
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
        ]
        assert kinds.count("response.done") == 1
        item = done["response"]["output"][0]
        assert encode_item(conn.runtime.store.items[-1]) == item
        assert next(e for e in events if e["type"] == "response.output_item.done")["item"] == item
        schema = TypeAdapter(RealtimeServerEvent)
        for event in ws.sent:
            schema.validate_python(event, strict=True)
        if mode == "audio":
            pcm = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta")
            assert pcm and len(pcm) % 2 == 0 and not pcm.startswith(b"RIFF")


@pytest.mark.parametrize(
    ("duplex", "profile", "has_duplex", "has_ga", "default", "expected"),
    [
        ("1", None, True, True, "openai-realtime", "duplex"),
        ("1", "openai-realtime", True, True, "qwen3-legacy", "incompatible_parameters"),
        ("1", "qwen3-legacy", True, True, "qwen3-legacy", "incompatible_parameters"),
        ("1", None, False, True, "openai-realtime", "unsupported"),
        (None, None, True, False, "qwen3-legacy", "duplex"),
        (None, "openai-realtime", True, False, "qwen3-legacy", "incompatible_parameters"),
        ("0", "openai-realtime", False, True, "qwen3-legacy", "ga"),
        (None, None, False, True, "qwen3-legacy", "legacy"),
        (None, None, False, True, "openai-realtime", "ga"),
        (None, "qwen3-legacy", False, True, "openai-realtime", "legacy"),
        (None, "unknown", False, True, "qwen3-legacy", "invalid_value"),
    ],
)
def test_route_precedence(duplex, profile, has_duplex, has_ga, default, expected):
    kwargs = dict(duplex=duplex, profile=profile, has_duplex=has_duplex, has_ga=has_ga, default_profile=default)
    if expected in {"ga", "legacy", "duplex"}:
        assert select_realtime_route(**kwargs) == expected
    else:
        with pytest.raises(RealtimeError) as exc:
            select_realtime_route(**kwargs)
        assert exc.value.code == expected


@pytest.mark.parametrize(
    "event,param",
    [
        ({"type": "input_audio_buffer.append", "audio": "", "video_frames": []}, "video_frames"),
        ({"type": "input_audio_buffer.commit", "final": True}, "final"),
        ({"type": "input_audio_buffer.append", "audio": "!!!"}, "audio"),
        ({"type": "session.update", "session": {"output_modalities": ["text", "audio"]}}, "session.output_modalities"),
        (
            {
                "type": "session.update",
                "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}},
            },
            "session.audio.input.format.rate",
        ),
    ],
)
def test_invalid_ga_fields(event, param):
    with pytest.raises(RealtimeError) as exc:
        decode_event(event, SessionConfig("test"))
    assert exc.value.param == param


@pytest.mark.asyncio
async def test_bad_item_is_correlated_and_connection_remains_usable():
    async with connection() as (ws, _):
        await ws.until("session.created")
        ws.input.put_nowait(
            {
                "type": "conversation.item.create",
                "event_id": "bad",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,eA=="}],
                },
            }
        )
        error = await ws.until("error")
        assert error["error"]["event_id"] == "bad"
        assert error["error"]["param"] == "item.content[0].image_url"
        ws.input.put_nowait({"type": "session.update", "session": {"output_modalities": ["text"]}})
        assert (await ws.until("session.updated"))["session"]["output_modalities"] == ["text"]


@pytest.mark.asyncio
async def test_failed_websocket_send_aborts_model_and_clears_state():
    class FailedSocket(Socket):
        async def send_json(self, item):
            if item["type"] == "response.output_text.delta":
                raise WebSocketDisconnect()
            await super().send_json(item)

    ws = FailedSocket()
    adapter = Adapter(hold=True)
    conn = RealtimeGAConnection(ws, adapter, "test")
    task = asyncio.create_task(conn.handle_connection())
    await ws.until("session.created")
    ws.input.put_nowait({"type": "session.update", "session": {"output_modalities": ["text"]}})
    ws.input.put_nowait(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "question"}],
            },
        }
    )
    ws.input.put_nowait({"type": "response.create"})
    await asyncio.wait_for(task, timeout=3)
    assert len(adapter.aborted) == 1 and adapter.closed == adapter.aborted
    assert conn.runtime.store.items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected",
    [
        ({"duplex": "1"}, "duplex"),
        ({"duplex": "1", "profile": "openai-realtime"}, "error"),
        ({"duplex": "1", "profile": "qwen3-legacy"}, "error"),
    ],
)
async def test_real_route_rejects_conflicts_before_allocating_qwen_state(monkeypatch, query, expected):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from vllm_omni.entrypoints.openai.api_server import realtime_websocket
    from vllm_omni.entrypoints.openai.realtime import connection as connection_module

    duplex = SimpleNamespace(handle_realtime_session=AsyncMock())
    constructor = Mock(side_effect=AssertionError("Qwen state must not be allocated"))
    monkeypatch.setattr(connection_module, "RealtimeGAConnection", constructor)
    ws = SimpleNamespace(
        query_params=query,
        accept=AsyncMock(),
        send_json=AsyncMock(),
        close=AsyncMock(),
        app=SimpleNamespace(
            state=SimpleNamespace(
                openai_serving_duplex=duplex,
                openai_serving_realtime_ga=object(),
                realtime_profile="openai-realtime",
            )
        ),
    )
    await realtime_websocket(ws)
    constructor.assert_not_called()
    if expected == "duplex":
        duplex.handle_realtime_session.assert_awaited_once_with(ws)
    else:
        duplex.handle_realtime_session.assert_not_awaited()
        assert ws.send_json.call_args.args[0]["error"]["code"] == "incompatible_parameters"
        ws.close.assert_awaited_once()


def test_streaming_pcm_conversion_preserves_chunk_boundaries():
    from vllm_omni.entrypoints.openai.realtime.contracts import Item, Response

    response = Response("resp", "req", Item("assistant", ()), "audio")
    samples = np.sin(np.arange(4800, dtype=np.float32) / 100)
    encoder = GAEventEncoder("session", "conversation")
    chunks = []
    for part in np.array_split(samples, 13):
        chunks.extend(encoder.pcm(response, part, 48000))
    chunks.extend(encoder.pcm(response, np.empty(0, dtype=np.float32), 48000, final=True))
    whole = GAEventEncoder("session", "conversation").pcm(response, samples, 48000, final=True)
    raw = b"".join(base64.b64decode(event["delta"]) for event in chunks)
    expected = b"".join(base64.b64decode(event["delta"]) for event in whole)
    assert raw == expected and len(raw) == 4800


def test_large_buffered_audio_is_split_below_sdk_limit_without_changing_samples():
    import json

    import numpy as np

    from vllm_omni.entrypoints.openai.realtime.contracts import Item, Response

    response = Response("resp", "req", Item("assistant", ()), "audio")
    samples = np.sin(np.arange(500_000, dtype=np.float32) / 30)
    events = GAEventEncoder("session", "conversation").pcm(response, samples, 24000)
    assert len(events) > 100
    assert max(len(json.dumps(event)) for event in events) < 1024 * 1024
    raw = b"".join(base64.b64decode(event["delta"]) for event in events)
    expected = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    assert raw == expected


@pytest.mark.asyncio
async def test_active_generation_is_not_an_idle_connection():
    ws = Socket()
    adapter = Adapter(hold=True)
    conn = RealtimeGAConnection(ws, adapter, "test", idle_timeout=0.02)
    task = asyncio.create_task(conn.handle_connection())
    try:
        await ws.until("session.created")
        ws.input.put_nowait(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
            }
        )
        ws.input.put_nowait({"type": "response.create"})
        await ws.until("response.output_audio.delta")
        await asyncio.sleep(0.06)
        assert not task.done()
        assert not any(event["type"] == "error" for event in ws.sent)
        ws.input.put_nowait({"type": "response.cancel"})
        await ws.until("response.done")
    finally:
        ws.input.put_nowait(WebSocketDisconnect())
        await asyncio.wait_for(task, timeout=3)
