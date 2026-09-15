# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Base WebSocket handler for streaming video input understanding.

Legacy WebSocket translation, EVS pre-filter and prewarm.
Conversation storage and response execution use the shared RealtimeRuntime;
Pipeline-specific prompt hooks adapt the legacy buffers to engine ``generate()`` streaming.
Subclasses supply trigger rules and prompt construction through :class:`VideoStreamPipelineHooks`.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "...", "frame_id": "...", "pts_ms": 0}
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "video.frame.ack", ...}          # when frame_id is provided
        {"type": "video.frames.consumed", ...}    # after first engine output
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.output_audio.delta", "data": "...", "format": "wav"}
        {"type": "response.output_audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

import asyncio
import base64
import hashlib
import io
import json
import time as _time
import uuid
import wave
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final, Protocol, TypeAlias, runtime_checkable

import torch
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai import video_stream_envs
from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter
from vllm_omni.entrypoints.openai.video_stream_context import (
    text_only_message,
)
from vllm_omni.entrypoints.realtime.contracts import CancelResponse, Content, CreateResponse, Item
from vllm_omni.entrypoints.realtime.legacy import HistoryView, LegacySession
from vllm_omni.entrypoints.realtime.qwen3 import Qwen3RealtimeAdapter
from vllm_omni.entrypoints.realtime.video import sample_frame_indices
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame
_MAX_BUFFER_FRAMES = 64
_MAX_AUDIO_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_MSG_QUEUE = 200
_CODEC_FRAME_SAMPLES = 1920  # CausalConv leading-edge artifact length


class _FrameStatus(Enum):
    BAD = "bad"


_BAD_FRAME: Final = _FrameStatus.BAD
PrewarmedFrame: TypeAlias = tuple[Any, str] | _FrameStatus


def _decode_frame_bytes(raw_bytes: bytes) -> Any:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


@runtime_checkable
class VideoStreamPipelineHooks(Protocol):
    """Pipeline-specific hooks for streaming video handlers."""

    def should_trigger_turn(self, trigger: "VideoStreamTurnTrigger") -> bool:
        """Return True to auto-start a turn after a new frame (no ``video.query``)."""
        ...

    def build_engine_prompt(
        self,
        config: "StreamingVideoSessionConfig",
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]] | HistoryView,
        query_text: str,
        prewarmed_frames: Mapping[str, PrewarmedFrame],
        *,
        frame_indices: list[int] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build messages using the supplied frame selection, or sample if omitted."""
        ...

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]] | HistoryView,
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        """Update session state after a successful turn."""
        ...


@dataclass(frozen=True)
class VideoStreamTurnTrigger:
    """Snapshot passed to :meth:`OmniStreamingVideoHandler.should_trigger_turn`."""

    frame_count: int
    is_generating: bool
    config: "StreamingVideoSessionConfig"


class StreamingVideoSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message."""

    model: str | None = None
    modalities: list[str] = Field(
        default_factory=lambda: ["text", "audio"],
        description="Output modalities: 'text', 'audio', or both.",
    )
    num_frames: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Max frames to sample from buffer for the model.",
    )
    max_frames: int = Field(
        default=50,
        ge=1,
        le=256,
        description="Max frames to keep in the buffer.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt.",
    )
    use_audio_in_video: bool = Field(
        default=True,
        description="Interleave audio chunks with video frames when audio input is present.",
    )
    sampling_params_list: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-stage sampling params [thinker, talker, code2wav].",
    )
    enable_frame_filter: bool = Field(
        default=True,
        description="EVS pixel-similarity pre-filter to drop near-duplicate frames.",
    )
    frame_filter_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="EVS similarity threshold (higher = keep more frames).",
    )


class OmniStreamingVideoHandler:
    """Base handler for WebSocket streaming video sessions.

    Subclasses implement :class:`VideoStreamPipelineHooks` to customize turn
    triggering, prompt construction, and history updates.
    """

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        """Auto-trigger after ``video.frame`` when True (default: never)."""
        return False

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]] | HistoryView,
        query_text: str,
        prewarmed_frames: Mapping[str, PrewarmedFrame],
        *,
        frame_indices: list[int] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]] | HistoryView,
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        raise NotImplementedError

    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        """Per-session conversation state (default: empty OpenAI-style list)."""
        return []

    def on_frame_buffered(
        self,
        raw_bytes: bytes,
        frame_b64: str,
        message_history: Any,
        config: StreamingVideoSessionConfig,
    ) -> None:
        """Hook after a frame is accepted into the session buffer."""
        del raw_bytes, frame_b64, message_history, config

    def __init__(
        self,
        chat_service: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
        engine_client: Any | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout
        self._engine_client = engine_client

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            legacy = LegacySession(
                Qwen3RealtimeAdapter(
                    self._chat_service, self._engine_client, preprocess=self._preprocess_to_engine_prompt
                ),
                model=config.model or "default",
                max_frames=config.max_frames,
            )
            frame_buffer = legacy.frames
            frame_metadata: list[dict[str, Any]] = []
            # Per-frame PIL cache + uuid for mm_hash reuse. Aligned with frame_buffer by index.
            frame_pil_cache: dict[str, PrewarmedFrame] = {}  # b64 -> (PIL.Image, uuid) or _BAD_FRAME
            frame_filter = (
                FrameSimilarityFilter(threshold=config.frame_filter_threshold) if config.enable_frame_filter else None
            )
            audio_buffer = legacy.store.audio_buffer
            message_history = legacy.history
            active_request_id: str | None = None
            interrupt_event = asyncio.Event()
            prewarm_tasks: set[asyncio.Task[Any]] = set()
            query_task: asyncio.Task[Any] | None = None

            msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_MAX_MSG_QUEUE)

            async def _reader() -> None:
                """Receive WebSocket messages and enqueue them."""
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                websocket.receive_text(),
                                timeout=self._idle_timeout,
                            )
                        except asyncio.TimeoutError:
                            await self._send_error(websocket, "Idle timeout")
                            await msg_queue.put(None)
                            return

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            await self._send_error(websocket, "Invalid JSON")
                            continue

                        if not isinstance(msg, dict):
                            await self._send_error(websocket, "Messages must be JSON objects")
                            continue

                        msg_type = str(msg.get("type", ""))
                        if msg_type.startswith("_internal."):
                            await self._send_error(websocket, f"Unknown type: {msg_type}")
                            continue
                        if msg_type == "video.frame":
                            msg["_receiver_received_ts_ms"] = _time.monotonic() * 1000

                        await msg_queue.put(msg)
                        if msg.get("type") == "video.done":
                            return
                except WebSocketDisconnect:
                    await msg_queue.put(None)
                except Exception:
                    await msg_queue.put(None)
                    raise

            async def _cancel_active_query() -> None:
                """Stop the projector; its cleanup waits for the shared runtime abort."""
                nonlocal active_request_id, query_task
                if active_request_id is not None:
                    interrupt_event.set()
                    logger.info("Interrupt signaled for %s", active_request_id)
                    if query_task is not None and not query_task.done():
                        query_task.cancel()
                        await asyncio.gather(query_task, return_exceptions=True)
                    query_task = None

            async def _start_query_turn(*, query_text: str) -> None:
                """Schedule a new inference turn from the current buffers."""
                nonlocal active_request_id, query_task

                await _cancel_active_query()

                if not frame_buffer:
                    await self._send_error(websocket, "No frames buffered")
                    return

                request_id = f"video-{uuid.uuid4().hex[:12]}"
                active_request_id = request_id
                interrupt_event.clear()
                query_frames = list(frame_buffer)
                query_frame_metadata = list(frame_metadata)
                query_audio_buffer = bytearray(audio_buffer)
                audio_buffer.clear()
                query_prewarmed_frames = dict(frame_pil_cache)

                async def _run_query() -> None:
                    nonlocal active_request_id
                    try:
                        process_kwargs: dict[str, Any] = {}
                        if any(metadata.get("frame_id") for metadata in query_frame_metadata):
                            process_kwargs["frame_metadata"] = query_frame_metadata
                        await self._process_query(
                            websocket,
                            config,
                            query_frames,
                            query_audio_buffer,
                            message_history,
                            query_text,
                            request_id,
                            interrupt_event,
                            query_prewarmed_frames,
                            **process_kwargs,
                        )
                    finally:
                        if active_request_id == request_id:
                            active_request_id = None

                query_task = asyncio.create_task(_run_query())

            async def _processor() -> None:
                """Process enqueued messages."""
                nonlocal active_request_id, query_task

                while True:
                    msg = await msg_queue.get()
                    if msg is None:
                        await _cancel_active_query()
                        return

                    msg_type = msg.get("type")

                    if msg_type == "_internal.frame_decode_failed":
                        frame_data = msg.get("b64", "")
                        removed = frame_data in frame_buffer
                        if removed:
                            retained_indices = [
                                index for index, frame in enumerate(frame_buffer) if frame != frame_data
                            ]
                            frame_buffer[:] = [frame_buffer[index] for index in retained_indices]
                            frame_metadata[:] = [frame_metadata[index] for index in retained_indices]
                        if frame_pil_cache.get(frame_data) is _BAD_FRAME:
                            frame_pil_cache.pop(frame_data, None)
                        if removed:
                            await self._send_error(websocket, "Frame decode failed")

                    elif msg_type == "video.frame":
                        frame_data = msg.get("data", "")
                        if not frame_data:
                            continue
                        if len(frame_data) > _MAX_FRAME_SIZE:
                            await self._send_error(websocket, "Frame too large")
                            continue
                        try:
                            raw_bytes = base64.b64decode(frame_data, validate=True)
                        except Exception:
                            await self._send_error(websocket, "Invalid image data")
                            continue
                        if frame_filter is not None:
                            try:
                                if not frame_filter.should_retain(raw_bytes):
                                    await self._send_frame_ack(
                                        websocket,
                                        msg,
                                        accepted=False,
                                        buffered_frames=len(frame_buffer),
                                        reason="filtered",
                                    )
                                    continue
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                        max_buf = config.max_frames
                        dropped_frame_id: str | None = None
                        if len(frame_buffer) >= max_buf:
                            dropped = frame_buffer.pop(0)
                            dropped_metadata = frame_metadata.pop(0)
                            dropped_frame_id = dropped_metadata.get("frame_id")
                            frame_pil_cache.pop(dropped, None)
                        frame_buffer.append(frame_data)
                        frame_metadata.append(
                            {
                                "frame_id": msg.get("frame_id"),
                                "pts_ms": msg.get("pts_ms"),
                                "source_pts_ms": msg.get("source_pts_ms"),
                                "quality_profile": msg.get("quality_profile"),
                                "capture_ts_ms": msg.get("capture_ts_ms"),
                                "receiver_received_ts_ms": msg.get("_receiver_received_ts_ms"),
                            }
                        )
                        self.on_frame_buffered(raw_bytes, frame_data, message_history, config)
                        await self._send_frame_ack(
                            websocket,
                            msg,
                            accepted=True,
                            buffered_frames=len(frame_buffer),
                            dropped_frame_id=dropped_frame_id,
                        )
                        # Prewarm: decode PIL off the event loop so query-time chat_template
                        # can skip base64+Image.open. uuid=md5 lets mm_cache dedupe identical frames.
                        if frame_data not in frame_pil_cache:
                            mm_uuid = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()

                            async def _prewarm(b64: str, b: bytes, u: str) -> None:
                                try:
                                    pil = await asyncio.to_thread(_decode_frame_bytes, b)
                                    # The frame may have been evicted while decoding.
                                    if b64 in frame_buffer:
                                        frame_pil_cache[b64] = (pil, u)
                                except Exception:
                                    if b64 not in frame_buffer:
                                        return
                                    frame_pil_cache[b64] = _BAD_FRAME
                                    logger.warning("prewarm decode failed for frame (len=%d)", len(b))
                                    try:
                                        msg_queue.put_nowait({"type": "_internal.frame_decode_failed", "b64": b64})
                                    except asyncio.QueueFull:
                                        logger.warning(
                                            "frame decode failure event dropped because message queue is full"
                                        )

                            task = asyncio.create_task(_prewarm(frame_data, raw_bytes, mm_uuid))
                            prewarm_tasks.add(task)
                            task.add_done_callback(prewarm_tasks.discard)

                        is_generating = active_request_id is not None or (
                            query_task is not None and not query_task.done()
                        )
                        if self.should_trigger_turn(
                            VideoStreamTurnTrigger(
                                frame_count=len(frame_buffer),
                                is_generating=is_generating,
                                config=config,
                            )
                        ):
                            await _start_query_turn(query_text="")

                    elif msg_type == "audio.chunk":
                        data_b64 = msg.get("data", "")
                        try:
                            pcm_bytes = base64.b64decode(data_b64)
                        except Exception:
                            continue
                        if len(audio_buffer) + len(pcm_bytes) > _MAX_AUDIO_BUFFER_BYTES:
                            await self._send_error(websocket, "Audio buffer overflow")
                            audio_buffer.clear()
                            continue
                        audio_buffer.extend(pcm_bytes)

                    elif msg_type == "video.query":
                        query_text = msg.get("text", "")
                        audio_data_b64 = msg.get("audio_data")
                        if audio_data_b64:
                            try:
                                decoded = base64.b64decode(audio_data_b64)
                                if len(audio_buffer) + len(decoded) <= _MAX_AUDIO_BUFFER_BYTES:
                                    audio_buffer.extend(decoded)
                                else:
                                    await self._send_error(websocket, "Audio buffer overflow")
                                    audio_buffer.clear()
                            except Exception:
                                pass

                        await _start_query_turn(query_text=query_text)

                    elif msg_type == "video.done":
                        if query_task is not None and not query_task.done():
                            await asyncio.gather(query_task, return_exceptions=True)
                            query_task = None
                        await websocket.send_json({"type": "session.done"})
                        return

                    elif msg_type == "ping":
                        try:
                            await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass

                    else:
                        await self._send_error(websocket, f"Unknown type: {msg_type}")

            reader_task = asyncio.create_task(_reader())
            try:
                await _processor()
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
                for t in list(prewarm_tasks):
                    t.cancel()
                if prewarm_tasks:
                    await asyncio.gather(*prewarm_tasks, return_exceptions=True)
                if query_task is not None and not query_task.done():
                    await _cancel_active_query()
                await legacy.runtime.close()

        except WebSocketDisconnect:
            logger.info("Streaming video: client disconnected")
        except Exception as e:
            logger.exception("Streaming video session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    async def _receive_config(self, websocket: WebSocket) -> StreamingVideoSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        config_data = {k: v for k, v in msg.items() if k != "type"}
        alias_map = {
            "num_sample_frames": "num_frames",
            "evs_enabled": "enable_frame_filter",
            "evs_threshold": "frame_filter_threshold",
        }
        for old_key, new_key in alias_map.items():
            if old_key in config_data and new_key not in config_data:
                config_data[new_key] = config_data[old_key]

        try:
            config = StreamingVideoSessionConfig(**config_data)
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _process_query(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]] | HistoryView,
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: Mapping[str, PrewarmedFrame],
        frame_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build prompt, run inference, stream text + audio response."""

        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        engine_kwargs: dict[str, Any] = {}
        if frame_metadata:
            engine_kwargs["frame_metadata"] = frame_metadata
        await self._process_query_engine(
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
            **engine_kwargs,
        )

    # ------------------------------------------------------------------
    # Engine-client path (async_chunk audio streaming)
    # ------------------------------------------------------------------

    async def _process_query_engine(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]] | HistoryView,
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: Mapping[str, PrewarmedFrame],
        frame_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Project a shared-runtime response into legacy text/WAV events."""
        from contextlib import aclosing

        frame_indices = self._sample_frame_indices(frame_buffer, config.num_frames, prewarmed_frames)
        messages, user_message = self.build_engine_prompt(
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
            frame_indices=frame_indices,
        )
        persistent = isinstance(message_history, HistoryView)
        legacy = (
            message_history.session
            if isinstance(message_history, HistoryView)
            else LegacySession(
                Qwen3RealtimeAdapter(
                    self._chat_service, self._engine_client, preprocess=self._preprocess_to_engine_prompt
                ),
                model=config.model or "default",
                max_frames=config.max_frames,
            )
        )
        runtime = legacy.runtime
        try:
            legacy.adapter.prepared = await legacy.adapter.adapter.prepare(
                messages,
                {
                    "model": config.model or "default",
                    "output_modalities": config.modalities,
                    "omni": {
                        "_legacy_output_modalities": True,
                        "use_audio_in_video": config.use_audio_in_video,
                        "sampling_params_list": config.sampling_params_list,
                        "_audio_chunk_mode": video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK,
                        "_audio_delta_mode": video_stream_envs.VLLM_VIDEO_AUDIO_DELTA_MODE,
                    },
                },
                has_audio=bool(audio_buffer),
            )
        except Exception as exc:
            await self._send_error(websocket, f"Failed to build or preprocess request: {exc}")
            if not persistent:
                await runtime.close()
            return
        decoded_ready_ts_ms = _time.monotonic() * 1000
        selected_metadata = [frame_metadata[index] for index in frame_indices] if frame_metadata else []
        model_selected_ts_ms = _time.monotonic() * 1000
        # Only text from previous legacy turns is retained; selected media lives
        # in the immutable prepared prompt until the shared iterator is closed.
        user_item = Item("user", (Content("text", text=query_text),))
        runtime.store.insert(user_item)
        response_config = replace(
            runtime.store.config, output_mode="audio" if config.modalities == ["audio"] else "text"
        )
        await runtime.submit(CreateResponse(config=response_config, request_id=request_id))
        audio_sent = False
        consumed_sent = False
        completed = False
        try:
            async with aclosing(runtime.events()) as events:
                async for event in events:
                    if event.delivery is not None and event.delivery.cancelled:
                        continue
                    if event.kind == "response_started":
                        await websocket.send_json({"type": "response.start"})
                    elif event.kind == "delta":
                        if not consumed_sent and frame_metadata:
                            await websocket.send_json(
                                {
                                    "type": "video.frames.consumed",
                                    "request_id": request_id,
                                    "model_selected_ts_ms": model_selected_ts_ms,
                                    "frame_ids": [
                                        m["frame_id"] for m in selected_metadata if isinstance(m.get("frame_id"), str)
                                    ],
                                    "frames": [
                                        {
                                            **{
                                                key: m.get(key)
                                                for key in (
                                                    "frame_id",
                                                    "pts_ms",
                                                    "source_pts_ms",
                                                    "quality_profile",
                                                    "receiver_received_ts_ms",
                                                )
                                            },
                                            "decoded_ready_ts_ms": decoded_ready_ts_ms,
                                        }
                                        for m in selected_metadata
                                    ],
                                    "latest_pts_ms": selected_metadata[-1].get("pts_ms") if selected_metadata else None,
                                }
                            )
                            consumed_sent = True
                        delta = event.value
                        if delta.text and video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on":
                            await websocket.send_json({"type": "response.text.delta", "delta": delta.text})
                        if delta.audio is not None:
                            await websocket.send_json(
                                {
                                    "type": "response.output_audio.delta",
                                    "data": self._encode_audio_wav_b64(delta.audio),
                                    "format": "wav",
                                }
                            )
                            audio_sent = True
                    elif event.kind == "response_finished":
                        response = event.value
                        if response.status == "cancelled":
                            break
                        text = response.item.content[0].text
                        await websocket.send_json({"type": "response.text.done", "text": text})
                        if audio_sent:
                            await websocket.send_json({"type": "response.output_audio.done"})
                        if response.status == "failed":
                            await self._send_error(websocket, "Query processing failed")
                        if not persistent:
                            self.on_turn_complete(message_history, user_message, text)
                        completed = True
                        break
        finally:
            if runtime.active_response:
                await asyncio.shield(runtime.submit(CancelResponse()))
            if persistent:
                if not completed:
                    for item in (user_item, runtime.store.last_response.item if runtime.store.last_response else None):
                        if item is not None and any(existing.id == item.id for existing in runtime.store.items):
                            runtime.store.delete(item.id)
                legacy.prune_history()
                # A cancelled projector leaves its terminal queued. Drain it
                # before the next response gets a new projector.
                runtime.discard_pending_events()
            else:
                await runtime.close()

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pcm_to_wav_b64(pcm_data: bytes, sample_rate: int = 16000) -> str:
        """Wrap raw PCM16 mono in a WAV container and return base64."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return base64.b64encode(buf.getvalue()).decode()

    @classmethod
    def _extract_audio_delta_b64(
        cls,
        result: OmniRequestOutput,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Return (base64 WAV of new samples, updated chunks_drained).

        `chunks_drained` is the number of per-step tensors in
        ``audio_data`` that have already been emitted. Each engine step appends
        one tensor, so new samples are ``audio_data[chunks_drained:]`` — no
        matter how many steps accumulated between reads (handles backpressure
        cleanly, unlike a simple ``audio_data[-1]``).

        Two paths, selected at runtime by ``VLLM_VIDEO_AUDIO_DELTA_MODE``:
          * fast — only D2H the new tail. Per-call cost ∝ new chunks.
          * slow — full cat + D2H each call. Per-call cost ∝ total history.
                   Retained for A/B; remove once downstream callers confirm.
        """
        audio_data = cls._get_audio_data(result)
        if audio_data is None:
            return None, chunks_drained

        if video_stream_envs.VLLM_VIDEO_AUDIO_DELTA_MODE == "slow":
            return cls._delta_slow(audio_data, chunks_drained)
        return cls._delta_fast(audio_data, chunks_drained)

    @staticmethod
    def _get_audio_data(result: OmniRequestOutput):
        """Navigate OmniRequestOutput → multimodal_output['audio']. None on miss."""
        request_output = result
        if request_output is None:
            return None
        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return None
        mm_output = getattr(outputs[0], "multimodal_output", None)
        if not isinstance(mm_output, Mapping):
            return None
        return mm_output.get("audio")

    @classmethod
    def _delta_fast(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Emit only tensors appended since the last call."""
        # Single tensor: output_processor hands us one tensor before it becomes a
        # list (see output_processor.py:89). Treat it as chunk #0.
        if not isinstance(audio_data, list):
            if chunks_drained >= 1:
                return None, chunks_drained
            tail_np = cls._tensor_to_1d_np(audio_data)
            return cls._encode_tail(tail_np, chunks_drained, new_drained=1, is_first=True)

        n = len(audio_data)
        if n <= chunks_drained:
            return None, chunks_drained

        new_chunks = audio_data[chunks_drained:]
        tail = new_chunks[0] if len(new_chunks) == 1 else torch.cat(new_chunks, dim=-1)
        tail_np = cls._tensor_to_1d_np(tail)
        return cls._encode_tail(tail_np, chunks_drained, new_drained=n, is_first=(chunks_drained == 0))

    @classmethod
    def _delta_slow(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Pre-fix behaviour: concat everything each call and slice on CPU."""
        if isinstance(audio_data, list):
            if not audio_data:
                return None, chunks_drained
            audio_tensor = torch.cat(audio_data, dim=-1)
            new_drained = len(audio_data)
        else:
            audio_tensor = audio_data
            new_drained = 1

        full_np = cls._tensor_to_1d_np(audio_tensor)
        if full_np is None:
            return None, chunks_drained
        # chunks_drained doesn't map directly to sample offset without tracking
        # per-chunk lengths, so we re-derive: replay the tail that corresponds
        # to chunks appended since last call by slicing off the part produced
        # by the already-drained prefix. For slow path this is intentionally
        # wasteful — the point is to reproduce the pre-fix hot loop.
        if chunks_drained == 0:
            tail_np = full_np
        else:
            # Recover prefix length by re-concatenating the already-drained
            # prefix tensors (cost intentionally identical to the baseline
            # implementation this was lifted from).
            if isinstance(audio_data, list) and chunks_drained < len(audio_data):
                prefix_len = sum(int(t.shape[-1]) for t in audio_data[:chunks_drained])
                tail_np = full_np[prefix_len:]
            else:
                tail_np = full_np[0:0]
        return cls._encode_tail(tail_np, chunks_drained, new_drained=new_drained, is_first=(chunks_drained == 0))

    @classmethod
    def _encode_tail(
        cls,
        tail_np,
        old_drained: int,
        *,
        new_drained: int,
        is_first: bool,
    ) -> tuple[str | None, int]:
        """Strip the CausalConv leading artifact on first emit, then b64-encode."""
        if tail_np is None or len(tail_np) == 0:
            return None, new_drained
        if is_first and len(tail_np) > _CODEC_FRAME_SAMPLES * 2:
            tail_np = tail_np[_CODEC_FRAME_SAMPLES:]
        if len(tail_np) == 0:
            return None, new_drained
        try:
            return cls._encode_audio_wav_b64(tail_np), new_drained
        except Exception:
            logger.exception("Failed to encode audio delta WAV")
            return None, old_drained

    @staticmethod
    def _tensor_to_1d_np(t):
        """Tensor → flat float32 numpy on CPU. None on failure."""
        if t is None or not hasattr(t, "float"):
            return None
        arr = t.float().detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr

    @staticmethod
    def _encode_audio_wav_b64(audio_np) -> str:
        """Encode numpy float32 audio to base64 WAV (24kHz)."""
        from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
        from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

        audio_obj = CreateAudio(
            audio_tensor=audio_np,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            base64_encode=True,
        )
        mixin = AudioMixin()
        resp = mixin.create_audio(audio_obj)
        # base64_encode=True selects the string result of create_audio().
        assert isinstance(resp.audio_data, str)
        return resp.audio_data

    @staticmethod
    def _extract_text_delta(
        result: OmniRequestOutput,
        previous_text: str,
    ) -> tuple[str, str]:
        """Extract incremental text delta from OmniRequestOutput."""
        if result.final_output_type != "text":
            return "", previous_text

        request_output = result
        if request_output is None:
            return "", previous_text

        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return "", previous_text

        text = getattr(outputs[0], "text", None)
        if not isinstance(text, str) or not text:
            return "", previous_text

        if text.startswith(previous_text):
            return text[len(previous_text) :], text
        return text, text

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    async def _preprocess_to_engine_prompt(self, request) -> Any:
        """Use the chat handler's preprocessing to build an engine prompt."""
        handler = self._chat_service
        renderer = handler.renderer

        _conversation, engine_prompts = await handler._preprocess_chat(
            request,
            request.messages,
            default_template=getattr(request, "chat_template", None) or handler.chat_template,
            default_template_content_format=handler.chat_template_content_format,
            renderer=renderer,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            add_special_tokens=request.add_special_tokens,
        )
        return engine_prompts[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    _text_only_message = staticmethod(text_only_message)

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass

    @staticmethod
    def _sample_frame_indices(
        frame_buffer: list[str],
        num_frames: int,
        prewarmed_frames: Mapping[str, PrewarmedFrame],
    ) -> list[int]:
        """Stride-sample with the last frame, then drop known bad frames without refilling."""
        indices = sample_frame_indices(len(frame_buffer), num_frames)
        return [index for index in indices if prewarmed_frames.get(frame_buffer[index]) is not _BAD_FRAME]

    @staticmethod
    async def _send_frame_ack(
        websocket: WebSocket,
        message: Mapping[str, Any],
        *,
        accepted: bool,
        buffered_frames: int,
        reason: str | None = None,
        dropped_frame_id: str | None = None,
    ) -> None:
        frame_id = message.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            return
        ack: dict[str, Any] = {
            "type": "video.frame.ack",
            "frame_id": frame_id,
            "pts_ms": message.get("pts_ms"),
            "capture_ts_ms": message.get("capture_ts_ms"),
            "accepted": accepted,
            "buffered_frames": buffered_frames,
            "server_receive_ts_ms": message.get("_receiver_received_ts_ms", _time.monotonic() * 1000),
        }
        if reason is not None:
            ack["reason"] = reason
        if dropped_frame_id is not None:
            ack["dropped_frame_id"] = dropped_frame_id
        await websocket.send_json(ack)
