# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render runtime facts as the GA response/item/content-part lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pybase64 as base64

from vllm_omni.entrypoints.openai.realtime.contracts import Content, Response, RuntimeEvent, new_id
from vllm_omni.utils.audio_resample import StreamingAudioResampler

from .codec import encode_item, session_config


def event(kind: str, **fields: Any) -> dict[str, Any]:
    return {"type": kind, "event_id": new_id("event"), **fields}


def error_event(error, client_event_id=None):
    return event(
        "error",
        error={
            "type": "server_error" if error.code == "cleanup_failed" else "invalid_request_error",
            "code": error.code,
            "message": str(error),
            "param": error.param,
            "event_id": client_event_id,
        },
    )


class GAEventEncoder:
    """Only wire projection state: identities and streaming PCM conversion."""

    def __init__(self, session_id: str, conversation_id: str):
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.responses: dict[str, Response] = {}
        self.resamplers: dict[str, StreamingAudioResampler] = {}

    def session(self, config):
        return {**session_config(config), "id": self.session_id, "object": "realtime.session"}

    def response(self, response: Response):
        details: dict[str, Any] | None = None
        if response.status in ("cancelled", "incomplete"):
            details = {"type": response.status, "reason": response.reason}
        elif response.status == "failed":
            details = {
                "type": "failed",
                "error": {"type": "server_error", "code": response.reason, "message": response.error},
            }
        return {
            "id": response.id,
            "object": "realtime.response",
            "status": response.status,
            "status_details": details,
            "output": [] if response.status == "in_progress" else [encode_item(response.item)],
            "conversation_id": self.conversation_id,
            "output_modalities": [response.mode],
            "metadata": None,
            "usage": None,
        }

    @staticmethod
    def part_ids(response):
        return {"response_id": response.id, "item_id": response.item.id, "output_index": 0, "content_index": 0}

    @staticmethod
    def part(content: Content):
        # GA part events use text/audio; message content uses output_text/output_audio.
        return (
            {"type": "audio", "transcript": content.text}
            if content.kind == "audio"
            else {"type": "text", "text": content.text}
        )

    def pcm(self, response, samples, rate, *, final=False):
        resampler = self.resamplers.get(response.id)
        if rate != 24000 or resampler is not None:
            if resampler is None:
                resampler = StreamingAudioResampler(rate, 24000)
                self.resamplers[response.id] = resampler
            samples = resampler.process(samples, final=final)
        if not samples.size:
            return []
        raw = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        # Non-streaming engines may return minutes of audio in one output.
        # Keep each GA message below ordinary SDK/WebSocket receive limits.
        chunk_bytes = 2400 * 2  # 100 ms of mono PCM16 at 24 kHz.
        return [
            event(
                "response.output_audio.delta",
                **self.part_ids(response),
                delta=base64.b64encode(raw[offset : offset + chunk_bytes]).decode("ascii"),
            )
            for offset in range(0, len(raw), chunk_bytes)
        ]

    def encode(self, fact: RuntimeEvent) -> list[dict[str, Any]]:
        kind, value = fact.kind, fact.value
        if kind == "configured":
            return [event("session.updated", session=self.session(value))]
        if kind in ("item_inserted", "audio_committed"):
            result = [event("conversation.item.deleted", item_id=item_id) for item_id in value.deleted_ids]
            if kind == "audio_committed":
                result.append(
                    event("input_audio_buffer.committed", item_id=value.item.id, previous_item_id=value.previous_id)
                )
            item = encode_item(value.item)
            result.extend(
                [
                    event("conversation.item.added", item=item, previous_item_id=value.previous_id),
                    event("conversation.item.done", item=item, previous_item_id=value.previous_id),
                ]
            )
            return result
        if kind == "item_deleted":
            return [event("conversation.item.deleted", item_id=value)]
        if kind == "audio_cleared":
            return [event("input_audio_buffer.cleared")]
        if kind == "response_started":
            response, previous = value
            self.responses[response.id] = response
            item = encode_item(response.item)
            part = Content("audio" if response.mode == "audio" else "text")
            return [
                event("response.created", response=self.response(response)),
                event("response.output_item.added", response_id=response.id, output_index=0, item=item),
                event("conversation.item.added", item=item, previous_item_id=previous),
                event("response.content_part.added", **self.part_ids(response), part=self.part(part)),
            ]
        if kind == "delta":
            assert fact.response_id is not None
            response = self.responses[fact.response_id]
            result = []
            text = value.transcript if response.mode == "audio" else value.text
            if text:
                name = (
                    "response.output_audio_transcript.delta"
                    if response.mode == "audio"
                    else "response.output_text.delta"
                )
                result.append(event(name, **self.part_ids(response), delta=text))
            if value.audio is not None and response.mode == "audio":
                result.extend(
                    self.pcm(response, np.asarray(value.audio, dtype=np.float32).reshape(-1), value.sample_rate)
                )
            return result
        if kind == "response_finished":
            response = value
            ids, part = self.part_ids(response), response.item.content[0]
            result = []
            resampler = self.resamplers.get(response.id)
            if resampler is not None and response.status != "cancelled":
                # Flush the same stream state; never resample chunks independently.
                result.extend(self.pcm(response, np.empty(0, dtype=np.float32), 24000, final=True))
            if response.mode == "audio":
                result.extend(
                    [
                        event("response.output_audio_transcript.done", **ids, transcript=part.text),
                        event("response.output_audio.done", **ids),
                    ]
                )
            else:
                result.append(event("response.output_text.done", **ids, text=part.text))
            item = encode_item(response.item)
            result.extend(
                [
                    event("response.content_part.done", **ids, part=self.part(part)),
                    event("response.output_item.done", response_id=response.id, output_index=0, item=item),
                    event("conversation.item.done", item=item),
                    event("response.done", response=self.response(response)),
                ]
            )
            self.responses.pop(response.id, None)
            self.resamplers.pop(response.id, None)
            return result
        raise TypeError(f"Unsupported runtime event: {kind}")
