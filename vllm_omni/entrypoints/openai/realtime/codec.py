# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GA field validation and conversion to transport-independent commands."""

from __future__ import annotations

import binascii
import copy
import io
import uuid
import warnings
from typing import Any

import pybase64 as base64
from PIL import Image, UnidentifiedImageError

from vllm_omni.entrypoints.realtime.contracts import (
    AppendAudio,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    Content,
    CreateResponse,
    DeleteItem,
    InsertItem,
    Item,
    RealtimeError,
    SessionConfig,
    UpdateSession,
    VisualPolicy,
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_APPEND_BYTES = 15 * 1024 * 1024
_PCM_FORMAT = {"type": "audio/pcm", "rate": 24000}


def _object(value: Any, param: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RealtimeError("Expected an object.", param)
    return value


def _known_fields(value: dict[str, Any], allowed: set[str], param: str) -> None:
    for key in value:
        if key not in allowed:
            raise RealtimeError(
                "This parameter is not supported by the Qwen3 GA profile.", f"{param}.{key}" if param else key
            )


def _integer(value: Any, minimum: int, maximum: int, param: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise RealtimeError(f"Expected an integer between {minimum} and {maximum}.", param)


def _modalities(value: Any, param: str) -> None:
    if value != ["text"] and value != ["audio"]:
        raise RealtimeError('Use ["text"] or ["audio"]; audio includes its spoken transcript.', param)


def _max_tokens(value: Any, param: str) -> None:
    if value != "inf":
        _integer(value, 1, 4096, param)


def _decode_base64(value: Any, param: str, limit: int | None = None) -> bytes:
    if not isinstance(value, str):
        raise RealtimeError("Expected a base64-encoded string.", param)
    if limit is not None and len(value) > ((limit + 2) // 3) * 4:
        raise RealtimeError(f"Decoded media exceeds the {limit}-byte limit.", param)
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RealtimeError("Invalid base64-encoded media.", param) from exc
    if limit is not None and len(decoded) > limit:
        raise RealtimeError(f"Decoded media exceeds the {limit}-byte limit.", param)
    return decoded


def _validate_pcm(value: Any, param: str) -> None:
    decoded = _decode_base64(value, param)
    if not decoded or len(decoded) % 2:
        raise RealtimeError("Expected non-empty, complete PCM16 samples at 24000 Hz.", param)


def _validate_image(value: Any, param: str) -> None:
    if not isinstance(value, str):
        raise RealtimeError("Expected a JPEG or PNG base64 data URL.", param)
    header, separator, encoded = value.partition(",")
    formats = {"data:image/jpeg;base64": "JPEG", "data:image/png;base64": "PNG"}
    if not separator or header not in formats:
        raise RealtimeError("Only JPEG and PNG base64 data URLs are supported.", param)
    decoded = _decode_base64(encoded, param, MAX_IMAGE_BYTES)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(decoded)) as image:
                if image.format != formats[header]:
                    raise RealtimeError("Image format does not match its data URL.", param)
                if image.width <= 0 or image.height <= 0:
                    raise RealtimeError("Image dimensions must be positive.", param)
                # Validate the pixels now, so a malformed image never receives
                # a successful item acknowledgement and fails only at inference.
                image.load()
    except RealtimeError:
        raise
    except (
        ValueError,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        # PNG metadata decompression limits raise ValueError rather than
        # OSError; invalid media must remain a recoverable client error.
        raise RealtimeError("Invalid image data or excessive image dimensions.", param) from exc


class _Validation:
    @staticmethod
    def _update_audio(audio: dict[str, Any], changes: Any) -> None:
        changes = _object(changes, "session.audio")
        _known_fields(changes, {"input", "output"}, "session.audio")
        for direction, values in changes.items():
            param = f"session.audio.{direction}"
            values = _object(values, param)
            allowed = (
                {"format", "turn_detection", "noise_reduction", "transcription"} if direction == "input" else {"format"}
            )
            _known_fields(values, allowed, param)
            for key, value in values.items():
                field = f"{param}.{key}"
                if key == "format":
                    value = _object(value, field)
                    _known_fields(value, {"type", "rate"}, field)
                    if value.get("type", "audio/pcm") != "audio/pcm":
                        raise RealtimeError("Only PCM16 audio/pcm is supported.", f"{field}.type")
                    if type(value.get("rate", 24000)) is not int or value.get("rate", 24000) != 24000:
                        raise RealtimeError("The GA PCM sample rate is 24000 Hz.", f"{field}.rate")
                    audio[direction][key] = dict(_PCM_FORMAT)
                else:
                    if value is not None:
                        raise RealtimeError("This feature is not supported; use null to disable it.", field)
                    audio[direction][key] = None

    @staticmethod
    def _update_omni(omni: dict[str, Any], changes: Any) -> None:
        changes = _object(changes, "session.omni")
        _known_fields(changes, set(omni), "session.omni")
        for key, value in changes.items():
            param = f"session.omni.{key}"
            if key == "input_image_retention":
                if value not in ("client", "rolling"):
                    raise RealtimeError("Use client or rolling image retention.", param)
            elif key == "input_image_max_items":
                _integer(value, 1, 256, param)
            elif key == "input_image_sample":
                _integer(value, 1, 128, param)
            elif key in ("input_image_evs", "use_audio_in_video"):
                if type(value) is not bool:
                    raise RealtimeError("Expected a boolean.", param)
            elif key == "input_image_evs_threshold":
                if type(value) not in (float, int) or not 0 <= value <= 1:
                    raise RealtimeError("Expected a finite number between 0 and 1.", param)
            omni[key] = copy.deepcopy(value)

    @staticmethod
    def _validate_item(item: Any) -> dict[str, Any]:
        item = _object(item, "item")
        _known_fields(item, {"id", "object", "type", "role", "status", "content"}, "item")
        if item.get("type") != "message":
            raise RealtimeError("Only message items are supported.", "item.type")
        if item.get("role") not in ("user", "system", "assistant"):
            raise RealtimeError("Use a user, system, or assistant message.", "item.role")
        if "object" in item and item["object"] != "realtime.item":
            raise RealtimeError("Expected realtime.item.", "item.object")
        if "status" in item and item["status"] not in ("completed", "incomplete", "in_progress"):
            raise RealtimeError("Invalid item status.", "item.status")
        if "id" in item:
            item_id = item["id"]
            if not isinstance(item_id, str) or not item_id or len(item_id) > 512 or item_id == "root":
                raise RealtimeError(
                    "Expected a non-empty item ID of at most 512 characters, other than root.", "item.id"
                )
        content = item.get("content")
        if not isinstance(content, list):
            raise RealtimeError("Expected a content array.", "item.content")
        allowed = {
            "user": {"input_text", "input_audio", "input_image"},
            "system": {"input_text"},
            "assistant": {"output_text"},
        }
        for index, part in enumerate(content):
            param = f"item.content[{index}]"
            part = _object(part, param)
            kind = part.get("type")
            if not isinstance(kind, str) or kind not in allowed[item["role"]]:
                raise RealtimeError("This content type is not supported for the message role.", f"{param}.type")
            if kind in ("input_text", "output_text"):
                _known_fields(part, {"type", "text"}, param)
                if not isinstance(part.get("text"), str):
                    raise RealtimeError("Expected a string.", f"{param}.text")
            elif kind == "input_audio":
                _known_fields(part, {"type", "audio", "transcript"}, param)
                _validate_pcm(part.get("audio"), f"{param}.audio")
                if "transcript" in part and not isinstance(part["transcript"], str):
                    raise RealtimeError("Expected a string.", f"{param}.transcript")
            else:
                _known_fields(part, {"type", "image_url", "detail"}, param)
                if "detail" in part and part["detail"] not in ("auto", "low", "high"):
                    raise RealtimeError("Use auto, low, or high image detail.", f"{param}.detail")
                _validate_image(part.get("image_url"), f"{param}.image_url")
        result = copy.deepcopy(item)
        result.setdefault("id", f"item_{uuid.uuid4().hex}")
        result["object"] = "realtime.item"
        result["status"] = "completed"
        return result


def session_config(config: SessionConfig) -> dict[str, Any]:
    visual = config.visual
    return {
        "type": "realtime",
        "model": config.model,
        "instructions": config.instructions,
        "output_modalities": [config.output_mode],
        "max_output_tokens": config.max_tokens if config.max_tokens is not None else "inf",
        "audio": {
            "input": {"format": dict(_PCM_FORMAT), "turn_detection": None},
            "output": {"format": dict(_PCM_FORMAT)},
        },
        "omni": {
            "input_image_retention": visual.retention,
            "input_image_max_items": visual.max_items,
            "input_image_sample": visual.sample,
            "input_image_evs": visual.evs,
            "input_image_evs_threshold": visual.threshold,
            "use_audio_in_video": config.use_audio_in_video,
        },
    }


def decode_config(changes: Any, current: SessionConfig, *, response: bool = False) -> SessionConfig:
    prefix = "response" if response else "session"
    changes = _object(changes, prefix)
    values = session_config(current)
    allowed = {"instructions", "output_modalities", "max_output_tokens"} if response else set(values)
    _known_fields(changes, allowed, prefix)
    for key, value in changes.items():
        param = f"{prefix}.{key}"
        if key == "type" and value != "realtime":
            raise RealtimeError("Only realtime sessions are supported.", param)
        elif key == "model" and value != current.model:
            raise RealtimeError("The model cannot change during a session.", param)
        elif key == "instructions" and not isinstance(value, str):
            raise RealtimeError("Expected a string.", param)
        elif key == "output_modalities":
            _modalities(value, param)
        elif key == "max_output_tokens":
            _max_tokens(value, param)
        elif key == "audio":
            _Validation._update_audio(values["audio"], value)
            continue
        elif key == "omni":
            _Validation._update_omni(values["omni"], value)
            continue
        values[key] = copy.deepcopy(value)
    visual = values["omni"]
    return SessionConfig(
        model=values["model"],
        instructions=values["instructions"],
        output_mode=values["output_modalities"][0],
        max_tokens=None if values["max_output_tokens"] == "inf" else values["max_output_tokens"],
        visual=VisualPolicy(
            retention=visual["input_image_retention"],
            max_items=visual["input_image_max_items"],
            sample=visual["input_image_sample"],
            evs=visual["input_image_evs"],
            threshold=visual["input_image_evs_threshold"],
        ),
        use_audio_in_video=visual["use_audio_in_video"],
    )


def decode_item(raw: Any) -> Item:
    item = _Validation._validate_item(raw)
    parts = []
    for part in item["content"]:
        kind = part["type"]
        if kind in ("input_text", "output_text"):
            parts.append(Content("text", text=part["text"]))
        elif kind == "input_audio":
            parts.append(Content("audio", text=part.get("transcript", ""), data=base64.b64decode(part["audio"])))
        else:
            header, data = part["image_url"].split(",", 1)
            parts.append(
                Content(
                    "image",
                    data=base64.b64decode(data),
                    media_type=header[5:].split(";")[0],
                    detail=part.get("detail", "auto"),
                )
            )
    return Item(item["role"], tuple(parts), id=item["id"])


def encode_item(item: Item) -> dict[str, Any]:
    content = []
    for part in item.content:
        if part.kind == "text":
            content.append({"type": "output_text" if item.role == "assistant" else "input_text", "text": part.text})
        elif part.kind == "audio":
            content.append(
                {"type": "output_audio" if item.role == "assistant" else "input_audio", "transcript": part.text}
            )
        else:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{part.media_type};base64,{base64.b64encode(part.data).decode('ascii')}",
                    "detail": part.detail,
                }
            )
    return {
        "id": item.id,
        "object": "realtime.item",
        "type": "message",
        "role": item.role,
        "status": item.status,
        "content": content,
    }


def decode_event(event: Any, config: SessionConfig):
    event = _object(event, "event")
    event_id = event.get("event_id")
    if event_id is not None and (not isinstance(event_id, str) or len(event_id) > 512):
        raise RealtimeError("event_id must be a string of at most 512 characters.", "event_id")
    kind = event.get("type")
    fields = {
        "session.update": {"session"},
        "conversation.item.create": {"item", "previous_item_id"},
        "conversation.item.delete": {"item_id"},
        "input_audio_buffer.append": {"audio", "video_frames", "sample_rate_hz"},
        "input_audio_buffer.commit": {"final"},
        "input_audio_buffer.clear": set(),
        "response.create": {"response"},
        "response.cancel": {"response_id"},
    }
    if isinstance(kind, str) and kind in fields:
        _known_fields(event, fields[kind] | {"type", "event_id"}, "")
    if kind == "session.update":
        return UpdateSession(decode_config(event.get("session"), config))
    if kind == "conversation.item.create":
        previous = event.get("previous_item_id")
        if previous is not None and not isinstance(previous, str):
            raise RealtimeError("Expected an item ID.", "previous_item_id")
        return InsertItem(decode_item(event.get("item")), previous)
    if kind == "conversation.item.delete":
        if not isinstance(event.get("item_id"), str):
            raise RealtimeError("Expected an item ID.", "item_id")
        return DeleteItem(event["item_id"])
    if kind == "input_audio_buffer.append":
        if "video_frames" in event:
            raise RealtimeError("Send input_image conversation items.", "video_frames")
        if "sample_rate_hz" in event:
            raise RealtimeError("GA PCM uses the session's 24000 Hz format.", "sample_rate_hz")
        return AppendAudio(_decode_base64(event.get("audio"), "audio", MAX_AUDIO_APPEND_BYTES))
    if kind == "input_audio_buffer.commit":
        if "final" in event:
            raise RealtimeError("commit.final belongs to qwen3-legacy; use response.create.", "final")
        return CommitAudio()
    if kind == "input_audio_buffer.clear":
        return ClearAudio()
    if kind == "response.create":
        return CreateResponse(decode_config(event.get("response", {}), config, response=True))
    if kind == "response.cancel":
        response_id = event.get("response_id")
        if response_id is not None and not isinstance(response_id, str):
            raise RealtimeError("Expected a response ID.", "response_id")
        return CancelResponse(response_id)
    raise RealtimeError(f"Unsupported event type: {kind!r}", "type", "unsupported_event")
