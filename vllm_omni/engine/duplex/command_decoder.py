# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Decode Realtime client messages into the duplex protocol's typed commands.

Shared protocol helpers validate formats, session fields and audio. This module
selects the command and the capabilities of the duplex engine. Per-session
checks, such as empty commits and response-id fallback, are resolved by the
runner through ``session_projection`` after the command has been queued.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from vllm_omni.protocol.duplex import (
    RealtimeInputDefaults,
    RealtimeProtocolCapabilities,
    RealtimeProtocolError,
    decode_audio_append,
    normalize_conversation_item,
    validate_conversation_item_audio_formats,
    validate_realtime_response_audio_formats,
    validate_session_payload,
)
from vllm_omni.protocol.duplex.commands import (
    AckPlayback,
    AppendAudio,
    AppendText,
    BargeIn,
    CancelInput,
    CancelResponse,
    ClearInput,
    ClearOutputAudio,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
    DeleteItem,
    Heartbeat,
    RealtimeCommand,
    SignalTurn,
    TruncateItem,
    UpdateSession,
)


def _validate_duplex_turn_detection(session_payload: Mapping[str, object]) -> str | None:
    """The duplex engine's answer for ``turn_detection`` on a session object.

    Imported lazily: the validator lives with the Silero VAD backend, and a
    connection that never sends a session object should not pay for loading it.
    """
    from vllm_omni.engine.duplex.turn_detection import validate_realtime_turn_detection

    return validate_realtime_turn_detection(session_payload)


#: What the duplex engine can serve, for the shared session-object validator.
DUPLEX_REALTIME_CAPABILITIES = RealtimeProtocolCapabilities(
    validate_turn_detection=_validate_duplex_turn_detection,
)


def duplex_response_format(realtime_format: str) -> str:
    """A Realtime output format -> the duplex ``response_format`` vocabulary."""
    normalized = realtime_format.lower()
    if normalized in {"pcm16", "pcm_s16le", "s16le"}:
        return "pcm"
    if normalized in {"g711_ulaw", "g711_alaw"}:
        return "pcm"
    if normalized in {"wav", "pcm"}:
        return normalized
    return "wav"


# ---- append / item audio conversion ----


def build_append_audio(
    event: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults,
    hints_source: Mapping[str, object] | None = None,
) -> AppendAudio:
    """Decode one audio append (shared codec) and pack it as the duplex command."""
    decoded = decode_audio_append(event, defaults=defaults, hints_source=hints_source)
    return AppendAudio(
        event_id=decoded.event_id,
        audio=decoded.audio,
        format=decoded.format,
        sample_rate_hz=decoded.sample_rate_hz,
        is_speech=decoded.is_speech,
        video_frames=decoded.video_frames,
        duration_ms=decoded.duration_ms,
        audio_end_ms=decoded.audio_end_ms,
        hints=decoded.hints,
    )


# ---- translation ----


def decode_command(
    payload: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults | None = None,
) -> RealtimeCommand:
    """Map one OpenAI Realtime client event onto a :class:`RealtimeCommand`.

    Raises :class:`RealtimeProtocolError` for malformed or unsupported payloads.
    ``session.resume`` and ``session.event_ack`` are transport concerns and are
    rejected with ``code="unknown_event"``.
    """
    defaults = defaults or RealtimeInputDefaults()
    event_type = payload.get("type")
    event_id = cast("str", payload.get("event_id")) if isinstance(payload.get("event_id"), str) else None
    if not isinstance(event_type, str):
        raise RealtimeProtocolError("Duplex event missing string type", code="bad_event", event_id=event_id)

    if event_type == "session.update":
        session = payload.get("session")
        session_payload: Mapping[str, object] = session if isinstance(session, dict) else payload
        rejection = validate_session_payload(session_payload, capabilities=DUPLEX_REALTIME_CAPABILITIES)
        if rejection is not None:
            raise RealtimeProtocolError(rejection.message, code=rejection.code, event_id=event_id)
        return UpdateSession(event_id=event_id, patch=dict(session_payload))

    if event_type == "conversation.item.create":
        item = payload.get("item")
        format_error = validate_conversation_item_audio_formats(item)
        if format_error is not None:
            raise RealtimeProtocolError(format_error, code="unsupported_audio_format", event_id=event_id)
        if not isinstance(item, dict):
            raise RealtimeProtocolError("conversation.item.create requires item", code="bad_event", event_id=event_id)
        previous_item_id = payload.get("previous_item_id")
        return CreateItem(
            event_id=event_id,
            item=normalize_conversation_item(item),
            previous_item_id=previous_item_id if isinstance(previous_item_id, str) else None,
        )

    if event_type == "conversation.item.delete":
        item_id = payload.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise RealtimeProtocolError(
                "conversation.item.delete requires item_id", code="missing_item_id", event_id=event_id
            )
        return DeleteItem(event_id=event_id, item_id=item_id)

    if event_type == "conversation.item.truncate":
        item_id = payload.get("item_id")
        audio_end_ms = payload.get("audio_end_ms")
        content_index = payload.get("content_index", 0)
        if not isinstance(item_id, str) or not item_id:
            raise RealtimeProtocolError(
                "conversation.item.truncate requires item_id", code="missing_item_id", event_id=event_id
            )
        if not isinstance(audio_end_ms, int | float):
            raise RealtimeProtocolError(
                "conversation.item.truncate requires numeric audio_end_ms", code="bad_event", event_id=event_id
            )
        return TruncateItem(
            event_id=event_id,
            item_id=item_id,
            audio_end_ms=int(audio_end_ms),
            content_index=int(content_index) if isinstance(content_index, int | float) else 0,
        )

    if event_type == "input_audio_buffer.append":
        return build_append_audio(dict(payload), defaults=defaults)

    if event_type in {"input_audio_buffer.commit", "input.commit"}:
        final = payload.get("final", True)
        create_response = payload.get("response_create", payload.get("create_response"))
        is_speech = payload.get("is_speech")
        return Commit(
            event_id=event_id,
            final=bool(final) if isinstance(final, bool) else True,
            create_response=(
                bool(create_response)
                if isinstance(create_response, bool)
                else (True if event_type == "input.commit" and create_response is None else None)
            ),
            is_speech=is_speech if isinstance(is_speech, bool) else None,
        )

    if event_type == "input_audio_buffer.clear":
        return ClearInput(event_id=event_id)

    if event_type == "output_audio_buffer.clear":
        response_id = payload.get("response_id")
        return ClearOutputAudio(
            event_id=event_id,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
        )

    if event_type == "response.cancel":
        response_id = payload.get("response_id")
        return CancelResponse(
            event_id=event_id,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
        )

    if event_type == "response.create":
        response_payload = payload.get("response")
        if isinstance(response_payload, dict):
            format_error = validate_realtime_response_audio_formats(response_payload)
            if format_error is not None:
                raise RealtimeProtocolError(format_error, code="unsupported_audio_format", event_id=event_id)
        return CreateResponse(
            event_id=event_id,
            options=dict(response_payload) if isinstance(response_payload, dict) else {},
        )

    if event_type in {"playback.ack", "audio.playback_ack"}:
        played_ms = payload.get("played_ms")
        if not isinstance(played_ms, int | float):
            raise RealtimeProtocolError("playback.ack requires numeric played_ms", code="bad_event", event_id=event_id)
        committed_ms = payload.get("committed_ms")
        response_id = payload.get("response_id")
        item_id = payload.get("item_id")
        return AckPlayback(
            event_id=event_id,
            played_ms=int(played_ms),
            committed_ms=int(committed_ms) if isinstance(committed_ms, int | float) else None,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
            item_id=item_id if isinstance(item_id, str) and item_id else None,
        )

    if event_type == "session.heartbeat":
        return Heartbeat(event_id=event_id)

    if event_type == "conversation.item.retrieve":
        # The projected conversation items live engine-side; the runner answers.
        return SignalTurn(event_id=event_id, event="conversation.item.retrieve", signal_payload=dict(payload))

    if event_type in {"session.close", "close", "close_session"}:
        reason = payload.get("reason")
        return CloseSession(event_id=event_id, reason=reason if isinstance(reason, str) and reason else "client_close")

    if event_type in {"input.text.append", "input_text.append", "push_text"}:
        text = payload.get("text")
        if not isinstance(text, str):
            raise RealtimeProtocolError("input.text.append requires text", code="bad_event", event_id=event_id)
        return AppendText(event_id=event_id, text=text)

    if event_type == "input.cancel":
        return CancelInput(event_id=event_id)

    if event_type == "barge_in":
        return BargeIn(event_id=event_id)

    if event_type in {"turn.signal", "signal_turn"}:
        signal_event = payload.get("event")
        if not isinstance(signal_event, str) or not signal_event:
            raise RealtimeProtocolError("turn.signal requires event", code="bad_event", event_id=event_id)
        signal_payload = payload.get("payload")
        signal_payload = dict(signal_payload) if isinstance(signal_payload, dict) else {}
        if signal_event == "input.cancel":
            return CancelInput(event_id=event_id)
        if signal_event == "barge_in":
            return BargeIn(event_id=event_id)
        if signal_event == "response.cancel":
            response_id = signal_payload.get("response_id", payload.get("response_id"))
            return CancelResponse(
                event_id=event_id,
                response_id=response_id if isinstance(response_id, str) and response_id else None,
            )
        if signal_event == "session.update":
            return UpdateSession(event_id=event_id, patch=signal_payload)
        if signal_event == "conversation.item.create":
            item = signal_payload.get("item")
            if not isinstance(item, dict):
                raise RealtimeProtocolError(
                    "conversation.item.create requires item", code="bad_event", event_id=event_id
                )
            previous_item_id = signal_payload.get("previous_item_id")
            return CreateItem(
                event_id=event_id,
                item=normalize_conversation_item(item),
                previous_item_id=previous_item_id if isinstance(previous_item_id, str) else None,
            )
        if signal_event == "conversation.item.delete":
            item_id = signal_payload.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise RealtimeProtocolError(
                    "conversation.item.delete requires item_id", code="missing_item_id", event_id=event_id
                )
            return DeleteItem(event_id=event_id, item_id=item_id)
        if signal_event == "conversation.item.truncate":
            item_id = signal_payload.get("item_id")
            audio_end_ms = signal_payload.get("audio_end_ms")
            content_index = signal_payload.get("content_index", 0)
            if not isinstance(item_id, str) or not item_id or not isinstance(audio_end_ms, int | float):
                raise RealtimeProtocolError(
                    "conversation.item.truncate requires item_id and numeric audio_end_ms",
                    code="bad_event",
                    event_id=event_id,
                )
            return TruncateItem(
                event_id=event_id,
                item_id=item_id,
                audio_end_ms=int(audio_end_ms),
                content_index=int(content_index) if isinstance(content_index, int | float) else 0,
            )
        return SignalTurn(event_id=event_id, event=signal_event, signal_payload=signal_payload)

    raise RealtimeProtocolError(f"Unknown duplex event type: {event_type}", code="unknown_event", event_id=event_id)
