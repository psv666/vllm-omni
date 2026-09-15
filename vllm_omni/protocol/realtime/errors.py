# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The OpenAI Realtime error envelope.

A Realtime server reports a rejected client event as one ``error`` event whose
``error`` object carries ``type`` / ``code`` / ``message`` (plus the client's
``event_id`` and an optional ``param``). ``type`` is not free-form: it is one
of OpenAI's four buckets, derived from our internal code by
:data:`REALTIME_ERROR_TYPES_BY_CODE`.

The table and :class:`RealtimeProtocolError` live here so a consumer that is
not the duplex engine can raise and render the same envelope. Rendering an
``error`` event for a duplex session stays on
``vllm_omni.engine.duplex.events.ErrorEvent``, which reads this table.
"""

from __future__ import annotations

__all__ = [
    "REALTIME_ERROR_TYPES_BY_CODE",
    "RealtimeProtocolError",
    "realtime_error_type",
]

#: OpenAI Realtime ``error.type`` for each internal error code.
REALTIME_ERROR_TYPES_BY_CODE: dict[str, str] = {
    "bad_event": "invalid_request_error",
    "bad_audio": "invalid_request_error",
    "config_timeout": "invalid_request_error",
    "invalid_json": "invalid_request_error",
    "event_too_large": "invalid_request_error",
    "unknown_event": "invalid_request_error",
    "internal_error": "server_error",
    "runtime_append_failed": "server_error",
    "runtime_append_task_failed": "server_error",
    "runtime_signal_failed": "server_error",
    "runtime_abort_failed": "server_error",
    "runtime_data_plane_stream_failed": "server_error",
    "runtime_data_plane_text_without_audio": "server_error",
    "resource_exhausted": "rate_limit_error",
    "session_exists": "invalid_request_error",
    "session_closed": "invalid_request_error",
    "unknown_session": "invalid_request_error",
    "invalid_duplex_runtime_config": "invalid_request_error",
    "instructions_update_unsupported": "invalid_request_error",
    "persona_update_unsupported": "invalid_request_error",
    "voice_update_unsupported": "invalid_request_error",
    "unsupported_nemotron_duplex_mode": "invalid_request_error",
    "unsupported_native_response_options": "invalid_request_error",
    "runtime_touch_failed": "server_error",
    "engine_error": "server_error",
    "input_backpressure": "rate_limit_error",
    "response_already_active": "invalid_request_error",
    "response_not_active": "invalid_request_error",
    "response_create_without_input": "invalid_request_error",
    "input_audio_buffer_empty": "invalid_request_error",
    "missing_item_id": "invalid_request_error",
    "item_not_found": "invalid_request_error",
    "playback_item_mismatch": "invalid_request_error",
    "playback_item_not_found": "invalid_request_error",
    "playback_ack_too_late": "invalid_request_error",
    "unsupported_audio_format": "invalid_request_error",
    "unsupported_turn_detection": "invalid_request_error",
    "unsupported_ref_audio_path": "invalid_request_error",
    "ref_audio_required": "invalid_request_error",
    "model_update_unsupported": "invalid_request_error",
    "voice_update_after_audio_unsupported": "invalid_request_error",
    "ref_audio_update_unsupported": "invalid_request_error",
    "native_text_append_unsupported": "invalid_request_error",
    "invalid_video_frames": "invalid_request_error",
    "invalid_function_call_output": "invalid_request_error",
    "server_vad_unavailable": "server_error",
}


def realtime_error_type(code: str) -> str:
    """The OpenAI ``error.type`` bucket for an internal error code."""
    return REALTIME_ERROR_TYPES_BY_CODE.get(code, "invalid_request_error")


class RealtimeProtocolError(ValueError):
    """A client payload could not be decoded into a valid Realtime intent.

    ``code`` is the internal error code that :data:`REALTIME_ERROR_TYPES_BY_CODE`
    maps to an OpenAI ``error.type``; ``event_id`` is the *client* event id the
    error answers. ``vllm_omni.engine.duplex.commands.DuplexCommandError`` is
    the duplex specialization, so a consumer catching either sees the same
    three attributes.
    """

    def __init__(self, message: str, *, code: str = "bad_event", event_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.event_id = event_id
