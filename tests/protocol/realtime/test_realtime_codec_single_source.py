# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""There is one Realtime codec, and the duplex engine uses it.

RFC #6592 P0a rejects the "copy the codec and fix it up" option because the two
copies drift. The extraction is only worth anything while the duplex names are
*the same objects* as the protocol ones, not lookalikes --- so assert identity,
which a re-implementation cannot satisfy.
"""

from __future__ import annotations

import pytest

from vllm_omni.engine.duplex import audio as duplex_audio
from vllm_omni.engine.duplex import events as duplex_events
from vllm_omni.engine.duplex import realtime_commands as duplex_codec
from vllm_omni.engine.duplex.commands import DuplexCommandError
from vllm_omni.protocol import realtime as protocol
from vllm_omni.protocol.realtime.errors import RealtimeProtocolError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

#: Names ``engine.duplex.realtime_commands`` re-exports for its existing importers.
_REEXPORTED_FROM_PROTOCOL = (
    "REALTIME_INPUT_AUDIO_FORMATS",
    "REALTIME_INPUT_HINT_KEYS",
    "REALTIME_OUTPUT_AUDIO_FORMATS",
    "RealtimeInputDefaults",
    "apply_realtime_session_defaults",
    "copy_realtime_input_hints",
    "input_audio_transcription_config",
    "input_explicitly_non_speech",
    "input_looks_like_speech",
    "input_transcript_from_item",
    "is_supported_realtime_input_format",
    "json_safe_realtime_payload",
    "normalize_conversation_item",
    "parse_realtime_audio_format",
    "realtime_audio_format_object",
    "realtime_max_output_tokens",
    "realtime_output_format",
    "realtime_overlap_fields",
    "text_chars_for_audio_ms_from_marks",
    "truncate_realtime_item_content",
    "validate_conversation_item_audio_formats",
    "validate_realtime_item_truncate",
    "validate_realtime_response_audio_formats",
    "validate_realtime_session_audio_formats",
    "validate_realtime_video_frames",
)


@pytest.mark.parametrize("name", _REEXPORTED_FROM_PROTOCOL)
def test_the_duplex_codec_name_is_the_protocol_object(name: str) -> None:
    assert getattr(duplex_codec, name) is getattr(protocol, name)


@pytest.mark.parametrize(
    "name",
    ("convert_input_audio_with_rate", "convert_output_audio", "resample_pcm16_mono", "wav_payload_to_pcm16"),
)
def test_the_duplex_audio_shim_is_the_protocol_object(name: str) -> None:
    assert getattr(duplex_audio, name) is getattr(protocol, name)


def test_the_error_type_table_is_defined_once() -> None:
    assert duplex_events.REALTIME_ERROR_TYPES_BY_CODE is protocol.REALTIME_ERROR_TYPES_BY_CODE


def test_a_duplex_command_error_is_a_realtime_protocol_error() -> None:
    # The duplex binding converts the codec's error into its own type; both
    # carry the same code/event_id, which is what the error envelope renders.
    error = DuplexCommandError("bad", code="bad_audio", event_id="event_1")

    assert isinstance(error, RealtimeProtocolError)
    assert (error.code, error.event_id) == ("bad_audio", "event_1")


def test_the_duplex_append_command_is_built_from_the_shared_decoder() -> None:
    defaults = protocol.RealtimeInputDefaults()
    event = {"event_id": "event_1", "audio": "", "format": "pcm16", "duration_ms": 40}

    decoded = protocol.decode_audio_append(event, defaults=defaults)
    command = duplex_codec.build_append_audio(event, defaults=defaults)

    assert (command.audio, command.format, command.sample_rate_hz) == (
        decoded.audio,
        decoded.format,
        decoded.sample_rate_hz,
    )
    assert (command.is_speech, command.video_frames, command.duration_ms) == (
        decoded.is_speech,
        decoded.video_frames,
        decoded.duration_ms,
    )
    assert (command.audio_end_ms, dict(command.hints), command.event_id) == (
        decoded.audio_end_ms,
        decoded.hints,
        decoded.event_id,
    )


def test_the_shared_decoder_error_reaches_the_client_as_a_duplex_command_error() -> None:
    defaults = protocol.RealtimeInputDefaults()
    event = {"event_id": "event_1", "audio": "AAAA", "format": "opus"}

    with pytest.raises(RealtimeProtocolError) as shared:
        protocol.decode_audio_append(event, defaults=defaults)
    with pytest.raises(DuplexCommandError) as duplex:
        duplex_codec.build_append_audio(event, defaults=defaults)

    assert (duplex.value.code, str(duplex.value)) == (shared.value.code, str(shared.value))
    assert duplex.value.event_id == shared.value.event_id == "event_1"


def test_the_duplex_capabilities_reject_unimplemented_turn_detection() -> None:
    # The duplex answer comes from engine.duplex.turn_detection, reached through
    # the capability object rather than imported by the codec.
    rejection = protocol.validate_session_payload(
        {"turn_detection": {"type": "semantic_vad"}},
        capabilities=duplex_codec.DUPLEX_REALTIME_CAPABILITIES,
    )

    assert rejection is not None
    assert rejection.code == "unsupported_turn_detection"
