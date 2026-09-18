# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""There is one Realtime codec, and the duplex engine uses it.

RFC #6592 P0a rejects the "copy the codec and fix it up" option because the two
copies drift. The extraction is only worth anything while the duplex names are
*the same objects* as the protocol ones, not lookalikes --- so assert identity,
which a re-implementation cannot satisfy.
"""

from __future__ import annotations

import dataclasses

import pytest

from vllm_omni.engine.duplex import command_decoder as duplex_codec
from vllm_omni.protocol import duplex as duplex_protocol
from vllm_omni.protocol import realtime as protocol
from vllm_omni.protocol.duplex import events as duplex_events
from vllm_omni.protocol.realtime.errors import RealtimeProtocolError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

#: Shared helpers exposed through the duplex protocol entry point.
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
    assert getattr(duplex_protocol, name) is getattr(protocol, name)


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


def test_the_shared_decoder_error_keeps_its_code_and_event_id() -> None:
    defaults = protocol.RealtimeInputDefaults()
    event = {"event_id": "event_1", "audio": "AAAA", "format": "opus"}

    with pytest.raises(RealtimeProtocolError) as shared:
        protocol.decode_audio_append(event, defaults=defaults)
    with pytest.raises(RealtimeProtocolError) as duplex:
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


# ---- the class split (RFC #6592 P0a, second half) ----
#
# Shared event types are reused, while duplex extensions add only their own fields.

# `docs/serving/realtime_duplex_api.md` sorts every message into three tiers.
# Tier 1 is pure OpenAI and lives in protocol/realtime; Tier 2 (OpenAI names
# carrying our extensions) and Tier 3 (ours alone) live in protocol/duplex.
_TIER1_EVENTS = ("SessionUpdated", "ItemAdded", "ItemCreated", "TextDelta", "OutputItemAdded")
_TIER2_EVENTS = (
    "SessionCreated",
    "ResponseCreated",
    "ResponseDone",
    "AudioDelta",
    "InputCommitted",
    "ItemDeleted",
    "ItemTruncated",
    "ErrorEvent",
)
_TIER3_EVENTS = ("Listen", "Speak", "OverlapDecision", "PlaybackAcknowledged", "SessionResumed")
_TIER1_COMMANDS = ("UpdateSession", "CreateItem", "CancelResponse", "ClearInput", "Commit")


@pytest.mark.parametrize("name", _TIER1_EVENTS)
def test_tier1_events_are_the_pure_openai_objects(name: str) -> None:
    from vllm_omni.protocol.realtime import events as realtime_events

    assert duplex_events.__dict__[name] is getattr(realtime_events, name)


@pytest.mark.parametrize("name", _TIER2_EVENTS + _TIER3_EVENTS)
def test_tier2_and_tier3_events_come_from_protocol_duplex(name: str) -> None:
    from vllm_omni.protocol.duplex import events as duplex_wire_events

    assert duplex_events.__dict__[name] is getattr(duplex_wire_events, name)


@pytest.mark.parametrize("name", _TIER2_EVENTS)
def test_tier2_extends_its_tier1_twin_rather_than_replacing_it(name: str) -> None:
    """A Tier 2 class must be its Tier 1 class plus fields, never a fork of it."""
    from vllm_omni.protocol.duplex import events as duplex_wire_events
    from vllm_omni.protocol.realtime import events as realtime_events

    tier1 = getattr(realtime_events, name)
    tier2 = getattr(duplex_wire_events, name)

    assert issubclass(tier2, tier1)
    assert tier2.wire_type == tier1.wire_type
    tier1_fields = {f.name for f in dataclasses.fields(tier1)}
    tier2_fields = {f.name for f in dataclasses.fields(tier2)}
    # Purely additive: Tier 2 adds, never drops or renames.
    assert tier1_fields < tier2_fields


@pytest.mark.parametrize("name", _TIER1_EVENTS + _TIER1_COMMANDS)
def test_tier1_carries_no_vllm_omni_extension_fields(name: str) -> None:
    """The point of the split: protocol/realtime must be honestly pure OpenAI."""
    from vllm_omni.protocol.realtime import commands as realtime_commands
    from vllm_omni.protocol.realtime import events as realtime_events

    cls = getattr(realtime_events, name, None) or getattr(realtime_commands, name)
    fields = {f.name for f in dataclasses.fields(cls)}
    # Extension fields catalogued as Tier 2 in the serving doc.
    assert not (
        fields
        & {
            "attachment_generation",
            "resume_token",
            "details",
            "extra",
            "is_speech",
            "video_frames",
            "hints",
            "realtime_item_id",
            "final",
        }
    )


def test_the_error_code_vocabulary_is_tier3() -> None:
    """OpenAI standardises the error classes, not our codes (doc: Tier 3)."""
    from vllm_omni.protocol.duplex.errors import REALTIME_ERROR_TYPES_BY_CODE
    from vllm_omni.protocol.realtime import errors as realtime_errors

    assert duplex_events.REALTIME_ERROR_TYPES_BY_CODE is REALTIME_ERROR_TYPES_BY_CODE
    assert not hasattr(realtime_errors, "REALTIME_ERROR_TYPES_BY_CODE")


def test_the_tier1_error_event_reports_only_openai_classes() -> None:
    from vllm_omni.protocol.duplex import events as duplex_wire_events
    from vllm_omni.protocol.realtime import events as realtime_events

    # Tier 1 knows the envelope shape but not our code vocabulary.
    assert realtime_events.ErrorEvent(code="resource_exhausted").error_type == "invalid_request_error"
    # Tier 2 resolves it through our table.
    assert duplex_wire_events.ErrorEvent(code="resource_exhausted").error_type == "rate_limit_error"
