# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI Realtime client events -> typed ``RealtimeCommand`` objects and internal handler payloads."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.engine.duplex.command_decoder import decode_command
from vllm_omni.engine.duplex.command_payload import to_internal_payload
from vllm_omni.protocol.duplex import RealtimeInputDefaults
from vllm_omni.protocol.duplex import commands as command_types
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
from vllm_omni.protocol.duplex.errors import RealtimeProtocolError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LOUD_PCM16 = base64.b64encode(b"\x00\x10" * 8).decode("ascii")
_SILENT_PCM16 = base64.b64encode(b"\x00\x00" * 8).decode("ascii")
_JPEG_FRAME = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 12).decode("ascii")

_MINIMAL_PAYLOADS: dict[str, tuple[dict[str, object], type[RealtimeCommand]]] = {
    "input_audio_buffer.append": ({"audio": _LOUD_PCM16}, AppendAudio),
    "input_audio_buffer.commit": ({}, Commit),
    "input_audio_buffer.clear": ({}, ClearInput),
    "output_audio_buffer.clear": ({}, ClearOutputAudio),
    "response.create": ({}, CreateResponse),
    "response.cancel": ({}, CancelResponse),
    "conversation.item.create": ({"item": {"type": "message", "role": "user", "content": []}}, CreateItem),
    "conversation.item.delete": ({"item_id": "item_1"}, DeleteItem),
    "conversation.item.truncate": ({"item_id": "item_1", "audio_end_ms": 100}, TruncateItem),
    "session.update": ({"session": {"instructions": "hi"}}, UpdateSession),
    "playback.ack": ({"played_ms": 10}, AckPlayback),
    "session.heartbeat": ({}, Heartbeat),
    "session.close": ({}, CloseSession),
    "turn.signal": ({"event": "user_started"}, SignalTurn),
    "input.text.append": ({"text": "hello"}, AppendText),
    "input.cancel": ({}, CancelInput),
    "barge_in": ({}, BargeIn),
}


def test_every_realtime_command_type_has_a_mapping_case():
    wire_types = {wire_type for name in command_types.__all__ if (wire_type := getattr(command_types, name).wire_type)}
    assert set(_MINIMAL_PAYLOADS) == wire_types


@pytest.mark.parametrize("event_type", sorted(_MINIMAL_PAYLOADS))
def test_decode_command_maps_each_type_to_its_dataclass_and_propagates_event_id(event_type: str):
    body, expected_cls = _MINIMAL_PAYLOADS[event_type]

    command = decode_command({"type": event_type, "event_id": f"evt-{event_type}", **body})

    assert type(command) is expected_cls
    assert command.event_id == f"evt-{event_type}"
    assert command.wire_type == event_type

    anonymous = decode_command({"type": event_type, **body})
    assert anonymous.event_id is None


@pytest.mark.parametrize(
    ("alias", "expected_cls", "body"),
    [
        ("push_text", AppendText, {"text": "hello"}),
        ("input_text.append", AppendText, {"text": "hello"}),
        ("signal_turn", SignalTurn, {"event": "user_started"}),
        ("close_session", CloseSession, {}),
        ("close", CloseSession, {}),
        ("audio.playback_ack", AckPlayback, {"played_ms": 10}),
        ("input.commit", Commit, {}),
    ],
)
def test_pre_realtime_client_event_aliases_map_to_their_canonical_command(
    alias: str, expected_cls: type[RealtimeCommand], body: dict[str, object]
):
    assert type(decode_command({"type": alias, **body})) is expected_cls


def test_append_audio_decodes_and_converts_pcm16_to_16k_float32():
    command = decode_command(
        {"type": "input_audio_buffer.append", "audio": _LOUD_PCM16, "format": "pcm16", "sample_rate_hz": 16000}
    )

    assert isinstance(command, AppendAudio)
    assert command.format == "pcm_f32le"
    assert command.sample_rate_hz == 16000
    samples = np.frombuffer(command.audio, dtype="<f4")
    assert samples.shape == (8,)
    assert np.allclose(samples, 4096 / 32768.0)
    assert command.is_speech is True
    assert command.video_frames == ()
    assert command.duration_ms is None
    assert command.audio_end_ms is None


def test_append_audio_defaults_come_from_the_session_payload():
    defaults = RealtimeInputDefaults().with_session_payload({"input_audio_format": "pcm16", "sample_rate_hz": 8000})
    assert defaults.input_audio_format == "pcm16"
    assert defaults.input_sample_rate_hz == 8000

    command = decode_command({"type": "input_audio_buffer.append", "audio": _LOUD_PCM16}, defaults=defaults)

    assert isinstance(command, AppendAudio)
    assert command.format == "pcm_f32le"
    # 8 kHz input is resampled to the model's 16 kHz float stream.
    assert command.sample_rate_hz == 16000
    assert len(command.audio) > 8 * 4

    passthrough = decode_command(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(np.zeros(4, dtype="<f4").tobytes()).decode()},
        defaults=RealtimeInputDefaults(input_audio_format="pcm_f32le", input_sample_rate_hz=16000),
    )
    assert passthrough.format == "pcm_f32le"
    assert passthrough.audio == np.zeros(4, dtype="<f4").tobytes()


def test_append_audio_speech_classification_uses_hints_then_rms():
    silent = decode_command({"type": "input_audio_buffer.append", "audio": _SILENT_PCM16})
    assert silent.is_speech is False

    forced = decode_command({"type": "input_audio_buffer.append", "audio": _SILENT_PCM16, "is_speech": True})
    assert forced.is_speech is True
    assert forced.hints == {"is_speech": True}

    vad = decode_command(
        {"type": "input_audio_buffer.append", "audio": _LOUD_PCM16, "vad": {"speech_probability": 0.1}}
    )
    assert vad.is_speech is False
    assert vad.hints["vad"] == {"speech_probability": 0.1}


def test_append_audio_carries_wire_hints_and_video_frames():
    command = decode_command(
        {
            "type": "input_audio_buffer.append",
            "event_id": "evt-append",
            "audio": _LOUD_PCM16,
            "transcript": "hi there",
            "duration_ms": 1000,
            "audio_end_ms": 3000,
            "video_frames": [_JPEG_FRAME],
            "unknown_key": "dropped",
        }
    )

    assert isinstance(command, AppendAudio)
    assert command.duration_ms == 1000
    assert command.audio_end_ms == 3000
    assert command.video_frames == (_JPEG_FRAME,)
    assert dict(command.hints) == {"transcript": "hi there", "duration_ms": 1000, "audio_end_ms": 3000}

    payload = to_internal_payload(command)
    assert payload["type"] == "input_audio_buffer.append"
    assert payload["realtime_event_id"] == "evt-append"
    assert payload["audio"] == base64.b64encode(command.audio).decode("ascii")
    assert payload["format"] == "pcm_f32le"
    assert payload["sample_rate_hz"] == 16000
    assert payload["is_speech"] is True
    assert payload["transcript"] == "hi there"
    assert payload["duration_ms"] == 1000
    assert payload["audio_end_ms"] == 3000
    assert payload["video_frames"] == [_JPEG_FRAME]
    assert "hints" not in payload
    assert "unknown_key" not in payload


def test_append_audio_payload_re_encodes_bytes_and_flattens_hints():
    command = AppendAudio(audio=b"\x01\x02", hints={"vad": {"is_speech": True}}, is_speech=True)

    assert to_internal_payload(command) == {
        "type": "input_audio_buffer.append",
        "audio": "AQI=",
        "format": "pcm16",
        "is_speech": True,
        "vad": {"is_speech": True},
    }


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"audio": _LOUD_PCM16, "format": "mp3"}, "unsupported_audio_format"),
        ({"audio": "@@@@", "format": "pcm_f32le"}, "bad_audio"),
        ({"audio": _LOUD_PCM16, "video_frames": ["aGVsbG8="]}, "invalid_video_frames"),
        ({"audio": _LOUD_PCM16, "video_frames": [_JPEG_FRAME], "max_slice_nums": 4}, "invalid_video_frames"),
        ({"audio": _LOUD_PCM16, "format": "pcm16", "sample_rate_hz": 3}, "bad_event"),
    ],
)
def test_malformed_append_raises_command_error_with_code(body: dict[str, object], code: str):
    with pytest.raises(RealtimeProtocolError) as excinfo:
        decode_command({"type": "input_audio_buffer.append", "event_id": "evt-bad", **body})

    assert excinfo.value.code == code
    assert excinfo.value.event_id == "evt-bad"


# ---- other malformed payloads ----


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({}, "bad_event"),
        ({"type": 7}, "bad_event"),
        ({"type": "session.resume", "session_id": "s", "resume_token": "t"}, "unknown_event"),
        ({"type": "session.event_ack", "server_event_seq": 1}, "unknown_event"),
        ({"type": "conversation.item.frobnicate", "item_id": "item_1"}, "unknown_event"),
        ({"type": "conversation.item.create"}, "bad_event"),
        (
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_audio", "format": "mp3"}]},
            },
            "unsupported_audio_format",
        ),
        ({"type": "conversation.item.delete"}, "missing_item_id"),
        ({"type": "conversation.item.truncate", "audio_end_ms": 1}, "missing_item_id"),
        ({"type": "conversation.item.truncate", "item_id": "item_1"}, "bad_event"),
        ({"type": "session.update", "session": {"output_audio_format": "opus"}}, "unsupported_audio_format"),
        (
            {"type": "session.update", "session": {"turn_detection": {"type": "semantic_vad"}}},
            "unsupported_turn_detection",
        ),
        ({"type": "response.create", "response": {"output_audio_format": "opus"}}, "unsupported_audio_format"),
        ({"type": "playback.ack"}, "bad_event"),
        ({"type": "input.text.append"}, "bad_event"),
        ({"type": "turn.signal"}, "bad_event"),
        ({"type": "turn.signal", "event": "conversation.item.delete", "payload": {}}, "missing_item_id"),
        ({"type": "turn.signal", "event": "conversation.item.truncate", "payload": {"item_id": "i"}}, "bad_event"),
    ],
)
def test_malformed_payloads_raise_command_error_with_code(payload: dict[str, object], code: str):
    with pytest.raises(RealtimeProtocolError) as excinfo:
        decode_command({"event_id": "evt-bad", **payload})

    assert excinfo.value.code == code
    assert excinfo.value.event_id == "evt-bad"


# ---- command fields and internal payload conversion ----


def test_commit_decodes_final_and_response_create():
    command = decode_command(
        {"type": "input_audio_buffer.commit", "event_id": "evt-commit", "response_create": True, "final": False}
    )

    assert command == Commit(event_id="evt-commit", final=False, create_response=True)
    assert Commit().final is True
    assert Commit(is_speech=False, realtime_item_id="item_1").is_speech is False
    assert decode_command({"type": "input_audio_buffer.commit", "create_response": False}).create_response is False


def test_create_response_renders_options_as_response_object():
    command = decode_command({"type": "response.create", "response": {"modalities": ["text"]}})

    assert isinstance(command, CreateResponse)
    assert dict(command.options) == {"modalities": ["text"]}
    assert to_internal_payload(command) == {"type": "response.create", "response": {"modalities": ["text"]}}
    assert to_internal_payload(decode_command({"type": "response.create"})) == {
        "type": "response.create",
        "response": {},
    }


def test_cancel_and_clear_commands_keep_optional_response_id():
    assert decode_command({"type": "response.cancel", "response_id": ""}).response_id is None
    cancel = decode_command({"type": "response.cancel", "response_id": "resp_1"})
    assert cancel == CancelResponse(response_id="resp_1")
    clear = decode_command({"type": "output_audio_buffer.clear", "response_id": "resp_2"})
    assert clear == ClearOutputAudio(response_id="resp_2")
    assert decode_command({"type": "input_audio_buffer.clear"}) == ClearInput()


def test_update_session_preserves_the_session_patch():
    command = decode_command({"type": "session.update", "session": {"instructions": "be brief", "voice": "alloy"}})

    assert command == UpdateSession(patch={"instructions": "be brief", "voice": "alloy"})


def test_create_item_normalizes_item_and_preserves_previous_id():
    command = decode_command(
        {
            "type": "conversation.item.create",
            "previous_item_id": "item_0",
            "item": {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        }
    )

    assert isinstance(command, CreateItem)
    assert command.previous_item_id == "item_0"
    assert command.item["id"].startswith("item_")
    assert command.item["object"] == "realtime.item"
    assert command.item["status"] == "completed"
    without_previous = decode_command({"type": "conversation.item.create", "item": {"content": "x"}})
    assert without_previous.item["role"] == "user"
    assert without_previous.item["content"] == []
    assert without_previous.previous_item_id is None


def test_delete_and_truncate_render_as_turn_signal_payloads():
    delete = decode_command({"type": "conversation.item.delete", "item_id": "item_1"})
    assert delete == DeleteItem(item_id="item_1")
    assert to_internal_payload(delete) == {
        "type": "turn.signal",
        "event": "conversation.item.delete",
        "payload": {"item_id": "item_1"},
    }

    truncate = decode_command(
        {"type": "conversation.item.truncate", "item_id": "item_1", "audio_end_ms": 1500.0, "content_index": 1}
    )
    assert truncate == TruncateItem(item_id="item_1", audio_end_ms=1500, content_index=1)
    assert to_internal_payload(truncate) == {
        "type": "turn.signal",
        "event": "conversation.item.truncate",
        "payload": {"item_id": "item_1", "audio_end_ms": 1500, "content_index": 1},
    }


def test_playback_ack_close_and_text_append_payloads():
    ack = decode_command(
        {"type": "playback.ack", "played_ms": 1200.0, "committed_ms": 1000, "response_id": "resp_1", "item_id": ""}
    )
    assert ack == AckPlayback(played_ms=1200, committed_ms=1000, response_id="resp_1")
    assert to_internal_payload(ack) == {
        "type": "playback.ack",
        "played_ms": 1200,
        "committed_ms": 1000,
        "response_id": "resp_1",
    }

    assert decode_command({"type": "session.close"}) == CloseSession(reason="client_close")
    assert decode_command({"type": "session.close", "reason": "done"}) == CloseSession(reason="done")
    assert decode_command({"type": "input.text.append", "text": "hello"}) == AppendText(text="hello")
    assert decode_command({"type": "session.heartbeat"}) == Heartbeat()


def test_turn_signal_renders_payload_only_when_present_and_dispatches_known_events():
    signal = decode_command({"type": "turn.signal", "event": "user_started", "payload": {"source": "client"}})
    assert signal == SignalTurn(event="user_started", signal_payload={"source": "client"})
    assert to_internal_payload(signal) == {
        "type": "turn.signal",
        "event": "user_started",
        "payload": {"source": "client"},
    }
    assert to_internal_payload(decode_command({"type": "turn.signal", "event": "user_started"})) == {
        "type": "turn.signal",
        "event": "user_started",
    }

    assert decode_command({"type": "turn.signal", "event": "barge_in"}) == BargeIn()
    assert decode_command({"type": "turn.signal", "event": "input.cancel"}) == CancelInput()
    assert decode_command(
        {"type": "turn.signal", "event": "response.cancel", "payload": {"response_id": "resp_1"}}
    ) == CancelResponse(response_id="resp_1")
    assert decode_command(
        {"type": "turn.signal", "event": "session.update", "payload": {"voice": "alloy"}}
    ) == UpdateSession(patch={"voice": "alloy"})
    assert decode_command(
        {"type": "turn.signal", "event": "conversation.item.delete", "payload": {"item_id": "item_1"}}
    ) == DeleteItem(item_id="item_1")
    assert decode_command(
        {"type": "turn.signal", "event": "conversation.item.truncate", "payload": {"item_id": "i", "audio_end_ms": 5}}
    ) == TruncateItem(item_id="i", audio_end_ms=5)
    created = decode_command(
        {"type": "turn.signal", "event": "conversation.item.create", "payload": {"item": {"role": "system"}}}
    )
    assert isinstance(created, CreateItem)
    assert created.item["role"] == "system"


def test_raw_wire_hints_cannot_override_the_normalized_typed_fields():
    """``build_append_audio`` normalizes; a client's raw value must not undo it.

    A wire ``"is_speech": 0`` used to land in ``hints`` and overwrite the
    computed ``bool | None`` in the rendered payload, so the runner's
    silent-commit fast path (``event.get("is_speech") is False``) missed and
    overlap classification followed the unvalidated value.
    """
    command = AppendAudio(
        audio=b"\x00\x00" * 8,
        is_speech=False,
        hints={"is_speech": 0, "rms": 0.001},
    )

    payload = to_internal_payload(command)

    assert payload["is_speech"] is False, "the normalized field wins over the raw hint"
    assert payload["rms"] == 0.001, "a hint with no typed counterpart still comes through"


def test_a_hint_survives_when_its_typed_field_is_unset():
    """Unset typed fields are absent from the payload, so the hint is the only value."""
    command = AppendAudio(audio=b"\x00\x00" * 8, hints={"is_speech": True})

    assert to_internal_payload(command)["is_speech"] is True


@pytest.mark.parametrize("command", [CancelInput(event_id="cancel"), BargeIn(event_id="barge")])
def test_internal_cancel_payload_preserves_event_correlation(command):
    assert to_internal_payload(command) == {"type": command.wire_type, "realtime_event_id": command.event_id}


@pytest.mark.parametrize("event_id", [None, "", "client_1"])
def test_internal_payload_uses_client_correlation_field(event_id):
    command = DeleteItem(item_id="item_1", event_id=event_id)
    expected = {"type": "turn.signal", "event": "conversation.item.delete", "payload": {"item_id": "item_1"}}
    if event_id is not None:
        expected["realtime_event_id"] = event_id
    assert to_internal_payload(command) == expected
