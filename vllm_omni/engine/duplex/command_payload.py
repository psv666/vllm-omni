# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Convert commands for runner handlers that still consume internal dictionaries.

The runner queues typed commands. Only the handlers that need a dictionary use
this conversion; session updates and stateful commit/cancel/item resolution read
the command fields directly. These dictionaries are never client wire events.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

import pybase64 as base64

from vllm_omni.protocol.duplex.commands import (
    AckPlayback,
    AppendAudio,
    BargeIn,
    CancelInput,
    CreateResponse,
    DeleteItem,
    SignalTurn,
    TruncateItem,
)


def to_internal_payload(
    command: AppendAudio
    | CreateResponse
    | CancelInput
    | BargeIn
    | SignalTurn
    | AckPlayback
    | DeleteItem
    | TruncateItem,
) -> dict[str, object]:
    """Render the input expected by a runner handler, preserving client correlation."""
    data: dict[str, object] = {"type": command.wire_type}
    for f in fields(command):
        value = getattr(command, f.name)
        if value is None:
            continue
        if f.name == "event_id":
            data["realtime_event_id"] = value
            continue
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, Mapping):
            value = dict(value)
        data[f.name] = value

    if isinstance(command, AppendAudio):
        data.pop("hints")
        # Normalized fields win over raw hints; an unset field can still be
        # supplied by a hint, matching the runner's existing speech handling.
        data = {**command.hints, **data}
        data["audio"] = base64.b64encode(command.audio).decode("ascii")
        if not command.video_frames:
            data.pop("video_frames", None)
    elif isinstance(command, CreateResponse):
        data["response"] = data.pop("options")
    elif isinstance(command, SignalTurn):
        signal_payload = data.pop("signal_payload")
        if signal_payload:
            data["payload"] = signal_payload
    elif isinstance(command, DeleteItem | TruncateItem):
        data["type"] = "turn.signal"
        data["event"] = command.wire_type
        payload = {"item_id": data.pop("item_id")}
        if isinstance(command, TruncateItem):
            payload["audio_end_ms"] = data.pop("audio_end_ms")
            payload["content_index"] = data.pop("content_index")
        data["payload"] = payload
    elif not isinstance(command, CancelInput | BargeIn | AckPlayback):
        raise TypeError(f"No internal payload handler for {type(command).__name__}")
    return data
