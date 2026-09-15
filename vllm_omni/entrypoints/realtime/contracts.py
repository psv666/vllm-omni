# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""In-process conversation contracts. No WebSocket or wire event types."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class RealtimeError(ValueError):
    def __init__(self, message: str, param: str | None = None, code: str = "invalid_value"):
        super().__init__(message)
        self.param = param
        self.code = code


@dataclass(frozen=True)
class Content:
    kind: Literal["text", "image", "audio"]
    text: str = ""
    data: bytes = b""
    sample_rate: int = 24000
    media_type: str = "image/jpeg"
    detail: str = "auto"


@dataclass(frozen=True)
class Item:
    role: Literal["user", "assistant", "system"]
    content: tuple[Content, ...]
    id: str = field(default_factory=lambda: new_id("item"))
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualPolicy:
    retention: Literal["client", "rolling"] = "client"
    max_items: int = 50
    sample: int = 4
    evs: bool = True
    threshold: float = 0.95


@dataclass(frozen=True)
class SessionConfig:
    model: str
    instructions: str = ""
    output_mode: Literal["text", "audio"] = "audio"
    max_tokens: int | None = None
    visual: VisualPolicy = field(default_factory=VisualPolicy)
    use_audio_in_video: bool = True


@dataclass(frozen=True)
class TurnSnapshot:
    items: tuple[Item, ...]
    config: SessionConfig


@dataclass(frozen=True)
class ModelDelta:
    text: str = ""
    # Spoken text is explicit: a transport must not guess from arbitrary text.
    transcript: str = ""
    audio: Any = None
    sample_rate: int = 24000
    finish_reason: str | None = None


class ModelAdapter(Protocol):
    def generate(self, snapshot: TurnSnapshot, request_id: str) -> AsyncGenerator[ModelDelta, None]: ...

    async def abort(self, request_id: str) -> None: ...


@dataclass(frozen=True)
class Response:
    id: str
    request_id: str
    item: Item
    mode: str
    status: str = "in_progress"
    reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class UpdateSession:
    config: SessionConfig


@dataclass(frozen=True)
class InsertItem:
    item: Item
    previous_id: str | None = None


@dataclass(frozen=True)
class DeleteItem:
    item_id: str


@dataclass(frozen=True)
class AppendAudio:
    data: bytes


@dataclass(frozen=True)
class CommitAudio:
    sample_rate: int = 24000


@dataclass(frozen=True)
class ClearAudio:
    pass


@dataclass(frozen=True)
class CreateResponse:
    config: SessionConfig | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class CancelResponse:
    response_id: str | None = None


Command = (
    UpdateSession | InsertItem | DeleteItem | AppendAudio | CommitAudio | ClearAudio | CreateResponse | CancelResponse
)


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    value: Any = None
    response_id: str | None = None
    delivery: DeliveryToken | None = None


@dataclass
class DeliveryToken:
    cancelled: bool = False


@dataclass(frozen=True)
class ItemInsertion:
    item: Item
    previous_id: str | None
    deleted_ids: tuple[str, ...] = ()
