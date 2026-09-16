# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ordered conversation storage with synchronous, atomic mutations."""

from __future__ import annotations

from .contracts import Content, Item, ItemInsertion, RealtimeError, Response, SessionConfig, TurnSnapshot, new_id

MAX_AUDIO_BUFFER_BYTES = 16 * 1024 * 1024
MAX_SESSION_MEDIA_BYTES = 64 * 1024 * 1024


def image_count(item: Item) -> int:
    return sum(part.kind == "image" for part in item.content)


def camera_item(item: Item) -> bool:
    return item.role == "user" and len(item.content) == 1 and item.content[0].kind == "image"


class SessionStore:
    def __init__(self, config: SessionConfig):
        self.id = new_id("sess")
        self.conversation_id = new_id("conv")
        self.config = config
        self._items: tuple[Item, ...] = ()
        self.audio_buffer = bytearray()
        self.last_response: Response | None = None

    @property
    def items(self) -> tuple[Item, ...]:
        return self._items

    def update(self, config: SessionConfig) -> None:
        if config.model != self.config.model:
            raise RealtimeError("The model cannot change during a session.", "session.model")
        if sum(map(image_count, self._items)) > config.visual.max_items:
            raise RealtimeError("Delete image items before lowering capacity.", "session.omni.input_image_max_items")
        self.config = config

    def insert(self, item: Item, previous_id: str | None = None) -> ItemInsertion:
        if any(existing.id == item.id for existing in self._items):
            raise RealtimeError("An item with this ID already exists.", "item.id")
        position = len(self._items)
        if previous_id == "root":
            position = 0
        elif previous_id is not None:
            for index, existing in enumerate(self._items):
                if existing.id == previous_id:
                    position = index + 1
                    break
            else:
                raise RealtimeError("The preceding item does not exist.", "previous_item_id")
        overflow = sum(map(image_count, self._items)) + image_count(item) - self.config.visual.max_items
        removed: tuple[str, ...] = ()
        if overflow > 0 and self.config.visual.retention == "rolling":
            removed = tuple(existing.id for existing in self._items if camera_item(existing))[:overflow]
        if overflow > len(removed):
            raise RealtimeError(
                "Image capacity exceeded; delete image items first.", "item.content", "image_limit_exceeded"
            )
        position -= sum(existing.id in removed for existing in self._items[:position])
        retained = [existing for existing in self._items if existing.id not in removed]
        retained.insert(position, item)
        if sum(len(part.data) for existing in retained for part in existing.content) > MAX_SESSION_MEDIA_BYTES:
            raise RealtimeError(
                "Conversation media capacity exceeded; delete old items.", "item.content", "media_limit_exceeded"
            )
        # No mutation above this point: a rejected insertion cannot evict items.
        self._items = tuple(retained)
        return ItemInsertion(item, retained[position - 1].id if position else None, removed)

    def replace_item(self, replacement: Item) -> None:
        """Replace metadata/media without changing its conversation position."""
        for index, item in enumerate(self._items):
            if item.id == replacement.id:
                if item.status == "in_progress":
                    raise RealtimeError("An active output item cannot be replaced.", "item_id")
                self._items = self._items[:index] + (replacement,) + self._items[index + 1 :]
                return
        raise RealtimeError("The item does not exist.", "item_id")

    def delete(self, item_id: str) -> Item:
        for item in self._items:
            if item.id == item_id:
                if item.status == "in_progress":
                    raise RealtimeError("Cancel the active response before deleting its item.", "item_id")
                self._items = tuple(existing for existing in self._items if existing.id != item_id)
                return item
        raise RealtimeError("The item does not exist.", "item_id")

    def append_audio(self, data: bytes) -> None:
        if len(self.audio_buffer) + len(data) > MAX_AUDIO_BUFFER_BYTES:
            raise RealtimeError(
                "Uncommitted audio buffer is full; commit or clear it.", "audio", "audio_limit_exceeded"
            )
        self.audio_buffer.extend(data)

    def commit_audio(self, sample_rate: int) -> ItemInsertion:
        if not self.audio_buffer:
            raise RealtimeError("The input audio buffer is empty.", "audio", "input_audio_buffer_commit_empty")
        if len(self.audio_buffer) % 2:
            raise RealtimeError("Expected complete PCM16 samples.", "audio")
        mutation = self.insert(
            Item("user", (Content("audio", data=bytes(self.audio_buffer), sample_rate=sample_rate),))
        )
        self.audio_buffer.clear()
        return mutation

    def snapshot(self, config: SessionConfig | None = None) -> TurnSnapshot:
        config = config or self.config
        items = tuple(item for item in self._items if item.status != "in_progress")
        if not config.instructions.strip() and not any(
            part.text.strip() or part.data for item in items for part in item.content
        ):
            raise RealtimeError("Add conversation input or response instructions before generating.", "response")
        # Content uses immutable bytes; snapshots retain references after deletion
        # and release them naturally when the response finishes, without PIL copies.
        return TurnSnapshot(items, config)

    def finalize(self, response: Response) -> bool:
        """Replace the reserved slot and response terminal together, without await.

        Only the runtime calls this method. Completed output is immutable and is
        installed before the terminal event can be observed by either transport.
        """
        if self.last_response is not None and self.last_response.id == response.id:
            return False
        for index, item in enumerate(self._items):
            if item.id == response.item.id:
                if item.status != "in_progress":
                    return False
                self._items = self._items[:index] + (response.item,) + self._items[index + 1 :]
                self.last_response = response
                return True
        raise RuntimeError("The response lost its reserved conversation position")

    def clear(self) -> None:
        self._items = ()
        self.audio_buffer.clear()
        self.last_response = None
