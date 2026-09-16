# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Legacy buffer views over the same conversation store used by GA.

These views support the existing video prompt hooks; they own no second copy
of the media or history. Request execution belongs to RealtimeRuntime.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import replace

import pybase64 as base64

from .contracts import Content, Item, SessionConfig, VisualPolicy
from .qwen3 import Qwen3RealtimeAdapter, _PreparedTurn
from .runtime import RealtimeRuntime


class PreparedAdapter:
    """A turn prepared by a legacy prompt hook uses the shared model iterator."""

    def __init__(self, adapter: Qwen3RealtimeAdapter):
        self.adapter = adapter
        self.prepared: _PreparedTurn | None = None

    async def generate(self, snapshot, request_id):
        from contextlib import aclosing

        prepared, self.prepared = self.prepared, None
        assert prepared is not None
        async with aclosing(self.adapter.generate_prepared(prepared, request_id)) as outputs:
            async for output in outputs:
                yield output

    async def abort(self, request_id):
        await self.adapter.abort(request_id)


class LegacySession:
    def __init__(self, adapter, *, model, max_frames):
        self.adapter = PreparedAdapter(adapter)
        self.runtime = RealtimeRuntime(
            self.adapter, SessionConfig(model, output_mode="text", visual=VisualPolicy("rolling", max_frames))
        )
        self.store = self.runtime.store
        self.frames = FrameView(self.store)
        self.history = HistoryView(self)

    def prune_history(self):
        history = [item for item in self.store.items if not item.metadata.get("legacy_frame")]
        for item in history[:-2]:
            self.store.delete(item.id)


class FrameView(MutableSequence):
    def __init__(self, store):
        self.store = store

    def items(self):
        return [item for item in self.store.items if item.metadata.get("legacy_frame")]

    def __len__(self):
        return len(self.items())

    def __getitem__(self, index):
        selected = self.items()[index]
        if isinstance(index, slice):
            return [base64.b64encode(item.content[0].data).decode() for item in selected]
        return base64.b64encode(selected.content[0].data).decode()

    def __delitem__(self, index):
        selected = self.items()[index]
        for item in selected if isinstance(index, slice) else [selected]:
            self.store.delete(item.id)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            retained = set(value)
            for item in self.items()[index]:
                if base64.b64encode(item.content[0].data).decode() not in retained:
                    self.store.delete(item.id)
        else:
            item = self.items()[index]
            self.store.replace_item(replace(item, content=(Content("image", data=base64.b64decode(value)),)))

    def insert(self, index, value):
        frames = self.items()
        previous = frames[index - 1].id if 0 < index < len(frames) else "root" if index == 0 else None
        self.store.insert(
            Item("user", (Content("image", data=base64.b64decode(value)),), metadata={"legacy_frame": True}), previous
        )


class HistoryView:
    def __init__(self, session):
        self.session = session

    def messages(self):
        return [
            {"role": item.role, "content": "".join(part.text for part in item.content)}
            for item in self.session.store.items
            if not item.metadata.get("legacy_frame") and item.status == "completed"
        ]

    def __len__(self):
        return len(self.messages())

    def __getitem__(self, index):
        return self.messages()[index]

    def append(self, message):
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
        self.session.store.insert(Item(message["role"], (Content("text", text=content),)))
        self.session.prune_history()
