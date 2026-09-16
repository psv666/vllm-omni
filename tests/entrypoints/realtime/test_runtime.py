# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the same command/event interface used by both transports."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.realtime.contracts import (
    AppendAudio,
    CancelResponse,
    CommitAudio,
    Content,
    CreateResponse,
    DeleteItem,
    InsertItem,
    Item,
    ModelDelta,
    RealtimeError,
    SessionConfig,
    UpdateSession,
    VisualPolicy,
)
from vllm_omni.entrypoints.openai.realtime.runtime import RealtimeRuntime
from vllm_omni.entrypoints.openai.realtime.session import SessionStore

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Adapter:
    def __init__(self, *, hold=False, fail=False, finish=None):
        self.release = asyncio.Event()
        if not hold:
            self.release.set()
        self.abort_gate = asyncio.Event()
        self.abort_gate.set()
        self.aborted = []
        self.snapshots = []
        self.closed = []
        self.fail = fail
        self.finish = finish

    async def generate(self, snapshot, request_id):
        self.snapshots.append(snapshot)
        try:
            yield ModelDelta(text="hello", transcript="hello", audio=np.zeros(20, dtype=np.float32))
            await self.release.wait()
            if self.fail:
                raise ValueError("controlled model failure")
            yield ModelDelta(text=" world", transcript=" world", finish_reason=self.finish)
        finally:
            self.closed.append(request_id)

    async def abort(self, request_id):
        self.aborted.append(request_id)
        await self.abort_gate.wait()


async def until(events, kind):
    while True:
        event = await asyncio.wait_for(anext(events), timeout=3)
        if event.kind == kind:
            return event


def text(value, id="question"):
    return Item("user", (Content("text", text=value),), id=id)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "audio"])
@pytest.mark.parametrize("status", ["completed", "failed", "incomplete", "cancelled"])
async def test_final_item_is_atomic_and_preserves_mid_response_input_order(mode, status):
    adapter = Adapter(hold=True, fail=status == "failed", finish="length" if status == "incomplete" else None)
    runtime = RealtimeRuntime(adapter, SessionConfig("test", output_mode=mode))
    events = runtime.events()
    try:
        await runtime.submit(InsertItem(text("first")))
        await until(events, "item_inserted")
        await runtime.submit(CreateResponse())
        started = (await until(events, "response_started")).value[0]
        await until(events, "delta")
        pending = runtime.store.items[-1]
        assert pending.id == started.item.id and pending.content == ()
        assert pending.status == "in_progress"
        await runtime.submit(InsertItem(text("second", "next_question")))
        await until(events, "item_inserted")
        if status == "cancelled":
            await runtime.submit(CancelResponse())
        else:
            adapter.release.set()
        finished = (await until(events, "response_finished")).value
        assert finished.status == status
        assert runtime.store.last_response is finished
        assert runtime.store.items[1] is finished.item
        assert [item.id for item in runtime.store.items] == ["question", started.item.id, "next_question"]
        assert pending.content == () and pending.status == "in_progress"
        assert finished.item.content[0].text == ("hello" if status in ("failed", "cancelled") else "hello world")
        assert adapter.closed == [started.request_id]
        # A second response must reconstruct the committed assistant and next user.
        adapter.release.set()
        await runtime.submit(CreateResponse())
        await until(events, "response_finished")
        assert adapter.snapshots[1].items[1] == finished.item
        assert adapter.snapshots[1].items[2].id == "next_question"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cancel_waits_for_abort_ack_and_fences_queued_deltas():
    adapter = Adapter(hold=True)
    runtime = RealtimeRuntime(adapter, SessionConfig("test", output_mode="text"))
    events = runtime.events()
    try:
        await runtime.submit(InsertItem(text("first")))
        await runtime.submit(CreateResponse())
        await until(events, "response_started")
        delta = await until(events, "delta")
        adapter.abort_gate.clear()
        cancel = asyncio.create_task(runtime.submit(CancelResponse()))
        while not adapter.aborted:
            await asyncio.sleep(0)
        assert delta.delivery.cancelled
        adapter.release.set()
        await asyncio.sleep(0)
        assert runtime.store.items[-1].status == "in_progress"
        assert runtime.active_response is not None
        adapter.abort_gate.set()
        await cancel
        assert (await until(events, "response_finished")).value.status == "cancelled"
        assert runtime.active_response is None
    finally:
        adapter.abort_gate.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_immediate_cancel_before_task_runs_still_finalizes_once():
    runtime = RealtimeRuntime(Adapter(hold=True), SessionConfig("test"))
    events = runtime.events()
    try:
        await runtime.submit(InsertItem(text("first")))
        await runtime.submit(CreateResponse())
        await runtime.submit(CancelResponse())
        response = (await until(events, "response_finished")).value
        assert response.status == "cancelled"
        with pytest.raises(RealtimeError, match="No matching"):
            await runtime.submit(CancelResponse())
        assert runtime.store.last_response is response
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cleanup_timeout_closes_session_and_never_admits_new_request():
    adapter = Adapter(hold=True)
    adapter.abort_gate.clear()
    runtime = RealtimeRuntime(adapter, SessionConfig("test"), cancel_timeout=0.03)
    try:
        await runtime.submit(InsertItem(text("first")))
        await runtime.submit(CreateResponse())
        with pytest.raises(RealtimeError) as exc:
            await runtime.submit(CancelResponse())
        assert exc.value.code == "cleanup_failed"
        with pytest.raises(RealtimeError, match="closed"):
            await runtime.submit(CreateResponse())
    finally:
        adapter.abort_gate.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_empty_and_concurrent_create_do_not_create_an_extra_response():
    runtime = RealtimeRuntime(Adapter(hold=True), SessionConfig("test"))
    try:
        with pytest.raises(RealtimeError, match="Add conversation"):
            await runtime.submit(CreateResponse())
        await runtime.submit(InsertItem(text("first")))
        await runtime.submit(CreateResponse())
        before = runtime.active_response
        with pytest.raises(RealtimeError, match="already active"):
            await runtime.submit(CreateResponse())
        assert runtime.active_response is before
    finally:
        await runtime.close()


def test_rolling_mixed_item_overflow_is_atomic_and_snapshots_hold_deleted_media():
    store = SessionStore(SessionConfig("test", visual=VisualPolicy(retention="rolling", max_items=2)))
    frame = Item("user", (Content("image", data=b"image"),), id="frame")
    store.insert(frame)
    mixed = Item("user", (Content("image", data=b"mixed"), Content("text", text="question")), id="mixed")
    store.insert(mixed)
    snapshot = store.snapshot()
    before = store.items
    too_big = Item("user", (Content("image", data=b"x"),) * 2)
    with pytest.raises(RealtimeError):
        store.insert(too_big)
    assert store.items is before
    mutation = store.insert(Item("user", (Content("image", data=b"new"),), id="new"))
    assert mutation.deleted_ids == ("frame",)
    assert [item.id for item in store.items] == ["mixed", "new"]
    assert snapshot.items[0].content[0].data == b"image"


@pytest.mark.asyncio
async def test_audio_commit_clear_and_configuration_snapshot():
    runtime = RealtimeRuntime(Adapter(), SessionConfig("test", output_mode="text"))
    events = runtime.events()
    try:
        await runtime.submit(AppendAudio(b"\x01"))
        with pytest.raises(RealtimeError, match="complete PCM"):
            await runtime.submit(CommitAudio())
        assert bytes(runtime.store.audio_buffer) == b"\x01"
        await runtime.submit(AppendAudio(b"\x00"))
        await runtime.submit(CommitAudio())
        item = (await until(events, "audio_committed")).value.item
        assert item.content[0].sample_rate == 24000
        assert runtime.active_response is None and not runtime.store.audio_buffer
        await runtime.submit(CreateResponse())
        await until(events, "response_finished")
        await runtime.submit(UpdateSession(replace(runtime.store.config, instructions="next turn")))
        assert runtime.adapter.snapshots[0].config.instructions == ""
        await runtime.submit(DeleteItem(item.id))
        assert all(retained.id != item.id for retained in runtime.store.items)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_close_unblocks_a_full_output_queue_and_releases_snapshot():
    class Flood(Adapter):
        def __init__(self):
            super().__init__()
            self.filled = asyncio.Event()

        async def generate(self, snapshot, request_id):
            try:
                for index in range(1000):
                    if index == 205:
                        self.filled.set()
                    yield ModelDelta(text="word")
            finally:
                self.closed.append(request_id)

    adapter = Flood()
    runtime = RealtimeRuntime(adapter, SessionConfig("test", output_mode="text"))
    await runtime.submit(InsertItem(text("question")))
    await runtime.submit(CreateResponse())
    request_id = runtime.active_response.request_id
    # No event consumer: the producer must eventually pause at the bounded FIFO.
    for _ in range(210):
        await asyncio.sleep(0)
    assert not adapter.filled.is_set()
    await asyncio.wait_for(runtime.close(), timeout=1)
    assert adapter.closed == [request_id]
    assert adapter.aborted == [request_id]
    assert runtime.active_response is None and runtime.store.items == ()


@pytest.mark.asyncio
async def test_cancel_timeout_with_full_queue_closes_instead_of_hanging():
    adapter = Adapter(hold=True)
    adapter.abort_gate.clear()
    runtime = RealtimeRuntime(adapter, SessionConfig("test"), cancel_timeout=0.02)
    await runtime.submit(InsertItem(text("question")))
    await runtime.submit(CreateResponse())
    for _ in range(210):
        await asyncio.sleep(0)
    # A disconnected consumer can leave the FIFO full of previously accepted
    # inputs, as well as model deltas.
    for index in range(197):
        await runtime.submit(InsertItem(text("later", str(index))))
    with pytest.raises(RealtimeError, match="cleanup failed"):
        await asyncio.wait_for(runtime.submit(CancelResponse()), timeout=1)
    await asyncio.wait_for(runtime.close(), timeout=1)
    assert runtime.store.items == ()


@pytest.mark.asyncio
async def test_generator_cleanup_timeout_does_not_keep_session_admission_open():
    class StuckClosing(Adapter):
        def __init__(self):
            super().__init__(hold=True)
            self.in_cleanup = asyncio.Event()
            self.allow_close = asyncio.Event()

        async def generate(self, snapshot, request_id):
            try:
                yield ModelDelta(text="partial")
                await self.release.wait()
            finally:
                self.in_cleanup.set()
                # Model cleanup may itself await an engine acknowledgement.
                await self.allow_close.wait()

    adapter = StuckClosing()
    runtime = RealtimeRuntime(adapter, SessionConfig("test"), cancel_timeout=0.02)
    events = runtime.events()
    await runtime.submit(InsertItem(text("question")))
    await runtime.submit(CreateResponse())
    await until(events, "delta")
    with pytest.raises(RealtimeError, match="cleanup failed"):
        await asyncio.wait_for(runtime.submit(CancelResponse()), timeout=1)
    assert adapter.in_cleanup.is_set()
    with pytest.raises(RealtimeError, match="closed"):
        await runtime.submit(CreateResponse())
    adapter.allow_close.set()
    await asyncio.wait_for(runtime.close(), timeout=1)
    await asyncio.sleep(0)
    assert runtime.store.items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["audio", "text"])
async def test_empty_model_output_finalizes_failed_once(mode):
    class Empty(Adapter):
        async def generate(self, snapshot, request_id):
            yield ModelDelta(finish_reason="stop")

    runtime = RealtimeRuntime(Empty(), SessionConfig("test", output_mode=mode))
    events = runtime.events()
    try:
        await runtime.submit(InsertItem(text("question")))
        await runtime.submit(CreateResponse())
        final = (await until(events, "response_finished")).value
        assert final.status == "failed" and final.reason == "empty_output"
        assert runtime.store.items[-1] is final.item
        assert runtime.active_response is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_disconnected_cancel_caller_and_close_join_one_cleanup():
    adapter = Adapter(hold=True)
    adapter.abort_gate.clear()
    runtime = RealtimeRuntime(adapter, SessionConfig("test"))
    events = runtime.events()
    await runtime.submit(InsertItem(text("question")))
    await runtime.submit(CreateResponse())
    await until(events, "delta")
    cancel = asyncio.create_task(runtime.submit(CancelResponse()))
    while not adapter.aborted:
        await asyncio.sleep(0)
    cancel.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancel
    close = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not close.done()
    adapter.abort_gate.set()
    await asyncio.wait_for(close, timeout=1)
    assert len(adapter.aborted) == 1
    assert adapter.closed == adapter.aborted
    assert runtime.store.items == () and runtime.active_response is None
