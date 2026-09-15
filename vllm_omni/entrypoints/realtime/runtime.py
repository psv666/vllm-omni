# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A transport-independent owner of one conversation and its response task."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field, replace

from .contracts import (
    AppendAudio,
    CancelResponse,
    ClearAudio,
    Command,
    CommitAudio,
    Content,
    CreateResponse,
    DeleteItem,
    DeliveryToken,
    InsertItem,
    Item,
    ModelAdapter,
    RealtimeError,
    Response,
    RuntimeEvent,
    SessionConfig,
    TurnSnapshot,
    UpdateSession,
    new_id,
)
from .session import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class _ResponseDraft:
    response: Response
    delivery: DeliveryToken = field(default_factory=DeliveryToken)
    text: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    audio_samples: int = 0
    task: asyncio.Task | None = None
    cleanup: asyncio.Task | None = None
    abort_finished: asyncio.Event = field(default_factory=asyncio.Event)
    finalized: bool = False
    started: bool = False
    iterator_closed: asyncio.Event = field(default_factory=asyncio.Event)


class RealtimeRuntime:
    """submit / events / close is the shared interface for both transports.

    Commands are serialized; model iteration runs separately with a fixed input
    snapshot. Finalization never awaits a network send. A FIFO of immutable
    events keeps the terminal ahead of a subsequently accepted response.
    """

    def __init__(self, adapter: ModelAdapter, config: SessionConfig, *, cancel_timeout: float = 10.0):
        self.adapter = adapter
        self.store = SessionStore(config)
        self.cancel_timeout = cancel_timeout
        self._commands = asyncio.Lock()
        self._events: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue(maxsize=200)
        self._active: _ResponseDraft | None = None
        self._closed = False

    @property
    def active_response(self) -> Response | None:
        return self._active.response if self._active else None

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        while (event := await self._events.get()) is not None:
            yield event

    async def _emit(self, event: RuntimeEvent) -> None:
        if not self._closed:
            await self._events.put(event)

    async def submit(self, command: Command) -> None:
        async with self._commands:
            if self._closed:
                raise RealtimeError("The session is closed.", code="session_closed")
            if isinstance(command, UpdateSession):
                self.store.update(command.config)
                await self._emit(RuntimeEvent("configured", self.store.config))
            elif isinstance(command, InsertItem):
                await self._emit(RuntimeEvent("item_inserted", self.store.insert(command.item, command.previous_id)))
            elif isinstance(command, DeleteItem):
                await self._emit(RuntimeEvent("item_deleted", self.store.delete(command.item_id).id))
            elif isinstance(command, AppendAudio):
                self.store.append_audio(command.data)
            elif isinstance(command, CommitAudio):
                await self._emit(RuntimeEvent("audio_committed", self.store.commit_audio(command.sample_rate)))
            elif isinstance(command, ClearAudio):
                self.store.audio_buffer.clear()
                await self._emit(RuntimeEvent("audio_cleared"))
            elif isinstance(command, CreateResponse):
                if self._active:
                    raise RealtimeError(
                        "A response is already active.", "response", "conversation_already_has_active_response"
                    )
                await self._start(self.store.snapshot(command.config), command.request_id)
            elif isinstance(command, CancelResponse):
                if self._active is None or command.response_id not in (None, self._active.response.id):
                    raise RealtimeError("No matching active response.", "response_id", "response_cancel_not_active")
                await self._cancel(self._active)
            else:
                raise TypeError(f"Unsupported runtime command: {type(command).__name__}")

    async def _start(self, snapshot: TurnSnapshot, request_id: str | None = None) -> None:
        response = Response(
            new_id("resp"),
            request_id or new_id("realtime"),
            Item("assistant", (), status="in_progress"),
            snapshot.config.output_mode,
        )
        mutation = self.store.insert(response.item)
        active = _ResponseDraft(response)
        self._active = active
        await self._emit(RuntimeEvent("response_started", (response, mutation.previous_id), response.id))
        active.task = asyncio.create_task(self._run(active, snapshot))

    async def _run(self, active: _ResponseDraft, snapshot: TurnSnapshot) -> None:
        active.started = True
        status, reason, error = "completed", None, None
        try:
            async with aclosing(self.adapter.generate(snapshot, active.response.request_id)) as outputs:
                async for delta in outputs:
                    if active.delivery.cancelled:
                        break
                    if delta.text:
                        active.text.append(delta.text)
                    if delta.transcript:
                        active.transcript.append(delta.transcript)
                    if delta.audio is not None:
                        active.audio_samples += len(delta.audio)
                    if delta.finish_reason == "length":
                        status, reason = "incomplete", "max_output_tokens"
                    if delta.text or delta.transcript or delta.audio is not None:
                        await self._emit(RuntimeEvent("delta", delta, active.response.id, active.delivery))
        except asyncio.CancelledError:
            status, reason = "cancelled", "client_cancelled"
        except Exception as exc:
            logger.exception("Response %s failed", active.response.id)
            status, reason, error = "failed", "server_error", str(exc)
        finally:
            active.iterator_closed.set()
            if active.delivery.cancelled:
                # A terminal can race its abort acknowledgement. Keep admission
                # fenced until abort has actually returned.
                await active.abort_finished.wait()
                status, reason = "cancelled", "client_cancelled"
            await self._finish(active, status, reason, error)

    async def _finish(self, active: _ResponseDraft, status: str, reason: str | None, error: str | None = None) -> None:
        if active.finalized:
            return
        audio = active.response.mode == "audio"
        has_output = active.audio_samples > 0 if audio else bool("".join(active.text))
        if status == "completed" and not has_output:
            status, reason, error = "failed", "empty_output", "The model returned no output for the requested modality."
        part = Content("audio" if audio else "text", text="".join(active.transcript if audio else active.text))
        item = replace(
            active.response.item, content=(part,), status="completed" if status == "completed" else "incomplete"
        )
        response = replace(active.response, item=item, status=status, reason=reason, error=error)
        # One non-awaiting commit: the draft never mutates the reserved item.
        self.store.finalize(response)
        active.finalized = True
        active.response = response
        try:
            await self._emit(RuntimeEvent("response_finished", response, response.id))
        finally:
            if self._active is active:
                self._active = None

    async def _cancel(self, active: _ResponseDraft) -> None:
        # Cleanup belongs to the runtime, even if the reader/projector that
        # submitted cancel disappears. close() joins this same operation.
        if active.cleanup is None:
            if not active.finalized:
                active.delivery.cancelled = True
            active.cleanup = asyncio.create_task(self._stop(active))
            active.cleanup.add_done_callback(self._consume_task_result)
        await asyncio.shield(active.cleanup)

    async def _stop(self, active: _ResponseDraft) -> None:
        async def stop() -> None:
            if active.finalized:
                if active.task:
                    await active.task
                return
            active.delivery.cancelled = True
            try:
                if active.task and not active.task.done():
                    active.task.cancel()
                if active.started:
                    await active.iterator_closed.wait()
                await self.adapter.abort(active.response.request_id)
            finally:
                active.abort_finished.set()
            if active.task:
                try:
                    await active.task
                except asyncio.CancelledError:
                    pass
            # A task cancelled before its first instruction skips finally.
            await self._finish(active, "cancelled", "client_cancelled")

        cleanup = asyncio.create_task(stop())
        try:
            # wait_for also waits for cancellation completion, so a model stuck
            # in generator cleanup could exceed its deadline indefinitely.
            done, _ = await asyncio.wait({cleanup}, timeout=self.cancel_timeout)
            if not done:
                raise TimeoutError("Model cleanup exceeded its deadline")
            cleanup.result()
        except BaseException as exc:
            self._closed = True
            self._drain_events()
            active.delivery.cancelled = True
            active.finalized = True
            active.abort_finished.set()
            self._active = None
            cleanup.cancel()
            cleanup.add_done_callback(self._consume_task_result)
            if active.task:
                active.task.cancel()
                active.task.add_done_callback(self._consume_task_result)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RealtimeError(
                "Response cleanup failed; this session is closed.", "response", "cleanup_failed"
            ) from exc

    @staticmethod
    def _consume_task_result(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    def discard_pending_events(self) -> None:
        """Discard a stopped consumer's events before handing off its stream."""
        if self._active:
            raise RuntimeError("Cannot discard events while a response is active")
        self._drain_events()

    def _drain_events(self) -> None:
        while not self._events.empty():
            self._events.get_nowait()

    async def close(self) -> None:
        self._closed = True
        self._drain_events()
        if self._active:
            try:
                await self._cancel(self._active)
            except RealtimeError:
                logger.exception("Failed to close realtime response")
        self.store.clear()
        while not self._events.empty():
            self._events.get_nowait()
        self._events.put_nowait(None)
