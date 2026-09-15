# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WebSocket I/O and GA translation; conversation execution lives in runtime."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.websockets import WebSocketDisconnect

from vllm_omni.entrypoints.realtime.contracts import RealtimeError, SessionConfig
from vllm_omni.entrypoints.realtime.runtime import RealtimeRuntime

from .codec import decode_event
from .events import GAEventEncoder, error_event, event


class RealtimeGAConnection:
    def __init__(
        self,
        websocket: Any,
        adapter: Any,
        model: str,
        *,
        runtime: RealtimeRuntime | None = None,
        idle_timeout: float = 60.0,
    ):
        self.websocket = websocket
        self.idle_timeout = idle_timeout
        self.runtime = runtime or RealtimeRuntime(adapter, SessionConfig(model))
        store = self.runtime.store
        self.encoder = GAEventEncoder(store.id, store.conversation_id)
        self._send_lock = asyncio.Lock()

    async def _send(self, payload):
        async with self._send_lock:
            await asyncio.wait_for(self.websocket.send_json(payload), timeout=10)

    async def _read(self):
        while True:
            try:
                raw = await asyncio.wait_for(self.websocket.receive_json(), timeout=self.idle_timeout)
            except json.JSONDecodeError:
                await self._send(error_event(RealtimeError("Expected a JSON object.", code="invalid_json")))
                continue
            except TimeoutError:
                if self.runtime.active_response is not None:
                    continue
                await self._send(error_event(RealtimeError("Idle timeout.", code="idle_timeout")))
                return
            client_event_id = raw.get("event_id") if isinstance(raw, dict) else None
            try:
                # Image validation and large base64 conversion must not block
                # the sender or an unrelated connection's cancellation.
                command = await asyncio.to_thread(decode_event, raw, self.runtime.store.config)
                await self.runtime.submit(command)
            except RealtimeError as exc:
                await self._send(error_event(exc, client_event_id if isinstance(client_event_id, str) else None))
                if exc.code == "cleanup_failed":
                    return

    async def _write(self):
        async for fact in self.runtime.events():
            if fact.delivery is not None and fact.delivery.cancelled:
                continue
            for payload in self.encoder.encode(fact):
                async with self._send_lock:
                    if fact.delivery is None or not fact.delivery.cancelled:
                        await asyncio.wait_for(self.websocket.send_json(payload), timeout=10)

    async def handle_connection(self) -> None:
        await self.websocket.accept()
        tasks = []
        try:
            await self._send(event("session.created", session=self.encoder.session(self.runtime.store.config)))
            tasks = [asyncio.create_task(self._read()), asyncio.create_task(self._write())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # close also handles caller-task cancellation and send failures.
            cleanup = asyncio.create_task(self.runtime.close())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            close = getattr(self.websocket, "close", None)
            if close is not None:
                try:
                    await close()
                except RuntimeError:
                    pass
