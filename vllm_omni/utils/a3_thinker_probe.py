# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, host-clock measurements for the A3 Thinker experiment."""

import json
import os
import time
from multiprocessing.util import Finalize
from pathlib import Path

TRACE_DIR_ENV = "VLLM_OMNI_A3_THINKER_TRACE_DIR"
MAX_SAMPLES = 100_000


def trace_window(phase: str, boundary: str, concurrency: int | None) -> None:
    directory = os.environ.get(TRACE_DIR_ENV)
    if directory:
        row = [phase, boundary, concurrency, time.monotonic_ns()]
        with (Path(directory) / "windows.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")


class ThinkerProbe:
    def __init__(self, directory: str, owner: str, *, finalize: bool = True):
        self.path = Path(directory) / f"{owner}-{os.getpid()}.json"
        self.prompt_lengths: dict[str, int] = {}
        self.rows: list[tuple[str, int, str, int, int]] = []
        self.dropped = 0
        # Multiprocessing can bypass atexit in child processes.
        if finalize:
            Finalize(None, self.flush, exitpriority=10)

    def classify(self, output):
        for req_id in output.finished_req_ids:
            self.prompt_lengths.pop(req_id, None)
        for request in output.scheduled_new_reqs:
            if request.prompt_token_ids is not None:
                self.prompt_lengths[request.req_id] = len(request.prompt_token_ids)
        cached = output.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
        computed.update((req.req_id, req.num_computed_tokens) for req in output.scheduled_new_reqs)
        batch = len(output.num_scheduled_tokens)
        phases = set()
        for req_id in output.num_scheduled_tokens:
            if req_id not in computed or req_id not in self.prompt_lengths:
                phases.add("unknown")
            elif computed[req_id] < self.prompt_lengths[req_id]:
                phases.add("prefill")
            else:
                phases.add("decode")
        if "unknown" in phases:
            return "unknown", batch
        return next(iter(phases)) if len(phases) == 1 else "mixed", batch

    def record(self, name, batch, started, ended):
        if batch[1] == 0:
            return
        if len(self.rows) == MAX_SAMPLES:
            self.dropped += 1
            return
        self.rows.append((batch[0], batch[1], name, started, ended - started))

    def flush(self):
        temp = self.path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as file:
            json.dump({"rows": self.rows, "dropped": self.dropped}, file, separators=(",", ":"))
        temp.replace(self.path)


def attach_engine_probe(core, directory: str) -> ThinkerProbe:
    probe = ThinkerProbe(directory, "engine", finalize=False)
    schedule = core.scheduler.schedule
    execute = core.model_executor.execute_model
    consume = core.scheduler.update_from_output
    pending = {}
    last_scheduled = None

    def timed_schedule(*args, **kwargs):
        nonlocal last_scheduled
        started = time.monotonic_ns()
        output = schedule(*args, **kwargs)
        ended = time.monotonic_ns()
        batch = probe.classify(output)
        last_scheduled = (id(output), batch)
        probe.record("schedule", batch, started, ended)
        return output

    def timed_dispatch(scheduler_output, *args, **kwargs):
        # Use the same classification captured before the scheduler advances again.
        if last_scheduled is not None and last_scheduled[0] == id(scheduler_output):
            batch = last_scheduled[1]
        else:
            batch = probe.classify(scheduler_output)
        if batch[1]:
            pending[id(scheduler_output)] = (batch, time.monotonic_ns())
        return execute(scheduler_output, *args, **kwargs)

    def timed_consume(scheduler_output, *args, **kwargs):
        sample = pending.pop(id(scheduler_output), None)
        if sample is not None:
            batch, started = sample
            probe.record("dispatch_to_consume", batch, started, time.monotonic_ns())
        return consume(scheduler_output, *args, **kwargs)

    core.scheduler.schedule = timed_schedule
    core.model_executor.execute_model = timed_dispatch
    core.scheduler.update_from_output = timed_consume
    return probe


def attach_worker_probe(runner, directory: str, rank: int) -> None:
    probe = ThinkerProbe(directory, f"tp{rank}")
    execute = runner.execute_model
    sample = runner.sample_tokens
    current_batch = ("unknown", 0)

    def timed_execute(scheduler_output, *args, **kwargs):
        nonlocal current_batch
        if scheduler_output is None or kwargs.get("dummy_run"):
            current_batch = ("unknown", 0)
            return execute(scheduler_output, *args, **kwargs)
        current_batch = probe.classify(scheduler_output)
        started = time.monotonic_ns()
        result = execute(scheduler_output, *args, **kwargs)
        probe.record("execute_model", current_batch, started, time.monotonic_ns())
        return result

    def timed_sample(*args, **kwargs):
        nonlocal current_batch
        batch = current_batch
        current_batch = ("unknown", 0)
        started = time.monotonic_ns()
        result = sample(*args, **kwargs)
        probe.record("sample_tokens", batch, started, time.monotonic_ns())
        return result

    runner.execute_model = timed_execute
    runner.sample_tokens = timed_sample
