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
DEVICE_BATCH_ENV = "VLLM_OMNI_A3_DEVICE_PROFILE_BATCH"
DEVICE_STEPS = 4


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
        self.decode_graphs: list[tuple] = []
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
            json.dump(
                {"rows": self.rows, "decode_graphs": self.decode_graphs, "dropped": self.dropped},
                file,
                separators=(",", ":"),
            )
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


class DeviceGraphCapture:
    """A short NPU trace around warmed graph calls of a single decode shape."""

    def __init__(self, directory: str, rank: int, batch: int):
        self.directory = Path(directory)
        self.output = self.directory / f"device-tp{rank}"
        self.batch = batch
        self.profiler = None
        self.measured = False
        self.finished = False
        self.matched = 0
        self.slow_streak = []
        self.steps = 0
        self.dispatch_ns = []
        Finalize(None, self.close, exitpriority=20)

    def call(self, call, args, kwargs, batch, shape):
        if self.finished:
            return call(*args, **kwargs)
        match = (
            batch == ("decode", self.batch)
            and shape is not None
            and shape[:3]
            == (
                self.batch,
                "PIECEWISE",
                16,
            )
        )
        if not match:
            if self.profiler is not None:
                self.close()
            self.matched = 0
            self.slow_streak = []
            return call(*args, **kwargs)
        if not self.measured:
            windows = self.directory / "windows.jsonl"
            self.measured = windows.is_file() and any(
                json.loads(line)[:3] == ["measured", "start", self.batch]
                for line in windows.read_text(encoding="utf-8").splitlines()
                if line.endswith("]")
            )
        if not self.measured:
            return call(*args, **kwargs)
        self.matched += 1
        if self.matched <= 8:
            return call(*args, **kwargs)
        if self.profiler is None and self.batch == 16 and len(self.slow_streak) < 2:
            started = time.monotonic_ns()
            result = call(*args, **kwargs)
            elapsed = time.monotonic_ns() - started
            if elapsed >= 100_000_000:
                self.slow_streak.append(elapsed)
            else:
                self.slow_streak.clear()
            return result
        if self.profiler is None:
            self.start()
        started = time.monotonic_ns()
        result = call(*args, **kwargs)
        self.dispatch_ns.append(time.monotonic_ns() - started)
        self.steps += 1
        if self.steps == DEVICE_STEPS:
            self.close()
        return result

    def start(self):
        import torch_npu

        backend = torch_npu.profiler
        self.output.mkdir()
        self.profiler = backend.profile(
            activities=[backend.ProfilerActivity.CPU, backend.ProfilerActivity.NPU],
            experimental_config=backend._ExperimentalConfig(
                export_type=backend.ExportType.Text,
                profiler_level=backend.ProfilerLevel.Level1,
                data_simplification=True,
            ),
            on_trace_ready=backend.tensorboard_trace_handler(str(self.output), analyse_flag=self.batch != 16),
        )
        self.profiler.start()

    def close(self):
        if self.profiler is None:
            return
        self.finished = True
        profiler, self.profiler = self.profiler, None
        profiler.stop()
        (self.output / "capture.json").write_text(
            json.dumps(
                {
                    "steps": self.steps,
                    "graph_tokens": 16,
                    "batch": self.batch,
                    "host_forward_ns": self.dispatch_ns,
                    "pre_capture_slow_ns": self.slow_streak,
                }
            ),
            encoding="utf-8",
        )


def attach_worker_probe(runner, directory: str, rank: int) -> None:
    probe = ThinkerProbe(directory, f"tp{rank}")
    target = os.environ.get(DEVICE_BATCH_ENV)
    device_capture = DeviceGraphCapture(directory, rank, int(target)) if target in {"8", "9", "16"} else None
    execute = runner.execute_model
    sample = runner.sample_tokens
    prepare = runner.prepare_inputs
    prepare_attn = runner.prepare_attn
    call_model = runner._call_model
    current_batch = ("unknown", 0)
    shape = None
    model_ns = None
    attn_ns = None
    graph_ns = None
    graph_manager = None

    def timed_prepare(scheduler_output, batch_req_state, batch_desc, *args, **kwargs):
        nonlocal shape
        if current_batch[0] != "decode" or current_batch[1] < 8:
            return prepare(scheduler_output, batch_req_state, batch_desc, *args, **kwargs)
        started = time.monotonic_ns()
        result = prepare(scheduler_output, batch_req_state, batch_desc, *args, **kwargs)
        shape = (
            scheduler_output.total_num_scheduled_tokens,
            batch_desc.cg_mode.name,
            result.num_tokens_after_padding,
            batch_desc.num_reqs,
            time.monotonic_ns() - started,
        )
        return result

    def timed_prepare_attn(*args, **kwargs):
        nonlocal attn_ns
        if shape is None:
            return prepare_attn(*args, **kwargs)
        started = time.monotonic_ns()
        result = prepare_attn(*args, **kwargs)
        attn_ns = time.monotonic_ns() - started
        return result

    def time_graph_call(call):
        def timed_graph_call(*args, **kwargs):
            nonlocal graph_ns
            if shape is None and device_capture is None:
                return call(*args, **kwargs)
            started = time.monotonic_ns()
            if device_capture is not None:
                result = device_capture.call(call, args, kwargs, current_batch, shape)
            else:
                result = call(*args, **kwargs)
            graph_ns = (graph_ns or 0) + time.monotonic_ns() - started
            return result

        return timed_graph_call

    def timed_call_model(*args, **kwargs):
        nonlocal model_ns
        if shape is None:
            return call_model(*args, **kwargs)
        started = time.monotonic_ns()
        result = call_model(*args, **kwargs)
        model_ns = (model_ns or 0) + time.monotonic_ns() - started
        return result

    def timed_execute(scheduler_output, *args, **kwargs):
        nonlocal current_batch, shape, model_ns, attn_ns, graph_ns, graph_manager
        shape = model_ns = attn_ns = graph_ns = None
        if scheduler_output is None or kwargs.get("dummy_run"):
            current_batch = ("unknown", 0)
            return execute(scheduler_output, *args, **kwargs)
        current_batch = probe.classify(scheduler_output)
        if current_batch[0] == "decode" and current_batch[1] >= 8 and graph_manager is None:
            graph_manager = getattr(runner, "cudagraph_manager", None)
            if graph_manager is not None:
                for name in ("run_pw_graph", "run_fullgraph"):
                    if (call := getattr(graph_manager, name, None)) is not None:
                        setattr(graph_manager, name, time_graph_call(call))
        started = time.monotonic_ns()
        result = execute(scheduler_output, *args, **kwargs)
        ended = time.monotonic_ns()
        probe.record("execute_model", current_batch, started, ended)
        if shape is not None and len(probe.decode_graphs) < MAX_SAMPLES:
            if graph_ns is not None:
                model_path, forward_ns = "graph_manager", graph_ns
            else:
                model_path, forward_ns = "_call_model" if model_ns is not None else "unmeasured", model_ns
            probe.decode_graphs.append(
                (started, current_batch[1], *shape, attn_ns, model_path, forward_ns, ended - started)
            )
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
    runner.prepare_inputs = timed_prepare
    runner.prepare_attn = timed_prepare_attn
    runner._call_model = timed_call_model
