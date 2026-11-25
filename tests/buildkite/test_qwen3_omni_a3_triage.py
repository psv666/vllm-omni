# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


triage = load_module(ROOT / ".buildkite/npu/scripts/qwen3_omni_a3_triage.py")
probe = load_module(ROOT / "vllm_omni/utils/a3_thinker_probe.py")


def make_artifacts(directory, slow):
    log = directory / "run.log"
    lines = []
    for c, n in ((8, 32), (16, 64)):
        tp = 168 if slow and c == 16 else 55 if c == 16 else 27
        thinker = 151 if slow and c == 16 else 50 if c == 16 else 25
        result = {
            "completed": n,
            "failed": 0,
            "num_tpot_samples": n,
            "mean_tpot_ms": tp,
            "median_tpot_ms": tp,
            "median_itl_ms": tp,
            "mean_e2el_ms": (thinker + 100) * 1000,
            "request_stage_metrics": [{"0": {"num_tokens_out": 900}, "1": {"num_tokens_out": 1536}}] * n,
        }
        triage.write_json(directory / f"result_qwen_random_{c}_{n}_in2500_out900_t.json", result)
        lines.extend(("Starting main benchmark run...", f"Maximum request concurrency: {c}"))
        for i in range(n):
            lines.append(f"[OmniTiming] req=chatcmpl-{i} total=200s stages=[0:{thinker}.00s,1:99.02s,2:0.5s]")
        lines.extend(("============ Serving Benchmark Result ============", f"Maximum request concurrency: {c}"))
        lines.append("[OmniTiming] req=warmup-ignored total=1s stages=[0:1.0s,1:0.0s,2:0.0s]")
    log.write_text("\n".join(lines), encoding="utf-8")
    return log


@pytest.mark.parametrize(("slow", "expected"), [(False, "near_normal"), (True, "reproduced")])
def test_replay_excludes_warmup_and_checks_token_counts(tmp_path, slow, expected):
    log = make_artifacts(tmp_path, slow)
    cases, _, _ = triage.performance(tmp_path, log, [8, 16])
    assert triage.verdict(cases) == expected
    assert cases[16]["timing_count"] == 64
    assert cases[16]["talker_s"] == 99.02

    (path,) = tmp_path.glob("result_*_16_64_*")
    data = triage.read_json(path)
    data["request_stage_metrics"][0]["1"]["num_tokens_out"] = 1535
    triage.write_json(path, data)
    cases, _, _ = triage.performance(tmp_path, log, [8, 16])
    assert triage.verdict(cases) == "invalid"


def test_create_config_retains_ci_sweep_order(tmp_path):
    (config,) = triage.make_config([1, 4, 8, 16], tmp_path)
    (workload,) = config["benchmark_params"]
    assert workload["max_concurrency"] == [1, 4, 8, 16]
    assert workload["num_prompts"] == [4, 16, 32, 64]
    assert workload["num_warmups"] == [2, 4, 8, 16]
    assert config["server_params"]["model"] == str(tmp_path)
    assert "--no-async-chunk" in config["server_params"]["extra_cli_args"]


def test_checkpoint_requires_complete_shards_and_provenance(tmp_path):
    triage.write_json(tmp_path / "config.json", {})
    index = tmp_path / "model.safetensors.index.json"
    triage.write_json(index, {"weight_map": {"fake-weight": "model.safetensors"}})
    with pytest.raises(triage.TriageError, match="missing weight shards"):
        triage.checkpoint(tmp_path)
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(triage.TriageError, match="lacks download provenance"):
        triage.checkpoint(tmp_path)
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for name in ("model.safetensors", "config.json"):
        (metadata / f"{name}.metadata").write_text(triage.MODEL_REV + "\n", encoding="utf-8")
    assert triage.checkpoint(tmp_path) == tmp_path.resolve()


def make_output(new=(), cached=(), finished=()):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=req, prompt_token_ids=[1, 2], num_computed_tokens=0) for req in new],
        scheduled_cached_reqs=SimpleNamespace(req_ids=list(cached), num_computed_tokens=[2] * len(cached)),
        num_scheduled_tokens=dict.fromkeys((*new, *cached), 1),
        finished_req_ids=set(finished),
        total_num_scheduled_tokens=len(new) + len(cached),
    )


def test_probes_preserve_async_future_and_capture_per_phase(tmp_path, monkeypatch):
    flushers = []
    monkeypatch.setattr(probe, "Finalize", lambda _, flush, **kwargs: flushers.append(flush))
    outputs = iter((make_output(new=["a"]), make_output(new=["b"], cached=["a"]), make_output(cached=["a"])))
    execution = object()
    scheduler = SimpleNamespace(schedule=lambda: next(outputs), update_from_output=lambda _, value: value)
    core = SimpleNamespace(
        scheduler=scheduler, model_executor=SimpleNamespace(execute_model=lambda *args, **kwargs: execution)
    )
    runner = SimpleNamespace(
        execute_model=lambda *args, **kwargs: execution,
        sample_tokens=lambda: execution,
        prepare_inputs=lambda *args, **kwargs: None,
        prepare_attn=lambda *args, **kwargs: None,
        _call_model=lambda *args, **kwargs: None,
    )
    engine_probe = probe.attach_engine_probe(core, str(tmp_path))
    probe.attach_worker_probe(runner, str(tmp_path), 0)

    for _ in range(3):
        schedule = core.scheduler.schedule()
        assert core.model_executor.execute_model(schedule, non_block=True) is execution
        assert runner.execute_model(schedule) is execution
        assert runner.sample_tokens() is execution
        assert core.scheduler.update_from_output(schedule, execution) is execution
    engine_probe.flush()
    for flush in flushers:
        flush()
    engine = triage.read_json(next(tmp_path.glob("engine-*.json")))
    worker = triage.read_json(next(tmp_path.glob("tp0-*.json")))
    for trace in (engine, worker):
        assert [row[:2] for row in trace["rows"][::2]] == [["prefill", 1], ["mixed", 2], ["decode", 1]]
        assert trace["dropped"] == 0


def test_trace_reports_missing_roles_and_separates_warmup(tmp_path):
    (tmp_path / "windows.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                ["warmup", "start", 8, 10],
                ["warmup", "end", 8, 30],
                ["measured", "start", 8, 40],
                ["measured", "end", 8, 60],
            )
        ),
        encoding="utf-8",
    )
    triage.write_json(
        tmp_path / "engine-1.json",
        {
            "dropped": 0,
            "rows": [
                ["decode", 8, "schedule", 20, 1_000_000],
                ["decode", 8, "schedule", 50, 2_000_000],
            ],
        },
    )
    summary, errors = triage.summarize_trace(tmp_path)
    assert summary["warmup/8/decode/8/engine/schedule"] == [1.0, 1.0, 1]
    assert summary["measured/8/decode/8/engine/schedule"] == [2.0, 2.0, 1]
    assert "missing trace from tp1" in errors
    assert any(error.startswith("c=16 missing") for error in errors)


def test_summary_budget(capsys):
    triage.emit(
        {
            "phase": "profile",
            "run": {
                "timing_ms_p50_p95_n": {f"8/prefill/{i}/engine/schedule": [1, 1, 1] for i in range(600)}
                | {"8/decode/8/engine/schedule": [1, 2, 3]}
            },
        }
    )
    printed = capsys.readouterr().out
    assert len(printed.encode()) <= 4096
    assert json.loads(printed.splitlines()[1])["run"]["timing_ms_p50_p95_n"] == {
        "8/decode/8/engine/schedule": [1, 2, 3]
    }


def test_recover_worker_only_profile_from_an_existing_trace(tmp_path, monkeypatch, capsys):
    windows = [
        ["measured", "start", 8, 10],
        ["measured", "end", 8, 30],
        ["measured", "start", 16, 40],
        ["measured", "end", 16, 60],
    ]
    (tmp_path / "windows.jsonl").write_text("\n".join(map(json.dumps, windows)), encoding="utf-8")
    triage.write_json(tmp_path / "timing-summary.json", {"measured/8/decode/8/tp0/execute_model": [5, 10, 1]})
    for rank in range(2):
        rows = [
            ["decode", batch, operation, timestamp, duration]
            for batch, timestamp, duration in [(8, 20, 5_000_000), (8, 48, 8_000_000), (16, 50, 25_000_000)]
            for operation in ["execute_model", "sample_tokens"]
        ]
        triage.write_json(tmp_path / f"tp{rank}-{100 + rank}.json", {"rows": rows, "dropped": 0})
    monkeypatch.setattr("sys.argv", ["a3_triage", "summarize-trace", "--trace-dir", str(tmp_path)])
    assert triage.main() == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[1])["run"]
    assert "missing trace from engine" in summary["probe_gaps"]
    assert summary["timing_ms_p50_p95_n"]["8/decode/8/tp0/execute_model"] == [5, 5, 1]
    assert summary["timing_ms_p50_p95_n"]["16/decode/8/tp1/execute_model"] == [8, 8, 1]
    assert summary["timing_ms_p50_p95_n"]["16/decode/16/tp1/execute_model"] == [25, 25, 1]


def test_graph_probe_records_selected_shape_and_measured_phases(tmp_path, monkeypatch):
    flushers = []
    monkeypatch.setattr(probe, "Finalize", lambda _, flush, **kwargs: flushers.append(flush))
    clock = iter(range(10, 150, 10))
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: next(clock))
    descriptor = SimpleNamespace(cg_mode=SimpleNamespace(name="PIECEWISE"), num_tokens=16, num_reqs=None)
    runner = SimpleNamespace(
        sample_tokens=lambda: None,
        prepare_inputs=lambda output, state, desc: SimpleNamespace(num_tokens_after_padding=desc.num_tokens),
        prepare_attn=lambda inputs: None,
        cudagraph_manager=SimpleNamespace(run_pw_graph=lambda: None),
        _call_model=lambda: None,
    )

    def execute(output):
        inputs = runner.prepare_inputs(output, None, descriptor)
        runner.prepare_attn(inputs)
        runner.cudagraph_manager.run_pw_graph()
        return output

    runner.execute_model = execute
    probe.attach_worker_probe(runner, str(tmp_path), 0)
    ids = [str(i) for i in range(9)]
    runner.execute_model(make_output(new=ids))
    step = make_output(cached=ids)
    assert runner.execute_model(step) is step
    flushers[0]()
    (tmp_path / "windows.jsonl").write_text('["measured","start",16,25]\n["measured","end",16,100]\n', encoding="utf-8")
    result, gaps = triage.summarize_graphs(tmp_path)
    assert result["9/tp0/9/PIECEWISE/16/None/graph_manager"]["n"] == 1
    assert "missing graph descriptor for batch 9/tp1" in gaps
    data = triage.read_json(next(tmp_path.glob("tp0-*.json")))
    assert len(data["decode_graphs"]) == 1
    assert data["decode_graphs"][0][1:6] == [9, 9, "PIECEWISE", 16, None]
    assert data["decode_graphs"][0][6:] == [10, 10, "graph_manager", 10, 70]


def test_graph_size_ab_configs_only_change_thinker_bucket(tmp_path):
    sizes = triage.THINKER_GRAPH_SIZES
    (a,) = triage.make_config([8], tmp_path, sizes)
    (b,) = triage.make_config([8], tmp_path, tuple(size for size in sizes if size != 8))
    a_compile = a["server_params"]["stage_overrides"]["0"]["compilation_config"]
    b_compile = b["server_params"]["stage_overrides"]["0"]["compilation_config"]
    assert a_compile["prefill_cudagraph_capture_token_sizes"] == list(sizes)
    assert b_compile["prefill_cudagraph_capture_token_sizes"] == [size for size in sizes if size != 8]
    a_compile["prefill_cudagraph_capture_token_sizes"].remove(8)
    assert a == b


@pytest.mark.parametrize("wrong_bucket", [False, True])
def test_graph_size_ab_validates_actual_dispatch(tmp_path, monkeypatch, wrong_bucket):
    calls = []

    def run_cases(directory, env, model, concurrencies, probe, graph_sizes):
        assert concurrencies == [8] and probe
        calls.append(directory.name)
        graph_tokens = 8 if 8 in graph_sizes or wrong_bucket else 16
        return {
            "verdict": "valid",
            "pytest_exit": 0,
            "errors": [],
            "probe_gaps": [],
            "cases": {8: {}},
            "graphs": {f"8/tp{rank}/8/PIECEWISE/{graph_tokens}/None/graph_manager": {"n": 33} for rank in range(2)},
        }

    monkeypatch.setattr(triage, "run_cases", run_cases)
    result = triage.run_graph_size_ab(tmp_path, {}, tmp_path)
    assert calls == ["A", "B"]
    assert result["verdict"] == ("invalid" if wrong_bucket else "valid")
    assert triage.read_json(tmp_path / "ab-results.json") == result["arms"]


def test_device_profile_groups_overlapping_kernels(tmp_path):
    triage.write_json(
        tmp_path / "capture.json", {"steps": 4, "batch": 9, "graph_tokens": 16, "host_forward_ns": [200_000] * 4}
    )
    (tmp_path / "kernel_details.csv").write_text(
        "Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
        "attention,FusedInferAttentionScore,AI_CORE,100,140\n"
        "collective,HcomAllReduce,HCCL,170,100\n"
        "mm,MatMul,AI_CORE,320,30\n",
        encoding="utf-8",
    )
    report = triage.device_profile_stats(tmp_path, 9)
    assert report["capture_span_ms"] == 0.25
    assert report["capture_device_busy_ms"] == 0.20
    assert report["capture_recognized_collective_busy_ms"] == 0.10
    assert report["capture_host_forward_ms"] == 0.8
    assert report["top_ops_summed_ms"][0] == ["FusedInferAttentionScore", 0.14]
    assert "error" in triage.device_profile_stats(tmp_path, 8)


def test_device_capture_ignores_warmup_and_stops_at_four_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "Finalize", lambda *args, **kwargs: None)
    capture = probe.DeviceGraphCapture(str(tmp_path), 0, 9)

    def start():
        capture.output.mkdir()
        capture.profiler = SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(capture, "start", start)
    shape = (9, "PIECEWISE", 16, None, 0)
    batch = ("decode", 9)
    capture.call(lambda: None, (), {}, batch, shape)
    assert capture.matched == 0
    (tmp_path / "windows.jsonl").write_text('["measured","start",9,1]\n', encoding="utf-8")
    for step in range(15):
        assert capture.call(lambda: step, (), {}, batch, shape) == step
    meta = triage.read_json(capture.output / "capture.json")
    assert meta["steps"] == 4
    assert capture.profiler is None and capture.finished


def test_device_slow_keeps_original_workload(tmp_path):
    (original,) = triage.make_config([16], tmp_path)
    (instrumented,) = triage.make_config([16], tmp_path, device_batch=16)
    assert original == instrumented
    (workload,) = instrumented["benchmark_params"]
    assert workload["num_prompts"] == [64] and workload["num_warmups"] == [16]


def test_slow_capture_waits_for_two_slow_matching_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "Finalize", lambda *args, **kwargs: None)
    cap = probe.DeviceGraphCapture(str(tmp_path), 0, 16)
    (tmp_path / "windows.jsonl").write_text('["measured","start",16,1]\n', encoding="utf-8")
    elapsed = 0
    ticks = iter([25, 160, 25, 160, 170, 160, 160, 160, 160])
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: elapsed)

    def start():
        cap.output.mkdir()
        cap.profiler = SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(cap, "start", start)
    shape, batch = (16, "PIECEWISE", 16, None, 0), ("decode", 16)
    for _ in range(8):
        cap.call(lambda: None, (), {}, batch, shape)

    def step():
        nonlocal elapsed
        elapsed += next(ticks) * 1_000_000

    for _ in range(5):
        cap.call(step, (), {}, batch, shape)
        assert not cap.output.exists()
    for _ in range(4):
        cap.call(step, (), {}, batch, shape)
    result = triage.read_json(cap.output / "capture.json")
    assert result["pre_capture_slow_ns"] == [160_000_000, 170_000_000]
    assert result["steps"] == 4
