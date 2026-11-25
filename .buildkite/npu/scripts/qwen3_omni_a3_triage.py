# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run or summarize the two-pass Qwen3-Omni A3 experiment from a source checkout."""

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "f602d0295852c430434f8cd2cd87da287db1a96e"
# The first field run used this revision. Subsequent changes touch only diagnostics.
FIRST_PASS_CHECKOUT = "2f0207881664c61d4b0d2ecaf262e49c46ebf77a"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REV = "26291f793822fb6be9555850f06dfe95f2d7e695"
CI_IMAGE = "sha256:fdcede880eeba2b93d0dc3fc8532d0cb88dd1c2586f5f61cff7e923b7f55cd69"
SOURCE_CONFIG = ROOT / "tests/dfx/perf/tests/test_qwen3_omni_no_async_chunk.json"
TRACE_ENV = "VLLM_OMNI_A3_THINKER_TRACE_DIR"
DEVICE_BATCH_ENV = "VLLM_OMNI_A3_DEVICE_PROFILE_BATCH"
COUNTS = {1: 4, 4: 16, 8: 32, 16: 64}
THINKER_GRAPH_SIZES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)
LIMIT = 4096
ANSI = re.compile(r"\x1b_bk;t=\d+\x07|\x1b\[[0-?]*[ -/]*[@-~]")
MEASUREMENT = re.compile(r"^Maximum request concurrency:\s*(\d+)\s*$")
TIMING = re.compile(r"\[OmniTiming\] req=(\S+) .*?stages=\[0:([\d.]+)s,1:([\d.]+)s,2:([\d.]+)s\]")
RESULT = re.compile(r"_random_(\d+)_(\d+)_in2500_out900_")
METRICS = ("mean_tpot_ms", "median_tpot_ms", "median_itl_ms", "mean_e2el_ms")
RUNTIME_CHECK = """
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path

import torch_npu

npu = torch_npu.npu
devices = [{"name": npu.get_device_name(i)} for i in range(npu.device_count())]
versions = {name: md.version(name) for name in ("vllm", "vllm-ascend", "torch", "torch-npu")}
cann = {}
for path in sorted(Path("/usr/local/Ascend/ascend-toolkit/latest").glob("*/*install.info")):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("version="):
            cann[path.name] = line.split("=", 1)[1].strip()
print("A3_RUNTIME=" + json.dumps({
    "versions": versions, "devices": devices, "cann": cann,
    "omni": importlib.util.find_spec("vllm_omni").origin,
}))
"""


class TriageError(Exception):
    """A missing prerequisite or incomplete experiment."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def checkpoint(model_dir):
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache:
        hf_home = os.environ.get("HF_HOME")
        xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache = (Path(hf_home) if hf_home else xdg_cache / "huggingface") / "hub"
    snapshot = model_dir or Path(cache) / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct" / "snapshots" / MODEL_REV
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file() or not (snapshot / "config.json").is_file():
        raise TriageError(f"cached checkpoint missing for {MODEL} at {MODEL_REV[:12]}")
    shards = set(read_json(index)["weight_map"].values())
    if not shards or any(not (snapshot / shard).is_file() for shard in shards):
        raise TriageError("cached checkpoint has missing weight shards")
    if snapshot.name != MODEL_REV or snapshot.parent.name != "snapshots":
        download_meta = snapshot / ".cache/huggingface/download"
        for name in {"config.json", *shards}:
            info = download_meta / f"{name}.metadata"
            if not info.is_file() or info.read_text(encoding="utf-8").partition("\n")[0].strip() != MODEL_REV:
                raise TriageError(f"local checkpoint lacks download provenance for {MODEL_REV[:12]}")
    return snapshot.resolve()


def preflight(directory, env, base_image_digest, model_dir):
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=False).returncode:
        raise TriageError(f"checkout must descend from {BASE[:12]}")
    model = checkpoint(model_dir)
    completed = subprocess.run(
        [sys.executable, "-c", RUNTIME_CHECK],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (directory / "preflight.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    line = next((s for s in completed.stdout.splitlines() if s.startswith("A3_RUNTIME=")), None)
    if completed.returncode or line is None:
        for name in ("torch_npu", "vllm", "vllm-ascend", "torch"):
            if (
                f"No module named '{name}'" in completed.stderr
                or f"No package metadata was found for {name}" in completed.stderr
            ):
                raise TriageError(f"runtime dependency missing: {name}")
        raise TriageError("runtime or device inspection failed; see preflight.log locally")
    runtime = json.loads(line.split("=", 1)[1])
    if Path(runtime.pop("omni")).resolve() != ROOT / "vllm_omni/__init__.py":
        raise TriageError("Python resolved Omni outside the diagnostic checkout")
    versions = runtime["versions"]
    if versions["vllm"].split("+", 1)[0] != "0.34.0":
        raise TriageError(f"expected vLLM 0.34.0, found {versions['vllm']}")
    if len(runtime["devices"]) < 3:
        raise TriageError("three logical A3 devices required for TP2 plus Talker/Code2Wav")
    runtime.update(
        checkout=git("rev-parse", "HEAD"),
        benchmark_base=BASE,
        model_rev=MODEL_REV,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        base_image_digest=base_image_digest,
        ci_reference_match=(base_image_digest == CI_IMAGE and versions["torch-npu"] == "2.10.0.post7"),
    )
    return runtime, model


def make_config(concurrencies, model, graph_sizes=None, device_batch=None):
    (config,) = read_json(SOURCE_CONFIG)
    (workload,) = (row for row in config["benchmark_params"] if row["dataset_name"] == "random")
    payload = workload["extra_body"]["sampling_params_list"]
    if (
        workload["random_input_len"] != 2500
        or workload["random_output_len"] != 900
        or any([p["min_tokens"], p["max_tokens"]] != [n, n] for p, n in zip(payload, (900, 1536)))
        or len(payload) != 2
        or not workload["extra_body"].get("return_stage_metrics")
        or not workload.get("ignore_eos")
    ):
        raise TriageError("pinned CI workload does not have the expected token contract")
    for concurrency in [8] if device_batch else concurrencies:
        index = workload["max_concurrency"].index(concurrency)
        if workload["num_prompts"][index] != COUNTS[concurrency]:
            raise TriageError(f"unexpected CI request count at c={concurrency}")
    workload["max_concurrency"] = concurrencies
    workload["num_prompts"] = [device_batch] if device_batch else [COUNTS[c] for c in concurrencies]
    workload["num_warmups"] = [2] if device_batch else [max(2, c) for c in concurrencies]
    workload.pop("baseline", None)
    config["benchmark_params"] = [workload]
    server = config["server_params"]
    server["model"] = str(model)
    server["extra_cli_args"] += [
        "--deploy-config",
        str(ROOT / "vllm_omni/deploy/qwen3_omni_moe.yaml"),
        "--stage-init-timeout",
        "900",
    ]
    if graph_sizes is not None:
        server["stage_overrides"] = {
            "0": {
                "compilation_config": {
                    "cudagraph_mode": "PIECEWISE",
                    "prefill_cudagraph_capture_token_sizes": list(graph_sizes),
                }
            },
        }
    return [config]


def log_metrics(path):
    rows = defaultdict(dict)
    active = None
    starting = False
    graph_roles = set()
    runner_roles = set()
    errors = []
    for line in ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).splitlines():
        line = line.strip()
        match = MEASUREMENT.fullmatch(line)
        if line == "Starting main benchmark run...":
            starting = True
        elif match and starting:
            active = int(match.group(1))
            starting = False
        elif line.startswith("Successful requests:") or "Serving Benchmark Result" in line:
            active = None
        elif active is not None and (timing := TIMING.search(line)):
            key, *values = timing.groups()
            rows[active][key] = tuple(map(float, values))
        if "Worker_TP" in line:
            role = line.split("Worker_TP", 1)[1].split()[0]
            if role in {"0", "1"}:
                if "ACL graph replay is active" in line:
                    graph_roles.add(role)
                if "NPU Model Runner V2 loaded" in line:
                    runner_roles.add(role)
        # Export fixed categories only. Container logs can contain internal paths and credentials.
        for indicator, category in (
            ("hcclGetRootInfo", "HCCL initialization"),
            ("OutOfMemoryError", "out of memory"),
            ("Warmup requests failed", "warmup failure"),
            ("warmup requests failed", "warmup failure"),
            ("Initial test run failed", "initial request failure"),
            ("timed out", "timeout"),
            ("ERROR collecting", "pytest collection"),
            ("RuntimeError:", "worker or runtime exception"),
            ("AssertionError", "benchmark assertion"),
            ("no tests ran", "no benchmark collected"),
        ):
            if indicator in line and category not in errors:
                errors.append(category)
    return rows, {"runner_v2_tp": sorted(runner_roles), "acl_graph_tp": sorted(graph_roles)}, errors


def performance(directory, log, concurrencies, expected_counts=COUNTS):
    timings, graph, errors = log_metrics(log)
    artifacts = defaultdict(list)
    for path in directory.glob("result_*.json"):
        if match := RESULT.search(path.name):
            artifacts[int(match.group(1))].append(path)
    cases = {}
    for c in concurrencies:
        count = expected_counts[c]
        paths = artifacts[c]
        if len(paths) != 1:
            cases[c] = {"error": f"expected one result artifact, found {len(paths)}"}
            continue
        result = read_json(paths[0])
        stages = result.get("request_stage_metrics", [])
        token_ok = len(stages) == count and all(
            stage.get("0", {}).get("num_tokens_out") == 900 and stage.get("1", {}).get("num_tokens_out") == 1536
            for stage in stages
        )
        values = timings[c].values()
        valid_metrics = all(isinstance(result.get(key), (int, float)) and math.isfinite(result[key]) for key in METRICS)
        case = {
            "completed": result.get("completed"),
            "failed": result.get("failed"),
            "timing_count": len(values),
            "tokens_900_1536": token_ok,
        }
        if valid_metrics:
            case.update({key: round(result[key], 2) for key in METRICS})
        if values:
            case["thinker_s"], case["talker_s"], case["code2wav_s"] = (
                round(statistics.median(column), 2) for column in zip(*values)
            )
        if (
            result.get("completed") != count
            or result.get("failed") != 0
            or result.get("num_tpot_samples") != count
            or len(values) != count
            or not token_ok
            or not valid_metrics
        ):
            case["error"] = "request, token, timing, or metric validation failed"
        cases[c] = case
    return cases, graph, errors


def verdict(cases):
    if any("error" in row for row in cases.values()) or not {8, 16} <= cases.keys():
        return "invalid"
    c8, c16 = cases[8], cases[16]
    if c8["median_tpot_ms"] < 40 and c16["median_tpot_ms"] >= 100 and c16["thinker_s"] >= 90:
        return "reproduced"
    if c8["median_tpot_ms"] < 40 and c16["median_tpot_ms"] < 75 and c16["thinker_s"] < 75:
        return "near_normal"
    return "inconclusive"


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def trace_bounds(trace):
    windows_path = trace / "windows.jsonl"
    bounds = defaultdict(dict)
    if not windows_path.is_file():
        return bounds
    for line in windows_path.read_text(encoding="utf-8").splitlines():
        phase, boundary, concurrency, timestamp = json.loads(line)
        bounds[(phase, concurrency)][boundary] = timestamp
    return bounds


def summarize_trace(trace):
    bounds = trace_bounds(trace)
    if not bounds:
        return {}, ["missing benchmark windows"]
    groups = defaultdict(list)
    errors = []
    owners = set()
    for path in sorted(trace.glob("*.json")):
        match = re.fullmatch(r"(engine|tp\d+)-\d+\.json", path.name)
        if match is None:
            continue
        owner = match.group(1)
        if owner in owners:
            errors.append(f"duplicate trace process {owner}")
        owners.add(owner)
        data = read_json(path)
        if data["dropped"]:
            errors.append(f"{owner} buffer dropped {data['dropped']} records")
        for kind, batch, metric, started, duration in data["rows"]:
            for (phase, concurrency), interval in bounds.items():
                if interval.get("start", math.inf) <= started < interval.get("end", -1):
                    groups[(phase, concurrency, kind, batch, owner, metric)].append(duration)
                    break
    for owner in sorted({"engine", "tp0", "tp1"} - owners):
        errors.append(f"missing trace from {owner}")
    for c in (8, 16):
        present = {
            (owner, metric)
            for phase, concurrency, kind, _, owner, metric in groups
            if phase == "measured" and concurrency == c and kind == "decode"
        }
        required = {("engine", "schedule"), ("engine", "dispatch_to_consume")}
        required.update((tp, name) for tp in ("tp0", "tp1") for name in ("execute_model", "sample_tokens"))
        if required - present:
            errors.append(
                f"c={c} missing measured decode timings for {','.join('/'.join(x) for x in sorted(required - present))}"
            )
    summary = {}
    for key, samples in sorted(groups.items()):
        summary["/".join(map(str, key))] = [
            round(statistics.median(samples) / 1e6, 4),
            round(percentile95(samples) / 1e6, 4),
            len(samples),
        ]
    return summary, errors


def summarize_graphs(trace, concurrency=16, batches=(8, 9, 16)):
    interval = trace_bounds(trace).get(("measured", concurrency), {})
    if "start" not in interval or "end" not in interval:
        return {}, [f"missing c={concurrency} measurement window"]
    groups = defaultdict(list)
    for path in sorted(trace.glob("tp*-*.json")):
        owner = path.stem.split("-", 1)[0]
        for row in read_json(path).get("decode_graphs", []):
            started, batch, tokens, mode, graph_tokens, graph_reqs, prep, attn, model_path, model, total = row
            if batch not in batches or not interval["start"] <= started < interval["end"]:
                continue
            key = f"{batch}/{owner}/{tokens}/{mode}/{graph_tokens}/{graph_reqs}/{model_path}"
            groups[key].append((prep, attn, model, total))

    def p50_ms(values):
        samples = [value for value in values if value is not None]
        return round(statistics.median(samples) / 1e6, 3) if samples else None

    report = {}
    omitted = 0
    groups_per_rank_batch = defaultdict(int)
    for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        rank_batch = key.rsplit("/", 5)[0]
        if groups_per_rank_batch[rank_batch] == 2:
            omitted += 1
            continue
        groups_per_rank_batch[rank_batch] += 1
        prep, attn, model, total = zip(*rows)
        forward = [value for value in model if value is not None]
        report[key] = {
            "n": len(rows),
            "prepare_inputs_p50_ms": p50_ms(prep),
            "prepare_attn_p50_ms": p50_ms(attn),
            "forward_p50_ms": p50_ms(forward),
            "forward_p95_ms": round(percentile95(forward) / 1e6, 3) if forward else None,
            "execute_p50_ms": p50_ms(total),
            "execute_p95_ms": round(percentile95(total) / 1e6, 3),
        }
    errors = [
        f"missing graph descriptor for batch {batch}/{rank}"
        for batch in batches
        for rank in ("tp0", "tp1")
        if not any(key.startswith(f"{batch}/{rank}/") for key in report)
    ]
    if omitted:
        errors.append(f"{omitted} lower-frequency descriptor groups kept in local traces")
    return report, errors


def compact_timings(timings):
    compact = {}
    for c in (8, 16):
        for kind in ("decode", "prefill", "mixed", "unknown"):
            prefix = f"measured/{c}/{kind}/"
            selected = [(key, values) for key, values in timings.items() if key.startswith(prefix)]
            if not selected:
                continue
            # In an incomplete run the worker traces can still identify useful batch sizes.
            counts = defaultdict(int)
            for key, values in selected:
                batch = int(key.split("/")[3])
                counts[batch] = max(counts[batch], values[-1])
            batch = c if kind == "decode" and c in counts else max(counts, key=counts.get)
            batches = {batch}
            if kind == "decode" and c == 16 and 8 in counts:
                batches.add(8)
            for key, values in selected:
                if int(key.split("/")[3]) in batches:
                    compact[key.removeprefix("measured/")] = values
    return compact


def run_cases(directory, env, model, concurrencies, probe, graph_sizes=None, device_batch=None):
    directory.mkdir()
    config = directory / "config.json"
    log = directory / "run.log"
    write_json(config, make_config(concurrencies, model, graph_sizes, device_batch))
    process_env = {**env, "BENCHMARK_DIR": str(directory)}
    if device_batch:
        process_env[DEVICE_BATCH_ENV] = str(device_batch)
    if probe:
        trace = directory / "trace"
        trace.mkdir()
        process_env[TRACE_ENV] = str(trace)
    print(f"Running c={concurrencies}. Full logs stay in {directory}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-s",
                "-v",
                "-m",
                "npu",
                "tests/dfx/perf/scripts/run_benchmark.py",
                "--test-config-file",
                str(config),
            ],
            cwd=ROOT,
            env=process_env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    cases, graph, errors = performance(
        directory, log, concurrencies, {device_batch: device_batch} if device_batch else COUNTS
    )
    result = {
        "concurrencies": concurrencies,
        "verdict": verdict(cases) if rc == 0 else "invalid",
        "pytest_exit": rc,
        "cases": cases,
        **graph,
        "errors": errors,
    }
    if graph_sizes is not None:
        result["verdict"] = "valid" if rc == 0 and not any("error" in case for case in cases.values()) else "invalid"
        target = device_batch or 8
        result["graphs"], result["probe_gaps"] = summarize_graphs(trace, concurrency=target, batches=(target,))
    elif probe:
        timings, gaps = summarize_trace(trace)
        write_json(trace / "timing-summary.json", timings)
        result["probe_gaps"] = gaps
        result["timing_ms_p50_p95_n"] = compact_timings(timings)
    return result


def run_graph_size_ab(directory, env, model):
    arms = {}
    for label, graph_tokens in (("A", 8), ("B", 16)):
        sizes = THINKER_GRAPH_SIZES if label == "A" else tuple(size for size in THINKER_GRAPH_SIZES if size != 8)
        result = run_cases(directory / label, env, model, [8], probe=True, graph_sizes=sizes)
        graphs = result["graphs"]
        gaps = list(result["probe_gaps"])
        expected = {f"8/tp{rank}/8/PIECEWISE/{graph_tokens}/None/graph_manager" for rank in range(2)}
        wrong = set(graphs) - expected
        if wrong or expected - set(graphs):
            gaps.append(f"expected two batch-8 rank groups dispatched to the PIECEWISE {graph_tokens}-token graph")
        if any(graphs[key]["n"] < 32 for key in expected & graphs.keys()):
            gaps.append("fewer than 32 measured decode steps on an expected rank")
        arms[label] = {
            "expected_graph_tokens": graph_tokens,
            "case": result["cases"][8],
            "pytest_exit": result["pytest_exit"],
            "observed_graphs": graphs,
            "errors": result["errors"],
            "probe_gaps": gaps,
        }
        write_json(directory / "ab-results.json", arms)
        if result["verdict"] != "valid" or gaps:
            return {"verdict": "invalid", "failed_arm": label, "arms": arms}
    return {"verdict": "valid", "arms": arms}


def span_us(intervals):
    intervals.sort()
    begin, end = intervals[0]
    busy = 0.0
    for left, right in intervals[1:]:
        if left > end:
            busy += end - begin
            begin, end = left, right
        else:
            end = max(end, right)
    return busy + end - begin, end - intervals[0][0]


def device_profile_stats(directory, batch):
    capture = directory / "capture.json"
    if not capture.is_file():
        return {"error": "profiler did not flush a capture"}
    metadata = read_json(capture)
    steps = metadata["steps"]
    if metadata["batch"] != batch or metadata["graph_tokens"] != 16 or steps != 4:
        return {"error": "expected four contiguous matching graph calls", "steps": steps}
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        return {"error": "expected one kernel_details.csv", "files": len(paths)}
    with paths[0].open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = {re.sub(r"[^a-z]", "", name.lower()): name for name in reader.fieldnames or []}
        start_key = columns.get("starttimeus") or columns.get("taskstarttimeus")
        duration_key = columns.get("durationus") or columns.get("taskdurationus")
        name_key = columns.get("type") or columns.get("optype") or columns.get("name")
        core_key = columns.get("acceleratorcore")
        if not all((start_key, duration_key, name_key)):
            return {"error": "unexpected CANN kernel CSV columns", "columns": list(columns)[:12]}
        intervals, collective_intervals = [], []
        operators = defaultdict(float)
        for row in reader:
            try:
                start, duration = float(row[start_key].strip("\t")), float(row[duration_key].strip("\t"))
            except (AttributeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                continue
            name = row[name_key] or "unknown"
            identity = (name + " " + (row.get(core_key) or "")).lower()
            intervals.append((start, start + duration))
            if any(
                token in identity
                for token in ("hccl", "hcom", "allreduce", "all_reduce", "allgather", "reduce_scatter")
            ):
                collective_intervals.append(intervals[-1])
            operators[re.sub(r"[^A-Za-z0-9_.:+/-]", "_", name)[:64]] += duration
    if not intervals:
        return {"error": "empty device kernel samples"}
    busy, span = span_us(intervals)
    collective_busy = span_us(collective_intervals)[0] if collective_intervals else 0.0
    return {
        "steps": steps,
        "kernel_count": len(intervals),
        "capture_host_forward_ms": round(sum(metadata["host_forward_ns"]) / 1e6, 2),
        "capture_span_ms": round(span / 1000, 2),
        "capture_device_busy_ms": round(busy / 1000, 2),
        "capture_recognized_collective_busy_ms": round(collective_busy / 1000, 2),
        "top_ops_summed_ms": [
            [name, round(total / 1000, 2)] for name, total in sorted(operators.items(), key=lambda item: -item[1])[:4]
        ],
    }


def run_device_compare(directory, env, model):
    arms = {}
    sizes = tuple(size for size in THINKER_GRAPH_SIZES if size != 8)
    for batch in (8, 9):
        arm = directory / f"batch-{batch}"
        result = run_cases(arm, env, model, [batch], probe=True, graph_sizes=sizes, device_batch=batch)
        graphs = result["graphs"]
        graph_p50 = {}
        expected_graphs = set()
        ranks = {}
        for rank in (0, 1):
            expected = f"{batch}/tp{rank}/{batch}/PIECEWISE/16/None/graph_manager"
            expected_graphs.add(expected)
            graph_p50[f"tp{rank}"] = graphs.get(expected, {}).get("forward_p50_ms")
            ranks[f"tp{rank}"] = device_profile_stats(arm / "trace" / f"device-tp{rank}", batch)
        if set(graphs) != expected_graphs:
            result["probe_gaps"].append(
                "measured graph dispatch differs from the expected fixed 16-token PIECEWISE graph"
            )
        arms[batch] = {
            "case": result["cases"][batch],
            "profile_gaps": result["probe_gaps"],
            "forward_p50_ms": graph_p50,
            "device": ranks,
            "errors": result["errors"],
        }
        write_json(directory / "device-results.json", arms)
        if (
            result["verdict"] != "valid"
            or result["probe_gaps"]
            or not all(graph_p50.values())
            or any("error" in row for row in ranks.values())
        ):
            return {"verdict": "invalid", "failed_batch": batch, "arms": arms}
    fast, slow = arms[8]["forward_p50_ms"], arms[9]["forward_p50_ms"]
    representative = all(slow[rank] >= 100 and slow[rank] >= 2 * fast[rank] for rank in fast)
    return {"verdict": "valid", "slow_step_reproduced": representative, "arms": arms}


def emit(summary):
    compact = json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    if len(compact.encode()) + len("A3_SUMMARY_BEGIN\n\nA3_SUMMARY_END\n") > LIMIT:
        run = summary.get("run", {})
        timings = run.get("timing_ms_p50_p95_n", {})
        run["timing_ms_p50_p95_n"] = {key: value for key, value in timings.items() if "/decode/" in key}
    compact = json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    if len(compact.encode()) + len("A3_SUMMARY_BEGIN\n\nA3_SUMMARY_END\n") > LIMIT:
        compact = json.dumps({"error": "summary exceeds 4 KiB; inspect local summary.json"})
    print(f"A3_SUMMARY_BEGIN\n{compact}\nA3_SUMMARY_END", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "check",
            "reproduce",
            "profile",
            "replay",
            "summarize-trace",
            "summarize-graphs",
            "graph-size-ab",
            "device-compare",
        ),
    )
    parser.add_argument("--output", type=Path, help="new directory for local logs and results")
    parser.add_argument("--first-pass", type=Path, help="first-pass summary.json required for profile")
    parser.add_argument("--base-image-digest", default=os.environ.get("A3_BASE_IMAGE_DIGEST"))
    parser.add_argument("--model-dir", type=Path, help="local verified model snapshot instead of the HF cache")
    parser.add_argument("--log", type=Path, help="existing complete job log for replay")
    parser.add_argument("--artifacts", type=Path, help="directory of existing result JSON for replay")
    parser.add_argument("--trace-dir", type=Path, help="local instrumented/trace directory from a completed profile")
    args = parser.parse_args()
    if args.phase in ("summarize-trace", "summarize-graphs") and args.trace_dir is None:
        parser.error(f"{args.phase} requires --trace-dir")
    if args.phase == "summarize-graphs":
        graphs, gaps = summarize_graphs(args.trace_dir)
        emit(
            {
                "phase": args.phase,
                "key": "batch/rank/scheduled_tokens/mode/graph_tokens/graph_reqs/forward_hook",
                "graphs": graphs,
                "probe_gaps": gaps,
            }
        )
        return 0 if graphs else 1
    if args.phase == "summarize-trace":
        timings, gaps = summarize_trace(args.trace_dir)
        compact = compact_timings(timings)
        emit({"phase": args.phase, "run": {"probe_gaps": gaps, "timing_ms_p50_p95_n": compact}})
        return 0 if compact else 1
    if args.phase == "replay":
        if args.log is None or args.artifacts is None:
            parser.error("replay requires --log and --artifacts")
        cases, graph, errors = performance(args.artifacts, args.log, [8, 16])
        emit({"phase": args.phase, "run": {"verdict": verdict(cases), "cases": cases, **graph, "errors": errors}})
        return 0 if verdict(cases) != "invalid" else 1
    if args.output is None:
        parser.error("benchmark phases require --output")
    if args.phase == "profile" and args.first_pass is None:
        parser.error("profile requires --first-pass")
    if args.base_image_digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", args.base_image_digest):
        parser.error("--base-image-digest must be sha256 followed by 64 lowercase hex digits")
    directory = args.output.resolve()
    output_created = False
    summary = {"phase": args.phase}
    try:
        directory.mkdir(parents=True)
        output_created = True
        env = os.environ.copy()
        env.pop(TRACE_ENV, None)
        env.pop(DEVICE_BATCH_ENV, None)
        env.update(
            PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH")))),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
            PYTHONUNBUFFERED="1",
        )
        if args.phase == "profile":
            first = read_json(args.first_pass)
            if first.get("phase") != "reproduce" or first.get("run", {}).get("verdict") != "reproduced":
                raise TriageError("profile requires a successful reproduction summary")
        runtime, model = preflight(directory, env, args.base_image_digest, args.model_dir)
        summary["runtime"] = runtime
        if args.phase == "check":
            run = {"verdict": "ready"}
        elif args.phase == "graph-size-ab":
            run = run_graph_size_ab(directory, env, model)
        elif args.phase == "device-compare":
            run = run_device_compare(directory, env, model)
        elif args.phase == "reproduce":
            run = run_cases(directory / "initial", env, model, [8, 16], probe=False)
            if run["verdict"] == "near_normal" and runtime["ci_reference_match"]:
                summary["initial_verdict"] = run["verdict"]
                run = run_cases(directory / "ci-order", env, model, [1, 4, 8, 16], probe=False)
        else:
            required_runtime = dict(first["runtime"])
            if required_runtime["checkout"] == FIRST_PASS_CHECKOUT:
                required_runtime["checkout"] = runtime["checkout"]
            if runtime != required_runtime:
                raise TriageError("runtime differs from first pass; repeat reproduce in this container")
            if first["run"]["concurrencies"] not in ([8, 16], [1, 4, 8, 16]):
                raise TriageError("invalid first-pass benchmark sequence")
            run = run_cases(directory / "instrumented", env, model, first["run"]["concurrencies"], probe=True)
            if run["verdict"] != "reproduced":
                run["probe_gaps"].append("the regression was not reproduced with the probe enabled")
        summary["run"] = run
    except TriageError as exc:
        summary["error"] = str(exc)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        if output_created:
            (directory / "launcher-error.log").write_text(str(exc) + "\n", encoding="utf-8")
        summary["error"] = f"{type(exc).__name__} during setup or result processing"
    if output_created:
        write_json(directory / "summary.json", summary)
    emit(summary)
    return int("error" in summary or summary["run"]["verdict"] == "invalid" or bool(summary["run"].get("probe_gaps")))


if __name__ == "__main__":
    sys.exit(main())
