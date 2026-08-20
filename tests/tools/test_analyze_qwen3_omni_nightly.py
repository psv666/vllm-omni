# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "nightly" / "analyze_qwen3_omni_nightly.py"
SPEC = importlib.util.spec_from_file_location("analyze_qwen3_omni_nightly", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_nightly_jobs_is_qwen_only() -> None:
    jobs = analysis.load_nightly_jobs(Path(".buildkite/cuda/test-nightly.yml"))

    assert [job.category for job in jobs].count("function") == 1
    assert [job.category for job in jobs].count("accuracy") == 1
    assert [job.category for job in jobs].count("performance") == 4
    assert [job.category for job in jobs].count("multi-replica-function") == 1
    assert all(job.gpu_count in {2, 3, 4} for job in jobs)
    assert not any("MiniCPM" in job.label or "Doc Test" in job.label for job in jobs)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        ({"branch": "main", "source": "schedule", "env": {"NIGHTLY": "1"}}, True),
        ({"branch": "main", "source": "ui", "env": {"NIGHTLY": "1"}}, False),
        ({"branch": "feature", "source": "schedule", "env": {"NIGHTLY": "1"}}, False),
        ({"branch": "main", "source": "schedule", "env": {}, "pull_request": None}, False),
        ({"branch": "main", "source": "schedule", "env": {"NIGHTLY": "1", "WEEKLY": "1"}}, False),
        (
            {
                "branch": "main",
                "source": "schedule",
                "env": {"NIGHTLY": "1"},
                "pull_request": {"id": 1},
            },
            False,
        ),
    ],
)
def test_is_scheduled_nightly_is_strict(build: dict, expected: bool) -> None:
    assert analysis.is_scheduled_nightly(build, branch="main") is expected


def test_summarize_jobs_counts_retry_cost_and_flake() -> None:
    job = analysis.NightlyJob("job", "job-key", "function", 2, 60, "pytest tests/demo.py")
    attempts = [
        analysis.Attempt(
            1,
            "deadbeef",
            "job-key",
            "a",
            "failed",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:10:00Z",
            600,
            True,
            0,
            gpu_count=2,
        ),
        analysis.Attempt(
            1,
            "deadbeef",
            "job-key",
            "b",
            "passed",
            "2026-01-01T00:11:00Z",
            "2026-01-01T00:21:00Z",
            600,
            False,
            1,
            gpu_count=2,
        ),
        analysis.Attempt(
            2,
            "deadbeef",
            "job-key",
            "c",
            "passed",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:20:00Z",
            1200,
            False,
            0,
            gpu_count=2,
        ),
    ]

    summary = analysis.summarize_jobs(attempts, [job])["job-key"]

    assert summary.executions == 2
    assert summary.successes == 2
    assert summary.retries == 1
    assert summary.flakes == 1
    assert summary.p50_seconds == 900
    assert summary.p95_seconds == 1200
    assert summary.gpu_hours == pytest.approx(4 / 3)
    assert summary.retry_gpu_hours == pytest.approx(1 / 3)


def test_summarize_cases_tracks_independent_failures() -> None:
    case_a = analysis.StaticCase("tests/a.py::test_a", "job", "job", "function", 2, {}, set(), 1, "s", "tests/a.py")
    case_b = analysis.StaticCase("tests/a.py::test_b", "job", "job", "function", 2, {}, set(), 1, "s", "tests/a.py")
    attempts = [
        analysis.Attempt(
            10,
            "deadbeef",
            "job",
            "a",
            "failed",
            None,
            None,
            None,
            False,
            0,
            log="tests/a.py::test_a FAILED\ntests/a.py::test_b PASSED\n",
        ),
        analysis.Attempt(
            11,
            "deadbeef",
            "job",
            "b",
            "failed",
            None,
            None,
            None,
            False,
            0,
            log="tests/a.py::test_a FAILED\ntests/a.py::test_b FAILED\n",
        ),
    ]

    summaries = analysis.summarize_cases(attempts, [case_a, case_b])

    assert summaries[case_a.case_id].executions == 2
    assert summaries[case_a.case_id].independent_failures == 1
    assert summaries[case_b.case_id].independent_failures == 0


def test_case_results_normalizes_ansi_and_carriage_returns() -> None:
    log = "\x1b[32mtests/a.py::test_a PASSED\x1b[0m\rtests/a.py::test_b FAILED\r\n"

    assert analysis._case_results(log) == [
        ("tests/a.py::test_a", "PASSED"),
        ("tests/a.py::test_b", "FAILED"),
    ]


def test_cli_defaults_to_twenty_builds() -> None:
    assert analysis.parse_args([]).max_builds == 20


def test_public_build_classification_requires_exact_nightly_evidence() -> None:
    build = {
        "branch_name": "main",
        "source": "schedule",
        "message": "Scheduled nightly build",
        "pull_request": None,
    }

    assert analysis.classify_public_build(build, branch="main") == "public_scheduled_nightly"
    assert analysis.classify_public_build({**build, "message": "Scheduled weekly build"}, branch="main") == (
        "schedule_unresolved"
    )
    assert analysis.classify_public_build({**build, "source": "ui"}, branch="main") == "excluded"


def test_public_build_list_search_paginates_and_deduplicates() -> None:
    class FakeClient:
        pipeline_path = "/vllm/vllm-omni"

        def __init__(self) -> None:
            self.calls = []

        def get_text(self, path, params=None):
            self.calls.append((path, params))
            if params["page"] == 1:
                return (
                    '<a href="/vllm/vllm-omni/builds/3">nightly</a>'
                    '<a href="/vllm/vllm-omni/builds/2">nightly</a>'
                    '<a rel="next">next</a>'
                )
            return '<a href="/vllm/vllm-omni/builds/2">nightly</a><a href="/vllm/vllm-omni/builds/1">nightly</a>'

    client = FakeClient()
    numbers, pages = analysis._public_nightly_build_numbers(client, branch="main", max_builds=3)

    assert numbers == [3, 2, 1]
    assert pages == 2
    assert client.calls[0][1]["query"] == "Scheduled nightly build"


def test_public_job_state_uses_outcome_fields() -> None:
    assert analysis._public_job_state({"state": "finished", "passed": True, "exit_status": 0}) == "passed"
    assert analysis._public_job_state({"state": "finished", "passed": False, "exit_status": 1}) == "failed"
    assert analysis._public_job_state({"state": "finished", "timed_out_at": "now"}) == "timed_out"
    assert analysis._public_job_state({"state": "finished", "canceled_at": "now"}) == "canceled"


def test_case_results_infers_success_from_public_log_nodes() -> None:
    log = "\x1b_bk;t=123\x07tests/a.py::test_a output from test\r\ntests/a.py::test_b output from test\n"

    assert analysis._case_results(log, fallback_status="PASSED") == [
        ("tests/a.py::test_a", "PASSED"),
        ("tests/a.py::test_b", "PASSED"),
    ]


def test_safe_history_cache_drops_logs_and_signed_urls() -> None:
    history = {
        "schema_version": 1,
        "builds": [
            {
                "number": 1,
                "jobs": [
                    {
                        "job_key": "x",
                        "log": "secret-ish output",
                        "artifacts": [{"filename": "result.json", "download_url": "https://signed.example"}],
                    }
                ],
            }
        ],
    }

    safe = analysis._safe_history_cache(history)

    job = safe["builds"][0]["jobs"][0]
    assert "log" not in job
    assert "download_url" not in job["artifacts"][0]


def test_history_loader_requires_builds(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"jobs": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="builds list"):
        analysis.load_history(invalid)


def test_schedule_without_nightly_env_is_unresolved() -> None:
    build = {"branch": "main", "source": "schedule", "env": {}, "pull_request": None}

    assert analysis.classify_build(build, branch="main") == "schedule_unresolved"


def test_jobs_endpoint_follows_cursor_and_requests_retries() -> None:
    class FakeClient:
        pipeline_path = "/organizations/o/pipelines/p"

        def __init__(self) -> None:
            self.calls = []

        def get_json(self, path, params=None):
            self.calls.append((path, params))
            if len(self.calls) == 1:
                return {"items": [{"id": "a"}], "links": {"next": "https://api.buildkite.com/v2/next"}}, {}
            return {"items": [{"id": "b"}], "links": {"next": None}}, {}

    client = FakeClient()

    jobs = analysis._list_job_attempts(client, 123)

    assert [job["id"] for job in jobs] == ["a", "b"]
    assert client.calls[0][1]["include_retried_jobs"] == "true"
    assert client.calls[0][1]["group_key"] == "nightly-omni-test-group"
    assert client.calls[1] == ("https://api.buildkite.com/v2/next", None)


def test_coverage_distinguishes_text_only_from_audio_input() -> None:
    job = analysis.NightlyJob("job", "job", "function", 2, 120, "pytest tests/e2e")
    text_node = "tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_to_text_audio_001[default]"
    audio_node = "tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_audio_to_text_audio_002[default]"

    text_dimensions, text_capabilities = analysis._coverage_for_node(Path.cwd(), text_node, job)
    audio_dimensions, audio_capabilities = analysis._coverage_for_node(Path.cwd(), audio_node, job)

    assert "audio-input" not in text_capabilities
    assert "audio-input" in audio_capabilities
    assert text_dimensions != audio_dimensions
