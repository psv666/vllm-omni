# Qwen3-Omni fixed-workload performance test

The `random` sweep in `tests/test_qwen3_omni_no_async_chunk.json` measures
2500 configured input tokens, 900 Thinker output tokens, and exactly 1536 Talker
output tokens. The actual chat-templated input token count can be larger.
Concurrency/request-count pairs remain `(1, 4)`, `(4, 16)`, `(8, 32)`, `(16, 64)`,
and `(32, 128)`. Each point uses two warmups and benchmark seed zero.
The same configuration is used by CUDA and NPU nightly jobs.

## What is fixed

The request supplies the first two entries of `sampling_params_list` explicitly.
They are complete sampling parameter objects, not partial overlays of deploy
YAML defaults. Thinker retains greedy sampling and `ignore_eos`; Talker retains
temperature 0.9, top-k 50 and repetition penalty 1.05, with request seed zero and
`min_tokens == max_tokens == 1536`. The omitted Code2Wav entry is copied from the
server defaults.

`min_tokens` suppresses the model's EOS and required stop tokens until the fixed
workload is complete. `max_tokens` alone is only an upper bound, and
`ignore_eos` alone does not disable explicit stop tokens. Model stop-token
constraints remain unchanged. The fixed workload can truncate or extend speech;
it is a performance measurement, not an audio completeness or quality test.
Natural termination and quality coverage remain in the existing model tests.

Benchmark results retain `request_stage_metrics`, one snapshot per formal
request in input order, including missing snapshots. Warmups are excluded.
The CI runner rejects failed requests, missing snapshots and any fixed stage
whose `num_tokens_out` differs from the requested length. These snapshots also
record audio frames/duration and stage timings without embedding audio data.
The field is retained even without `--save-detailed` so the runner can validate
normal nightly artifacts.

Fixed Talker token counts do not guarantee identical waveform sample counts.
Record the distribution of `audio_frames` and `audio_duration` as well: Code2Wav
batching and tail cropping can affect the number of returned samples. Investigate
duration changes separately instead of treating a passing token check as proof
that audio length or quality is unchanged.

## Baseline migration: pending H100 calibration

The five H100 values currently in the JSON are the historical **variable-length**
baselines. They are deliberately unchanged until fixed-workload H100 measurements
are available. **Do not merge this workload change with those old baselines.**
Do not replace them with local L20X results or the approximately 0.21 RTF measured
for the old variable-length workload.

Before updating the baseline:

1. Pin and record the model snapshot, container digest, installed dependencies,
   driver, GPU model and both source revisions. On the same H100 environment,
   compare commit `e3be42e052` and the candidate with this same fixed workload at
   concurrency 1, using three independent server starts per revision. Stop and
   investigate if the candidate's median RTF regresses by more than 10%.
2. Run the candidate's full five-point sweep three times with independent server
   starts, preserving two warmups per point. Check every Thinker length (900),
   Talker length (1536), length finish reason, nonempty audio, and audio frame-count
   stability. Record the frame-count distribution and investigate differences.
   Check request success and fixed lengths in NPU CI too.
3. For each concurrency, require `(max RTF - min RTF) / median RTF <= 0.05` across
   the three runs. Otherwise investigate variability before calibration.
4. Replace each H100 baseline with the median of the three corresponding
   `mean_audio_rtf` measurements, keeping the existing alert threshold. Attach
   all measurements and environment metadata to the PR. Mark the workload change
   explicitly; old and new baseline numbers are not directly comparable.

The existing CUDA/NPU nightly commands already use this JSON. For calibration,
use a copy containing only the first `benchmark_params` entry to avoid unrelated
multimodal runs; for the old/new comparison also narrow its concurrency and
request-count arrays to `[1]` and `[4]`. Preserve the sampling parameters above
for both revisions. The old runner does not enforce the new length check, so
verify its per-stage output from the response metrics before accepting a run.

Current main targets vLLM 0.30.0, whereas historical commit `e3be42e052` targets
0.29.0. Check revision/runtime compatibility before the historical comparison;
a run that fails initialization is not a performance measurement. If one runtime
cannot run both revisions, report the dependency difference explicitly. Use
unmodified current main plus the identical fixed-length request configuration as
an additional control for this PR on the 0.30.0 environment; do not attribute a
cross-runtime performance change to this benchmark-only patch.
