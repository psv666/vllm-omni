# Qwen3-Omni Seed-TTS Realtime turn-trigger comparison

The performance configuration at
[`tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json`](../../tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json)
compares the two ways a Realtime turn can be ended for the same Seed-TTS audio
and target text:

| Case | Input and trigger |
| --- | --- |
| `explicit` | WebSocket `/v1/realtime?duplex=1`; send text and paced speech PCM, then `input_audio_buffer.commit` and `response.create` at the end of the speech |
| `vad` | Same text and speech PCM, followed by the silent tail; server VAD ends the turn and creates the response, without a client commit or `response.create` |

Both run on `qwen3_omni_duplex.yaml` with async chunking, temperature 0 for text
generation, a 256-token output limit, the first four English Seed-TTS entries in
fixed order, and concurrency 1. Each entry is an independent session carrying one
utterance. The standard benchmark warmup runs separately from the four measured
requests.

`--seed-tts-reference-as-input` sends the dataset's reference speech as actual
user audio, rather than as `ref_audio` voice-cloning metadata. The instruction
asks the model to read the target text, not transcribe or answer the audio.
Reference audio is normalized once to mono 24 kHz PCM16 with a one-second silent
tail (`SEED_TTS_SILENT_TAIL_MS`); both cases send the identical speech samples in
chunks of at most 200 ms at real-time speed. Each chunk is sent at the end of its
capture interval, so the server cannot consume future audio or silence. A chunk
is split at the reference-speech boundary when needed, keeping that boundary
exact for clips whose length is not a multiple of 200 ms.

**Only VAD streams the silent tail.** The tail exists so server VAD can detect
the endpoint, and its 800 ms silence threshold must stay shorter than the tail or
the turn never ends. An explicit client ends the turn itself when the speaker
stops, so making it stream a second of silence first would charge it for latency
no real caller would pay — and would understate how much the VAD endpoint
actually costs. VAD may trim audio at its detected speech boundaries before model
execution. The public VAD contract requires `interrupt_response=true` and
`barge_in_on_speech`; these flags are accepted, but this workload never sends
overlapping speech, so it does not exercise or measure interruption.

This checks turn-trigger performance for one response per input. It does not
measure interruptions, overlapping speech, cross-turn context, or load scaling.
A VAD split before the reference audio ends is rejected rather than included as
a successful single-turn comparison.

## Run

Use the repository environment and reserve GPUs according to your host rules.
On this shared NVIDIA host:

```bash
gpu run --gpus 2 --nonblock --timeout 3h --note "Qwen3 Seed-TTS triggers" -- \
  uv run --no-sync pytest -s -v tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_seed_tts.json \
  --run-level full_model
```

The checked-in CUDA nightly job runs both cases on two H100/B200 GPUs.
`BENCHMARK_DIR` selects the output directory (default `tests/dfx/perf/results`).
For local cached weights/data, copy the JSON and replace `server_params.model`
and every `benchmark_params[].dataset_path` with local paths. Preserve sample
count, order, prompt, and generation settings across cases.

## Read the measurements

TTFT, audio TTFP, E2EL, and audio RTF share one client-side origin: the end of
reference-speech capture, immediately before the last speech chunk is sent.
They exclude session setup and the capture/upload intervals preceding that
boundary. Both cases use the same speech samples and capture pacing, so their
post-speech response timings are comparable.

- **TTFT / audio TTFP:** origin to first non-empty text delta / first audio packet
  received. This includes transport and response processing after the speech-end
  boundary and, for VAD, endpoint detection during the silent tail.
- **E2EL:** the same origin to `response.done` reception.
- **Audio RTF:** origin to the last audio packet divided by output audio duration.
  It includes post-speech waiting and endpoint detection; it is not model-only
  compute RTF.
- **Audio duration:** generated audio length, received as PCM16.

The per-request rows in `duplex_request_metrics` retain `measurement_origin`,
utterance identity, trigger, and these diagnostic fields:

- `input_audio_ms`: prepared clip duration, including the silent tail.
- `input_content_ms`: reference-speech duration, excluding the tail.
- `input_uploaded_ms`: audio duration actually sent; only VAD includes the tail.
- `input_upload_ms` / `session_setup_ms`: client wall-clock upload / setup time.
- `session_start_to_first_audio_ms` / `session_start_to_response_done_ms`:
  timings from before text submission and audio upload, retained for diagnosis.
- `explicit_commit_to_first_audio_ms` (explicit only): sending the explicit
  trigger to receiving the first audio packet.
- `vad_stop_received_ms` (VAD only): speech-end origin to the client's receipt of
  `speech_stopped`.
- `vad_stop_to_first_audio_ms` (VAD only): that event's receipt to the first audio
  packet. Both VAD fields use client event timestamps, not server GPU timestamps.

TTFT/TTFP here are client event timestamps. The server also attaches its own
`response_request_metrics` to the first text delta, measured from "accepted
native-append start" — a different origin, and one that is not defined for the
VAD case at all. The MiniCPM-o Seed-TTS duplex benchmark
(`test_minicpmo_4_5_duplex_seed_tts.json`) prefers those server values and
derives TPOT from Stage-0 engine metrics. **The two configurations do not share
a metric origin, so their TTFT/TTFP/RTF numbers are not comparable across
models.**

TPOT/ITL are not reported here, and cannot currently be: this path returns
`stage_metrics: {}` on every delta (verified against a live Qwen3-Omni duplex
server), so there is no engine token timing to read. The backend explicitly
disables derived TPOT even if a tokenizer counts transcript tokens.
`num_tpot_samples` stays `0`, so a configured TPOT baseline rejects the missing
measurement instead of comparing a value derived from client response latency.

## Historical measurements before the capture-pacing fix

The following numbers were collected on 2x L20X before chunks were moved to the
end of their capture intervals. That sender gave the server up to one chunk of
lookahead and could understate VAD endpoint latency by up to 200 ms. These are
historical results, not measurements of the current sender; rerun both triggers
before using them as a performance reference.

The effect of choosing a timing origin was recomputed over the same four
historical measured requests:

| RTF origin | `explicit` mean | `vad` mean | worst single request |
| --- | --- | --- | --- |
| end of reference speech | 0.13 | 0.21 | 0.24 |
| session start | 0.71 | 0.84 | 0.98 |

Timing from session start does not fail the `< 1` SLO outright, but roughly 85%
of what it reports is the client's own real-time upload, so the model's share is
diluted into a near-constant offset. Over these requests a hypothetical 100 ms
generation regression moves the reported RTF by 8.9% from the speech-end origin
and by 1.6% from session start — a 5.7x difference in how visible a regression
is. The worst measured VAD request already sits at 0.98 under the session-start
origin, with the margin consumed by upload rather than by the model, so a batch
of longer reference clips would cross the line for no model-side reason.

The test requires all requests to complete, exactly one completed response with
text and audio per request, and, for VAD, actual speech-start and speech-stop
events. No performance regression baseline is set: the numbers below were taken
on 2x L20X, not on the H100/B200 the nightly job targets, and four requests are
a small smoke test rather than a statistically stable latency study.

Reference run, 2x L20X, one warmup then four requests at concurrency 1:

| | `explicit` | `vad` |
| --- | --- | --- |
| Mean TTFT | 347 ms | 912 ms |
| Mean audio TTFP | 419 ms | 1035 ms |
| Mean audio RTF | 0.13 | 0.21 |
| Mean E2EL | 1213 ms | 1864 ms |

The roughly 600 ms VAD difference in this historical run must not be interpreted
as an unbiased measurement of the configured 800 ms silence threshold: the old
sender made the first 200 ms of silence available at the speech-end origin.
Current capture pacing removes that lookahead. Note the spread within each
run (explicit median TTFT 54 ms against a mean of 347 ms): the first measured
request is still paying warmup, so one `--num-warmups` is not enough to read
per-request numbers, only the means.
