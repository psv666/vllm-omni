# Qwen3 Realtime session and transport boundaries

The Qwen3 GA profile is a turn-based frontend over AsyncOmni. It does not
require persistent model-side sessions or cross-turn KV-cache reuse.

## Code layout

All components live under `vllm_omni/entrypoints/openai/realtime/`:

- `contracts.py`, `session.py`, and `runtime.py`: conversation state and response execution.
- `qwen3.py`, `legacy.py`, and `video.py`: model adaptation and legacy video support.
- `codec.py`, `events.py`, `connection.py`, and `routing.py`: GA protocol and connection handling.

Transport code calls the shared runtime; the runtime does not import transport
modules. The parent OpenAI package exposes its serving objects lazily to
preserve this import boundary within the common directory.

## Request path

~~~text
WebSocket client
  -> RealtimeGAConnection._read
  -> codec.decode_event: JSON -> typed command
  -> RealtimeRuntime.submit
       -> SessionStore: ordered input items and one reserved assistant slot
       -> Qwen3RealtimeAdapter.generate: immutable TurnSnapshot -> model input
            -> existing chat preprocessing
            -> AsyncOmni.generate -> Thinker -> Talker -> Code2Wav
       <- ModelDelta: text, spoken-text source, waveform samples
  <- runtime.events
  <- GAEventEncoder: GA item/part lifecycle and PCM24 messages
  <- RealtimeGAConnection._write
~~~

The WebSocket reader and writer are separate asyncio tasks in the API
process. The runtime owns a separate task for the active response. AsyncOmni
dispatches work to the existing orchestrator and stage processes; this change
adds no model process or duplex engine implementation.

The legacy video handler prepares its existing prompt, then uses the same
runtime through PreparedAdapter. Its outer query task translates events;
only the shared runtime owns model iteration and cancellation.

## State ownership

| Owner | State | Lifetime |
| --- | --- | --- |
| SessionStore | Ordered immutable items, current configuration, uncommitted PCM, last response | One WebSocket connection |
| Runtime response draft | Fixed input snapshot, private output accumulation, model task, cancellation fence | One response |
| Qwen prepared turn | Model prompt, sampling parameters, waveform offsets, output mode | One model iteration |
| GA encoder | Wire IDs and PCM resampler | Until the response terminal is encoded |
| Legacy shim | Frame acknowledgement metadata and bounded PIL prewarm cache | One legacy connection |

The GA connection has no second message history or generation loop.
The runtime imports neither GA event types nor MiniCPM duplex code.

## Atomic completion

Suppose the client inserts user item U1 and starts response R1:

1. The runtime freezes a snapshot containing U1.
2. It inserts an empty, in-progress assistant item A1 after U1.
3. Model deltas accumulate in a private draft; the stored A1 stays unchanged.
4. A newly received user item U2 goes after A1 and affects the next response.
5. Finalization synchronously replaces A1 with the complete item and records
   the Response object. There is no await between these state mutations.
6. The runtime then queues the terminal fact. GA item terminals and
   response.done use that same finalized item.

The final order is U1, A1, U2 regardless of how U2's arrival overlaps model
generation. Completed, cancelled, failed and token-limited responses all
close the item/part lifecycle. Failed or interrupted assistant items have
incomplete item status.

## Cancellation and cleanup

A connection admits one active response. Cancellation immediately marks
queued deltas undeliverable, closes the model iterator, and waits for engine
abort acknowledgement before admitting another response. No fixed sleep
stands in for that acknowledgement. Cancelling a completed response returns
a recoverable error.

A cleanup deadline closes admission if the engine cannot stop. A send failure
or disconnect also closes the runtime and releases its conversation state.
Backpressure uses a bounded event queue. A slow sender has a send deadline;
an active model response is not treated as an idle connection.

When async_chunk is disabled, the backend can finish synthesis before the
first waveform is returned. The GA encoder splits large waveforms into
100 ms PCM deltas so ordinary SDK message limits still work. This does not
turn buffered model generation into incremental synthesis, and a late cancel
does not undo a response already committed to Conversation.

## Model and media policy

GA client retention keeps all accepted item content until explicit deletion.
Rolling retention is an explicit session.omni extension: only single-image
user items can be evicted. EVS and sampling select model input without
rewriting retained items; the newest selected image survives.

Each input snapshot holds references to immutable media bytes. Deleting an
item therefore affects future responses without invalidating an active one.
Committed PCM is resampled as a continuous buffer for Qwen's 16 kHz input.
Output resampling keeps continuity across model chunks before transport
splitting into PCM24 messages.

For the Instruct pipeline, Thinker text conditions Talker through
thinker2talker_async_chunk (or the corresponding non-streaming path).
That text supplies output_audio_transcript. It is not ASR of the emitted
waveform and does not provide playback alignment or word timing.

## Validation

CPU tests exercise the public submit/events/close interface, atomic writeback,
cancel/finish races, queue backpressure, SDK event schemas, image policies,
audio conversion, route precedence, and import boundaries.

The Qwen GA E2E module starts one full server per async_chunk mode and uses
the official SDK for text, image, audio, mixed input, history, deletion,
cancellation, repeated requests and concurrent sessions. It also exercises
Chat Completions, the legacy Realtime profile, and legacy video on that server.
The merge pipeline explicitly runs this matrix in its own Qwen GA job.

See [the wire protocol and user guide](../../serving/openai_realtime_qwen3.md).
