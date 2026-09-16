# OpenAI Realtime GA for Qwen3-Omni

The `openai-realtime` profile provides a stateful, turn-based subset of the
OpenAI Realtime GA WebSocket API. Add text, audio, images, or mixed messages
to a conversation, then explicitly request a text or spoken response. Each
response rebuilds the model prompt from a conversation snapshot, including
previous user inputs and assistant replies.

## Start the server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml \
    --realtime-profile openai-realtime --port 8091
```

Use the full thinker, talker, and code2wav pipeline. The supplied deployment
uses two GPUs. The default profile remains `qwen3-legacy`, as described in
[Realtime Audio WebSocket API](realtime_api.md). The profile is selected on
the server; SDK clients only need the usual model argument and base URL.

## Official OpenAI SDK example

Install `openai[realtime]`. This example uses the GA `client.realtime`
interface available in OpenAI Python SDK 2.45.0.

```python
import asyncio
import base64
from pathlib import Path

from openai import AsyncOpenAI


async def main():
    image = base64.b64encode(Path("frame.jpg").read_bytes()).decode()
    async with AsyncOpenAI(base_url="http://localhost:8091/v1/", api_key="EMPTY") as client:
        async with client.realtime.connect(model="Qwen/Qwen3-Omni-30B-A3B-Instruct") as connection:
            assert (await connection.recv()).type == "session.created"
            await connection.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "audio": {"input": {"turn_detection": None}},
                }
            )
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image}"},
                        {"type": "input_text", "text": "What is in this image?"},
                    ],
                }
            )
            await connection.response.create()
            async for event in connection:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "error":
                    raise RuntimeError(event.error.message)
                elif event.type == "response.done":
                    print()
                    break


asyncio.run(main())
```

The supplied client supports repeated `--image`, video-file frames, audio,
and saving spoken output:

```bash
python examples/online_serving/realtime/openai_realtime_client.py \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --image frame.jpg --query "What is happening?" --output audio
```

For video, install `opencv-python-headless` and pass `--video clip.mp4`.
This explicitly enables the rolling camera-frame extension below. Add
`--input-wav input_24k_mono.wav` for audio; `--query ""` omits text input.

## Supported events and audio

| Client event | Behavior |
| --- | --- |
| `session.update` | Updates supported settings and returns resolved `session.updated`. |
| `conversation.item.create` | Adds a message, then emits `conversation.item.added` and `.done`. |
| `conversation.item.delete` | Removes a retained item and its media. |
| `input_audio_buffer.append` | Buffers base64 raw PCM16. |
| `input_audio_buffer.commit` | Creates a user audio item and emits `input_audio_buffer.committed`; does not generate a response. |
| `input_audio_buffer.clear` | Clears uncommitted audio. |
| `response.create` | Starts one response from a conversation snapshot. |
| `response.cancel` | Cancels the active model request. |

The server emits the following lifecycle (one output item and one content
part per response in this implementation):

```text
response.created
response.output_item.added
conversation.item.added
response.content_part.added
response.output_text.delta* OR output_audio_transcript.delta* / output_audio.delta*
response.output_text.done OR output_audio_transcript.done / output_audio.done
response.content_part.done
response.output_item.done
conversation.item.done
response.done
```

All response events share the response/item IDs and index 0. Content-part
events use part types `text`/`audio`; conversation message content uses
`output_text`/`output_audio`. The runtime reserves the assistant's position
at response start. It accumulates output privately and replaces that item
atomically before publishing terminal events. Input accepted during a response
therefore stays after the reserved assistant item. The finalized item is the
same in Conversation, item terminals, and `response.done.output`.

The server emits response, output-item, and content-part start/end events,
incremental output, and exactly one `response.done` per started response.
Terminal statuses include `completed`, `cancelled`, `failed`, and `incomplete`
when the Thinker output-token limit is reached. This limit changes the
comprehension stage budget; Talker/Code2Wav retain their deployment limits. A second active `response.create`
returns an error without replacing the current response. Disconnecting
aborts the active model request.

Cancellation stops generation. Clients must stop their own queued audio
playback. The server does not track which audio the client has played or
roll back conversation history to that playback position;
`conversation.item.truncate` is not supported by this profile.

`output_modalities` accepts `["text"]` or `["audio"]`. Spoken output includes
its transcript through `response.output_audio_transcript.delta` and `.done`;
`["text", "audio"]` is not a GA output mode. Text output uses
`response.output_text.delta` and `.done`. The spoken-output transcript comes
from Qwen3-Omni's Thinker text output, rather than transcription of the
generated waveform. Speech can deviate from that text, including extra
words in very short answers. This profile does not verify speech with ASR
or provide word timing and playback correction.

Input and output PCM are **mono PCM16 at 24 kHz**, represented by
`{"type": "audio/pcm", "rate": 24000}`. Send raw PCM in
`input_audio_buffer.append.audio`; receive it in
`response.output_audio.delta.delta`. Neither payload contains a WAV header.
The client adds one when saving the complete waveform. PCM delta events contain
at most 100 ms each, including when the engine returns an entire waveform.
With `async_chunk: false`, synthesis may already have finished when the first
audio delta arrives; cancelling that finished response returns
`response_cancel_not_active`. Clients still stop their own playback.

Supported session settings include `instructions`, `output_modalities`,
`max_output_tokens` (1–4096 or `"inf"`), PCM format, and disabled
`audio.input.turn_detection` (`null`). Unsupported options return a
recoverable `error` with the offending field in `error.param`.

## Image retention and camera frames

Image parts use JPEG or PNG base64 data URLs. Each base64-decoded JPEG or
PNG file is limited to 10 MiB. This limit applies to the encoded image file
bytes, not to its decompressed pixel data. The server also decodes and
validates the image before acknowledging an item. Remote image URLs and
inline video payloads are not supported. Image acknowledgements retain the
accepted data URL. For images exceeding the SDK's default 1 MiB message limit,
pass `websocket_connection_options={"max_size": 128 * 1024 * 1024}` to the
standard SDK `connect` method, as the example client does. Uncommitted audio is limited to
16 MiB per connection and retained compressed media to 64 MiB. Exceeding a
limit returns an error without partially applying the operation.

By default, accepted items preserve their content and order until deletion.
Capacity defaults to 50 image parts across the conversation. At capacity,
new images receive an error; `conversation.item.delete` frees space. A mixed
image/text/audio message remains one item. Pure image, text, or audio input
is also valid.

For a long camera stream, explicitly enable the vLLM-Omni extension:

```python
await connection.send({
    "type": "session.update",
    "session": {
        "type": "realtime",
        "omni": {
            "input_image_retention": "rolling",
            "input_image_max_items": 50,
            "input_image_sample": 4,
            "input_image_evs": True,
            "input_image_evs_threshold": 0.95,
        },
    },
})
```

`session.omni` is a vendor extension. Use SDK `connection.send` for this
payload; SDK 2.45 `session.update` has no `extra_body` argument. The resolved
values are echoed in `session.updated`.

Rolling mode may remove the oldest **single-image user items**, emitting
`conversation.item.deleted`. Mixed-content and multi-image items are never
automatically removed. If those protected items leave insufficient room,
insertion fails without partially deleting the conversation. Capacity can
be configured from 1 to 256 images.

EVS removes near-duplicate frames from the model input for a response;
accepted conversation content remains intact. Sampling selects up to
`input_image_sample` images (1–128, default 4) and preserves the newest
selected frame. These optimizations are enabled only in rolling mode.

## Routing and ownership

The server default is `--realtime-profile qwen3-legacy`. A Qwen connection
can override it with `?profile=openai-realtime` or `?profile=qwen3-legacy`.
Starting with the GA default allows ordinary SDK `connect(model=...)`.

`duplex=1` takes precedence over the server's Qwen profile default. An
explicit `duplex=1&profile=...` is rejected; the two protocols cannot be
combined. An unsupported duplex request fails without falling back to Qwen.
Native MiniCPM duplex selection and `/v1/duplex` stay separate.

All Realtime components live under `entrypoints/openai/realtime/`.
`session.py` and `runtime.py` own the conversation store, input snapshot,
response task, and cancellation; `qwen3.py` adapts turns to Qwen model execution.
`codec.py`, `events.py`, and `connection.py` handle GA messages and WebSocket I/O.
The session/runtime modules do not import these transport modules or the
MiniCPM duplex runtime. The parent OpenAI package loads its public serving
exports lazily so importing the runtime does not initialize the API server.
The legacy video shim shares the conversation store and response runtime,
while retaining its original audio format, context policy, and frame
acknowledgements. Closing either connection releases its session state.

See [the runtime design](../design/feature/qwen3_realtime.md) for the execution
path, state owners, and cancellation sequence.

## Compatibility and limits

- The default legacy Realtime profile keeps its 16 kHz input and
  `input_audio_buffer.commit.final` contract; GA rejects those legacy fields.
- `input_audio_buffer.append.video_frames` is rejected with
  `error.param: "video_frames"`. Send `input_image` items instead.
- The [legacy video endpoint](video_stream_api.md) and
  `streaming_video_client.py` remain available through a compatibility shim
  in this change, preserving the event vocabulary, 16 kHz audio input, and
  WAV output chunks. Their owners will decide any future deprecation or
  removal as clients migrate; no sunset date is set here.
- This implementation uses explicit response turns. Server VAD, tool calls,
  WebRTC, SIP, session resume, playback acknowledgements, and native duplex
  execution are outside this profile.
- Conversation state belongs to the WebSocket. Reconnecting starts a new
  conversation. Cross-turn visual KV-cache reuse and incremental visual
  prefill are not implemented.

See the [OpenAI Realtime conversations guide](https://developers.openai.com/api/docs/guides/realtime-conversations)
and [Realtime event reference](https://developers.openai.com/api/reference/resources/realtime/).
