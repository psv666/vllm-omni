# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Image/audio conversations with the official OpenAI Realtime SDK.

Install openai[realtime]. --video also needs opencv-python-headless and opts
into the vLLM-Omni rolling camera-frame extension.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import mimetypes
import os
import wave
from pathlib import Path

from openai import AsyncOpenAI


async def receive_until(connection, event_type):
    while True:
        event = await asyncio.wait_for(connection.recv(), timeout=180)
        if event.type == "error":
            raise RuntimeError(f"{event.error.code}: {event.error.message}")
        if event.type == event_type:
            return event


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type not in {"image/jpeg", "image/png"}:
        raise ValueError("Use a JPEG or PNG image.")
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode()}"


def video_frames(path: str, *, fps: float, maximum: int):
    import cv2

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(source_fps / fps))
        frame_index = sent = 0
        while sent < maximum:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                encoded, jpeg = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise ValueError("Could not encode a video frame as JPEG.")
                yield "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
                sent += 1
            frame_index += 1
    finally:
        capture.release()


async def add_image(connection, image_url):
    await connection.conversation.item.create(
        item={"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": image_url}]}
    )
    await receive_until(connection, "conversation.item.done")


async def run(args):
    async with AsyncOpenAI(base_url=args.base_url, api_key=args.api_key) as client:
        async with client.realtime.connect(
            model=args.model,
            websocket_connection_options={"max_size": 128 * 1024 * 1024},
        ) as connection:
            await receive_until(connection, "session.created")
            await connection.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": [args.output],
                    "audio": {"input": {"turn_detection": None}},
                }
            )
            await receive_until(connection, "session.updated")
            if args.video:
                # session.omni is an extension; SDK session.update has no extra_body argument.
                await connection.send(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "omni": {"input_image_retention": "rolling"}},
                    }
                )
                await receive_until(connection, "session.updated")
                for image_url in video_frames(args.video, fps=args.fps, maximum=args.max_frames):
                    await add_image(connection, image_url)
            for path in args.image:
                await add_image(connection, image_data_url(path))
            if args.input_wav:
                with wave.open(str(args.input_wav), "rb") as source:
                    if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 24000):
                        raise ValueError("GA input WAV must contain mono PCM16 at 24000 Hz.")
                    while chunk := source.readframes(2400):
                        await connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode())
                await connection.input_audio_buffer.commit()
                await receive_until(connection, "conversation.item.done")
            if args.query:
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": args.query}],
                    }
                )
                await receive_until(connection, "conversation.item.done")
            await connection.response.create()
            chunks = []
            while True:
                event = await asyncio.wait_for(connection.recv(), timeout=180)
                if event.type == "error":
                    raise RuntimeError(f"{event.error.code}: {event.error.message}")
                if event.type in {"response.output_text.delta", "response.output_audio_transcript.delta"}:
                    print(event.delta, end="", flush=True)
                elif event.type == "response.output_audio.delta":
                    chunks.append(base64.b64decode(event.delta, validate=True))
                elif event.type == "response.done":
                    print()
                    if event.response.status != "completed":
                        raise RuntimeError(f"Response ended with status {event.response.status}")
                    break
            if chunks:
                with wave.open(str(args.output_wav), "wb") as destination:
                    destination.setnchannels(1)
                    destination.setsampwidth(2)
                    destination.setframerate(24000)
                    destination.writeframes(b"".join(chunks))
                print(f"Saved audio to {args.output_wav}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8091/v1/")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True, help="Model name exposed by the Realtime server")
    parser.add_argument("--image", type=Path, action="append", default=[], help="JPEG/PNG; repeat for multiple images")
    parser.add_argument("--video", help="Video file; enables rolling camera-frame retention")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames uploaded per second of source video")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--input-wav", type=Path, help="Optional mono PCM16 24 kHz input")
    parser.add_argument(
        "--query", default="Describe what you see.", help="Use an empty string for image/audio-only input"
    )
    parser.add_argument("--output", choices=["text", "audio"], default="text")
    parser.add_argument("--output-wav", type=Path, default=Path("response.wav"))
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames <= 0:
        parser.error("--fps and --max-frames must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
