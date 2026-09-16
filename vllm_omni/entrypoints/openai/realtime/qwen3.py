# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3 Realtime turn execution, independent of either WebSocket protocol."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pybase64 as base64
from vllm.sampling_params import RequestOutputKind

from vllm_omni.entrypoints.openai.realtime.video import FrameSimilarityFilter, sample_frame_indices
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.outputs import OmniRequestOutput

from .contracts import ModelDelta, TurnSnapshot

_CODEC_FRAME_SAMPLES = 1920


def pcm_to_wav_b64(raw: bytes, sample_rate: int = 16000) -> str:
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def resample_pcm16(raw: bytes, source_rate: int, target_rate: int) -> bytes:
    """Resample a committed buffer once, retaining continuity across appends."""
    if source_rate == target_rate or not raw:
        return raw
    from vllm_omni.utils.audio_resample import StreamingAudioResampler

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    converted = StreamingAudioResampler(source_rate, target_rate).process(samples, final=True)
    return np.clip(np.rint(converted * 32768.0), -32768, 32767).astype("<i2").tobytes()


def _audio_payload(output: OmniRequestOutput) -> Mapping[str, Any]:
    # Current OmniRequestOutput exposes a property that handles both
    # completion-level and output-level multimodal payloads.
    payload = getattr(output, "multimodal_output", None)
    result = dict(payload) if isinstance(payload, Mapping) else {}
    if result.get("audio") is None:
        audio = getattr(output, "audio_data", None)
        if audio is not None:
            result["audio"] = audio
    for key in ("sr", "sample_rate", "audio_sample_rate"):
        if key not in result and getattr(output, key, None) is not None:
            result[key] = getattr(output, key)
    return result


def audio_sample_rate(payload: Mapping[str, Any], default: int = 24000) -> int:
    for key in ("sr", "sample_rate", "audio_sample_rate"):
        value = payload.get(key)
        while isinstance(value, (list, tuple)) and value:
            value = value[-1]
        scalar = getattr(value, "item", None)
        if callable(scalar):
            value = scalar()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
    return default


def audio_delta_samples(
    audio: Any, sample_offset: int, *, is_first: bool = True, mode: str = "fast"
) -> tuple[np.ndarray | None, int]:
    """Extract a cumulative waveform tail by samples, supporting lists/tensors.

    OutputProcessor consolidates cumulative lists into one tensor, so a chunk
    count cannot identify an already-emitted prefix. DELTA callers pass zero.
    """
    if audio is None:
        return None, sample_offset
    chunks = audio if isinstance(audio, list) else [audio]
    sizes = [int(chunk.numel()) if hasattr(chunk, "numel") else int(np.asarray(chunk).size) for chunk in chunks]
    total = sum(sizes)
    if total <= sample_offset:
        return None, sample_offset
    arrays: list[np.ndarray] = []
    cursor = 0
    for chunk, size in zip(chunks, sizes):
        start = 0 if mode == "slow" else max(0, sample_offset - cursor)
        cursor += size
        if start >= size:
            continue
        if hasattr(chunk, "detach"):
            chunk = chunk.reshape(-1)[start:].float().detach().cpu().numpy()
        else:
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)[start:]
        arrays.append(chunk)
    samples = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    if mode == "slow":
        samples = samples[sample_offset:]
    if is_first and samples.size > _CODEC_FRAME_SAMPLES * 2:
        samples = samples[_CODEC_FRAME_SAMPLES:]
    return samples if samples.size else None, total


@dataclass
class _PreparedTurn:
    prompt: Any
    modalities: list[str]
    sampling_kwargs: dict[str, Any]
    buffer_audio: bool = False
    first_text_at: float | None = None
    first_audio_at: float | None = None
    audio_chunks: int = 0
    output_kinds: list[RequestOutputKind] = field(default_factory=list)
    audio_delta_mode: str = "fast"

    def is_delta(self, output: OmniRequestOutput) -> bool:
        stage_id = output.stage_id
        if not isinstance(stage_id, int):
            stage_id = len(self.output_kinds) - 1 if output.final_output_type == "audio" else 0
        kind = self.output_kinds[stage_id] if 0 <= stage_id < len(self.output_kinds) else RequestOutputKind.DELTA
        return kind == RequestOutputKind.DELTA


class Qwen3RealtimeAdapter:
    def __init__(
        self, chat_service: Any, engine_client: Any, *, preprocess: Callable[[Any], Awaitable[Any]] | None = None
    ) -> None:
        self._chat_service = chat_service
        self._engine_client = engine_client
        self._preprocess_override = preprocess

    async def abort(self, request_id: str) -> None:
        await self._engine_client.abort(request_id)

    @staticmethod
    def _selected_images(snapshot: TurnSnapshot) -> set[tuple[int, int]]:
        images = [
            (i, j)
            for i, item in enumerate(snapshot.items)
            for j, part in enumerate(item.content)
            if part.kind == "image"
        ]
        policy = snapshot.config.visual
        if policy.retention != "rolling":
            return set(images)
        if policy.evs:
            frame_filter = FrameSimilarityFilter(threshold=policy.threshold)
            retained = [
                pos for pos in images if frame_filter.should_retain(snapshot.items[pos[0]].content[pos[1]].data)
            ]
            # The newest camera view must survive duplicate filtering too.
            if images and images[-1] not in retained:
                retained.append(images[-1])
            images = retained
        return {images[i] for i in sample_frame_indices(len(images), policy.sample)}

    def build_messages(self, snapshot: TurnSnapshot) -> tuple[list[dict[str, Any]], bool]:
        selected = self._selected_images(snapshot)
        messages: list[dict[str, Any]] = []
        if snapshot.config.instructions:
            messages.append({"role": "system", "content": snapshot.config.instructions})
        has_audio = False
        for i, item in enumerate(snapshot.items):
            content: list[dict[str, Any]] = []
            for j, part in enumerate(item.content):
                if part.kind == "text" or (part.kind == "audio" and item.role == "assistant"):
                    content.append({"type": "text", "text": part.text})
                elif part.kind == "image" and (i, j) in selected:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{part.media_type};base64,{base64.b64encode(part.data).decode('ascii')}",
                                "detail": part.detail,
                            },
                        }
                    )
                elif part.kind == "audio" and part.data:
                    pcm = resample_pcm16(part.data, part.sample_rate, 16000)
                    content.append(
                        {"type": "input_audio", "input_audio": {"data": pcm_to_wav_b64(pcm), "format": "wav"}}
                    )
                    has_audio = True
            if content:
                messages.append({"role": item.role, "content": content})
        return messages, has_audio

    async def _preprocess(self, request: Any) -> Any:
        if self._preprocess_override is not None:
            return await self._preprocess_override(request)
        handler = self._chat_service
        _, prompts = await handler._preprocess_chat(
            request,
            request.messages,
            default_template=getattr(request, "chat_template", None) or handler.chat_template,
            default_template_content_format=handler.chat_template_content_format,
            default_template_kwargs=handler._effective_chat_template_kwargs(request),
            renderer=handler.renderer,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            add_special_tokens=request.add_special_tokens,
        )
        return prompts[0]

    async def generate(self, snapshot: TurnSnapshot, request_id: str) -> AsyncGenerator[ModelDelta, None]:
        # PIL decode/EVS and committed-buffer resampling must not stall the
        # connection reader while it accepts cancellation or the next input.
        messages, has_audio = await asyncio.to_thread(self.build_messages, snapshot)
        prepared = await self.prepare(
            messages,
            {
                "model": snapshot.config.model,
                "instructions": snapshot.config.instructions,
                "output_modalities": [snapshot.config.output_mode],
                "max_output_tokens": snapshot.config.max_tokens,
                "omni": {"use_audio_in_video": snapshot.config.use_audio_in_video},
            },
            has_audio=has_audio,
        )
        async with aclosing(self.generate_prepared(prepared, request_id)) as outputs:
            async for output in outputs:
                yield output

    async def prepare(
        self, messages: list[dict[str, Any]], config: dict[str, Any], *, has_audio: bool
    ) -> _PreparedTurn:
        """Use the same chat preprocessing for GA and the legacy video shim."""
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        modalities = ["text", "audio"] if "audio" in config.get("output_modalities", ["audio"]) else ["text"]
        if config.get("omni", {}).get("_legacy_output_modalities"):
            modalities = list(config["output_modalities"])
        request_kwargs: dict[str, Any] = {
            "model": config.get("model") or "default",
            "messages": messages,
            "stream": True,
            "modalities": modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        omni = config.get("omni", {})
        if has_audio and omni.get("use_audio_in_video", True):
            request_kwargs["mm_processor_kwargs"] = {"use_audio_in_video": True}
        limit = config.get("max_output_tokens", "inf")
        if isinstance(limit, int):
            request_kwargs["max_tokens"] = limit
        request = ChatCompletionRequest(**request_kwargs)
        sampling_kwargs: dict[str, Any] = {}
        overrides = omni.get("sampling_params_list")
        if overrides:
            params = self._chat_service._to_sampling_params_list(overrides)
            sampling_kwargs["sampling_params_list"] = coerce_param_message_types(params, is_streaming=True)
        elif isinstance(limit, int):
            params = self._chat_service._build_sampling_params_list_from_request(request)
            sampling_kwargs["sampling_params_list"] = coerce_param_message_types(params, is_streaming=True)
        prompt = await self._preprocess(request)
        # AsyncOmni coerces omitted/default params to DELTA too, then applies
        # per-stage pipeline constraints. Mirror that contract without changing
        # the omitted-parameter behavior at the actual generate boundary.
        stages = getattr(self._engine_client, "stage_configs", [])
        stages = stages if isinstance(stages, (list, tuple)) else []
        defaults = getattr(self._engine_client, "default_sampling_params_list", [])
        defaults = defaults if isinstance(defaults, (list, tuple)) else []
        explicit = sampling_kwargs.get("sampling_params_list", [])
        count = max(len(stages), len(defaults), len(explicit))
        output_kinds = [RequestOutputKind.DELTA] * count
        constraints = getattr(self._engine_client, "sampling_constraints_list", None)
        if not isinstance(constraints, (list, tuple)):
            constraints = [getattr(stage, "sampling_constraints", {}) for stage in stages]
        for index in range(count):
            if index < len(explicit):
                output_kinds[index] = explicit[index].output_kind
            if index < len(constraints) and isinstance(constraints[index], Mapping):
                value = constraints[index].get("output_kind")
                if value is not None:
                    output_kinds[index] = (
                        RequestOutputKind[value.upper()] if isinstance(value, str) else RequestOutputKind(value)
                    )
        return _PreparedTurn(
            prompt,
            modalities,
            sampling_kwargs,
            omni.get("_audio_chunk_mode") == "off",
            output_kinds=output_kinds,
            audio_delta_mode=omni.get("_audio_delta_mode", "fast"),
        )

    async def generate_prepared(self, prepared: _PreparedTurn, request_id: str) -> AsyncGenerator[ModelDelta, None]:
        """Stream model facts after preparation, without knowing either protocol."""
        previous_text = ""
        audio_offset = 0
        audio_started = False
        finish_reason = None
        buffered_audio: list[np.ndarray] = []
        buffered_rate: int | None = None
        results = self._engine_client.generate(
            prompt=prepared.prompt,
            request_id=request_id,
            output_modalities=prepared.modalities,
            **prepared.sampling_kwargs,
        )
        async with aclosing(results):
            async for output in results:
                if not isinstance(output, OmniRequestOutput):
                    continue
                if output.error:
                    raise RuntimeError(output.error)
                outputs = getattr(output, "outputs", [])
                if outputs and getattr(outputs[0], "finish_reason", None) and finish_reason != "length":
                    finish_reason = outputs[0].finish_reason
                is_delta = prepared.is_delta(output)
                if getattr(output, "final_output_type", "text") == "audio":
                    prepared.audio_chunks += 1
                    if prepared.first_audio_at is None:
                        prepared.first_audio_at = time.monotonic()
                    payload = _audio_payload(output)
                    audio = payload.get("audio")
                    if audio is None:
                        continue
                    samples, audio_offset = audio_delta_samples(
                        audio,
                        0 if is_delta else audio_offset,
                        is_first=not audio_started and not prepared.buffer_audio,
                        mode=prepared.audio_delta_mode,
                    )
                    if samples is None:
                        continue
                    audio_started = True
                    rate = audio_sample_rate(payload)
                    if prepared.buffer_audio:
                        if buffered_rate is not None and rate != buffered_rate:
                            raise ValueError("Audio sample rate changed within a buffered response")
                        buffered_rate = rate
                        buffered_audio.append(samples)
                    else:
                        yield ModelDelta(audio=samples, sample_rate=rate)
                elif outputs and getattr(output, "stage_id", None) in (None, 0):
                    text = getattr(outputs[0], "text", "") or ""
                    delta = text if is_delta else text[len(previous_text) :] if text.startswith(previous_text) else text
                    previous_text = text
                    if delta:
                        if prepared.first_text_at is None:
                            prepared.first_text_at = time.monotonic()
                        yield ModelDelta(text=delta, transcript=delta if "audio" in prepared.modalities else "")
        if buffered_audio:
            coalesced = np.concatenate(buffered_audio)
            if coalesced.size > _CODEC_FRAME_SAMPLES * 2:
                coalesced = coalesced[_CODEC_FRAME_SAMPLES:]
            yield ModelDelta(audio=coalesced, sample_rate=buffered_rate or 24000)
        yield ModelDelta(finish_reason=finish_reason or "stop")
