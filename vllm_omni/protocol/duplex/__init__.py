# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""vLLM-Omni's full-duplex extension of the OpenAI Realtime wire protocol.

OpenAI's Realtime API assumes one turn at a time: the client commits input, the
server answers. A full-duplex session does not --- the model listens while it
speaks, it decides when to take a turn, the client reports playback progress,
and a dropped socket re-attaches to a session that kept running. Those need
events and commands the GA vocabulary does not have.

This package is that delta. It builds on ``vllm_omni.protocol.realtime`` (same
base classes, same pure wire rendering, no duplication) and, like it, owns no
session and imports no runtime: the duplex engine depends on this package, not
the other way round.

The one door
------------
It is also the *only* door a duplex consumer uses. ``protocol.duplex.commands``
and ``protocol.duplex.events`` carry the complete vocabulary --- Tier 1 classes
re-exported unchanged alongside the Tier 2 and Tier 3 ones --- and this module
re-exports the Tier 1 helper functions. So the layering is a chain, not a mesh::

    protocol.realtime   (OpenAI's surface)
        ^
        | imports
        |
    protocol.duplex     (this package: re-exports Tier 1, adds Tier 2 and 3)
        ^
        | imports
        |
    duplex engine / entrypoints / clients

Duplex consumers import this package rather than the Tier 1 package directly.
The facade centralizes imports; it does not replace dependencies already bound
inside shared helpers. A consumer-specific conversion must be passed explicitly
to a helper when that behavior is introduced.

Commands here carry decoded client intent. The runner queues these objects and
reads their fields. Handlers that still consume internal dictionaries use
``vllm_omni.engine.duplex.command_payload.to_internal_payload``. Session state
and command resolution stay in the engine.
"""

from vllm_omni.protocol.duplex.errors import (
    REALTIME_ERROR_TYPES_BY_CODE,
    RealtimeProtocolError,
)
from vllm_omni.protocol.realtime.audio import (
    MAX_INPUT_SAMPLE_RATE_HZ,
    MIN_INPUT_SAMPLE_RATE_HZ,
    convert_input_audio_with_rate,
    convert_output_audio,
    decode_g711_alaw,
    decode_g711_ulaw,
    encode_g711_alaw,
    encode_g711_ulaw,
    resample_pcm16_mono,
    validate_input_sample_rate_hz,
    wav_payload_to_pcm16,
)
from vllm_omni.protocol.realtime.audio_input import (
    REALTIME_INPUT_HINT_KEYS,
    RealtimeAudioAppend,
    copy_realtime_input_hints,
    decode_audio_append,
    input_explicitly_non_speech,
    input_looks_like_speech,
)
from vllm_omni.protocol.realtime.capabilities import (
    RealtimeProtocolCapabilities,
    RealtimeSessionRejection,
    validate_session_payload,
)
from vllm_omni.protocol.realtime.formats import (
    REALTIME_INPUT_AUDIO_FORMATS,
    REALTIME_OUTPUT_AUDIO_FORMATS,
    is_supported_realtime_input_format,
    parse_realtime_audio_format,
    realtime_audio_format_object,
    realtime_output_format,
    validate_conversation_item_audio_formats,
    validate_realtime_response_audio_formats,
    validate_realtime_session_audio_formats,
)
from vllm_omni.protocol.realtime.items import (
    input_transcript_from_item,
    normalize_conversation_item,
    text_chars_for_audio_ms_from_marks,
    truncate_realtime_item_content,
    validate_realtime_item_truncate,
    validate_realtime_video_frames,
)
from vllm_omni.protocol.realtime.session import (
    RealtimeInputDefaults,
    apply_realtime_session_defaults,
    input_audio_transcription_config,
    json_safe_realtime_payload,
    realtime_max_output_tokens,
    realtime_overlap_fields,
)

__all__ = [
    "MAX_INPUT_SAMPLE_RATE_HZ",
    "MIN_INPUT_SAMPLE_RATE_HZ",
    "REALTIME_ERROR_TYPES_BY_CODE",
    "REALTIME_INPUT_AUDIO_FORMATS",
    "REALTIME_INPUT_HINT_KEYS",
    "REALTIME_OUTPUT_AUDIO_FORMATS",
    "RealtimeAudioAppend",
    "RealtimeInputDefaults",
    "RealtimeProtocolCapabilities",
    "RealtimeProtocolError",
    "RealtimeSessionRejection",
    "apply_realtime_session_defaults",
    "convert_input_audio_with_rate",
    "convert_output_audio",
    "copy_realtime_input_hints",
    "decode_audio_append",
    "decode_g711_alaw",
    "decode_g711_ulaw",
    "encode_g711_alaw",
    "encode_g711_ulaw",
    "input_audio_transcription_config",
    "input_explicitly_non_speech",
    "input_looks_like_speech",
    "input_transcript_from_item",
    "is_supported_realtime_input_format",
    "json_safe_realtime_payload",
    "normalize_conversation_item",
    "parse_realtime_audio_format",
    "realtime_audio_format_object",
    "realtime_max_output_tokens",
    "realtime_output_format",
    "realtime_overlap_fields",
    "resample_pcm16_mono",
    "text_chars_for_audio_ms_from_marks",
    "truncate_realtime_item_content",
    "validate_conversation_item_audio_formats",
    "validate_input_sample_rate_hz",
    "validate_realtime_item_truncate",
    "validate_realtime_response_audio_formats",
    "validate_realtime_session_audio_formats",
    "validate_realtime_video_frames",
    "validate_session_payload",
    "wav_payload_to_pcm16",
]
