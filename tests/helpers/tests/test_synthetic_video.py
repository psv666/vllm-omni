# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import io
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from tests.helpers import media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def synthetic_audio(mocker):
    """Keep video tests independent of the host's TTS installation."""
    sample_rate = 48000
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 440 * samples)).reshape(-1, 1)
    return mocker.patch.object(media, "generate_synthetic_audio", return_value={"np_array": waveform})


def test_audio_generation_failure_is_not_cached(mocker, tmp_path):
    failure = RuntimeError("TTS dependency unavailable")
    mocker.patch.object(media, "generate_synthetic_audio", side_effect=failure)

    with pytest.raises(RuntimeError, match="TTS dependency unavailable") as caught:
        media.generate_synthetic_video(64, 64, 6, embed_audio=True, cache_dir=tmp_path)

    assert caught.value is failure
    assert not list(tmp_path.glob("*.mp4"))


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("FFmpeg unavailable"),
        subprocess.CalledProcessError(1, ["ffmpeg"]),
        subprocess.TimeoutExpired(["ffmpeg"], 300),
    ],
    ids=["missing-ffmpeg", "ffmpeg-error", "ffmpeg-timeout"],
)
def test_audio_mux_failure_is_propagated(mocker, synthetic_audio, failure):
    mocker.patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    mocker.patch.object(media.subprocess, "run", side_effect=failure)

    with pytest.raises(type(failure)) as caught:
        media._mux_mp4_bytes_with_synthetic_audio(b"video", num_frames=6)

    assert caught.value is failure


def test_audio_in_video_repairs_video_only_cache(synthetic_audio, tmp_path):
    video_only = media.generate_synthetic_video(64, 64, 6, cache_dir=tmp_path)
    # Reproduce a cache entry written by the old silent audio-mux fallback.
    cache_path = tmp_path / "synth_video_w64_h64_nf6_ea1.mp4"
    original_bytes = Path(video_only["file_path"]).read_bytes()
    cache_path.write_bytes(original_bytes)

    result = media.generate_synthetic_video(64, 64, 6, embed_audio=True, cache_dir=tmp_path)

    assert result["file_path"] == str(cache_path)
    raw = base64.b64decode(result["base64"])
    assert raw == cache_path.read_bytes()
    assert raw != original_bytes
    assert result["np_array"].shape == (6, 64, 64, 3)
    with av.open(io.BytesIO(raw)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        audio_frames = list(container.decode(audio=0))
        assert sum(frame.samples for frame in audio_frames) > 0
        assert any(np.any(frame.to_ndarray()) for frame in audio_frames)


@pytest.mark.parametrize("embed_audio", [False, True], ids=["video-only", "with-audio"])
def test_valid_video_cache_is_reused(mocker, synthetic_audio, tmp_path, embed_audio):
    result = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)
    encode = mocker.patch("imageio.get_writer", side_effect=AssertionError("valid cache was regenerated"))

    cached = media.generate_synthetic_video(64, 64, 6, embed_audio=embed_audio, cache_dir=tmp_path)

    encode.assert_not_called()
    assert cached["base64"] == result["base64"]
    assert cached["file_path"] == result["file_path"]
    assert np.array_equal(cached["np_array"], media._np_array_from_mp4_bytes(base64.b64decode(result["base64"])))
