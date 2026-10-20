# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from vllm.config import DeviceConfig, VllmConfig
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _codec(mocker, dtype, device="cuda"):
    config = Qwen3OmniMoeCode2WavConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        decoder_dim=32,
        upsample_rates=[2, 2],
        upsampling_ratios=[2],
        codebook_size=16,
        num_quantizers=2,
    )
    vllm_config = mocker.Mock(
        spec=VllmConfig,
        model_config=mocker.Mock(spec=OmniModelConfig, hf_config=config, stage_connector_config=None),
        device_config=mocker.Mock(spec=DeviceConfig, device=torch.device(device)),
        quant_config=None,
    )
    # Simulate the loader's dtype context, but keep all tensors on CPU.
    with set_default_torch_dtype(dtype):
        return Qwen3OmniMoeCode2Wav(vllm_config=vllm_config).eval()


@pytest.fixture
def blackwell_cudnn(mocker):
    mocker.patch.object(current_omni_platform, "is_cuda", return_value=True)
    mocker.patch.object(torch.backends.cudnn, "version", return_value=92000)
    mocker.patch.object(torch.accelerator, "current_device_index", return_value=1)
    return mocker.patch.object(current_omni_platform, "get_device_capability", return_value=DeviceCapability(10, 3))


@pytest.mark.parametrize("capability", [DeviceCapability(10, 0), DeviceCapability(10, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_affected_codec_uses_float32(mocker, blackwell_cudnn, capability, dtype):
    blackwell_cudnn.return_value = capability
    with set_default_torch_dtype(dtype):
        model = _codec(mocker, dtype)
        # A nested Code2Wav copy must not change the enclosing Talker's dtype.
        assert torch.get_default_dtype() == dtype
        assert torch.nn.Linear(2, 2).weight.dtype == dtype

    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert model.code_offset.dtype == torch.long
    # Device 0 can be a different GPU: query the worker's selected device.
    blackwell_cudnn.assert_called_once_with(1)


@pytest.mark.parametrize(
    ("cuda", "device", "capability", "cudnn", "dtype"),
    [
        (False, "cpu", DeviceCapability(10, 3), 92000, torch.bfloat16),
        (True, "cpu", DeviceCapability(10, 3), 92000, torch.bfloat16),
        (True, "cuda", DeviceCapability(9, 0), 92000, torch.bfloat16),
        (True, "cuda", DeviceCapability(12, 0), 92000, torch.bfloat16),
        (True, "cuda", None, 92000, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), None, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 91900, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 92100, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 92400, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 92000, torch.float32),
    ],
)
def test_other_environments_keep_dtype(mocker, blackwell_cudnn, cuda, device, capability, cudnn, dtype):
    mocker.patch.object(current_omni_platform, "is_cuda", return_value=cuda)
    mocker.patch.object(torch.backends.cudnn, "version", return_value=cudnn)
    blackwell_cudnn.return_value = capability
    model = _codec(mocker, dtype, device)

    assert {p.dtype for p in model.parameters()} == {dtype}


def test_explicit_device_index_takes_precedence(mocker, blackwell_cudnn):
    model = _codec(mocker, torch.bfloat16, torch.device("cuda:2"))

    blackwell_cudnn.assert_called_once_with(2)
    assert model.code_embedding.weight.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fallback_loads_checkpoint_and_matches_explicit_float32(mocker, blackwell_cudnn, dtype):
    torch.manual_seed(8935)
    reference = _codec(mocker, torch.float32)
    checkpoint = [("code2wav." + name, p.to(dtype)) for name, p in reference.named_parameters()]
    model = _codec(mocker, dtype)
    assert model.load_weights(checkpoint) == reference.load_weights(checkpoint)
    model.precompute_snake_caches()
    reference.precompute_snake_caches()

    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert all(b.dtype == torch.float32 for b in model.buffers() if b.is_floating_point())
    for frames in (4, 29, 50):
        codes = torch.randint(0, 16, (1, 2, frames))
        actual = model(codes)
        assert actual.dtype == torch.float32
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference(codes), rtol=0, atol=0)
