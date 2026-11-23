# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

from vllm.logger import init_logger
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.platforms.npu.ascend_warmup_patch import apply_ascend_warmup_patch
from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase
from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)

# Install at import time: this module is imported while the stage worker is
# built, which is earlier than the ascend Triton kernel warmup (that runs
# after load_model in the same process). The guard resolves the SoC per call,
# so installing this early is safe -- the probe only has to be right when a
# warmup actually runs.
apply_ascend_warmup_patch()


class NPUARWorker(OmniWorkerMixin, OmniNPUWorkerBase):
    """NPU AR worker for thinker/talker stages in Omni model."""

    model_runner_cls = NPUOmniARModelRunner

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        logger.info("Using MR v2 NPUOmniARModelRunner.")
        self.model_runner = self.model_runner_cls(self.vllm_config, self.device)
        if (trace_dir := os.environ.get("VLLM_OMNI_A3_THINKER_TRACE_DIR")) and (
            self.vllm_config.model_config.model_stage == "thinker"
        ):
            from vllm_omni.utils.a3_thinker_probe import attach_worker_probe

            attach_worker_probe(self.model_runner, trace_dir, self.rank)
