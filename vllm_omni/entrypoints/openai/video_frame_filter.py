# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility import for the shared video sampling implementation."""

from vllm_omni.entrypoints.openai.realtime.video import FrameSimilarityFilter

__all__ = ["FrameSimilarityFilter"]
