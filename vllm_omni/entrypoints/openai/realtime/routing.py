# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Select a transport before allocating any connection/session state."""

from vllm_omni.entrypoints.realtime.contracts import RealtimeError

PROFILES = ("qwen3-legacy", "openai-realtime")


def select_realtime_route(*, duplex, profile, default_profile, has_duplex, has_ga):
    enabled = duplex is not None and duplex.lower() in {"1", "true", "on"}
    if duplex is not None and duplex.lower() not in {"1", "true", "on", "0", "false", "off"}:
        raise RealtimeError("Invalid duplex selector.", "duplex")
    if enabled:
        if profile is not None:
            raise RealtimeError("duplex and Qwen profile cannot be combined.", "profile", "incompatible_parameters")
        if not has_duplex:
            raise RealtimeError("This model does not support duplex.", "duplex", "unsupported")
        return "duplex"
    if has_duplex and duplex is None:
        if profile is not None:
            raise RealtimeError(
                "A Qwen profile cannot select a native duplex model.", "profile", "incompatible_parameters"
            )
        return "duplex"
    selected = profile if profile is not None else default_profile
    if selected not in PROFILES:
        raise RealtimeError("Unknown realtime profile.", "profile")
    if selected == "openai-realtime":
        if not has_ga:
            raise RealtimeError("This model does not support the Qwen3 GA profile.", "profile", "unsupported")
        return "ga"
    return "legacy"
