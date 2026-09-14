# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guards for the duplex-only API server (unified full-duplex framework).

A model whose pipeline declares ``duplex_plugin`` is served through
``DuplexOmni`` and nothing else: sessions run over ``/v1/realtime?duplex=1``
(alias ``/v1/duplex``) and every turn-based route reports "not available".
These tests lock that shape, which ``test_api_server_guards.py`` covers for the
diffusion and multi-stage servers:

* the pipeline probe that decides duplex vs turn-based at startup,
* the ``app.state`` snapshot of a duplex server (what is wired, what stays
  ``None``),
* both duplex websocket routes reaching the session handler, and
  ``/v1/realtime`` without the flag still reporting the turn-based Realtime API
  as unavailable,
* the startup-warmup gate that holds real clients while the warmup session runs.

CPU-only: no engine is started; ``DuplexOmni`` is subclassed with just the
accessors the app-state wiring reads.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import State
from starlette.websockets import WebSocketDisconnect

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni
from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ``app.state`` of a duplex-only server. Presence is not enough: a route whose
# handler is missing answers "not available", so "not wired" is written as None.
_DUPLEX_APP_STATE_KEYS = {
    "engine_client",
    "log_stats",
    "args",
    "sleeping_stages",
    "stage_configs",
    "vllm_config",
    "diffusion_engine",
    "openai_serving_models",
    "serving_tokenization",
    "serving_tokens",
    "online_renderer",
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_completion",
    "openai_serving_responses",
    "openai_serving_embedding",
    "openai_serving_pooling",
    "openai_serving_classification",
    "openai_serving_scores",
    "openai_serving_transcription",
    "openai_serving_translation",
    "openai_serving_speech",
    "openai_serving_audio_generate",
    "openai_serving_video",
    "openai_streaming_speech",
    "openai_streaming_video",
    "openai_streaming_video_output",
    "openai_serving_realtime",
    "openai_serving_realtime_robot",
    "anthropic_serving_messages",
    "openai_serving_duplex",
    "enable_server_load_tracking",
    "server_load_metrics",
}
#: Every turn-based service, plus the Realtime route that is not the duplex one.
_DUPLEX_MUST_BE_NONE = _DUPLEX_APP_STATE_KEYS - {
    "engine_client",
    "log_stats",
    "args",
    "sleeping_stages",
    "stage_configs",
    "vllm_config",
    "openai_serving_models",
    "openai_serving_duplex",
    "enable_server_load_tracking",
    "server_load_metrics",
}
_DUPLEX_MUST_BE_WIRED = {
    "engine_client",
    "args",
    "stage_configs",
    "vllm_config",
    "openai_serving_models",
    "openai_serving_duplex",
}
#: A duplex model is a turn-based model underneath, so it keeps
#: ``/v1/chat/completions`` when its pipeline generates (RFC #7181 D9). Chat and
#: the renderer it needs are the only turn-based services that may be wired.
_CHAT_WHEN_GENERATING = {"openai_serving_chat", "online_renderer"}


class _FakeDuplexOmni(DuplexOmni):
    """A ``DuplexOmni`` without an engine: only what the app-state wiring reads."""

    def __init__(self, supported_tasks: tuple[str, ...] = ("generate",)) -> None:
        self.model = "demo-duplex-model"
        self._supported_tasks = supported_tasks
        self._stage_configs = [object(), object(), object()]
        self._vllm_config = SimpleNamespace(
            lora_config=None,
            model_config=SimpleNamespace(),
            parallel_config=SimpleNamespace(_api_process_rank=0),
        )
        self._duplex_session_config = DuplexSessionRuntimeConfig()

    async def get_supported_tasks(self) -> tuple[str, ...]:
        return self._supported_tasks

    async def get_tokenizer(self):
        return None

    @property
    def renderer(self):
        """Read when building the chat serving a generating duplex model keeps."""
        return object()

    @property
    def model_config(self):
        return SimpleNamespace()

    async def get_vllm_config(self):
        return self._vllm_config

    @property
    def stage_configs(self) -> list[object]:
        return self._stage_configs

    @property
    def duplex_session_config(self) -> DuplexSessionRuntimeConfig:
        return self._duplex_session_config


class _FakeModels:
    def __init__(self, *args, **kwargs) -> None:
        self.base_model_paths = kwargs.get("base_model_paths") or []


def _minimal_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        model="demo-duplex-model",
        served_model_name=None,
        disable_log_stats=True,
        enable_log_requests=False,
        max_log_len=None,
        enable_server_load_tracking=False,
        trust_remote_code=True,
        deploy_config=None,
        # Read by the chat serving a generating duplex model keeps.
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        default_chat_template_kwargs=None,
        response_role="assistant",
        return_tokens_as_token_ids=False,
        enable_auto_tool_choice=False,
        exclude_tools_when_tool_choice_none=False,
        tool_call_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        enable_log_deltas=False,
        log_error_stack=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------- #
# Startup: which entrypoint class serves the model                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pipeline_config", "expected"),
    [
        (SimpleNamespace(duplex_plugin="pkg.mod.Plugin"), True),
        (SimpleNamespace(duplex_plugin=None), False),
        (SimpleNamespace(duplex_plugin=""), False),
        (SimpleNamespace(), False),
        (None, False),
    ],
)
def test_duplex_model_probe_follows_the_pipeline_plugin(monkeypatch, pipeline_config, expected: bool) -> None:
    """``duplex_plugin`` alone decides whether ``serve`` builds ``DuplexOmni``."""
    seen: dict[str, object] = {}

    def fake_get_pipeline_config(*, model, trust_remote_code, deploy_config_path=None, **_kwargs):
        seen.update(model=model, trust_remote_code=trust_remote_code, deploy_config_path=deploy_config_path)
        return pipeline_config

    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", fake_get_pipeline_config)

    kwargs = {"trust_remote_code": True, "deploy_config": "deploy.yaml"}
    assert api_server._is_duplex_model("demo-duplex-model", kwargs) is expected
    assert seen == {
        "model": "demo-duplex-model",
        "trust_remote_code": True,
        "deploy_config_path": "deploy.yaml",
    }


def test_duplex_model_probe_propagates_resolution_errors(monkeypatch) -> None:
    """A pipeline that cannot be resolved must fail startup, not fall back to turn-based."""

    def fake_get_pipeline_config(**_kwargs):
        raise ValueError("needs trust_remote_code=True")

    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", fake_get_pipeline_config)

    with pytest.raises(ValueError, match="trust_remote_code"):
        api_server._is_duplex_model("demo-duplex-model", {})


# --------------------------------------------------------------------------- #
# app.state of a duplex-only server                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_duplex_app_state_wires_the_session_handler_and_chat(monkeypatch) -> None:
    """Lock the duplex ``app.state``: sessions plus chat, and nothing else.

    Fails if a turn-based service other than chat is wired into a duplex server
    (its route would answer instead of reporting "not available"), or if a key
    the routes read disappears.
    """
    monkeypatch.setattr(api_server, "OpenAIServingModels", _FakeModels)
    monkeypatch.setattr(api_server, "OnlineRenderer", lambda **kwargs: object())
    monkeypatch.setattr(api_server, "OmniOpenAIServingChat", lambda **kwargs: object())
    engine = _FakeDuplexOmni()
    state = State()

    await api_server.omni_init_app_state(engine, state, _minimal_args())

    present = {key for key in _DUPLEX_APP_STATE_KEYS if hasattr(state, key)}
    assert present == _DUPLEX_APP_STATE_KEYS
    not_wired = sorted(key for key in _DUPLEX_MUST_BE_WIRED if getattr(state, key) is None)
    assert not not_wired, f"duplex app.state keys registered but not wired: {not_wired}"
    unexpectedly_set = sorted(
        key for key in _DUPLEX_MUST_BE_NONE - _CHAT_WHEN_GENERATING if getattr(state, key) is not None
    )
    assert not unexpectedly_set, f"turn-based services wired into a duplex server: {unexpectedly_set}"
    assert isinstance(state.openai_serving_duplex, OmniDuplexSessionHandler)
    assert state.openai_serving_chat is not None, "a generating duplex model must keep /v1/chat/completions"
    assert state.engine_client is engine
    assert state.vllm_config is engine._vllm_config


async def test_a_duplex_model_that_cannot_generate_gets_no_chat_route(monkeypatch) -> None:
    """Support follows the pipeline's tasks, not the model's name.

    A duplex pipeline that does not report ``generate`` has nothing to answer a
    chat request with, so the route stays off and the startup log says why.
    """
    monkeypatch.setattr(api_server, "OpenAIServingModels", _FakeModels)
    engine = _FakeDuplexOmni(supported_tasks=())
    state = State()

    await api_server.omni_init_app_state(engine, state, _minimal_args())

    assert state.openai_serving_chat is None
    assert isinstance(state.openai_serving_duplex, OmniDuplexSessionHandler)


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #


class _RecordingHandler:
    """Stands in for ``OmniDuplexSessionHandler`` on the app state."""

    def __init__(self) -> None:
        self.queries: list[dict[str, str]] = []

    async def handle_realtime_session(self, websocket) -> None:
        await websocket.accept()
        self.queries.append(dict(websocket.query_params))
        await websocket.send_json({"type": "session.created", "session": {"id": "duplex-test"}})
        await websocket.close()


def _duplex_app(handler: object) -> FastAPI:
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.openai_serving_duplex = handler
    # A duplex server serves no turn-based Realtime route.
    app.state.openai_serving_realtime = None
    return app


@pytest.mark.parametrize("path", ["/v1/duplex", "/v1/realtime?duplex=1"])
def test_both_duplex_routes_reach_the_session_handler(path: str) -> None:
    """``/v1/duplex`` is an alias of ``/v1/realtime?duplex=1``: same handler, same protocol."""
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(path) as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert len(handler.queries) == 1


@pytest.mark.parametrize("flag", ["1", "true", "on"])
def test_realtime_route_switches_to_duplex_only_for_the_documented_flag_values(flag: str) -> None:
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(f"/v1/realtime?duplex={flag}") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert handler.queries == [{"duplex": flag}]


@pytest.mark.parametrize("query", ["", "?model=openbmb%2FMiniCPM-o-4_5"])
def test_realtime_without_a_duplex_flag_reaches_the_session_handler(query: str) -> None:
    """A bare connection to a duplex deployment is a duplex session.

    ``?duplex=1`` is the spelling the server advertises, but it is not a
    requirement: a deployment that declares a duplex plugin answers plain
    ``/v1/realtime`` too, so a stock Realtime client needs no vendor query
    parameter.
    """
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(f"/v1/realtime{query}") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert len(handler.queries) == 1


@pytest.mark.parametrize("query", ["?duplex=0", "?duplex=false", "?duplex=off"])
def test_realtime_opted_out_of_duplex_falls_through_to_the_turn_based_route(query: str) -> None:
    """An explicit opt-out selects the legacy handler, which a duplex server does not mount."""
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(f"/v1/realtime{query}") as websocket:
            assert websocket.receive_json() == {
                "type": "error",
                "error": "Realtime API is not available",
                "code": "unsupported",
            }
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert handler.queries == []


# --------------------------------------------------------------------------- #
# Startup warmup gate                                                         #
# --------------------------------------------------------------------------- #


def _warmup_websocket(warmup_done: asyncio.Event | None, query: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(duplex_warmup_done=warmup_done)),
        query_params=query,
    )


@pytest.mark.asyncio
async def test_warmup_gate_holds_clients_until_the_warmup_session_finishes() -> None:
    warmup_done = asyncio.Event()
    websocket = _warmup_websocket(warmup_done, {})

    waiting = asyncio.create_task(api_server._wait_for_duplex_warmup(websocket))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiting), timeout=0.05)

    warmup_done.set()
    await asyncio.wait_for(waiting, timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("warmup_done", "query"),
    [
        (None, {}),
        (asyncio.Event(), {"vllm_omni_warmup": "1"}),
    ],
    ids=["no-warmup-configured", "the-warmup-connection-itself"],
)
async def test_warmup_gate_lets_the_warmup_connection_and_plain_servers_through(warmup_done, query) -> None:
    websocket = _warmup_websocket(warmup_done, query)

    await asyncio.wait_for(api_server._wait_for_duplex_warmup(websocket), timeout=1.0)
