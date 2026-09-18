# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Protocol extensions preserve inheritance and the shared command contract."""

from dataclasses import fields

import pytest

from vllm_omni.protocol.duplex.commands import Commit as DuplexCommit
from vllm_omni.protocol.duplex.events import ErrorEvent, InputCommitted, ItemDeleted, ItemTruncated
from vllm_omni.protocol.realtime.commands import Commit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("event_class", [InputCommitted, ItemDeleted, ItemTruncated])
def test_an_event_subclass_preserves_wire_serialization(event_class):
    CustomEvent = type("CustomEvent", (event_class,), {})

    kwargs = {"event_id": "evt_1", "item_id": "item_1", "details": {"source": "client"}}
    expected = event_class(**kwargs).to_realtime()
    actual = CustomEvent(**kwargs).to_realtime()
    assert actual == expected
    assert actual["item_id"] == "item_1"
    assert actual["event"] == {"source": "client"}


def test_error_subclass_preserves_error_fields_and_extensions():
    class CustomError(ErrorEvent):
        pass

    event = CustomError(
        code="resource_exhausted",
        message="Session capacity reached",
        related_event_id="client_1",
        param="session",
        extra={"retryable": True},
    )
    assert event.to_realtime()["error"] == {
        "type": "rate_limit_error",
        "code": "resource_exhausted",
        "message": "Session capacity reached",
        "event_id": "client_1",
        "param": "session",
        "retryable": True,
    }


def test_nonfinal_commit_is_a_duplex_extension():
    assert {field.name for field in fields(Commit)} == {"event_id"}
    assert DuplexCommit().final is True
    assert DuplexCommit(final=False).final is False
