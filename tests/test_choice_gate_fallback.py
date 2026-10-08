"""A choice gate must survive a websocket reconnect.

A pending ``AskActionMessage`` waits on ``sio.call(..., to=sid)`` bound to the
socket id current when it was sent. A reconnect rebinds the session but not that
awaiting call, so the acknowledgement never arrives while the buttons stay
clickable -- and the click lands on the HTTP action endpoint instead, which 404s
unless a callback is registered for that action name.
"""

import asyncio

import chainlit as cl
import pytest
from chainlit.config import config as chainlit_config

from lakegen.tracing import HumanGate
from lakegen.ui import workflow as wf


class _FakeSession:
    id = "session-under-test"


class _NeverAnsweredAsk:
    """Stands in for an ask whose acknowledgement can never arrive."""

    def __init__(self, *args, **kwargs):
        self.actions = kwargs.get("actions") or []
        self.removed = False

    async def send(self):
        await asyncio.Event().wait()  # never resolves

    async def remove(self):
        self.removed = True


class _Recorder:
    def __init__(self):
        self.calls = []

    def record_approval(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def gate_env(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(wf.cl, "AskActionMessage", _NeverAnsweredAsk)
    monkeypatch.setattr(wf.cl, "context", type("C", (), {"session": _FakeSession()})())
    monkeypatch.setattr(
        wf, "get_session", lambda: type("S", (), {"intervention_recorder": recorder})()
    )
    wf._PENDING_CHOICES.clear()
    yield recorder
    wf._PENDING_CHOICES.clear()


def _action(name, value, gate):
    return cl.Action(name=name, payload={"value": value, "gate": gate}, label=name)


async def _start_gate():
    task = asyncio.create_task(
        wf._ask_choice(
            "Approve?",
            [
                ("approve_selection", "approve", "Approve Selection"),
                ("recalculate_selection", "recalculate", "Recalculate"),
            ],
            phase="discovery",
            gate=HumanGate.DATASET_APPROVAL,
            approved_value="approve",
            remove_after_answer=True,
        )
    )
    for _ in range(200):  # let the gate register its pending future
        await asyncio.sleep(0)
        if wf._PENDING_CHOICES:
            break
    assert wf._PENDING_CHOICES, "gate never registered a pending choice"
    nonce = wf._PENDING_CHOICES[_FakeSession.id][0]
    return task, nonce


@pytest.mark.asyncio
async def test_orphaned_ask_is_answered_through_the_action_endpoint(gate_env):
    task, nonce = await _start_gate()

    # The websocket reply can never arrive; the click arrives over HTTP instead.
    assert await wf._on_choice_action(_action("approve_selection", "approve", nonce)) == ""

    assert await asyncio.wait_for(task, timeout=2) == "approve"
    assert gate_env.calls[0]["approved"] is True
    assert not wf._PENDING_CHOICES, "pending choice was not cleaned up"


@pytest.mark.asyncio
async def test_rejection_travels_the_same_path(gate_env):
    task, nonce = await _start_gate()
    await wf._on_choice_action(_action("recalculate_selection", "recalculate", nonce))

    assert await asyncio.wait_for(task, timeout=2) == "recalculate"
    assert gate_env.calls[0]["approved"] is False


@pytest.mark.asyncio
async def test_a_click_on_a_superseded_prompt_is_refused(gate_env):
    task, nonce = await _start_gate()

    stale = await wf._on_choice_action(
        _action("approve_selection", "approve", "a-different-gate")
    )
    assert stale == "This prompt is no longer active."
    assert not task.done(), "a stale click must not answer the open question"

    await wf._on_choice_action(_action("approve_selection", "approve", nonce))
    assert await asyncio.wait_for(task, timeout=2) == "approve"


@pytest.mark.asyncio
async def test_click_with_no_gate_open_is_refused(gate_env):
    assert (
        await wf._on_choice_action(_action("approve_selection", "approve", "x"))
        == "This prompt is no longer active."
    )


def test_every_choice_action_has_a_registered_callback():
    for name in wf._CHOICE_ACTIONS:
        assert name in chainlit_config.code.action_callbacks, (
            f"{name} would 404 on the HTTP action endpoint"
        )
    # The names must match what the gates actually send (the driver's).
    import re

    from lakegen.interfaces.chainlit import driver

    source = open(driver.__file__, encoding="utf-8").read()
    sent = set(re.findall(r'\("(\w+)", "(?:approve|recalculate|force)", ', source))
    assert sent, "the driver's gates send no choice actions"
    assert sent <= set(wf._CHOICE_ACTIONS), sent - set(wf._CHOICE_ACTIONS)
