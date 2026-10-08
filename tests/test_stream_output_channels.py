"""Agent output reaches the terminal exactly once.

Each phase writes streamed deltas to the terminal itself and forwards them to
``stream_callback`` as an additional UI channel. A caller that also prints --
as the CLI used to, via its own ``_stream_to_terminal`` -- duplicates every
token. Keeping the two roles separate is what stops that recurring.
"""

import re
from pathlib import Path

import pytest

from lakegen.phases import orchestrated_discovery


def test_a_delta_is_printed_once_and_forwarded_once(monkeypatch, capsys):
    forwarded = []

    def fake_workflow(*, emit_stream, **_kwargs):
        emit_stream("hello ")
        emit_stream("world")
        emit_stream("")  # empty deltas must reach neither channel
        return "done"

    monkeypatch.setattr(orchestrated_discovery, "run_agent_workflow", fake_workflow)
    monkeypatch.setattr(orchestrated_discovery, "reset_llm_token_usage", lambda _llm: None)
    monkeypatch.setattr(
        orchestrated_discovery, "get_llm_token_usage", lambda _llm: 0
    )

    response, stream, _tokens = orchestrated_discovery._run_tool_free_turn(
        llm=object(),
        system_prompt="s",
        user_prompt="u",
        agent_name="test_agent",
        stream_callback=forwarded.append,
    )

    # The streamed payload is authoritative: some FunctionAgent versions
    # return an empty/wrapper final response after streaming the protocol.
    assert response == "hello world"
    assert stream == "hello world"
    assert forwarded == ["hello ", "world"]
    # Printed exactly once, not once per channel.
    assert capsys.readouterr().out == "hello world"


def test_the_terminal_still_gets_output_with_no_callback(monkeypatch, capsys):
    """service.py calls the phases without a stream_callback."""

    def fake_workflow(*, emit_stream, **_kwargs):
        emit_stream("server side")
        return "done"

    monkeypatch.setattr(orchestrated_discovery, "run_agent_workflow", fake_workflow)
    monkeypatch.setattr(orchestrated_discovery, "reset_llm_token_usage", lambda _llm: None)
    monkeypatch.setattr(orchestrated_discovery, "get_llm_token_usage", lambda _llm: 0)

    orchestrated_discovery._run_tool_free_turn(
        llm=object(), system_prompt="s", user_prompt="u", agent_name="a",
        stream_callback=None,
    )

    assert capsys.readouterr().out == "server side"


def test_the_cli_does_not_add_a_second_printing_channel():
    """The phases own the terminal, so a printing callback would double output."""
    source = Path("src/lakegen/interfaces/cli/ask.py").read_text(encoding="utf-8")

    assert "_stream_to_terminal" not in source
    printing_callbacks = re.findall(
        r"stream_callback=(\w+)", source
    )
    assert printing_callbacks == [], printing_callbacks


@pytest.mark.parametrize(
    "module_path",
    [
        "src/lakegen/discovery/unified.py",
        "src/lakegen/discovery/divided.py",
        "src/lakegen/discovery/orchestrated.py",
    ],
)
def test_every_phase_owns_the_terminal(module_path):
    """If a phase stopped printing, callers passing no callback would go silent."""
    source = Path(module_path).read_text(encoding="utf-8")
    body = re.search(
        r"def emit(?:_stream)?\(delta[^)]*\)[^\n]*\n((?:[ \t]{8,}.*\n|\n)+)", source
    )
    assert body, f"{module_path}: no streaming emit function found"
    assert "print(delta" in body.group(1), f"{module_path}: emit no longer prints"
