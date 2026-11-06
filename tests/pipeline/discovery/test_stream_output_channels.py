"""Agent output reaches the terminal exactly once.

Each phase writes streamed deltas to the terminal itself and forwards them to
``stream_callback`` as an additional UI channel. A caller that also prints --
as the CLI used to, via its own ``_stream_to_terminal`` -- duplicates every
token. Keeping the two roles separate is what stops that recurring.
"""

import re
from pathlib import Path

import pytest

from lakegen.pipeline.discovery.divided import streamer


def test_a_delta_is_printed_once_and_forwarded_once(capsys):
    forwarded = []
    emit = streamer(forwarded.append)
    for delta in ("hello ", "world", ""):  # empty deltas must reach neither channel
        emit(delta)

    assert forwarded == ["hello ", "world"]
    # Printed exactly once, not once per channel.
    assert capsys.readouterr().out == "hello world"


def test_the_terminal_still_gets_output_with_no_callback(capsys):
    """service.py calls the phases without a stream_callback."""
    streamer(None)("server side")

    assert capsys.readouterr().out == "server side"


def test_the_cli_does_not_add_a_second_printing_channel():
    """The phases own the terminal, so a printing callback would double output."""
    source = Path("src/lakegen/entrypoints/cli/ask.py").read_text(encoding="utf-8")

    assert "_stream_to_terminal" not in source
    printing_callbacks = re.findall(r"stream_callback=(\w+)", source)
    assert printing_callbacks == [], printing_callbacks


@pytest.mark.parametrize("module_path", ["src/lakegen/pipeline/discovery/divided.py"])
def test_every_phase_owns_the_terminal(module_path):
    """If a phase stopped printing, callers passing no callback would go silent (the unified agent streams through divided's)."""
    source = Path(module_path).read_text(encoding="utf-8")
    body = re.search(
        r"def emit(?:_stream)?\((?:self, )?delta[^)]*\)[^\n]*\n((?:[ \t]{8,}.*\n|\n)+)", source
    )
    assert body, f"{module_path}: no streaming emit function found"
    assert "print(delta" in body.group(1), f"{module_path}: emit no longer prints"
