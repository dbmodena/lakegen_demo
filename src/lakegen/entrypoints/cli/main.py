"""The ``lakegen`` command: one entry for the commands the package offers.

    lakegen ask [options] "question"      one question, step by step or autonomous
    lakegen config show|validate|diff     the resolved configuration of a file

Each command keeps its own parser; this one only routes to it. Experiment
specs, services, the run store and the retrieval benchmark are the
experiment harness's: ``python -m experiment_runner``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Command = Callable[[list[str]], int | None]


def _ask(rest: list[str]) -> int:
    from lakegen.entrypoints.cli import ask

    ask.main(rest)
    return 0


def _config(rest: list[str]) -> int:
    from lakegen.entrypoints.cli import config as cli

    return cli.main(rest)


COMMANDS: dict[str, tuple[Command, str]] = {
    "ask": (_ask, "answer one question from the terminal"),
    "config": (_config, "show, validate or diff configuration files"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Route ``lakegen <command>`` to the command's own parser; returns its exit code."""
    parser = argparse.ArgumentParser(prog="lakegen", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS),
        help="; ".join(f"{name}: {summary}" for name, (_, summary) in COMMANDS.items()),
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="the command's own arguments")
    args = parser.parse_args(argv)
    command, _summary = COMMANDS[args.command]
    return int(command(list(args.rest)) or 0)


if __name__ == "__main__":
    sys.exit(main())
