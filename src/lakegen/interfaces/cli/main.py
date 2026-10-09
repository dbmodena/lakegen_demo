"""The ``lakegen`` command: one entry for the commands the package offers.

    lakegen ask [options] "question"      one question, step by step or autonomous
    lakegen config show|validate|diff     the resolved configuration of a file
    lakegen exp plan|run|status SPEC      an experiment spec against the run store
    lakegen runs index-legacy JOB_DIR...  the old submit scripts' jobs into the store
    lakegen services status|up|down       the services experiments need
    lakegen bench [options]               the retrieval benchmark
    lakegen index [options]               table embeddings into a Solr core

Each command keeps its own parser; this one only routes to it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Command = Callable[[list[str]], int | None]


def _ask(rest: list[str]) -> int:
    from lakegen.interfaces.cli import ask

    ask.main(rest)
    return 0


def _config(rest: list[str]) -> int:
    from lakegen.interfaces.cli import config as cli

    return cli.main(rest)


def _experiments(rest: list[str]) -> int:
    from lakegen.experiments import cli

    return cli.main(rest)


def _runs(rest: list[str]) -> int:
    from lakegen.experiments import cli

    return cli.main(rest)


def _services(rest: list[str]) -> int:
    from lakegen.experiments import cli

    return cli.main(["services", *rest])


def _bench(rest: list[str]) -> int:
    from lakegen.retrieval import benchmark

    return benchmark.main(rest)


def _index(rest: list[str]) -> int:
    from lakegen.interfaces.cli import index

    index.main(rest)
    return 0


COMMANDS: dict[str, tuple[Command, str]] = {
    "ask": (_ask, "answer one question from the terminal"),
    "config": (_config, "show, validate or diff configuration files"),
    "exp": (_experiments, "plan, run or follow an experiment spec"),
    "runs": (_runs, "index the old submit scripts' jobs into the run store"),
    "services": (_services, "status, up or down of the services experiments need"),
    "bench": (_bench, "run the retrieval benchmark"),
    "index": (_index, "index a lake's table embeddings into its Solr core"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lakegen", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command", choices=sorted(COMMANDS),
        help="; ".join(f"{name}: {summary}" for name, (_, summary) in COMMANDS.items()),
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="the command's own arguments")
    args = parser.parse_args(argv)
    command, _summary = COMMANDS[args.command]
    return int(command(list(args.rest)) or 0)


if __name__ == "__main__":
    sys.exit(main())
