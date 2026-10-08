"""Inspect experiment configuration files, until the ``lakegen`` command exists.

    python -m lakegen.config.cli show FILE [--format flat|v2] [--set KEY=VALUE ...]
    python -m lakegen.config.cli validate FILE [FILE ...]
    python -m lakegen.config.cli diff FILE_A FILE_B

``show`` prints the resolved configuration (every default applied), in the
flat schema or the nested layout; ``validate`` loads each file and reports the
first error; ``diff`` lists the settings two files resolve differently.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from lakegen.config.files import ConfigFileError, parse_set_values, to_v2


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def _resolved(path: Path, set_values: list[str] | None = None) -> dict[str, Any]:
    from lakegen.experiment_config import load_experiment_config  # the schema; imported late

    return load_experiment_config(path, overrides=parse_set_values(set_values)).model_dump(mode="json")


def show(path: Path, layout: str, set_values: list[str] | None) -> int:
    flat = _resolved(path, set_values)
    if layout == "v2":
        document, dropped = to_v2(flat)
        for key, value in dropped.items():
            print(f"# not carried by the nested layout for this retrieval mode: {key} = {value!r}")
        print(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), end="")
    else:
        print(yaml.safe_dump(flat, sort_keys=False, allow_unicode=True), end="")
    return 0


def validate(paths: list[Path]) -> int:
    failures = 0
    for path in paths:
        try:
            _resolved(path)
        except (ConfigFileError, ValueError, OSError) as exc:
            failures += 1
            print(f"error  {path}: {exc}")
        else:
            print(f"ok     {path}")
    return 1 if failures else 0


def diff(left: Path, right: Path) -> int:
    before, after = _flatten(_resolved(left)), _flatten(_resolved(right))
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    for key in changed:
        print(f"{key}: {before.get(key, '<absent>')!r} -> {after.get(key, '<absent>')!r}")
    print(f"{len(changed)} settings differ")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lakegen.config.cli", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show_command = commands.add_parser("show", help="print the resolved configuration")
    show_command.add_argument("file", type=Path)
    show_command.add_argument("--format", choices=("flat", "v2"), default="flat")
    show_command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    validate_command = commands.add_parser("validate", help="load each file and report errors")
    validate_command.add_argument("files", nargs="+", type=Path)
    diff_command = commands.add_parser("diff", help="settings two files resolve differently")
    diff_command.add_argument("left", type=Path)
    diff_command.add_argument("right", type=Path)
    args = parser.parse_args(argv)
    if args.command == "show":
        return show(args.file, args.format, args.set)
    if args.command == "validate":
        return validate(args.files)
    return diff(args.left, args.right)


if __name__ == "__main__":
    sys.exit(main())
