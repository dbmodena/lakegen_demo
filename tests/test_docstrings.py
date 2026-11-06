"""Every multi-line function docstring documents its arguments.

Ruff checks the rest of the Google style (pyproject ``[tool.ruff.lint]``): that
public code has docstrings, that an ``Args:`` section names every argument, and
that returns, yields and raises are documented. It cannot require the ``Args:``
section itself, so this test does, for private functions too, as ruff's
pydoclint checks do. A one-line docstring is Google style's allowance for a
function whose name and signature say it all; a longer one says what each
argument is.
"""

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = ("src", "pneuma_engine", "solr_builder", "experiment_runner", "scripts")


def _sources() -> list[Path]:
    # The files git would commit: tracked or new, never ignored ones.
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *CHECKED],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        ROOT / name
        for name in sorted(set(listed))
        if name.endswith(".py") and "tests" not in Path(name).parts and (ROOT / name).exists()
    ]


def _missing_args_sections(path: Path) -> list[str]:
    missing = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        docstring = ast.get_docstring(node) or ""
        arguments = [
            argument.arg
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            if argument.arg not in ("self", "cls")
        ]
        if "\n" in docstring.strip() and arguments and "Args:" not in docstring:
            missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")
    return missing


def test_multi_line_docstrings_document_their_arguments():
    missing = [entry for path in _sources() for entry in _missing_args_sections(path)]

    assert not missing, "multi-line docstrings without an Args: section:\n" + "\n".join(missing)
