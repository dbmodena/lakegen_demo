"""Resolve generated column aliases against exact table schemas."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(value)).casefold()
    return re.findall(r"[a-z0-9]+", normalized)


def _key(value: str) -> str:
    return "".join(_tokens(value))


def resolve_column_name(requested: str, available: Iterable[str]) -> str | None:
    """Return the unique exact schema name matching a generated name.

    Matching is deterministic: exact, case-insensitive, punctuation-insensitive,
    with spaces/underscores/punctuation ignored. Semantic aliases and
    abbreviations are deliberately not guessed. Ambiguous matches are rejected.
    """

    names = list(dict.fromkeys(str(name) for name in available))
    if requested in names:
        return requested
    strategies = (
        lambda value: value.casefold(),
        lambda value: _key(value),
    )
    for strategy in strategies:
        wanted = strategy(requested)
        matches = [name for name in names if strategy(name) == wanted]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


@dataclass(frozen=True)
class ColumnResolutionResult:
    code: str
    replacements: Mapping[str, str]
    unresolved_required: tuple[str, ...]


_COLUMN_METHODS = {
    "agg",
    "drop",
    "drop_duplicates",
    "dropna",
    "groupby",
    "melt",
    "merge",
    "pivot",
    "pivot_table",
    "set_index",
    "sort_values",
}
_COLUMN_KEYWORDS = {
    "by",
    "columns",
    "id_vars",
    "index",
    "left_on",
    "on",
    "right_on",
    "subset",
    "values",
    "value_vars",
}


class _ColumnLiteralResolver(ast.NodeTransformer):
    def __init__(self, available: Sequence[str], generated: set[str] | None = None) -> None:
        self.available = available
        self.generated = generated or set()
        self.replacements: dict[str, str] = {}
        self.unresolved_required: set[str] = set()

    def _replace_constant(self, node: ast.Constant, *, required: bool = False) -> ast.Constant:
        if not isinstance(node.value, str):
            return node
        resolved = resolve_column_name(node.value, self.available)
        if resolved is not None and resolved != node.value:
            self.replacements[node.value] = resolved
            return ast.copy_location(ast.Constant(value=resolved), node)
        if required and resolved is None and node.value not in self.generated:
            self.unresolved_required.add(node.value)
        return node

    def _replace_value(self, node: ast.AST, *, required: bool = False) -> ast.AST:
        if isinstance(node, ast.Constant):
            return self._replace_constant(node, required=required)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            node.elts = [self._replace_value(item, required=required) for item in node.elts]
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        required = any(
            isinstance(target, ast.Name)
            and re.search(r"(?:required|source|input)?_?col(?:umn)?s?$", target.id, re.I)
            for target in node.targets
        )
        if required:
            node.value = self._replace_value(node.value, required=True)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        node.slice = self._replace_value(node.slice)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if method in _COLUMN_METHODS:
            node.args = [self._replace_value(arg) for arg in node.args]
            for keyword in node.keywords:
                if keyword.arg in _COLUMN_KEYWORDS:
                    keyword.value = self._replace_value(keyword.value)
        return node


def resolve_generated_code_columns(
    code: str,
    schemas: Mapping[str, Sequence[str]] | Sequence[str],
) -> ColumnResolutionResult:
    """Rewrite recognizable dataframe column literals to their exact names."""

    if isinstance(schemas, Mapping):
        available = list(
            dict.fromkeys(
                column for columns in schemas.values() for column in map(str, columns)
            )
        )
    else:
        available = list(dict.fromkeys(map(str, schemas)))
    tree = ast.parse(code)
    generated = _generated_column_names(tree)
    resolver = _ColumnLiteralResolver(available, generated)
    resolved = resolver.visit(tree)
    ast.fix_missing_locations(resolved)
    rewritten = ast.unparse(resolved) if resolver.replacements else code
    return ColumnResolutionResult(
        code=rewritten,
        replacements=dict(sorted(resolver.replacements.items())),
        unresolved_required=tuple(sorted(resolver.unresolved_required)),
    )


def _generated_column_names(tree: ast.AST) -> set[str]:
    """Collect explicit dataframe columns created by the generated program.

    This deliberately recognizes only unambiguous write contexts. Merely using
    a string in a read, groupby, filter, or required-column declaration never
    makes it generated and therefore cannot bypass source-schema validation.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    names.add(target.slice.value)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method in {"assign", "agg", "aggregate"}:
            names.update(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None
            )
        elif method == "reset_index":
            for keyword in node.keywords:
                if (
                    keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    names.add(keyword.value.value)
        elif method == "rename":
            for keyword in node.keywords:
                if keyword.arg != "columns" or not isinstance(keyword.value, ast.Dict):
                    continue
                names.update(
                    value.value
                    for value in keyword.value.values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
    return names
