"""Ground truth for join/union evidence, taken from OrQa's generated benchmark (no LLM, no network).

Two kinds of label, kept separate because they are not equally trustworthy:

  candidate relationships  OrQa's `merge` / `union` relationships in query_candidates_semantic.json.
                           An LLM proposed the columns and a name-based COMA check accepted them, so
                           recall against them is partly circular.
  executed operations      `.merge(...)` / `pd.concat([...])` calls in the reference code of
                           judge-approved benchmark queries (generated_queries_semantic.json). These
                           actually ran and produced the benchmark answer: the independent evidence.

Data location: $ORQA_DATA_DIR (default /home/bilel/gits/orqa/data), `<portal>/candidates_discovery/`.
SQL joins are not extracted: their keys are wrapped in LOWER(...), REGEXP_REPLACE(...) or `ON TRUE`.
"""

from __future__ import annotations

import ast
import itertools
import json
import os
from pathlib import Path

ORQA_DATA = Path(os.environ.get("ORQA_DATA_DIR", "/home/bilel/gits/orqa/data"))


def candidates_dir(portal: str) -> Path:
    return ORQA_DATA / portal / "candidates_discovery"


def _root_name(expr: ast.AST) -> str | None:
    while True:
        if isinstance(expr, (ast.Attribute, ast.Subscript)):
            expr = expr.value
        elif isinstance(expr, ast.Call):
            expr = expr.func
        else:
            return expr.id if isinstance(expr, ast.Name) else None


def _chain_combines_tables(expr: ast.AST) -> bool:
    while True:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            if expr.func.attr in ("merge", "join", "concat"):
                return True
        if isinstance(expr, (ast.Attribute, ast.Subscript)):
            expr = expr.value
        elif isinstance(expr, ast.Call):
            expr = expr.func
        else:
            return False


def _variable_map(tree: ast.AST, aliases: dict[str, str]) -> dict[str, str]:
    """Variables derived from ONE benchmark table (filtered / copied / aggregated) keep that table's
    key columns; frames built by merge/concat are ambiguous and stay out."""
    varmap = {alias: alias for alias in aliases}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            source = _root_name(node.value)
            if source in varmap and not _chain_combines_tables(node.value):
                varmap[node.targets[0].id] = varmap[source]
    return varmap


def merge_calls(code: str, aliases: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Direct `A.merge(B, on / left_on+right_on)` calls between benchmark tables, with `how`."""
    out: list[dict] = []
    skipped: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out, ["syntax"]
    varmap = _variable_map(tree, aliases)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "merge"
        ):
            continue
        func = node.func
        if isinstance(func.value, ast.Name) and func.value.id in ("pd", "pandas"):
            if len(node.args) < 2:
                skipped.append("pd.merge args")
                continue
            left, right = node.args[0], node.args[1]
        else:
            left = func.value
            right = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "right"), None
            )
        left_name, right_name = _root_name(left), _root_name(right) if right is not None else None
        if (
            left_name not in varmap
            or right_name not in varmap
            or _chain_combines_tables(left)
            or _chain_combines_tables(right)
        ):
            skipped.append("merged/combined frame")
            continue
        if varmap[left_name] == varmap[right_name]:
            skipped.append("self-join")
            continue
        keywords: dict[str, list] = {}
        for keyword in node.keywords:
            if keyword.arg in ("on", "left_on", "right_on"):
                try:
                    value = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    value = None
                keywords[keyword.arg] = [value] if isinstance(value, str) else value
        if keywords.get("on"):
            left_on = right_on = keywords["on"]
        elif keywords.get("left_on") and keywords.get("right_on"):
            left_on, right_on = keywords["left_on"], keywords["right_on"]
        else:
            skipped.append("non-literal keys")
            continue
        how = next(
            (str(getattr(k.value, "value", "inner")) for k in node.keywords if k.arg == "how"),
            "inner",
        )
        out.append({
            "left": varmap[left_name], "right": varmap[right_name],
            "left_on": list(left_on), "right_on": list(right_on), "how": how,
        })
    return out, skipped


def concat_groups(code: str, aliases: dict[str, str]) -> list[list[str]]:
    """Alias groups combined by `pd.concat([...])` (projected columns are not tracked)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    varmap = _variable_map(tree, aliases)
    groups: list[list[str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "concat"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            names = [varmap.get(_root_name(element)) for element in node.args[0].elts]
            names = [n for n in names if n]
            if len(set(names)) >= 2:
                groups.append(sorted(set(names)))
    return groups


def _executed_queries(portal: str):
    """(aliases, code) for every judge-approved multi-table pandas query."""
    generated = json.load(open(candidates_dir(portal) / "generated_queries_semantic.json"))
    for group in generated["PANDAS"]["multi_table"].values():
        aliases = group["_meta"]["tables"]
        for key, query in group.items():
            if key != "_meta" and isinstance(query, dict):
                yield aliases, query.get("code") or ""


def join_labels(portal: str) -> dict[tuple, dict]:
    """{(left_table, right_table, left_on, right_on): {"source": cand | cand+exec | exec, "how": ...}}"""
    labels: dict[tuple, dict] = {}
    candidates = json.load(open(candidates_dir(portal) / "query_candidates_semantic.json"))
    for group in candidates:
        for relationship in group["relationships"]:
            if relationship["type"] not in ("merge", "merge_correlation"):
                continue
            left = group["aliases"][relationship["left"]]
            right = group["aliases"][relationship["right"]]
            key = (left, right, tuple(relationship["left_on"]), tuple(relationship["right_on"]))
            labels.setdefault(key, {"source": "cand", "how": relationship.get("how", "inner")})
    for aliases, code in _executed_queries(portal):
        for call in merge_calls(code, aliases)[0]:
            left, right = aliases[call["left"]], aliases[call["right"]]
            key = (left, right, tuple(call["left_on"]), tuple(call["right_on"]))
            twin = (right, left, key[3], key[2])
            if key in labels:
                labels[key]["source"] = "cand+exec"
            elif twin in labels:
                labels[twin]["source"] = "cand+exec"
            else:
                labels[key] = {"source": "exec", "how": call["how"]}
    return labels


def union_positives(portal: str) -> tuple[dict, set, list[str]]:
    """(positives, related_pairs, candidate_tables); positives map an unordered table pair to
    {"source": cand | cand+exec | exec, "L", "R", "labelled": [(left_col, right_col), ...]}."""
    positives: dict[tuple, dict] = {}
    related: set[tuple] = set()
    tables: set[str] = set()
    candidates = json.load(open(candidates_dir(portal) / "query_candidates_semantic.json"))
    for group in candidates:
        tables.update(group["aliases"].values())
        for relationship in group["relationships"]:
            left = group["aliases"][relationship["left"]]
            right = group["aliases"][relationship["right"]]
            pair = tuple(sorted((left, right)))
            related.add(pair)
            if relationship["type"] == "union":
                entry = positives.setdefault(
                    pair, {"source": "cand", "L": left, "R": right, "labelled": []}
                )
                entry["labelled"] += list(zip(relationship["left_cols"], relationship["right_cols"]))
    for aliases, code in _executed_queries(portal):
        for group in concat_groups(code, aliases):
            for a, b in itertools.combinations([aliases[x] for x in group], 2):
                pair = tuple(sorted((a, b)))
                if pair in positives:
                    positives[pair]["source"] = "cand+exec"
                else:
                    positives[pair] = {"source": "exec", "L": a, "R": b, "labelled": []}
    return positives, related, sorted(tables)
