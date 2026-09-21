"""Value-grounding check: the direct fix for the motivating bug. AST-walks
the ACTUALLY GENERATED code for Compare/.isin/.eq()-family literal constants
(pure-AST, no pandas dependency in the collection step, modeled on
lakegen.column_resolution's _ColumnLiteralResolver pattern), cross-checks
each against real observed column values. Neither LakeGen nor OrQa had this
before it was built and validated here -- confirmed by direct exploration of
both codebases.

Validated over four rounds of live 100-question scratchpad testing. Two real
regressions were caught and fixed during that validation, both covered
below: a malformed-header table where a date column was silently mis-parsed
to 100% NaT (`check_date_parse_sanity`), and a `.eq()`-method-call filter the
AST collector originally missed because it only recognized `==`/`!=`/etc. as
`ast.Compare` nodes, not the equivalent pandas method-call form
(`_root_column`'s chain-unwrapping + the `_METHOD_OP` table below).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import pandas as pd

from lakegen.distribution_check import _detect_encoded_span


@dataclass
class LiteralComparison:
    column: str
    operator: str  # Eq | NotEq | Gt | Lt | GtE | LtE | In (isin)
    literal: object
    lineno: int


class _LiteralComparisonCollector(ast.NodeVisitor):
    """Out of scope for v1 (deliberately, not silently): comparisons against
    variables, other columns, function-call results, or .query() string
    expressions."""

    def __init__(self) -> None:
        self.comparisons: list[LiteralComparison] = []

    @staticmethod
    def _subscript_column(node: ast.AST) -> str | None:
        # df['col'] or df["col"]  -> a Subscript whose slice is a string Constant
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                return sl.value
        return None

    @staticmethod
    def _root_column(node: ast.AST, _depth: int = 0) -> str | None:
        """Walk back through a chain like
        df['col'].astype(str).str.strip().str.lower() to find the underlying
        df['col'] subscript -- catches filters like `.eq('open')` on a chain
        rooted at df['DEFECT'], semantically identical to df['DEFECT'] ==
        'open' but invisible to visit_Compare alone since it's a method
        call, not a Compare node. Type/case-transform methods (astype/
        str.strip/str.lower/str.upper/...) don't change WHICH column is
        referenced, so unwrapping through them is safe."""
        if _depth > 12:
            return None
        col = _LiteralComparisonCollector._subscript_column(node)
        if col is not None:
            return col
        if isinstance(node, ast.Call):
            return _LiteralComparisonCollector._root_column(node.func, _depth + 1)
        if isinstance(node, ast.Attribute):
            return _LiteralComparisonCollector._root_column(node.value, _depth + 1)
        return None

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and len(node.comparators) == 1:
            left, op, right = node.left, node.ops[0], node.comparators[0]
            op_name = type(op).__name__
            if op_name in ("Eq", "NotEq", "Gt", "Lt", "GtE", "LtE"):
                col = self._root_column(left)
                lit_node = right
                if col is None:
                    col = self._root_column(right)
                    lit_node = left
                if col is not None and isinstance(lit_node, ast.Constant):
                    self.comparisons.append(LiteralComparison(col, op_name, lit_node.value, node.lineno))
        self.generic_visit(node)

    _METHOD_OP = {"eq": "Eq", "ne": "NotEq", "gt": "Gt", "lt": "Lt", "ge": "GtE", "le": "LtE"}

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            # df['col'].isin([...]) / .astype(str)...isin([...])
            if attr == "isin" and node.args and isinstance(node.args[0], (ast.List, ast.Tuple, ast.Set)):
                col = self._root_column(node.func.value)
                if col is not None:
                    literals = [elt.value for elt in node.args[0].elts if isinstance(elt, ast.Constant)]
                    for lit in literals:
                        self.comparisons.append(LiteralComparison(col, "In", lit, node.lineno))
            # df['col'].eq('x') / .ne(...) / .gt(...) / ... -- the pandas
            # method-call equivalents of ==, !=, >, <, >=, <=.
            elif attr in self._METHOD_OP and node.args and isinstance(node.args[0], ast.Constant):
                col = self._root_column(node.func.value)
                if col is not None:
                    self.comparisons.append(
                        LiteralComparison(col, self._METHOD_OP[attr], node.args[0].value, node.lineno)
                    )
        self.generic_visit(node)


def extract_literal_comparisons(code: str) -> list[LiteralComparison]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    collector = _LiteralComparisonCollector()
    collector.visit(tree)
    return collector.comparisons


@dataclass
class GroundingViolation:
    column: str
    operator: str
    literal: object
    confidence: str  # HIGH | SOFT
    message: str


def check_value_grounding(comparisons: list[LiteralComparison], df: pd.DataFrame) -> list[GroundingViolation]:
    violations: list[GroundingViolation] = []
    for comp in comparisons:
        if comp.column not in df.columns:
            continue  # column-name resolution is a separate, already-existing check
        series = df[comp.column]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if comp.operator in ("Eq", "NotEq", "In"):
            # Hard presence check for equality-style comparisons.
            span = _detect_encoded_span(series)
            if span and isinstance(comp.literal, (int, str)) and len(str(comp.literal)) == 4 and str(comp.literal).isdigit():
                violations.append(GroundingViolation(
                    comp.column, comp.operator, comp.literal, "HIGH",
                    f"{comp.column!r} holds fiscal-year-span codes like {span['example']!r}, but "
                    f"the code compares it to bare year {comp.literal!r} which never appears -- "
                    f"did you mean the span code for that year?",
                ))
                continue
            normalized_literal = str(comp.literal).strip().casefold() if not is_numeric else comp.literal
            if is_numeric:
                present = (series.dropna() == comp.literal).any()
            else:
                present = series.dropna().astype(str).str.strip().str.casefold().eq(normalized_literal).any()
            if not present:
                violations.append(GroundingViolation(
                    comp.column, comp.operator, comp.literal, "HIGH",
                    f"{comp.column!r} is compared to {comp.literal!r}, but that value never "
                    f"appears in the column's real observed values.",
                ))
        elif is_numeric and comp.operator in ("Gt", "Lt", "GtE", "LtE"):
            non_null = series.dropna()
            if non_null.empty:
                continue
            lo, hi = float(non_null.min()), float(non_null.max())
            margin = (hi - lo) * 0.5 if hi > lo else max(abs(hi), 1.0)
            try:
                lit = float(comp.literal)
            except (TypeError, ValueError):
                continue
            if lit < lo - margin or lit > hi + margin:
                violations.append(GroundingViolation(
                    comp.column, comp.operator, comp.literal, "SOFT",
                    f"{comp.column!r} ranges {lo}-{hi} in the real data; the filter constant "
                    f"{comp.literal!r} is far outside that range -- likely wrong, but could be "
                    f"an intentionally-empty-result threshold.",
                ))
    return violations


class _DateParseCollector(ast.NodeVisitor):
    """Finds pd.to_datetime(df['col'], format=..., ...) calls -- a DIFFERENT,
    complementary bug shape from direct literal comparisons: the code isn't
    comparing to a wrong VALUE, it's parsing the WRONG COLUMN as a date (or
    the right column with the wrong format string), which silently produces
    all-NaT and makes every downstream date filter match nothing. A
    direct-comparison check (extract_literal_comparisons) can't see this on
    its own: the downstream filter is typically `.dt.year == 2018`, a
    chained accessor, not `df[col] == literal`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, int]] = []  # (column, format, lineno)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_to_datetime = (
            (isinstance(func, ast.Attribute) and func.attr == "to_datetime")
            or (isinstance(func, ast.Name) and func.id == "to_datetime")
        )
        if is_to_datetime and node.args:
            col = _LiteralComparisonCollector._subscript_column(node.args[0])
            if col is not None:
                fmt = None
                for kw in node.keywords:
                    if kw.arg == "format" and isinstance(kw.value, ast.Constant):
                        fmt = kw.value.value
                self.calls.append((col, fmt, node.lineno))
        self.generic_visit(node)


def check_date_parse_sanity(code: str, df: pd.DataFrame) -> list[GroundingViolation]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    collector = _DateParseCollector()
    collector.visit(tree)
    violations: list[GroundingViolation] = []
    for col, fmt, _lineno in collector.calls:
        if col not in df.columns:
            continue
        source = df[col]
        non_null_source = int(source.notna().sum())
        if non_null_source == 0:
            continue
        try:
            parsed = pd.to_datetime(source, format=fmt, errors="coerce") if fmt else pd.to_datetime(source, errors="coerce")
        except Exception:  # noqa: BLE001
            continue
        nat_ratio = parsed.isna().sum() / non_null_source
        if nat_ratio > 0.9:
            violations.append(GroundingViolation(
                col, "to_datetime", fmt, "HIGH",
                f"pd.to_datetime(df[{col!r}], format={fmt!r}) produces "
                f"{nat_ratio:.0%} NaT out of {non_null_source} non-null source values -- this "
                f"is very likely the wrong column or wrong format string, not genuinely missing "
                f"dates. A sample real value in {col!r}: {source.dropna().iloc[0]!r}.",
            ))
    return violations


@dataclass
class _StringContainsCall:
    column: str
    literal: str
    regex: bool
    lineno: int


class _StringContainsCollector(ast.NodeVisitor):
    """Finds `df['col'].str.contains(LITERAL, ...)` calls that are
    case-sensitive by construction (no `case=False` kwarg, and no
    `.str.lower()/.str.casefold()/.str.upper()` normalization anywhere in
    the chain between the column and `.contains(...)`) -- a DIFFERENT bug
    shape from `check_value_grounding`'s Eq/NotEq presence check: the
    literal DOES appear in the column, so a "does this value exist at all"
    check passes cleanly, but a case-sensitive `.contains()` silently
    under-matches when the SAME real-world value is inconsistently
    capitalized across rows (confirmed live: a `LocationName` column with
    both "(Scarborough)" and "(scarborough)" entries -- a case-sensitive
    `.contains('Scarborough')` silently dropped 8 of 61 otherwise-matching
    rows, undercounting a total).

    Deliberately scoped to `.str.contains(...)` only, not bare `==`/`!=`
    equality (`check_value_grounding` already covers those with its own,
    intentionally case-INsensitive presence check) -- an exact `==` match
    is far more often used to target a specific canonical code/category
    where a differently-cased variant may be a genuinely different value
    (e.g. status codes), so auto-flagging every such case gap there would
    be noisy in a way `.contains()` isn't: `.contains()` is inherently a
    "does this concept appear in this text" match, where tolerating case is
    almost always the intended behavior."""

    _NORMALIZING_METHODS = {"lower", "casefold", "upper"}

    def __init__(self) -> None:
        self.calls: list[_StringContainsCall] = []

    @classmethod
    def _chain_is_case_normalized(cls, node: ast.AST, _depth: int = 0) -> bool:
        """True if `.str.lower()/.str.casefold()/.str.upper()` appears
        anywhere between the root column and this node -- the author
        already made case an explicit, deliberate choice, so don't flag."""
        if _depth > 12:
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in cls._NORMALIZING_METHODS:
                return True
            return cls._chain_is_case_normalized(node.func.value, _depth + 1)
        if isinstance(node, ast.Attribute):
            return cls._chain_is_case_normalized(node.value, _depth + 1)
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "contains":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                col = _LiteralComparisonCollector._root_column(node.func.value)
                if col is not None and not self._chain_is_case_normalized(node.func.value):
                    case_sensitive = True
                    regex = True
                    for kw in node.keywords:
                        if kw.arg == "case" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                            case_sensitive = False
                        if kw.arg == "regex" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                            regex = False
                    if case_sensitive:
                        self.calls.append(_StringContainsCall(col, node.args[0].value, regex, node.lineno))
        self.generic_visit(node)


def check_case_sensitivity_gaps(code: str, df: pd.DataFrame) -> list[GroundingViolation]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    collector = _StringContainsCollector()
    collector.visit(tree)
    violations: list[GroundingViolation] = []
    for call in collector.calls:
        if call.column not in df.columns:
            continue
        series = df[call.column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        non_null = series.dropna().astype(str)
        if non_null.empty:
            continue
        try:
            case_sensitive_hits = int(non_null.str.contains(call.literal, case=True, regex=call.regex, na=False).sum())
            case_insensitive_hits = int(non_null.str.contains(call.literal, case=False, regex=call.regex, na=False).sum())
        except Exception:  # noqa: BLE001
            continue  # an invalid regex literal, etc. -- not this check's concern
        if case_insensitive_hits > case_sensitive_hits:
            gap = case_insensitive_hits - case_sensitive_hits
            violations.append(GroundingViolation(
                call.column, "Contains", call.literal, "SOFT",
                f"{call.column!r}.str.contains({call.literal!r}) is case-sensitive and matches "
                f"{case_sensitive_hits} row(s), but {gap} additional row(s) would also match if "
                f"case-insensitive -- the column has inconsistent capitalization for what looks like "
                f"the same value. Verify this exclusion is intentional, not a data-casing artifact.",
            ))
    return violations
