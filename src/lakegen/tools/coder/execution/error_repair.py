"""What a failed run tells the coder: column and join errors enriched with the tables' real names and types."""

from __future__ import annotations

import ast
import difflib
import re
from typing import Any

from lakegen.integrations.table_io import read_table
from lakegen.tools.coder.quality.column_resolution import generated_column_names
from lakegen.tools.coder.quality.data_quality import profile_table_quality
from lakegen.tools.coder.state import CoderLifecycle


def classify_execution_error(message: str, *, stage: str = "execution") -> dict[str, Any]:
    """Convert an execution/preflight failure into a stable, compact schema."""

    text = str(message or "Unknown execution error")
    lowered = text.casefold()
    category = "runtime_error"
    # Imports are blocked by the execution sandbox, but a coder can safely
    # repair them by removing the import.  Keep genuinely dangerous operations
    # (subprocess, eval, exec, and similar) in the terminal security category.
    if re.search(r"forbidden code fragment ['\"]import\s+", lowered):
        category = "forbidden_import"
    elif stage == "preflight":
        category = "column_resolution_error"
    elif "security error" in lowered or "forbidden code" in lowered:
        category = "security_error"
    elif "timed out" in lowered or "timeout" in lowered:
        category = "timeout"
    elif "keyerror" in lowered or "missing required column" in lowered:
        category = "missing_column"
    elif "filenotfounderror" in lowered or "no such file" in lowered:
        category = "missing_file"
    elif "syntaxerror" in lowered or "indentationerror" in lowered:
        category = "syntax_error"
    elif "cartesian_product_detected" in lowered:
        category = "join_error"
    elif "mergeerror" in lowered or "join" in lowered and "error" in lowered:
        category = "join_error"
    elif "typeerror" in lowered or "valueerror" in lowered:
        category = "type_error"

    column = None
    match = re.search(r"KeyError(?: for column)?:?\s*['\"]?([^'\"\n]+)", text)
    if match:
        column = match.group(1).strip()
    return {
        "stage": stage,
        "category": category,
        "message": text[-1200:],
        "column": column,
        "retryable": category not in {"security_error", "timeout"},
    }


class ErrorRepairMixin:
    """The error-enrichment methods of ``Phase3ToolsManager``."""

    def _all_columns(self) -> list[str]:
        columns: list[str] = []
        for table in self.tables:
            try:
                columns.extend(str(c) for c in read_table(self.csv_dir / table.strip(), nrows=0).columns)
            except Exception:
                continue
        return list(dict.fromkeys(columns))

    def _rename_hints(self, column: str) -> list[dict[str, str]]:
        """Find literal Pandas rename mappings relevant to a failed column."""

        try:
            tree = ast.parse(self.state.clean_code or self.state.code_raw)
        except SyntaxError:
            return []
        hints: list[dict[str, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "rename":
                continue
            columns_kw = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "columns"),
                None,
            )
            if not isinstance(columns_kw, ast.Dict):
                continue
            for old_node, new_node in zip(columns_kw.keys, columns_kw.values):
                if not (
                    isinstance(old_node, ast.Constant)
                    and isinstance(new_node, ast.Constant)
                    and isinstance(old_node.value, str)
                    and isinstance(new_node.value, str)
                ):
                    continue
                if old_node.value == column:
                    hints.append({"renamed_from": old_node.value, "renamed_to": new_node.value})
        return hints

    @staticmethod
    def _auto_repair_renamed_column(code: str, error: str) -> str | None:
        """Patch the one failing line that uses a label after renaming it.

        Only fires for an unambiguous literal ``rename(columns={old: new})``
        that precedes the failing line; the model repeatedly failed to apply
        this one-token fix even when told exactly what to change.
        """

        if "KeyError" not in error:
            return None
        key = re.search(r"KeyError(?: for column)?:?\s*['\"]([^'\"\n]+)['\"]", error)
        line = re.search(r"Failing line (\d+):", error)
        if not key or not line:
            return None
        old = key.group(1)
        lines = code.splitlines()
        index = int(line.group(1)) - 1
        if not 0 <= index < len(lines):
            return None
        renamed = {
            match.group(3)
            for text in lines[:index] if "rename(" in text
            for match in re.finditer(
                rf"(['\"]){re.escape(old)}\1\s*:\s*(['\"])([^'\"]+)\2", text
            )
        }
        if len(renamed) != 1:
            return None
        new = next(iter(renamed))
        patched = re.sub(
            rf"\[(['\"]){re.escape(old)}\1\]",
            lambda match: f"[{match.group(1)}{new}{match.group(1)}]",
            lines[index],
        )
        if patched == lines[index]:
            return None
        lines[index] = patched
        return "\n".join(lines)

    def _enrich_column_error(self, error: dict[str, Any]) -> None:
        column = error.get("column")
        if not column:
            return
        columns = self._all_columns()
        error["closest_columns"] = difflib.get_close_matches(
            str(column).strip(), columns, n=5, cutoff=0.5
        )
        error["source_columns"] = columns[:100]
        try:
            derived_columns = generated_column_names(self.state.clean_code)
        except SyntaxError:
            derived_columns = set()
        if str(column).strip() in derived_columns:
            error["generated_column"] = True
            error["repair_hint"] = (
                "This is a derived downstream label, not a missing source column. "
                "Inspect the preceding filter/groupby/aggregation for empty output "
                "and ensure the derived column is created before using it."
            )
            return
        rename_hints = self._rename_hints(str(column).strip())
        if rename_hints:
            error["rename_hints"] = rename_hints
            # The source-schema match is exactly the stale label; listing it as
            # the closest column contradicts the rename evidence.
            error["closest_columns"] = [item["renamed_to"] for item in rename_hints]
            replacements = ", ".join(
                f"{item['renamed_from']} -> {item['renamed_to']}"
                for item in rename_hints
            )
            error["repair_hint"] = (
                "The missing label was renamed earlier in the generated pipeline. "
                f"Use the post-rename label in downstream expressions: {replacements}."
            )

    def _enrich_join_error(self, error: dict[str, Any]) -> None:
        if error.get("category") != "join_error":
            return
        error.setdefault(
            "repair_hint",
            "Aggregate each side to one row per join key (e.g. "
            "groupby(key).agg(...)) before merging, or confirm the join keys are "
            "true unique identifiers. Do not merge row-by-row on broad "
            "category/geo codes shared by many rows on either side.",
        )
        error.setdefault(
            "next_actions", ["aggregate_before_join", "verify_unique_join_keys"]
        )

    def _table_quality_profile(self, table: str) -> dict[str, dict]:
        frame = self.state.table_samples.get(table)
        if frame is None:
            try:
                frame = read_table(self.csv_dir / table.strip(), nrows=500)
            except Exception:
                return {}
        try:
            return profile_table_quality(frame)
        except Exception:
            return {}

    def _enrich_type_error(self, error: dict[str, Any]) -> None:
        if error.get("category") != "type_error":
            return
        message = str(error.get("message") or "")
        match = re.search(
            r"(?:could not convert string to float|invalid literal for \w+\(\) "
            r"with base \d+|could not convert)[^:]*:?\s*'([^']*)'",
            message,
        )
        offending_value = match.group(1).strip().casefold() if match else ""
        culprits: dict[str, dict[str, Any]] = {}
        for table in self.tables:
            for column, evidence in self._table_quality_profile(table).items():
                tokens = {
                    str(token).casefold()
                    for token in (evidence.get("sentinel_tokens_observed") or {})
                }
                has_numeric_noise = bool(evidence.get("numeric_format_examples"))
                if offending_value:
                    if offending_value not in tokens:
                        continue
                elif not tokens and not has_numeric_noise:
                    continue
                culprits[f"{table}::{column}"] = evidence
        if not culprits:
            return
        error["dirty_columns_detected"] = culprits
        example_cols = ", ".join(sorted(culprits)[:5])
        error["repair_hint"] = (
            "Numeric conversion failed because of literal non-numeric tokens "
            f"observed during profiling in these columns: {example_cols}. Replace "
            "exactly those observed tokens (do not invent new ones) with NaN "
            "before pd.to_numeric, e.g. "
            "df[col] = df[col].replace(<observed tokens>, pd.NA); "
            "df[col] = pd.to_numeric(df[col], errors='coerce')."
        )
        error.setdefault(
            "next_actions", ["clean_using_observed_tokens", "retry_conversion"]
        )

    def _reject_after_repeated_missing_column(self, error: dict[str, Any]) -> None:
        """Escalate a repeated, genuinely absent column back to discovery."""
        if error.get("category") != "missing_column":
            return
        if error.get("generated_column"):
            return
        column = str(error.get("column") or "").strip()
        if not column or error.get("rename_hints"):
            return
        available = {
            re.sub(r"[^a-z0-9]", "", item.casefold())
            for item in error.get("source_columns", [])
        }
        normalized = re.sub(r"[^a-z0-9]", "", column.casefold())
        if normalized in available or error.get("closest_columns"):
            return
        failures = self.state.missing_column_failures.get(normalized, 0) + 1
        self.state.missing_column_failures[normalized] = failures
        if failures < 2:
            return
        self.state.rejected_reason = (
            f"Required column {column!r} is absent from every selected table after "
            "two corrected execution attempts."
        )
        self.state.rejection_details = {
            "missing_requirements": [f"source column {column}"],
            "inspected_evidence": (
                "The repeated KeyError has no exact, close, or rename-mapped match "
                f"among {len(available)} selected-table columns."
            ),
        }
        self.state.lifecycle = CoderLifecycle.TABLES_INSUFFICIENT
        error["retryable"] = False
        error["escalate_to_discovery"] = True
        error["next_actions"] = ["retry_table_discovery"]
