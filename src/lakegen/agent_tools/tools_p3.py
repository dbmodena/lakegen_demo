"""Bounded tools used by the agentic code-generation phase."""

from __future__ import annotations

import json
import ast
import difflib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
import pandas as pd

from lakegen.core.table_io import read_table, table_load_command


ReviewStatus = Literal["verified", "not_applicable", "needs_revision"]


class CoderLifecycle(str, Enum):
    NEEDS_CODE = "needs_code"
    NEEDS_INSPECTION = "needs_inspection"
    NEEDS_REVISION = "needs_revision"
    READY_TO_FINISH = "ready_to_finish"
    TABLES_INSUFFICIENT = "tables_insufficient"
    FINISHED = "finished"


class FinishCodeSchema(BaseModel):
    filters: ReviewStatus = Field(description="Verify every requested filter and time range.")
    measures: ReviewStatus = Field(description="Verify measures, aggregation and units.")
    grouping: ReviewStatus = Field(description="Verify requested groups and category coverage.")
    ordering: ReviewStatus = Field(description="Verify ranking, sorting, top/bottom and limits.")
    output_shape: ReviewStatus = Field(description="Verify that the output shape answers the question.")
    requirement_reviews: dict[str, str] = Field(description="Map every exact requirement reported by inspect_result to concrete evidence from the result.")
    review: str = Field(description="Brief evidence-based final review of the computed result.")


class RejectTablesSchema(BaseModel):
    reason: str = Field(description="Why the selected tables cannot answer the question.")
    missing_requirements: list[str] = Field(description="Concrete missing columns, periods, categories, or join keys.")
    inspected_evidence: str = Field(description="Evidence observed through inspect_table or run_code.")


class AnalysisContractSchema(BaseModel):
    filters: list[str] = Field(description="Requested filters and time ranges, using exact values from the question.")
    measures: list[str] = Field(description="Requested measures with aggregation, units, and distinctness semantics.")
    group_by: list[str] = Field(description="Requested grouping dimensions.")
    distinct_counts: list[str] = Field(description="Entities that must be counted distinctly.")
    joins: list[str] = Field(description="Required table relationships or join keys; empty when no join is needed.")
    ordering: str = Field(description="Requested ordering direction and measure, or 'none'.")
    limit: int | None = Field(description="Requested top/bottom N, or null.")
    output_columns: list[str] = Field(description="Semantic columns required in the final output.")


def classify_execution_error(message: str, *, stage: str = "execution") -> dict[str, Any]:
    """Convert an execution/preflight failure into a stable, compact schema."""

    text = str(message or "Unknown execution error")
    lowered = text.casefold()
    category = "runtime_error"
    if "forbidden code fragment 'import sys'" in lowered:
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


@dataclass
class P3State:
    max_runs: int = 3
    run_count: int = 0
    result_version: int = 0
    inspected_version: int = 0
    code_raw: str = ""
    clean_code: str = ""
    raw_result: str | None = None
    structured_result: object | None = None
    structured_result_error: str = ""
    error: str | None = None
    execution_error: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    review: dict[str, Any] = field(default_factory=dict)
    coverage_requirements: list[str] = field(default_factory=list)
    coverage_warnings: list[str] = field(default_factory=list)
    inspected_tables: set[str] = field(default_factory=set)
    table_samples: dict[str, Any] = field(default_factory=dict)
    table_profiled_columns: dict[str, set[str]] = field(default_factory=dict)
    rejected_reason: str = ""
    rejection_details: dict[str, Any] = field(default_factory=dict)
    lifecycle: CoderLifecycle = CoderLifecycle.NEEDS_CODE
    stop_reason: str = ""
    finalization_mode: str = ""
    analysis_contract: dict[str, Any] = field(default_factory=dict)
    contract_code_warnings: list[str] = field(default_factory=list)
    contract_advisories: list[str] = field(default_factory=list)
    contract_evidence: list[dict[str, Any]] = field(default_factory=list)
    contract_evidence_advisories: list[str] = field(default_factory=list)
    missing_column_failures: dict[str, int] = field(default_factory=dict)
    architect_contract_locked: bool = False

    def ready_for_finalization(self) -> bool:
        """Return whether a closure-only turn is safe and meaningful."""
        return (
            not self.finished
            and self.error is None
            and self.structured_result is not None
            and self.result_version > 0
            and self.inspected_version == self.result_version
            and not self.coverage_warnings
            and self.lifecycle == CoderLifecycle.READY_TO_FINISH
        )

    def ready_for_degraded_finalization(self) -> bool:
        """Return whether a computed result can be preserved with advisories."""
        return (
            not self.finished
            and self.error is None
            and self.structured_result is not None
            and self.result_version > 0
            and self.inspected_version == self.result_version
            and self.lifecycle in {
                CoderLifecycle.NEEDS_REVISION,
                CoderLifecycle.READY_TO_FINISH,
            }
        )


class Phase3ToolsManager:
    """Stateful, bounded execute/inspect/finish loop for the coder."""

    def __init__(
        self,
        state: P3State,
        *,
        tables: list[str],
        csv_dir: Path,
        run_dir: Path | None,
        evaluation_result_type: str | None,
        question: str = "",
        table_metadata: dict[str, Any] | None = None,
        selection_plan: dict[str, Any] | None = None,
        resolve_code: Callable[[str, list[str], Path], tuple[str, str | None]],
        execute_code: Callable[..., tuple[str | None, str | None, str]],
        extract_payload: Callable[[str], tuple[str, object | None, str | None]],
    ):
        self.state = state
        self.tables = tables
        self.csv_dir = Path(csv_dir)
        self.run_dir = run_dir
        self.evaluation_result_type = evaluation_result_type
        self.question = question
        self.table_metadata = table_metadata or {}
        self.selection_plan = selection_plan or {}
        self.resolve_code = resolve_code
        self.execute_code = execute_code
        self.extract_payload = extract_payload
        semantic_plan = self.selection_plan.get("semantic_plan")
        if isinstance(semantic_plan, dict):
            self.state.analysis_contract = self._contract_from_semantic_plan(
                semantic_plan
            )
            self.state.architect_contract_locked = True

    @staticmethod
    def _contract_from_semantic_plan(plan: dict[str, Any]) -> dict[str, Any]:
        filters = []
        for binding in plan.get("filters", []):
            if not isinstance(binding, dict):
                continue
            expression = " ".join(filter(None, [
                str(binding.get("column") or "").strip(),
                str(binding.get("operator") or "").strip(),
                str(binding.get("value") or "").strip(),
            ]))
            filters.append(expression)
        measures = []
        distinct_counts = []
        for binding in plan.get("measures", []):
            if not isinstance(binding, dict):
                continue
            operation = str(binding.get("operation") or "").strip()
            output = str(binding.get("output") or "").strip()
            columns = ", ".join(map(str, binding.get("columns", [])))
            measures.append(" ".join(filter(None, [operation, columns, "as", output])))
            if operation == "count_distinct":
                distinct_counts.extend(map(str, binding.get("columns", [])))
        dimensions = [
            str(binding.get("output") or binding.get("column") or "").strip()
            for binding in plan.get("dimensions", []) if isinstance(binding, dict)
        ]
        ordering = "; ".join(
            " ".join(filter(None, [
                str(item.get("output") or "").strip(),
                str(item.get("direction") or "").strip(),
            ]))
            for item in plan.get("ordering", []) if isinstance(item, dict)
        ) or "none"
        return {
            "filters": list(dict.fromkeys(filter(None, filters))),
            "measures": list(dict.fromkeys(filter(None, measures))),
            "group_by": list(dict.fromkeys(filter(None, dimensions))),
            "distinct_counts": list(dict.fromkeys(filter(None, distinct_counts))),
            "joins": [str(item).strip() for item in plan.get("joins", []) if str(item).strip()],
            "ordering": ordering,
            "limit": plan.get("limit"),
            "output_columns": [
                str(item).strip() for item in plan.get("output_columns", [])
                if str(item).strip()
            ],
            "source": "architect_semantic_plan",
        }

    def set_analysis_contract(
        self,
        filters: list[str],
        measures: list[str],
        group_by: list[str],
        distinct_counts: list[str],
        joins: list[str],
        ordering: str,
        limit: int | None,
        output_columns: list[str],
    ) -> str:
        """Declare the question semantics before writing or executing code."""
        if self.state.architect_contract_locked:
            return json.dumps({
                "ok": False,
                "error": "architect_semantic_plan_is_authoritative",
                "contract": self.state.analysis_contract,
                "next_action": "Write the program and call run_code.",
            }, ensure_ascii=False)
        contract = {
            "filters": self._clean_contract_items(filters),
            "measures": self._clean_contract_items(measures),
            "group_by": self._clean_contract_items(group_by),
            "distinct_counts": self._clean_contract_items(distinct_counts),
            "joins": self._clean_contract_items(joins),
            "ordering": str(ordering or "none").strip(),
            "limit": limit,
            "output_columns": self._clean_contract_items(output_columns),
        }
        problems = self._contract_problems(contract)
        self.state.analysis_contract = contract
        self.state.contract_advisories = problems
        return json.dumps({
            "ok": True,
            "contract": contract,
            "advisories": problems,
            "next_action": "Write the complete program and call run_code.",
        }, ensure_ascii=False)

    @staticmethod
    def _clean_contract_items(items: list[str]) -> list[str]:
        return list(dict.fromkeys(
            str(item).strip() for item in items if str(item).strip()
        ))

    def _contract_problems(self, contract: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        question = self.question.casefold()
        contract_text = json.dumps(contract, ensure_ascii=False).casefold()
        for year in self._requested_filter_years(question):
            if year not in contract_text:
                problems.append(f"missing requested year {year}")
        if re.search(r"\b(?:distinct|different|unique)\b", question) and not contract["distinct_counts"]:
            problems.append("distinct/unique count requested but distinct_counts is empty")
        expected_top = self._expected_top_n(question)
        if expected_top is not None and contract["limit"] != expected_top:
            problems.append(f"requested top/bottom limit is {expected_top}")
        if not contract["measures"]:
            problems.append("at least one requested measure is required")
        if not contract["output_columns"]:
            problems.append("at least one semantic output column is required")
        return problems

    @staticmethod
    def _requested_filter_years(question: str) -> set[str]:
        """Return row-filter years, excluding years that name a dataset edition."""
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
        edition_years = set(re.findall(
            r"\b(?:using|from|in)\s+(?:the\s+)?((?:19|20)\d{2})\b"
            r"[^?.]{0,80}\b(?:dataset|data\s+set|release|edition|inventory)\b",
            question,
            flags=re.IGNORECASE,
        ))
        return years - edition_years

    @staticmethod
    def _expected_top_n(question: str) -> int | None:
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", question)
        if match:
            return int(match.group(1))
        match = re.search(
            rf"\b(?:top|bottom)\s+({'|'.join(number_words)})\b", question
        )
        if match:
            return number_words[match.group(1)]
        match = re.search(
            rf"\bwhich\s+({'|'.join(number_words)})\b[^?.]{{0,100}}"
            r"\b(?:highest|lowest|most|least|largest|smallest)\b",
            question,
        )
        return number_words[match.group(1)] if match else None

    def infer_analysis_contract(self) -> None:
        """Create a conservative contract only for fenced-code protocol recovery."""
        question = self.question
        lowered = question.casefold()
        years = sorted(self._requested_filter_years(question))
        limit = self._expected_top_n(lowered)
        distinct = ["requested entity"] if re.search(
            r"\b(?:distinct|different|unique)\b", lowered
        ) else []
        measure = "requested numeric measure"
        if "average" in lowered or "mean" in lowered:
            measure = "average of the requested measure"
        elif "percentage" in lowered or "percent" in lowered:
            measure = "requested percentage"
        elif "how many" in lowered or "count" in lowered:
            measure = "count of requested entities"
        self.state.analysis_contract = {
            "filters": [f"year = {year}" for year in years],
            "measures": [measure],
            "group_by": [],
            "distinct_counts": distinct,
            "joins": [],
            "ordering": "question-defined ordering" if limit else "none",
            "limit": limit,
            "output_columns": ["requested result"],
            "inferred_for_protocol_recovery": True,
        }

    def _allowed_paths(self) -> dict[Path, str]:
        return {
            (self.csv_dir / table.strip()).resolve(): table.strip()
            for table in self.tables
        }

    def _validate_load_paths(self, code: str) -> dict[str, Any] | None:
        """Reject literal dataframe paths outside the selected table set."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None  # The existing preflight returns the richer syntax error.
        allowed = self._allowed_paths()
        invalid: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"read_csv", "read_parquet", "read_json", "read_pickle"}:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                continue
            supplied = Path(node.args[0].value).expanduser()
            if not supplied.is_absolute():
                supplied = self.csv_dir / supplied
            if supplied.resolve() not in allowed:
                invalid.append(str(node.args[0].value))
        if not invalid:
            return None
        return {
            "stage": "preflight",
            "category": "invalid_table_path",
            "message": "Generated code uses a path that is not one of the selected tables.",
            "invalid_paths": invalid,
            "allowed_load_commands": [
                table_load_command(self.csv_dir / table.strip()) for table in self.tables
            ],
            "retryable": True,
        }

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

    def _enrich_column_error(self, error: dict[str, Any]) -> None:
        column = error.get("column")
        if not column:
            return
        columns = self._all_columns()
        error["closest_columns"] = difflib.get_close_matches(
            str(column).strip(), columns, n=5, cutoff=0.5
        )
        error["source_columns"] = columns[:100]
        rename_hints = self._rename_hints(str(column).strip())
        if rename_hints:
            error["rename_hints"] = rename_hints
            replacements = ", ".join(
                f"{item['renamed_from']} -> {item['renamed_to']}"
                for item in rename_hints
            )
            error["repair_hint"] = (
                "The missing label was renamed earlier in the generated pipeline. "
                f"Use the post-rename label in downstream expressions: {replacements}."
            )

    def _reject_after_repeated_missing_column(self, error: dict[str, Any]) -> None:
        """Escalate a repeated, genuinely absent column back to discovery."""
        if error.get("category") != "missing_column":
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

    def inspect_table(self, file_name: str, columns: str = "") -> str:
        """Inspect a selected table from one cached sample. Up to eight columns may be profiled progressively."""
        table = file_name.strip()
        supplied = Path(table).expanduser()
        if table not in self.tables and supplied.is_absolute():
            supplied_resolved = supplied.resolve()
            table = next(
                (
                    selected
                    for selected in self.tables
                    if (self.csv_dir / selected).resolve() == supplied_resolved
                ),
                table,
            )
        if table not in self.tables:
            return json.dumps({"ok": False, "error": "Only selected tables may be inspected.", "selected_tables": self.tables})
        path = self.csv_dir / table
        cached = table in self.state.table_samples
        if cached:
            frame = self.state.table_samples[table]
        else:
            try:
                frame = read_table(path, nrows=5000)
            except Exception as exc:
                return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            self.state.inspected_tables.add(table)
            self.state.table_samples[table] = frame
            self.state.table_profiled_columns[table] = set()
        requested = [item.strip() for item in columns.split(",") if item.strip()]
        exact = {str(column): column for column in frame.columns}
        details: dict[str, Any] = {}
        resolved_requests: list[tuple[str, str]] = []
        for requested_column in requested:
            matches = difflib.get_close_matches(requested_column.strip(), list(exact), n=1, cutoff=0.55)
            if not matches:
                details[requested_column] = {"error": "column_not_found"}
                continue
            name = matches[0]
            resolved_requests.append((requested_column, name))
        already_profiled = self.state.table_profiled_columns[table]
        new_names = {name for _requested, name in resolved_requests} - already_profiled
        if len(already_profiled | new_names) > 8:
            return json.dumps({
                "ok": False,
                "error": "Column profile limit exceeded: at most 8 distinct columns per table.",
                "already_profiled": sorted(already_profiled),
                "remaining_profile_slots": max(0, 8 - len(already_profiled)),
                "next_action": "Use the existing evidence and choose run_code or reject_tables.",
            })
        already_profiled.update(new_names)
        for requested_column, name in resolved_requests:
            series = frame[exact[name]]
            non_null = series.dropna()
            detail: dict[str, Any] = {
                "exact_name": name,
                "dtype": str(series.dtype),
                "null_count_in_sample": int(series.isna().sum()),
                "distinct_count_in_sample": int(non_null.nunique()),
                "sample_values": [str(value)[:100] for value in non_null.drop_duplicates().head(12)],
            }
            metadata_description = self._column_description(table, name)
            if metadata_description:
                detail["metadata_description"] = metadata_description
            if not non_null.empty and ("date" in name.casefold() or "year" in name.casefold()):
                parsed = pd.to_datetime(non_null, errors="coerce").dropna()
                if not parsed.empty:
                    detail["temporal_min"] = str(parsed.min())
                    detail["temporal_max"] = str(parsed.max())
            details[requested_column] = detail
        temporal_candidates = [
            str(column) for column in frame.columns
            if re.search(
                r"(?:^|[_\W])(year|date|time|period|fy|sy)(?:$|[_\W])",
                str(column), re.IGNORECASE,
            )
            or re.search(r"(?:year|date)$", str(column), re.IGNORECASE)
        ]
        response = json.dumps({
            "ok": True,
            "cached_sample": cached,
            "table": table,
            "load_command": table_load_command(path),
            "sampled_rows": len(frame),
            "columns": [str(column) for column in frame.columns],
            "resource_metadata": self._resource_metadata_preview(table),
            "semantic_ambiguities": ({
                "temporal_columns": temporal_candidates[:8],
                "guidance": "Choose using resource/column meaning and inspected values, not the column name alone.",
            } if len(temporal_candidates) > 1 else {}),
            "requested_column_profiles": details,
            "profiled_columns_total": sorted(already_profiled),
            "remaining_profile_slots": 8 - len(already_profiled),
            "next_action": (
                "Choose one: run_code if the evidence is sufficient, or "
                "reject_tables if required data is missing."
            ),
        }, ensure_ascii=False, default=str)
        return response

    def reject_tables(
        self,
        reason: str,
        missing_requirements: list[str],
        inspected_evidence: str,
    ) -> str:
        """Reject the selected tables with structured evidence so discovery can retry."""
        if self.state.finished:
            raise ValueError("Tables cannot be rejected after finish_code.")
        if self.state.rejected_reason:
            return "REJECT_TABLES: " + json.dumps({
                "reason": self.state.rejected_reason,
                **self.state.rejection_details,
            }, ensure_ascii=False)
        if not self.state.inspected_tables and self.state.run_count == 0:
            raise ValueError("Inspect a selected table or run code before rejecting it.")
        missing = [str(item).strip() for item in missing_requirements if str(item).strip()]
        if not missing:
            raise ValueError("Provide at least one concrete missing requirement.")
        if len(reason.strip()) < 20 or len(inspected_evidence.strip()) < 20:
            raise ValueError("Provide a concrete reason and inspected evidence.")
        blockers = self._rejection_blockers(missing, reason, inspected_evidence)
        if blockers:
            self.state.lifecycle = CoderLifecycle.NEEDS_REVISION
            payload = json.dumps({
                "ok": False,
                "error": {
                    "category": "rejection_not_proven",
                    "message": "The selected tables have not been proven insufficient.",
                    "blockers": blockers,
                    "next_actions": ["run_code", "inspect_table"],
                },
            }, ensure_ascii=False)
            raise ValueError("REJECTION_NOT_PROVEN: " + payload)
        self.state.rejected_reason = reason.strip()
        self.state.rejection_details = {
            "missing_requirements": missing,
            "inspected_evidence": inspected_evidence.strip(),
        }
        self.state.lifecycle = CoderLifecycle.TABLES_INSUFFICIENT
        return "REJECT_TABLES: " + json.dumps({
            "reason": self.state.rejected_reason,
            **self.state.rejection_details,
        }, ensure_ascii=False)

    def _metadata_text(self) -> str:
        def flatten(value: Any) -> list[str]:
            if isinstance(value, dict):
                return [str(key) for key in value] + [
                    item for child in value.values() for item in flatten(child)
                ]
            if isinstance(value, (list, tuple, set)):
                return [item for child in value for item in flatten(child)]
            return [str(value)] if value is not None else []

        return " ".join(flatten(self.table_metadata)).casefold()

    def _resource_metadata_preview(self, table: str) -> dict[str, Any]:
        metadata = self.table_metadata.get(
            table, self.table_metadata.get(Path(table).stem, {})
        )
        if not isinstance(metadata, dict):
            return {}
        preview = {
            "title": " ".join(str(metadata.get("title") or "").split())[:160],
            "description": " ".join(
                str(metadata.get("description") or "").split()
            )[:320],
        }
        return {key: value for key, value in preview.items() if value}

    def _column_description(self, table: str, column_name: str) -> str:
        metadata = self.table_metadata.get(
            table, self.table_metadata.get(Path(table).stem, {})
        )
        columns = metadata.get("columns", []) if isinstance(metadata, dict) else []
        normalized = re.sub(r"[^a-z0-9]", "", column_name.casefold())
        for column in columns if isinstance(columns, list) else []:
            if not isinstance(column, dict):
                continue
            candidate = re.sub(
                r"[^a-z0-9]", "", str(column.get("name") or "").casefold()
            )
            if candidate == normalized:
                return " ".join(
                    str(column.get("description") or "").split()
                )[:240]
        return ""

    def _rejection_blockers(
        self,
        missing: list[str],
        reason: str,
        inspected_evidence: str,
    ) -> list[str]:
        """Reject only when a missing base fact, rather than a derivation, is proven."""
        claim = " ".join([reason, inspected_evidence, *missing]).casefold()
        blockers: list[str] = []
        metadata = self._metadata_text()
        requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", self.question))
        edition_years = requested_years - self._requested_filter_years(self.question)
        if edition_years and re.search(
            r"(?:year|date|temporal)[^.]*(?:missing|lack|without|does not contain)|"
            r"(?:missing|lack|without|does not contain)[^.]*(?:year|date|temporal)",
            claim,
        ):
            blockers.append(
                "The year identifies the requested dataset edition, not a row-level "
                "filter; a dedicated year/date column is not required."
            )
        if (
            requested_years
            and ("year column" in claim or "fiscal-year column" in claim or "fiscal year column" in claim)
            and any(year in metadata for year in requested_years)
        ):
            blockers.append(
                "The requested period appears in selected-table metadata; a dedicated year column is not required."
            )
        if re.search(
            r"(?:only|lack|missing|without)[^.]*(?:all five|five nyc)?[^.]*(?:borough|boro)",
            claim,
        ) or "zero-count" in claim or "zero count" in claim:
            blockers.append(
                "Absent borough rows do not prove missing data; derive observed groups and add valid zero-count groups when required."
            )
        if re.search(r"not enough distinct|fewer than\s+(?:five|5)|only\s+\d+\s+(?:distinct\s+)?(?:categories|codes)", claim):
            blockers.append(
                "Fewer observed categories than a requested top-N is not by itself proof that the tables are insufficient."
            )
        if ("column" in claim and ("missing" in claim or "does not contain" in claim)):
            all_columns = self._all_columns()
            normalized = {re.sub(r"[^a-z0-9]", "", column.casefold()) for column in all_columns}
            requirement_words = {
                re.sub(r"[^a-z0-9]", "", word)
                for word in re.findall(r"[A-Za-z][A-Za-z _-]{3,40}", " ".join(missing))
            }
            if any(word and any(word in column or column in word for column in normalized) for word in requirement_words):
                blockers.append(
                    "A selected schema contains a plausible equivalent column; inspect or use the exact available name before rejection."
                )
        return list(dict.fromkeys(blockers))

    def run_code(self, code: str) -> str:
        """Execute a complete Python analysis. At most three calls are allowed; fix structured errors before retrying."""
        if self.state.finished:
            return json.dumps({"ok": False, "error": {"category": "already_finished"}})
        if self.state.rejected_reason:
            return json.dumps({"ok": False, "error": {"category": "tables_already_rejected"}})
        if not self.state.analysis_contract:
            self.infer_analysis_contract()
        if self.state.run_count >= self.state.max_runs:
            return json.dumps({"ok": False, "error": {"category": "run_limit", "message": "Maximum of 3 executions reached."}})

        self.state.run_count += 1
        self.state.lifecycle = CoderLifecycle.NEEDS_REVISION
        self.state.code_raw = code
        path_error = self._validate_load_paths(code)
        if path_error:
            self.state.error = path_error["message"]
            self.state.execution_error = path_error
            return json.dumps({"ok": False, "attempt": self.state.run_count, "error": path_error})
        resolved, preflight_error = self.resolve_code(code, self.tables, self.csv_dir)
        self.state.clean_code = resolved
        self.state.contract_code_warnings = self._validate_contract_code(resolved)
        if preflight_error:
            self.state.error = preflight_error
            self.state.execution_error = classify_execution_error(preflight_error, stage="preflight")
            return json.dumps({"ok": False, "attempt": self.state.run_count, "error": self.state.execution_error})

        raw_result, error, clean_code = self.execute_code(resolved, run_dir=self.run_dir)
        self.state.clean_code = clean_code
        if error is not None:
            self.state.error = error
            self.state.raw_result = None
            self.state.execution_error = classify_execution_error(error)
            self._enrich_column_error(self.state.execution_error)
            self._reject_after_repeated_missing_column(self.state.execution_error)
            return json.dumps({"ok": False, "attempt": self.state.run_count, "error": self.state.execution_error})

        self.state.error = None
        self.state.execution_error = {}
        self.state.raw_result = raw_result
        self.state.structured_result = None
        self.state.structured_result_error = ""
        if raw_result is not None and self.evaluation_result_type:
            display, structured, payload_error = self.extract_payload(raw_result)
            self.state.raw_result = display
            self.state.structured_result = structured
            self.state.structured_result_error = payload_error or ""
            diagnostic_text = display.strip().casefold()
            diagnostic_markers = (
                "columns:", "unique ", "unique values", "sample values",
                "matches count", "potential ", "roof cols:", "cellar cols:",
                "empty dataframe", "dtype:", "length:",
            )
            if structured is None and any(
                marker in diagnostic_text for marker in diagnostic_markers
            ):
                self.state.error = (
                    "Diagnostic output is not a final answer. Use inspect_table "
                    "for schema/value evidence, then choose run_code with corrected "
                    "analysis or reject_tables if required data is unavailable."
                )
                self.state.execution_error = {
                    "stage": "result_validation",
                    "category": "diagnostic_output",
                    "message": self.state.error,
                    "retryable": self.state.run_count < self.state.max_runs,
                    "next_actions": ["inspect_table", "run_code", "reject_tables"],
                }
                self.state.raw_result = None
                return json.dumps({
                    "ok": False,
                    "attempt": self.state.run_count,
                    "error": self.state.execution_error,
                })
        self.state.result_version += 1
        self.state.lifecycle = CoderLifecycle.NEEDS_INSPECTION
        return json.dumps({
            "ok": True,
            "attempt": self.state.run_count,
            "result_available": self.state.raw_result is not None,
            "next_action": "Call inspect_result before finish_code.",
        })

    def _validate_contract_code(self, code: str) -> list[str]:
        contract = self.state.analysis_contract
        lowered = code.casefold()
        warnings: list[str] = []
        contract_years = set(re.findall(
            r"\b(?:19|20)\d{2}\b",
            " ".join(contract.get("filters", [])),
        ))
        for year in contract_years:
            if not self._code_represents_year(code, year, contract_years):
                warnings.append(f"contract_filter_missing_in_code: year {year}")
        if contract.get("distinct_counts") and not any(
            marker in lowered for marker in ("nunique(", "drop_duplicates(", ".unique(")
        ):
            warnings.append("contract_distinct_count_missing_in_code")
        question = self.question.casefold()
        count_requested = bool(re.search(r"\b(?:how many|count|number of)\b", question))
        distinct_requested = bool(re.search(r"\b(?:distinct|unique|different)\b", question))
        if (count_requested or distinct_requested) and distinct_requested and not any(
            marker in lowered for marker in ("nunique(", "drop_duplicates(", ".unique(")
        ):
            warnings.append(
                "count_semantics_check: question requests distinct entities but "
                "code does not show a distinct-count operation"
            )
        if not distinct_requested and any(
            marker in lowered for marker in ("nunique(", "drop_duplicates(")
        ):
            warnings.append(
                "unsupported_distinct_semantics: code deduplicates or counts unique "
                "values although the question does not request distinct entities"
            )
        measures = " ".join(contract.get("measures", [])).casefold()
        has_mean_aggregation = bool(re.search(
            r"(?:\.mean\s*\(|\b(?:np|numpy)\.mean\s*\(|['\"]mean['\"])",
            lowered,
        ))
        if ("average" in measures or "mean" in measures) and not has_mean_aggregation:
            warnings.append("contract_average_missing_in_code")
        if any(marker in lowered for marker in ("pd.qcut(", ".qcut(", "pd.cut(", ".cut(")) and not re.search(
            r"\b(?:quantile|quartile|quintile|decile|tertile|equal[- ](?:sized|width)|bins?)\b",
            question,
        ):
            warnings.append(
                "unsupported_bucket_assumption: code invents statistical bucket "
                "boundaries that are not specified by the question"
            )
        if contract.get("joins") and not any(
            marker in lowered for marker in (".merge(", ".join(", "pd.merge(")
        ):
            warnings.append("contract_join_missing_in_code")
        if contract.get("group_by") and not any(
            marker in lowered for marker in (".groupby(", ".pivot(", ".pivot_table(")
        ):
            warnings.append("contract_grouping_missing_in_code")
        requested_years = self._requested_filter_years(question)
        for year in sorted(requested_years):
            if not self._code_represents_year(code, year, requested_years):
                warnings.append(
                    f"time_range_check: requested year {year} is not evident in code"
                )
        limit = contract.get("limit")
        if limit is not None:
            has_limit = bool(re.search(
                rf"(?:head|nlargest|nsmallest)\s*\(\s*{int(limit)}\b", lowered
            ))
            if not has_limit:
                warnings.append(f"contract_limit_missing_in_code: {limit}")
            if not any(marker in lowered for marker in ("sort_values(", "nlargest(", "nsmallest(")):
                warnings.append("contract_ordering_missing_in_code")
        expected_top = self._expected_top_n(question)
        if expected_top is not None and not re.search(
            rf"(?:head|nlargest|nsmallest)\s*\(\s*{expected_top}\b", lowered
        ):
            warnings.append(
                f"top_n_check: question requests exactly {expected_top} ranked items"
            )
        categorical_terms = (
            "borough", "district", "category", "status", "type", "program"
        )
        if any(term in question for term in categorical_terms) and not any(
            marker in lowered
            for marker in (
                ".str.casefold(", ".str.lower(", ".str.upper(",
                ".casefold(", ".replace(", ".map(",
            )
        ):
            warnings.append(
                "category_normalization_check: verify case/alias normalization "
                "for requested categorical filters or groups"
            )
        unloaded = [table for table in self.tables if table.casefold() not in lowered]
        if unloaded:
            warnings.append("selected_tables_not_loaded_in_code: " + ", ".join(unloaded))
        loaded_table_count = len(self.tables) - len(unloaded)
        if loaded_table_count > 1 and not any(
            marker in lowered
            for marker in ("rename(", "columns =", "columns=", "reindex(")
        ):
            warnings.append(
                "partition_schema_check: multiple tables are loaded; verify exact "
                "column names and harmonize schema differences before combining"
            )
        strategy = str(self.selection_plan.get("combination_strategy", ""))
        markers_by_strategy = {
            "join": (".merge(", ".join(", "pd.merge("),
            "lookup": (".merge(", ".join(", ".map(", ".replace("),
            "concat_partitions": ("pd.concat(", ".concat("),
        }
        markers = markers_by_strategy.get(strategy)
        if len(self.tables) > 1 and markers and not any(marker in lowered for marker in markers):
            warnings.append(f"selection_strategy_not_evident_in_code: {strategy}")
        return warnings

    @staticmethod
    def _code_represents_year(code: str, year: str, requested_years: set[str]) -> bool:
        """Recognize literal years and compact school/fiscal-year encodings."""
        if year in code:
            return True
        short = year[-2:]
        lowered = code.casefold()
        for other in requested_years - {year}:
            other_short = other[-2:]
            compact_pairs = {
                f"sy{short}{other_short}",
                f"sy{other_short}{short}",
                f"fy{short}{other_short}",
                f"fy{other_short}{short}",
            }
            if any(pair in lowered for pair in compact_pairs):
                return True
            if re.search(
                rf"(?<!\d)(?:{short}\s*[-/]\s*{other_short}|"
                rf"{other_short}\s*[-/]\s*{short})(?!\d)",
                lowered,
            ):
                return True
        return False

    def _semantic_item_count(self, value: Any) -> int | None:
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            nested = [len(item) for item in value.values() if isinstance(item, list)]
            if nested:
                return max(nested)
            if value and all(not isinstance(item, (dict, list)) for item in value.values()):
                return len(value)
            return 1
        if value is not None:
            return 1
        return None

    def _flatten_values(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            return [str(key) for key in value] + [
                item for child in value.values() for item in self._flatten_values(child)
            ]
        if isinstance(value, list):
            return [item for child in value for item in self._flatten_values(child)]
        return [str(value)] if value is not None else []

    def _dimension_values(self, value: Any, dimension: str) -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                if dimension in str(key).casefold():
                    found.extend(self._flatten_values(child))
                found.extend(self._dimension_values(child, dimension))
        elif isinstance(value, list):
            for child in value:
                found.extend(self._dimension_values(child, dimension))
        return found

    def _result_field_names(self, value: Any) -> list[str]:
        """Return bounded semantic field names visible in a structured result."""
        names: list[str] = []
        if isinstance(value, dict):
            names.extend(map(str, value.keys()))
            for child in value.values():
                names.extend(self._result_field_names(child))
        elif isinstance(value, list):
            for child in value[:10]:
                names.extend(self._result_field_names(child))
        return list(dict.fromkeys(names))[:60]

    @staticmethod
    def _label_is_evident(label: str, texts: list[str]) -> bool:
        stop = {
            "the", "and", "for", "from", "with", "requested", "result",
            "count", "number", "value", "values", "total", "each", "per",
        }
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", label.casefold())
            if len(token) >= 3 and token not in stop
        ]
        haystack = " ".join(texts).casefold()
        return bool(tokens) and any(token in haystack for token in tokens)

    def _contract_evidence(self, value: Any) -> tuple[list[dict[str, Any]], list[str]]:
        """Map declared semantics to observable code/result evidence.

        Missing evidence is advisory only. It is deliberately not added to
        coverage_warnings and therefore cannot block a valid coder result.
        """
        contract = self.state.analysis_contract
        code = self.state.clean_code or self.state.code_raw
        lowered = code.casefold()
        result_fields = self._result_field_names(value)
        rows: list[dict[str, Any]] = []
        advisories: list[str] = []

        def add(kind: str, requirement: str, code_evidence: list[str], result_evidence: list[str]) -> None:
            missing: list[str] = []
            if not code_evidence:
                missing.append("code")
            if not result_evidence:
                missing.append("result")
            advisory = ""
            if missing:
                advisory = (
                    f"No explicit {' or '.join(missing)} evidence for {kind} "
                    f"requirement '{requirement}'. Verify it before finalizing."
                )
                advisories.append(advisory)
            rows.append({
                "kind": kind,
                "requirement": requirement,
                "code_evidence": code_evidence,
                "result_evidence": result_evidence,
                "advisory": advisory,
            })

        years = set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(contract.get("filters", []))))
        for item in contract.get("filters", []):
            item_years = set(re.findall(r"\b(?:19|20)\d{2}\b", item))
            evident = (
                all(self._code_represents_year(code, year, years) for year in item_years)
                if item_years else self._label_is_evident(item, [code])
            )
            add("filter", item, ["filter expression found in code"] if evident else [], ["filter affects computed structured result"] if evident else [])

        measures = contract.get("measures", [])
        for item in measures:
            item_lower = item.casefold()
            operation = ""
            if any(term in item_lower for term in ("average", "mean")) and re.search(r"\.mean\s*\(|['\"]mean['\"]", lowered):
                operation = "mean aggregation found"
            elif any(term in item_lower for term in ("count", "number", "total")) and any(marker in lowered for marker in (".count(", ".size(", "nunique(", "len(")):
                operation = "count aggregation found"
            elif self._label_is_evident(item, [code]):
                operation = "measure terms found in code"
            result_match = [field for field in result_fields if self._label_is_evident(item, [field])]
            if not result_match and len(measures) == 1 and value is not None and not result_fields:
                result_match = ["structured scalar result"]
            add("measure", item, [operation] if operation else [], result_match)

        for item in contract.get("group_by", []):
            grouped = any(marker in lowered for marker in (".groupby(", ".pivot(", ".pivot_table("))
            label_in_code = self._label_is_evident(item, [code])
            result_match = [field for field in result_fields if self._label_is_evident(item, [field])]
            add("grouping", item, ["grouping operation and dimension found"] if grouped and label_in_code else [], result_match)

        for item in contract.get("distinct_counts", []):
            distinct = any(marker in lowered for marker in ("nunique(", "drop_duplicates(", ".unique("))
            result_match = [field for field in result_fields if self._label_is_evident(item, [field])]
            add("distinct count", item, ["distinct-count operation found"] if distinct else [], result_match or (["structured result"] if distinct and value is not None else []))

        for item in contract.get("joins", []):
            joined = any(marker in lowered for marker in (".merge(", ".join(", "pd.merge("))
            add("join", item, ["table join operation found"] if joined else [], ["joined structured result"] if joined and value is not None else [])

        limit = contract.get("limit")
        if limit is not None:
            limited = bool(re.search(rf"(?:head|nlargest|nsmallest)\s*\(\s*{int(limit)}\b", lowered))
            count = self._semantic_item_count(value)
            add("limit", str(limit), [f"top/bottom limit {limit} found"] if limited else [], [f"{count} semantic items returned"] if count == limit else [])

        for item in contract.get("output_columns", []):
            result_match = [field for field in result_fields if self._label_is_evident(item, [field])]
            add("output", item, ["output field terms found in code"] if self._label_is_evident(item, [code]) else [], result_match)
        return rows, advisories

    def _coverage(self, value: Any) -> tuple[list[str], list[str], dict[str, Any]]:
        question = self.question.casefold()
        requirements: list[str] = []
        warnings: list[str] = []
        facts: dict[str, Any] = {}
        count = self._semantic_item_count(value)
        facts["semantic_item_count"] = count
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        ranking_terms = "highest|lowest|most|least|largest|smallest"
        top_match = re.search(r"\btop\s+(\d+)\b", question)
        expected_top = int(top_match.group(1)) if top_match else None
        if expected_top is None:
            top_word = re.search(
                rf"\btop\s+({'|'.join(number_words)})\b", question
            )
            if top_word:
                expected_top = number_words[top_word.group(1)]
        if expected_top is None:
            ranked_number = re.search(
                rf"\b({'|'.join(number_words)})\s+(?:\w+[\s-]+){{0,4}}(?:{ranking_terms})\b",
                question,
            )
            if ranked_number:
                expected_top = number_words[ranked_number.group(1)]
        if expected_top is None:
            which_number = re.search(
                rf"\bwhich\s+({'|'.join(number_words)})\b[^?.]{{0,80}}\b(?:{ranking_terms})\b",
                question,
            )
            if which_number:
                expected_top = number_words[which_number.group(1)]
        if expected_top is not None:
            requirement = f"return {expected_top} ranked semantic items"
            requirements.append(requirement)
            facts["expected_ranked_items"] = expected_top
            if count is not None and count < expected_top:
                warnings.append(f"coverage_shortfall: expected {expected_top} ranked items, found {count}")

        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        if "each month" in question or "all months" in question:
            month_values = " ".join(self._dimension_values(value, "month")).casefold()
            present = [month for month in months if month in month_values]
            numeric_months = {
                int(match) for match in re.findall(r"(?<!\d)(1[0-2]|[1-9])(?!\d)", month_values)
            }
            period_months = {
                int(match) for match in re.findall(r"\b\d{4}[-/](1[0-2]|0?[1-9])\b", month_values)
            }
            month_count = max(len(present), len(numeric_months | period_months))
            requirements.append("cover all 12 calendar months")
            facts["months_present"] = present or sorted(numeric_months | period_months)
            if month_count < 12:
                warnings.append(f"coverage_shortfall: expected 12 months, found {month_count}")
        boroughs = ["bronx", "brooklyn", "manhattan", "queens", "staten island"]
        if "each borough" in question or "all boroughs" in question:
            borough_values = " ".join(self._dimension_values(value, "borough")).casefold()
            present = [borough for borough in boroughs if borough in borough_values]
            # NYC DOE and several city datasets use these canonical one-letter
            # borough codes. Count them as equivalent semantic categories.
            code_aliases = {
                "x": "bronx", "k": "brooklyn", "m": "manhattan",
                "q": "queens", "r": "staten island",
            }
            raw_codes = {
                item.strip().casefold()
                for item in self._dimension_values(value, "boro")
                if len(item.strip()) == 1
            }
            present = list(dict.fromkeys([
                *present,
                *(code_aliases[code] for code in raw_codes if code in code_aliases),
            ]))
            requirements.append("cover all five NYC boroughs, including valid zero-count groups")
            facts["boroughs_present"] = present
            if len(present) < 5:
                warnings.append(f"coverage_shortfall: expected 5 boroughs, found {len(present)}")
        if self.evaluation_result_type == "number":
            requirements.append("return one scalar numeric result")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                warnings.append("output_type_mismatch: expected one scalar number")
        return requirements, warnings, facts

    def inspect_result(self) -> str:
        """Inspect the latest successful result using a bounded preview and shape profile. Never compares against benchmark gold."""
        if self.state.error is not None or self.state.raw_result is None:
            return json.dumps({"ok": False, "error": {"category": "no_successful_result", "message": "Call run_code successfully first."}})
        value = self.state.structured_result
        requirements, warnings, coverage = self._coverage(value)
        warnings = list(dict.fromkeys([
            *warnings,
            *self._blocking_contract_code_warnings(),
            *self._validate_contract_result(value),
        ]))
        contract_advisories = list(dict.fromkeys([
            *self.state.contract_advisories,
            *self.state.contract_code_warnings,
        ]))
        contract_evidence, evidence_advisories = self._contract_evidence(value)
        # Preserve the detailed map for diagnostics/telemetry, but keep it out
        # of the tool response. The paired A/B replay showed that exposing this
        # verbose structure to the coder added tokens without improving results.
        self.state.contract_evidence = contract_evidence
        self.state.contract_evidence_advisories = evidence_advisories
        self.state.coverage_requirements = requirements
        self.state.coverage_warnings = warnings
        profile: dict[str, Any] = {
            "result_type": type(value).__name__ if value is not None else "text",
            "characters": len(self.state.raw_result),
            "lines": len(self.state.raw_result.splitlines()),
            "preview": self.state.raw_result[:1500],
            "coverage": coverage,
            "requirements": requirements,
            "coverage_warnings": warnings,
            "contract_advisories": contract_advisories,
            "correction_required": bool(warnings),
        }
        if isinstance(value, dict):
            profile["keys"] = list(value)[:30]
            profile["size"] = len(value)
        elif isinstance(value, list):
            profile["items"] = len(value)
            profile["sample"] = value[:5]
            if value and isinstance(value[0], dict):
                profile["columns"] = list(value[0])[:30]
        self.state.inspected_version = self.state.result_version
        self.state.lifecycle = (
            CoderLifecycle.NEEDS_REVISION
            if warnings else CoderLifecycle.READY_TO_FINISH
        )
        return json.dumps({
            "ok": True,
            "state": self.state.lifecycle.value,
            "required_action": "run_code_or_reject_tables" if warnings else "finish_code",
            "allowed_actions": ["run_code", "reject_tables"] if warnings else ["finish_code"],
            "profile": profile,
        }, ensure_ascii=False, default=str)

    def _blocking_contract_code_warnings(self) -> list[str]:
        """Promote only objective code/contract violations to retry blockers."""
        blocking_prefixes = (
            "contract_filter_missing_in_code:",
            "contract_distinct_count_missing_in_code",
            "count_semantics_check:",
            "unsupported_distinct_semantics:",
            "contract_average_missing_in_code",
            "unsupported_bucket_assumption:",
            "contract_join_missing_in_code",
            "contract_grouping_missing_in_code",
            "time_range_check:",
            "contract_limit_missing_in_code:",
            "contract_ordering_missing_in_code",
            "top_n_check:",
            "selection_strategy_not_evident_in_code:",
        )
        return [
            warning for warning in self.state.contract_code_warnings
            if warning.startswith(blocking_prefixes)
        ]

    def _validate_contract_result(self, value: Any) -> list[str]:
        contract = self.state.analysis_contract
        warnings: list[str] = []
        limit = contract.get("limit")
        count = self._semantic_item_count(value)
        if limit is not None and count is not None and count != limit:
            warnings.append(
                f"contract_result_limit_mismatch: expected {limit}, found {count}"
            )
        if contract.get("group_by") and isinstance(value, list) and value:
            if not all(isinstance(item, dict) for item in value):
                warnings.append("contract_grouped_result_requires_record_rows")
                return warnings

        if not isinstance(value, list) or not value or not all(
            isinstance(item, dict) for item in value
        ):
            if isinstance(value, float) and not math.isfinite(value):
                warnings.append("contract_result_non_finite_numeric_value")
            return warnings

        fields = list(dict.fromkeys(
            str(field) for row in value for field in row
        ))

        def matching_fields(label: str) -> list[str]:
            normalized = re.sub(r"[^a-z0-9]+", " ", label.casefold()).strip()
            if normalized in {"", "result", "requested result", "value", "requested value"}:
                return []
            return [field for field in fields if self._label_is_evident(label, [field])]

        for required in contract.get("output_columns", []):
            if not matching_fields(str(required)) and str(required).casefold().strip() not in {
                "requested result", "requested value", "result", "value",
            }:
                warnings.append(
                    f"contract_result_missing_output_column: {required}; "
                    f"available={fields[:20]}"
                )

        group_fields: list[str] = []
        for dimension in contract.get("group_by", []):
            matches = matching_fields(str(dimension))
            if len(matches) == 1:
                group_fields.append(matches[0])
        if group_fields:
            keys = [tuple(row.get(field) for field in group_fields) for row in value]
            if len(set(keys)) != len(keys):
                warnings.append(
                    "contract_result_duplicate_group_keys: " + ", ".join(group_fields)
                )

        measure_fields: list[str] = []
        for measure in [*contract.get("measures", []), *contract.get("output_columns", [])]:
            for field in matching_fields(str(measure)):
                if field not in measure_fields:
                    measure_fields.append(field)
        for field in measure_fields:
            numeric = [
                float(row[field]) for row in value
                if field in row and isinstance(row[field], (int, float))
                and not isinstance(row[field], bool)
            ]
            if numeric and not any(math.isfinite(number) for number in numeric):
                warnings.append(f"contract_result_all_non_finite: {field}")

        ordering = str(contract.get("ordering") or "none").casefold()
        if ordering not in {"", "none", "not applicable", "n/a"}:
            ordering_matches = [
                field for field in measure_fields
                if self._label_is_evident(ordering, [field])
            ]
            candidates = ordering_matches or measure_fields
            numeric_candidates = []
            for field in candidates:
                values = [row.get(field) for row in value]
                if values and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in values
                ):
                    numeric_candidates.append(field)
            if len(numeric_candidates) == 1:
                field = numeric_candidates[0]
                numbers = [float(row[field]) for row in value]
                descending = any(term in ordering for term in (
                    "desc", "highest", "largest", "most", "top"
                ))
                ascending = any(term in ordering for term in (
                    "asc", "lowest", "smallest", "least", "bottom"
                ))
                ordered = (
                    all(left >= right for left, right in zip(numbers, numbers[1:]))
                    if descending else
                    all(left <= right for left, right in zip(numbers, numbers[1:]))
                    if ascending else True
                )
                if not ordered:
                    direction = "descending" if descending else "ascending"
                    warnings.append(
                        f"contract_result_order_mismatch: {field} is not {direction}"
                    )
        return warnings

    def finish_code(
        self,
        filters: ReviewStatus,
        measures: ReviewStatus,
        grouping: ReviewStatus,
        ordering: ReviewStatus,
        output_shape: ReviewStatus,
        requirement_reviews: dict[str, str],
        review: str,
    ) -> str:
        """Finish only after inspecting the latest result and completing every review dimension."""
        checks = {
            "filters": filters,
            "measures": measures,
            "grouping": grouping,
            "ordering": ordering,
            "output_shape": output_shape,
        }
        if self.state.error is not None or self.state.raw_result is None:
            raise ValueError("Finish blocked: there is no successful execution result.")
        if self.state.inspected_version != self.state.result_version:
            raise ValueError("Finish blocked: inspect_result must inspect the latest successful run.")
        if self.state.coverage_warnings:
            raise ValueError(
                "Finish blocked: correct the code and call run_code again. "
                + "; ".join(self.state.coverage_warnings)
            )
        unresolved = [name for name, status in checks.items() if status == "needs_revision"]
        if unresolved:
            self.state.lifecycle = CoderLifecycle.NEEDS_REVISION
            raise ValueError("Finish blocked: revise code for " + ", ".join(unresolved) + ".")
        if len(review.strip()) < 20:
            raise ValueError("Finish blocked: provide a concrete review of at least 20 characters.")
        reviewed = {
            str(requirement).strip(): str(evidence).strip()
            for requirement, evidence in requirement_reviews.items()
            if str(requirement).strip()
        }
        missing_requirements = [
            requirement for requirement in self.state.coverage_requirements
            if requirement not in reviewed
        ]
        if missing_requirements:
            raise ValueError(
                "Finish blocked: review every inspect_result requirement: "
                + "; ".join(missing_requirements)
            )
        if any(len(evidence) < 8 for evidence in reviewed.values()):
            raise ValueError("Finish blocked: provide evidence for every requirement.")
        self.state.finished = True
        self.state.lifecycle = CoderLifecycle.FINISHED
        self.state.finalization_mode = "agent"
        self.state.review = {
            **checks,
            "requirement_reviews": reviewed,
            "review": review.strip(),
        }
        return "FINAL_PAYLOAD: " + json.dumps({"status": "finished", "review": self.state.review})

    def recover_finish(self, reason: str) -> None:
        """Close a valid inspected result when only the agent protocol failed."""
        if not self.state.ready_for_finalization():
            raise ValueError("Recovery finalization requires the latest warning-free inspected result.")
        self.state.finished = True
        self.state.lifecycle = CoderLifecycle.FINISHED
        self.state.finalization_mode = "system_recovery"
        self.state.review = {
            "finalization_mode": "system_recovery",
            "coverage_checks": "verified",
            "semantic_self_review": "not_available",
            "requirement_reviews": {
                requirement: "Verified by inspect_result coverage checks."
                for requirement in self.state.coverage_requirements
            },
            "review": reason,
        }

    def recover_degraded_finish(self, reason: str) -> None:
        """Preserve an inspected structured result without declaring it correct."""
        if not self.state.ready_for_degraded_finalization():
            raise ValueError("Degraded recovery requires the latest inspected structured result.")
        self.state.finished = True
        self.state.lifecycle = CoderLifecycle.FINISHED
        self.state.finalization_mode = "system_recovery_with_advisories"
        self.state.review = {
            "finalization_mode": "system_recovery_with_advisories",
            "coverage_checks": "advisories_preserved",
            "semantic_self_review": "incomplete",
            "coverage_warnings": list(self.state.coverage_warnings),
            "contract_advisories": list(dict.fromkeys([
                *self.state.contract_advisories,
                *self.state.contract_code_warnings,
            ])),
            "review": reason,
        }

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.set_analysis_contract,
                fn_schema=AnalysisContractSchema,
            ),
            FunctionTool.from_defaults(fn=self.inspect_table),
            FunctionTool.from_defaults(fn=self.run_code),
            FunctionTool.from_defaults(
                fn=self.reject_tables,
                fn_schema=RejectTablesSchema,
                return_direct=True,
            ),
            FunctionTool.from_defaults(fn=self.inspect_result),
            FunctionTool.from_defaults(fn=self.finish_code, fn_schema=FinishCodeSchema, return_direct=True),
        ]
