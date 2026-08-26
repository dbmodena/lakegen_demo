"""Bounded tools used by the agentic code-generation phase."""

from __future__ import annotations

import json
import ast
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
import pandas as pd

from lakegen.core.table_io import read_table, table_load_command


ReviewStatus = Literal["verified", "not_applicable", "needs_revision"]


class FinishCodeSchema(BaseModel):
    filters: ReviewStatus = Field(description="Verify every requested filter and time range.")
    measures: ReviewStatus = Field(description="Verify measures, aggregation and units.")
    grouping: ReviewStatus = Field(description="Verify requested groups and category coverage.")
    ordering: ReviewStatus = Field(description="Verify ranking, sorting, top/bottom and limits.")
    output_shape: ReviewStatus = Field(description="Verify that the output shape answers the question.")
    requirement_reviews: dict[str, str] = Field(description="Map every exact requirement reported by inspect_result to concrete evidence from the result.")
    review: str = Field(description="Brief evidence-based final review of the computed result.")


def classify_execution_error(message: str, *, stage: str = "execution") -> dict[str, Any]:
    """Convert an execution/preflight failure into a stable, compact schema."""

    text = str(message or "Unknown execution error")
    lowered = text.casefold()
    category = "runtime_error"
    if stage == "preflight":
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
    table_inspection_cache: dict[str, str] = field(default_factory=dict)

    def ready_for_finalization(self) -> bool:
        """Return whether a closure-only turn is safe and meaningful."""
        return (
            not self.finished
            and self.error is None
            and self.structured_result is not None
            and self.result_version > 0
            and self.inspected_version == self.result_version
            and not self.coverage_warnings
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
        self.resolve_code = resolve_code
        self.execute_code = execute_code
        self.extract_payload = extract_payload

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

    def _enrich_column_error(self, error: dict[str, Any]) -> None:
        column = error.get("column")
        if not column:
            return
        columns = self._all_columns()
        error["closest_columns"] = difflib.get_close_matches(
            str(column).strip(), columns, n=5, cutoff=0.5
        )
        error["available_columns"] = columns[:100]

    def inspect_table(self, file_name: str, columns: str = "") -> str:
        """Inspect one selected table once, without consuming a run_code attempt. Columns is a comma-separated shortlist."""
        table = file_name.strip()
        if table not in self.tables:
            return json.dumps({"ok": False, "error": "Only selected tables may be inspected.", "selected_tables": self.tables})
        if table in self.state.table_inspection_cache:
            cached = json.loads(self.state.table_inspection_cache[table])
            cached["cached"] = True
            cached["next_action"] = (
                "This table was already inspected. Do not inspect it again; "
                "write or correct the analysis with run_code."
            )
            return json.dumps(cached, ensure_ascii=False, default=str)
        self.state.inspected_tables.add(table)
        path = self.csv_dir / table
        try:
            frame = read_table(path, nrows=5000)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        requested = [item.strip() for item in columns.split(",") if item.strip()]
        exact = {str(column): column for column in frame.columns}
        details: dict[str, Any] = {}
        for requested_column in requested[:8]:
            matches = difflib.get_close_matches(requested_column.strip(), list(exact), n=1, cutoff=0.55)
            if not matches:
                details[requested_column] = {"error": "column_not_found"}
                continue
            name = matches[0]
            series = frame[exact[name]]
            non_null = series.dropna()
            detail: dict[str, Any] = {
                "exact_name": name,
                "dtype": str(series.dtype),
                "null_count_in_sample": int(series.isna().sum()),
                "distinct_count_in_sample": int(non_null.nunique()),
                "sample_values": [str(value)[:100] for value in non_null.drop_duplicates().head(12)],
            }
            if not non_null.empty and ("date" in name.casefold() or "year" in name.casefold()):
                parsed = pd.to_datetime(non_null, errors="coerce").dropna()
                if not parsed.empty:
                    detail["temporal_min"] = str(parsed.min())
                    detail["temporal_max"] = str(parsed.max())
            details[requested_column] = detail
        response = json.dumps({
            "ok": True,
            "table": table,
            "load_command": table_load_command(path),
            "sampled_rows": len(frame),
            "columns": [str(column) for column in frame.columns],
            "requested_column_profiles": details,
        }, ensure_ascii=False, default=str)
        self.state.table_inspection_cache[table] = response
        return response

    def run_code(self, code: str) -> str:
        """Execute a complete Python analysis. At most three calls are allowed; fix structured errors before retrying."""
        if self.state.finished:
            return json.dumps({"ok": False, "error": {"category": "already_finished"}})
        if self.state.run_count >= self.state.max_runs:
            return json.dumps({"ok": False, "error": {"category": "run_limit", "message": "Maximum of 3 executions reached."}})

        self.state.run_count += 1
        self.state.code_raw = code
        path_error = self._validate_load_paths(code)
        if path_error:
            self.state.error = path_error["message"]
            self.state.execution_error = path_error
            return json.dumps({"ok": False, "attempt": self.state.run_count, "error": path_error})
        resolved, preflight_error = self.resolve_code(code, self.tables, self.csv_dir)
        self.state.clean_code = resolved
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
        self.state.result_version += 1
        return json.dumps({
            "ok": True,
            "attempt": self.state.run_count,
            "result_available": self.state.raw_result is not None,
            "next_action": "Call inspect_result before finish_code.",
        })

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
        top_match = re.search(r"\btop\s+(\d+)\b", question)
        expected_top = int(top_match.group(1)) if top_match else None
        if expected_top is None and any(term in question for term in ("highest", "lowest", "most", "least", "largest", "smallest")):
            for word, number in number_words.items():
                if re.search(rf"\b{word}\b", question):
                    expected_top = number
                    break
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
        return json.dumps({"ok": True, "profile": profile}, ensure_ascii=False, default=str)

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
        self.state.review = {
            **checks,
            "requirement_reviews": reviewed,
            "review": review.strip(),
        }
        return "FINAL_PAYLOAD: " + json.dumps({"status": "finished", "review": self.state.review})

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.inspect_table),
            FunctionTool.from_defaults(fn=self.run_code),
            FunctionTool.from_defaults(fn=self.inspect_result),
            FunctionTool.from_defaults(fn=self.finish_code, fn_schema=FinishCodeSchema, return_direct=True),
        ]
