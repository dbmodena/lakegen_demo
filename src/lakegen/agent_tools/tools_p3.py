"""Bounded tools used by the agentic code-generation phase."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field


ReviewStatus = Literal["verified", "not_applicable", "needs_revision"]


class FinishCodeSchema(BaseModel):
    filters: ReviewStatus = Field(description="Verify every requested filter and time range.")
    measures: ReviewStatus = Field(description="Verify measures, aggregation and units.")
    grouping: ReviewStatus = Field(description="Verify requested groups and category coverage.")
    ordering: ReviewStatus = Field(description="Verify ranking, sorting, top/bottom and limits.")
    output_shape: ReviewStatus = Field(description="Verify that the output shape answers the question.")
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
    review: dict[str, str] = field(default_factory=dict)


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
        resolve_code: Callable[[str, list[str], Path], tuple[str, str | None]],
        execute_code: Callable[..., tuple[str | None, str | None, str]],
        extract_payload: Callable[[str], tuple[str, object | None, str | None]],
    ):
        self.state = state
        self.tables = tables
        self.csv_dir = Path(csv_dir)
        self.run_dir = run_dir
        self.evaluation_result_type = evaluation_result_type
        self.resolve_code = resolve_code
        self.execute_code = execute_code
        self.extract_payload = extract_payload

    def run_code(self, code: str) -> str:
        """Execute a complete Python analysis. At most three calls are allowed; fix structured errors before retrying."""
        if self.state.finished:
            return json.dumps({"ok": False, "error": {"category": "already_finished"}})
        if self.state.run_count >= self.state.max_runs:
            return json.dumps({"ok": False, "error": {"category": "run_limit", "message": "Maximum of 3 executions reached."}})

        self.state.run_count += 1
        self.state.code_raw = code
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

    def inspect_result(self) -> str:
        """Inspect the latest successful result using a bounded preview and shape profile. Never compares against benchmark gold."""
        if self.state.error is not None or self.state.raw_result is None:
            return json.dumps({"ok": False, "error": {"category": "no_successful_result", "message": "Call run_code successfully first."}})
        value = self.state.structured_result
        profile: dict[str, Any] = {
            "result_type": type(value).__name__ if value is not None else "text",
            "characters": len(self.state.raw_result),
            "lines": len(self.state.raw_result.splitlines()),
            "preview": self.state.raw_result[:1500],
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
        unresolved = [name for name, status in checks.items() if status == "needs_revision"]
        if unresolved:
            raise ValueError("Finish blocked: revise code for " + ", ".join(unresolved) + ".")
        if len(review.strip()) < 20:
            raise ValueError("Finish blocked: provide a concrete review of at least 20 characters.")
        self.state.finished = True
        self.state.review = {**checks, "review": review.strip()}
        return "FINAL_PAYLOAD: " + json.dumps({"status": "finished", "review": self.state.review})

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.run_code),
            FunctionTool.from_defaults(fn=self.inspect_result),
            FunctionTool.from_defaults(fn=self.finish_code, fn_schema=FinishCodeSchema, return_direct=True),
        ]
