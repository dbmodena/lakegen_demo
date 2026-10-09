"""The coder's tool manager: the brief it starts from, the tools it offers (inspect, run, reject)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from lakegen.coding.tools.contract import ContractMixin
from lakegen.coding.tools.error_repair import ErrorRepairMixin
from lakegen.coding.tools.finish import FinishMixin
from lakegen.coding.tools.inspection import InspectionMixin
from lakegen.coding.tools.plan_grounding import PlanGroundingMixin
from lakegen.coding.tools.rejection import RejectionMixin
from lakegen.coding.tools.result_checks import ResultChecksMixin
from lakegen.coding.tools.sandbox import SandboxMixin
from lakegen.coding.tools.state import P3State
from lakegen.integrations.table_io import read_table


class RejectTablesSchema(BaseModel):
    reason: str = Field(description="Why the selected tables cannot answer the question.")
    missing_requirements: list[str] = Field(description="Concrete missing columns, periods, categories, or join keys.")
    inspected_evidence: str = Field(description="Evidence observed through inspect_table or run_code.")
    ban_tables: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Selected tables you judge highly irrelevant -- proven to satisfy "
            "none of the requirements -- mapped to the concrete evidence "
            "proving that. Every other selected table is kept and carries "
            "over to the next attempt, even if it does not fully answer the "
            "question -- only actively-proven-irrelevant tables belong here. "
            "A table without a concrete justification is not banned."
        ),
    )


class Phase3ToolsManager(PlanGroundingMixin, ContractMixin, SandboxMixin, ErrorRepairMixin, InspectionMixin, RejectionMixin, ResultChecksMixin, FinishMixin):
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
        require_semantic_plan: bool = True,
        require_analysis_manifest: bool = False,
        allow_semantic_revision: bool = True,
        brief_advisory: bool = False,
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
        self.require_semantic_plan = require_semantic_plan
        self.require_analysis_manifest = require_analysis_manifest
        self.allow_semantic_revision = allow_semantic_revision
        self.brief_advisory = brief_advisory
        self._snapshot_years_cache: set[str] | None = None
        self.original_coder_brief_status = (
            "provided" if isinstance(self.selection_plan.get("coder_brief"), dict)
            and bool(self.selection_plan.get("coder_brief")) else
            "legacy_plan" if isinstance(self.selection_plan.get("semantic_plan"), dict)
            and bool(self.selection_plan.get("semantic_plan")) else
            "missing"
        )
        coder_brief = self.selection_plan.get("coder_brief")
        if (
            not coder_brief
            and isinstance(self.selection_plan.get("semantic_plan"), dict)
            and self.selection_plan["semantic_plan"]
        ):
            coder_brief = self._legacy_semantic_plan_to_coder_brief(
                self.selection_plan["semantic_plan"]
            )
            self.selection_plan["coder_brief"] = coder_brief
        if not coder_brief and self.tables and self.require_semantic_plan:
            coder_brief = self._runtime_fallback_coder_brief()
            self.selection_plan["coder_brief"] = coder_brief
        if isinstance(coder_brief, dict):
            coder_brief = self._normalize_coder_brief(coder_brief)
            selected = coder_brief.get("selected_columns")
            if not isinstance(selected, dict) or not any(
                isinstance(columns, list) and columns for columns in selected.values()
            ):
                coder_brief["selected_columns"] = {
                    table: [str(column) for column in read_table(
                        self.csv_dir / table.strip(), nrows=0
                    ).columns]
                    for table in self.tables
                }
                coder_brief.setdefault("runtime_obligations", []).append(
                    "Choose the exact columns needed by the question from the runtime schema."
                )
            self._infer_unique_join(coder_brief)
            self.selection_plan["coder_brief"] = coder_brief
        if not isinstance(self.selection_plan.get("requirement_ledger"), list):
            self.selection_plan["requirement_ledger"] = (
                self._runtime_fallback_requirement_ledger()
            )
        brief_validation = self._validate_coder_brief_grounding(coder_brief)
        if brief_validation.get("valid"):
            semantic_validation = self._validate_semantic_plan_grounding(coder_brief)
            combined_warnings = [
                *brief_validation.get("warnings", []),
                *semantic_validation.get("warnings", []),
            ]
            if combined_warnings and semantic_validation.get("status") == "verified":
                semantic_validation["status"] = "executable_with_obligations"
                semantic_validation["required_action"] = "resolve_runtime_obligations"
            semantic_validation["warnings"] = combined_warnings
            semantic_validation["runtime_obligations"] = combined_warnings
            self.state.plan_validation = {
                **semantic_validation,
                "contract_type": "coder_brief",
                "data_reference_evidence": brief_validation.get("evidence", []),
            }
            self.state.architect_contract_locked = bool(
                self.state.plan_validation.get("valid")
            )
            if self.state.architect_contract_locked:
                self.state.analysis_contract = self._contract_from_coder_brief(coder_brief)
        else:
            self.state.plan_validation = brief_validation

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.inspect_data),
            FunctionTool.from_defaults(fn=self.run_analysis),
            FunctionTool.from_defaults(
                fn=self.reject_data,
                fn_schema=RejectTablesSchema,
                return_direct=True,
            ),
        ]
