"""Running the coder's code: the only place generated code executes, with its load-path and manifest checks."""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any

from lakegen.integrations.table_io import table_load_command
from lakegen.tools.coder.execution.error_repair import classify_execution_error
from lakegen.tools.coder.quality.data_quality import snapshot_stamp_years
from lakegen.tools.coder.quality.revision_policy import classify_revision, technical_repair_eligible
from lakegen.tools.coder.state import CoderLifecycle


class SandboxMixin:
    """Part of ``Phase3ToolsManager``; the manager supplies the attributes it reads."""

    def _record_execution_attempt(self) -> None:
        """Persist one benchmark-blind snapshot for post-hoc attempt metrics."""
        self.state.execution_attempts.append({
            "attempt": self.state.run_count,
            "generation_success": True,
            "execution_success": self.state.error is None and self.state.raw_result is not None,
            "structured_output_valid": self.state.structured_result is not None,
            "code": self.state.clean_code,
            "structured_result": copy.deepcopy(self.state.structured_result),
            "error": self.state.error or self.state.structured_result_error,
            "execution_error": copy.deepcopy(self.state.execution_error),
        })

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

    def run_code(self, code: str) -> str:
        """Execute analysis after a verified plan or an executable plan with obligations."""
        refusal = self._refusal_before_run()
        if refusal is not None:
            return refusal
        self.state.lifecycle = CoderLifecycle.NEEDS_REVISION
        self.state.code_raw = code
        resolved, preflight_failure = self._preflight(code)
        if preflight_failure is not None:
            return preflight_failure
        self.state.run_count += 1
        raw_result, error, clean_code = self._execute_with_repair(resolved)
        if error is not None:
            return self._failed_run(error)
        self._reset_result(raw_result)
        manifest_failure = self._take_manifest(clean_code)
        if manifest_failure is not None:
            return manifest_failure
        diagnostic_failure = self._take_payload()
        if diagnostic_failure is not None:
            return diagnostic_failure
        self.state.result_version += 1
        self.state.operation_trace = self._build_operation_trace(clean_code)
        self.state.lifecycle = CoderLifecycle.NEEDS_INSPECTION
        self._record_execution_attempt()
        return json.dumps({
            "ok": True,
            "attempt": self.state.run_count,
            "result_available": self.state.raw_result is not None,
            "next_action": "run_analysis inspects the result; finalization is automatic when validation passes.",
        })

    def _refusal_before_run(self) -> str | None:
        """Why no run happens now: the run is over, the tables rejected, the plan not grounded, the runs spent."""
        if self.state.finished:
            return json.dumps({"ok": False, "error": {"category": "already_finished"}})
        if self.state.rejected_reason:
            return json.dumps({"ok": False, "error": {"category": "tables_already_rejected"}})
        validation = self.state.plan_validation
        grounded = (
            validation.get("valid")
            and validation.get("status") in {"verified", "executable_with_obligations"}
            and self.state.architect_contract_locked
        )
        if self.require_semantic_plan and not grounded:
            return json.dumps({
                "ok": False,
                "error": {
                    "category": "semantic_plan_not_grounded",
                    "diagnostics": validation.get("diagnostics", []),
                    "correction_required": (
                        "Call inspect_data first, then run_analysis; "
                        "or call reject_data with evidence."
                    ),
                    "next_actions": (
                        ["run_analysis", "reject_data"]
                        if self.state.inspected_tables else
                        ["inspect_data", "reject_data"]
                    ),
                },
            }, ensure_ascii=False, default=str)
        if not self.state.analysis_contract:
            self.infer_analysis_contract()
        if self.state.run_count >= self.state.max_runs:
            return json.dumps({"ok": False, "error": {
                "category": "run_limit",
                "message": f"Maximum of {self.state.max_runs} executions reached.",
            }})
        return None

    def _not_consumed(self, error: dict[str, Any], **dumps: Any) -> str:
        """A refusal before execution: the attempt is not spent."""
        self.state.error = error["message"]
        self.state.execution_error = error
        return json.dumps({
            "ok": False, "attempt": self.state.run_count,
            "execution_consumed": False, "error": error,
        }, **dumps)

    def _preflight(self, code: str) -> tuple[str, str | None]:
        """The code resolved for execution, or the refusal that stops it before any run."""
        path_error = self._validate_load_paths(code)
        if path_error:
            return "", self._not_consumed(path_error)
        edition_filter_error = self._validate_dataset_edition_filter(code)
        if edition_filter_error:
            return "", self._not_consumed(edition_filter_error)
        resolved, preflight_error = self.resolve_code(code, self.tables, self.csv_dir)
        self.state.clean_code = resolved
        self.state.contract_code_warnings = self._validate_contract_code(resolved)
        if preflight_error:
            self.state.error = preflight_error
            self.state.execution_error = classify_execution_error(preflight_error, stage="preflight")
            return resolved, json.dumps({
                "ok": False, "attempt": self.state.run_count,
                "execution_consumed": False, "error": self.state.execution_error,
            })
        contract_gaps = self._blocking_contract_code_warnings() if self.require_semantic_plan else []
        if contract_gaps and not self.state.preflight_revision_used:
            self.state.preflight_revision_used = True
            message = "Code does not yet satisfy the compiled runtime checklist."
            return resolved, self._not_consumed({
                "stage": "preflight", "category": "pre_execution_contract_gap",
                "message": message, "missing": contract_gaps,
                "runtime_checklist": self._runtime_checklist(),
                "retryable": True, "revision_budget_remaining": 0,
                "repair_hint": "Correct every listed gap in one revision, then call run_analysis again.",
            }, ensure_ascii=False, default=str)
        return resolved, None

    def _execute_with_repair(self, resolved: str) -> tuple[str | None, str | None, str]:
        """One execution; a renamed column that the code can be patched for is retried once."""
        raw_result, error, clean_code = self.execute_code(resolved, run_dir=self.run_dir)
        self.state.clean_code = clean_code
        if error is not None:
            patched = self._auto_repair_renamed_column(clean_code, error)
            if patched is not None:
                retry_result, retry_error, retry_code = self.execute_code(patched, run_dir=self.run_dir)
                if retry_error is None:
                    raw_result, error, clean_code = retry_result, None, retry_code
                    self.state.clean_code = clean_code
                    self.state.result_adaptations.append("auto_repair_renamed_column")
        return raw_result, error, clean_code

    def _failed_run(self, error: str) -> str:
        """The error enriched for the coder, the revision decided, a technical repair credited once."""
        state = self.state
        state.error = error
        state.raw_result = None
        state.execution_error = classify_execution_error(error)
        self._enrich_column_error(state.execution_error)
        self._enrich_join_error(state.execution_error)
        self._enrich_type_error(state.execution_error)
        self._reject_after_repeated_missing_column(state.execution_error)
        decision = classify_revision(
            execution_error=state.execution_error,
            rejection_details=state.rejection_details,
        )
        if (
            technical_repair_eligible(state.execution_error)
            and not state.column_repair_credit_used
            and not state.best_result_snapshot
            and not state.rejected_reason
        ):
            state.column_repair_credit_used = True
            state.run_count = max(0, state.run_count - 1)
            state.execution_error["execution_consumed"] = False
            state.execution_error["dedicated_technical_repair"] = True
            state.execution_error["repair_runs_remaining"] = state.max_runs - state.run_count
            decision["credit_consumed"] = "technical"
        state.revision_decisions.append(decision)
        self._record_execution_attempt()
        return json.dumps({"ok": False, "attempt": state.run_count, "error": state.execution_error})

    def _reset_result(self, raw_result: str | None) -> None:
        state = self.state
        state.error = None
        state.execution_error = {}
        state.raw_result = raw_result
        state.structured_result = None
        state.structured_result_error = ""
        state.result_adaptations = []
        state.result_adaptation_notes = []
        state.analysis_manifest = {}
        state.analysis_manifest_validation = {}

    def _take_manifest(self, clean_code: str) -> str | None:
        """Read the analysis manifest out of the output when one is required; the failure when it is wrong."""
        state = self.state
        if state.raw_result is None:
            return None
        manifest_match = re.search(
            r"^__LAKEGEN_ANALYSIS_MANIFEST__(\{.*\})$", state.raw_result, flags=re.MULTILINE,
        )
        if self.require_analysis_manifest and manifest_match:
            try:
                candidate = json.loads(manifest_match.group(1))
                if isinstance(candidate, dict):
                    allowed = {
                        "used_tables", "used_columns", "filters", "joins",
                        "grouping", "aggregations", "ordering", "limit",
                        "output_columns", "result_type",
                    }
                    state.analysis_manifest = {key: candidate[key] for key in allowed if key in candidate}
            except (TypeError, ValueError, json.JSONDecodeError):
                state.analysis_manifest = {}
            state.raw_result = re.sub(
                r"^__LAKEGEN_ANALYSIS_MANIFEST__\{.*\}\s*$", "", state.raw_result, flags=re.MULTILINE,
            ).strip()
        manifest_errors, manifest_warnings = self._validate_analysis_manifest(state.analysis_manifest, clean_code)
        state.analysis_manifest_validation = {
            "valid": not manifest_errors,
            "errors": manifest_errors,
            "warnings": manifest_warnings,
        }
        if not (self.require_analysis_manifest and manifest_errors):
            return None
        state.error = (
            "A final analysis must emit a complete, trace-compatible "
            "__LAKEGEN_ANALYSIS_MANIFEST__."
        )
        state.raw_result = None
        state.execution_error = {
            "stage": "result_validation",
            "category": "manifest_missing" if not state.analysis_manifest else "manifest_invalid",
            "message": state.error,
            "diagnostics": manifest_errors,
            "retryable": state.run_count < state.max_runs,
            "repair_hint": (
                "Emit one compact JSON line prefixed by "
                "__LAKEGEN_ANALYSIS_MANIFEST__ before the evaluation payload."
            ),
        }
        self._record_execution_attempt()
        return json.dumps({
            "ok": False, "attempt": state.run_count, "error": state.execution_error,
        }, ensure_ascii=False)

    DIAGNOSTIC_MARKERS = (
        "columns:", "unique ", "unique values", "sample values",
        "matches count", "potential ", "roof cols:", "cellar cols:",
        "empty dataframe", "dtype:", "length:",
    )
    """What an output that only describes the data, not an answer, tends to say."""

    def _take_payload(self) -> str | None:
        """The structured result out of the output; the failure when the output is only a diagnostic."""
        state = self.state
        if state.raw_result is None:
            return None
        display, structured, payload_error = self.extract_payload(state.raw_result)
        structured, adaptations = self._adapt_structured_result(structured)
        state.raw_result = display
        state.structured_result = structured
        state.result_adaptations = adaptations
        state.result_adaptation_notes = self._result_adapter_notes(structured, adaptations)
        state.structured_result_error = payload_error or ""
        diagnostic_text = display.strip().casefold()
        if structured is not None or not any(marker in diagnostic_text for marker in self.DIAGNOSTIC_MARKERS):
            return None
        state.error = (
            "Diagnostic output is not a final answer. Use inspect_data "
            "for schema/value evidence, then choose run_analysis with corrected "
            "analysis or reject_data if required data is unavailable."
        )
        state.execution_error = {
            "stage": "result_validation",
            "category": "diagnostic_output",
            "message": state.error,
            "retryable": state.run_count < state.max_runs,
            "next_actions": ["inspect_data", "run_analysis", "reject_data"],
        }
        state.raw_result = None
        self._record_execution_attempt()
        return json.dumps({
            "ok": False,
            "attempt": state.run_count,
            "error": state.execution_error,
        })

    def _validate_analysis_manifest(
        self, manifest: dict[str, Any], code: str
    ) -> tuple[list[str], list[str]]:
        """Validate required runtime claims and compare them with the executed trace."""
        if not manifest:
            return ["manifest is missing"], []
        required = {"used_tables", "used_columns", "filters", "aggregations", "result_type"}
        errors = [f"missing required field: {key}" for key in sorted(required - set(manifest))]
        list_fields = ("used_tables", "used_columns", "filters", "aggregations")
        for key in list_fields:
            if key in manifest and not isinstance(manifest[key], list):
                errors.append(f"{key} must be a list")
        if not isinstance(manifest.get("result_type"), str) or not str(manifest.get("result_type", "")).strip():
            errors.append("result_type must be a non-empty string")
        declared_tables = [str(item) for item in manifest.get("used_tables", [])]
        declared_columns = [str(item) for item in manifest.get("used_columns", [])]
        unknown_tables = sorted(set(declared_tables) - set(self.tables))
        unknown_columns = sorted(set(declared_columns) - set(self._all_columns()))
        if unknown_tables:
            errors.append("unknown used_tables: " + ", ".join(unknown_tables))
        if unknown_columns:
            errors.append("unknown used_columns: " + ", ".join(unknown_columns))
        lowered = code.casefold()
        warnings = [
            f"declared table not visible in executed code: {table}"
            for table in declared_tables if table.casefold() not in lowered
        ]
        warnings.extend(
            f"declared column not visible in executed code: {column}"
            for column in declared_columns if column.casefold() not in lowered
        )
        return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))

    def _build_operation_trace(self, code: str) -> dict[str, Any]:
        """Build bounded, non-gold evidence from the executed program."""
        lowered = code.casefold()
        contract = self.state.analysis_contract
        contract_text = json.dumps(contract, ensure_ascii=False, default=str).casefold()
        added_operations: list[str] = []
        operation_markers = {
            "correlation": (".corr(", "pearsonr(", "spearmanr("),
            "ratio": ("ratio",),
            "final_reduction": (".mean(", ".sum(", ".median(", ".max(", ".min("),
            "ordering": ("sort_values(", "nlargest(", "nsmallest("),
            "limit": ("head(", "nlargest(", "nsmallest("),
        }
        declared_terms = {
            "correlation": ("correlation", "pearson", "spearman"),
            "ratio": ("ratio", "divide"),
            "final_reduction": ("mean", "average", "sum", "median", "max", "min"),
            "ordering": ("ordering", "sort", "highest", "lowest"),
            "limit": ("limit", "top"),
        }
        for operation, markers in operation_markers.items():
            if any(marker in lowered for marker in markers) and not any(
                term in contract_text for term in declared_terms[operation]
            ):
                added_operations.append(operation)
        trace = {
            "used_tables": [t for t in self.tables if t.casefold() in lowered],
            "used_columns": [c for c in self._all_columns() if c.casefold() in lowered],
            "applied_filters": list(contract.get("filters", [])),
            "rows_before_filter": None,
            "rows_after_filter": None,
            "joins": list(contract.get("joins", [])) if any(m in lowered for m in (".merge(", ".join(", "pd.merge(")) else [],
            "grouping_columns": list(contract.get("group_by", [])) if ".groupby(" in lowered else [],
            "aggregations": list(contract.get("measures", [])),
            "output_columns": list(contract.get("output_columns", [])),
            "ordering": contract.get("ordering") if any(m in lowered for m in ("sort_values(", "nlargest(", "nsmallest(")) else None,
            "limit": contract.get("limit") if any(m in lowered for m in ("head(", "nlargest(", "nsmallest(")) else None,
            "result_type": type(self.state.structured_result).__name__ if self.state.structured_result is not None else "text",
            "contract_source": "coder_brief",
            "result_adapter_attempted": True,
            "result_adaptations": list(self.state.result_adaptations),
            "result_adapter_not_applied_reasons": list(self.state.result_adaptation_notes),
            "inferred_plan_completion": {
                "added_operations": added_operations,
                "final_result_type": (
                    type(self.state.structured_result).__name__
                    if self.state.structured_result is not None else "text"
                ),
                "source": "executed_code",
            },
        }
        manifest = self.state.analysis_manifest if self.require_analysis_manifest else {}
        if manifest:
            trace.update({
                "manifest_provided": True,
                "manifest_validation": dict(self.state.analysis_manifest_validation),
            })
            declared_tables = [
                table for table in manifest.get("used_tables", []) if table in self.tables
            ]
            declared_columns = [
                column for column in manifest.get("used_columns", []) if column in self._all_columns()
            ]
            trace.update({
                "used_tables": declared_tables or trace["used_tables"],
                "used_columns": declared_columns or trace["used_columns"],
                "filter_steps": manifest.get("filters", []),
                "joins": manifest.get("joins", trace["joins"]),
                "grouping_columns": manifest.get("grouping", trace["grouping_columns"]),
                "aggregations": manifest.get("aggregations", trace["aggregations"]),
                "ordering": manifest.get("ordering", trace["ordering"]),
                "limit": manifest.get("limit", trace["limit"]),
                "output_columns": manifest.get("output_columns", trace["output_columns"]),
                "result_type": manifest.get("result_type", trace["result_type"]),
            })
        return trace

    def _result_adapter_notes(
        self, value: Any, adaptations: list[str]
    ) -> list[str]:
        if adaptations:
            return []
        notes = ["result already matched a supported runtime representation"]
        ordering = str(self.state.analysis_contract.get("ordering") or "").strip().casefold()
        if ordering and ordering not in {"none", "n/a", "not applicable"}:
            notes.append("ordering field was absent, non-numeric, or ambiguous")
        return notes

    def _snapshot_years(self) -> set[str]:
        if self._snapshot_years_cache is None:
            self._snapshot_years_cache = set().union(*(
                snapshot_stamp_years(self.csv_dir / table.strip())
                for table in self.tables
            )) if self.tables else set()
        return self._snapshot_years_cache

    def run_analysis(self, code: str) -> str:
        """Run code, infer plan completion, and return one compact decision."""
        if (
            (self.selection_plan.get("coder_brief") or {}).get("source") == "runtime_fallback"
            and not self.state.data_inspection_used
        ):
            return json.dumps({
                "status": "inspection_required",
                "message": "The selection supplied no verified computational brief. Use inspect_data once to verify the required columns and filter values before running the analysis.",
                "tables": self.tables,
                "remaining_runs": self.state.max_runs - self.state.run_count,
            })
        execution = json.loads(self.run_code(code))
        if not execution.get("ok"):
            error = execution.get("error", {})
            category = str(error.get("category") or "execution_error")
            if category == "security_error":
                return json.dumps({"status": "security_blocked", "error": error})
            if self.state.run_count >= self.state.max_runs and self.state.best_result_snapshot:
                self._restore_best_result()
                self.recover_degraded_finish(
                    "Preserved the best executed result after the final repair failed."
                )
                return json.dumps({
                    "status": "completed_with_warnings",
                    "warnings": self.state.coverage_warnings,
                    "selected_best_previous_result": True,
                }, ensure_ascii=False)
            return json.dumps({
                "status": "revision_required", "all_problems": [error],
                "remaining_runs": self.state.max_runs - self.state.run_count,
                "revision_decision": (
                    self.state.revision_decisions[-1]
                    if self.state.revision_decisions else
                    classify_revision(execution_error=error)
                ),
            }, ensure_ascii=False, default=str)

        inspection = json.loads(self.inspect_result())
        self._remember_best_result()
        warnings = list(self.state.coverage_warnings)
        if not warnings:
            self.state.revision_decisions.append(classify_revision())
            self.finalize_validated_result(
                "run_analysis finalized a warning-free inspected result."
            )
            return json.dumps({
                "status": "completed", "warnings": [],
                "adaptations": self.state.result_adaptations,
            }, ensure_ascii=False)
        if not self.allow_semantic_revision and self.state.ready_for_degraded_finalization(
            allow_unusable=True
        ):
            self.state.revision_decisions.append({
                "action": "CONTINUE",
                "reason": "semantic_revision_disabled_by_experiment",
                "retryable": False,
            })
            self.recover_degraded_finish(
                "Experimental ablation preserved the first inspected result "
                "without a semantic revision.",
                allow_unusable=True,
            )
            return json.dumps({
                "status": "completed_with_warnings",
                "warnings": self.state.coverage_warnings,
                "revision_experiment_mode": "no_semantic_revision",
            }, ensure_ascii=False)
        if self.state.run_count >= self.state.max_runs and self.state.best_result_snapshot:
            self._restore_best_result()
            self.recover_degraded_finish(
                "run_analysis finalized the best result after two executions."
            )
            return json.dumps({
                "status": "completed_with_warnings",
                "warnings": self.state.coverage_warnings,
                "adaptations": self.state.result_adaptations,
            }, ensure_ascii=False)
        if self.state.run_count >= self.state.max_runs:
            self.state.error = "The final output is diagnostic, has the wrong shape, or contains no finite measure."
            self.state.execution_error = {
                "stage": "result_validation", "category": "result_needs_revision",
                "coverage_warnings": warnings, "retryable": False,
            }
        return json.dumps({
            "status": "revision_required",
            "all_problems": inspection.get("profile", {}).get("correction_items", warnings),
            "remaining_runs": self.state.max_runs - self.state.run_count,
            "revision_decision": self._record_semantic_revision_decision(warnings),
        }, ensure_ascii=False, default=str)
