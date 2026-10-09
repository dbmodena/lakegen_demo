"""The coder's way back: rejecting tables or the plan, with the justification the architect needs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lakegen.agent_tools.tools_p2 import MIN_BAN_JUSTIFICATION_CHARS
from lakegen.coding.revision_policy import classify_revision
from lakegen.coding.tools.state import CoderLifecycle


class RejectionMixin:
    """Part of ``Phase3ToolsManager``; the manager supplies the attributes it reads."""

    def reject_tables(
        self,
        reason: str,
        missing_requirements: list[str],
        inspected_evidence: str,
        ban_tables: dict[str, str] | None = None,
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
                    "next_actions": ["run_analysis", "inspect_data"],
                },
            }, ensure_ascii=False)
            raise ValueError("REJECTION_NOT_PROVEN: " + payload)
        by_fold = {table.casefold(): table for table in self.tables}
        unjustified = [
            table for table, justification in (ban_tables or {}).items()
            if table.casefold() in by_fold
            and len(str(justification).strip()) < MIN_BAN_JUSTIFICATION_CHARS
        ]
        if unjustified:
            raise ValueError(
                "Selection blocked: ban_tables must map each highly-irrelevant "
                "table to the concrete evidence proving it satisfies nothing, "
                "not a bare name. Missing or too-short justification for: "
                f"{', '.join(unjustified)}. Give it a real justification or "
                "drop it from ban_tables."
            )
        banned = [
            by_fold[table.casefold()]
            for table in (ban_tables or {})
            if table.casefold() in by_fold
        ]
        self.state.rejection_skip_tables = banned
        self.state.rejection_keep_tables = [
            table for table in self.tables if table not in banned
        ]
        self.state.rejected_reason = reason.strip()
        self.state.rejection_details = {
            "missing_requirements": missing,
            "inspected_evidence": inspected_evidence.strip(),
            "category": "data_proven_missing",
        }
        self.state.plan_validation = {
            **self.state.plan_validation,
            "valid": False, "locked": False, "status": "rejected",
            "rejected": True,
        }
        self.state.architect_contract_locked = False
        self.state.lifecycle = CoderLifecycle.TABLES_INSUFFICIENT
        return "REJECT_TABLES: " + json.dumps({
            "reason": self.state.rejected_reason,
            **self.state.rejection_details,
        }, ensure_ascii=False)

    def plan_conflict(
        self, reason: str, missing_requirements: list[str], inspected_evidence: str
    ) -> str:
        """Stop when the locked semantic plan contradicts question or evidence."""
        if not self.selection_plan.get("semantic_plan"):
            return json.dumps({"ok": False, "error": "no_architect_plan"})
        conflict = {
            "reason": reason.strip(),
            "missing_requirements": missing_requirements,
            "inspected_evidence": inspected_evidence,
            "category": "PLAN_CONFLICT",
        }
        self.state.rejection_details = conflict
        self.state.architect_contract_locked = False
        self.state.analysis_contract = {}
        self.state.plan_validation = {
            **self.state.plan_validation,
            "valid": False, "locked": False, "status": "needs_runtime_evidence",
            "rejected": False, "conflict": conflict,
            "required_action": "inspect_data_then_run_analysis_or_reject_data",
        }
        self.state.lifecycle = CoderLifecycle.NEEDS_REVISION
        return json.dumps({
            "ok": True, "status": "PLAN_CONFLICT_RECORDED",
            "conflict": conflict,
            "next_actions": ["inspect_data", "run_analysis", "reject_data"],
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
        all_columns = self._all_columns()
        normalized_columns = {
            re.sub(r"[^a-z0-9]", "", column.casefold()) for column in all_columns
        }
        requirement_words = {
            re.sub(r"[^a-z0-9]", "", word.casefold())
            for word in re.findall(
                r"[A-Za-z][A-Za-z _-]{3,40}", " ".join(missing)
            )
        }
        plausible_equivalent = any(
            word and any(
                word in column or column in word
                for column in normalized_columns
            )
            for word in requirement_words
        )
        proven_missing_column = bool(
            "column" in claim
            and ("missing" in claim or "does not contain" in claim)
            and requirement_words
            and not plausible_equivalent
        )
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
        if (
            requested_years
            and not proven_missing_column
            and re.search(r"(?:year|date|period|temporal)", claim)
        ):
            temporal_columns = [
                column for column in self._all_columns()
                if re.search(r"(?:year|date|time|period|fiscal|school|^fy|^sy)", column, re.IGNORECASE)
            ]
            if temporal_columns:
                blockers.append(
                    "Temporal requirements still need runtime resolution against "
                    f"candidate columns: {temporal_columns[:8]}."
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
            if plausible_equivalent:
                blockers.append(
                    "A selected schema contains a plausible equivalent column; inspect or use the exact available name before rejection."
                )
        return list(dict.fromkeys(blockers))

    def _record_semantic_revision_decision(
        self, warnings: list[str]
    ) -> dict[str, object]:
        decision = classify_revision(coverage_warnings=warnings)
        self.state.revision_decisions.append(decision)
        return decision

    def reject_data(
        self,
        reason: str,
        missing_requirements: list[str],
        inspected_evidence: str,
        ban_tables: dict[str, str] | None = None,
    ) -> str:
        """Terminally reject data only after consolidated inspection evidence."""
        return self.reject_tables(
            reason, missing_requirements, inspected_evidence, ban_tables
        )
