"""The analysis contract: what the final result must contain, set by the coder or inferred from the question."""

from __future__ import annotations

import json
import re
from typing import Any


class ContractMixin:
    """Part of ``CoderTools``; the manager supplies the attributes it reads."""

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
            return json.dumps(
                {
                    "ok": False,
                    "error": "architect_semantic_plan_is_authoritative",
                    "contract": self.state.analysis_contract,
                    "next_action": "Write the program and call run_analysis.",
                },
                ensure_ascii=False,
            )
        if self.state.plan_validation and not self.state.plan_validation.get("valid"):
            # A failed architect plan is provisional and may be replaced after
            # runtime inspection; its diagnostics remain available for audit.
            self.state.contract_advisories.extend(
                str(item.get("category"))
                for item in self.state.plan_validation.get("diagnostics", [])
            )
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
        if self.state.plan_validation and not self.state.plan_validation.get("valid"):
            revised_verified = not problems and bool(self.state.inspected_tables)
            self.state.plan_validation = {
                **self.state.plan_validation,
                "valid": revised_verified,
                "locked": revised_verified,
                "revised_after_runtime_inspection": bool(self.state.inspected_tables),
                "revision_validated": True,
                "status": ("verified" if revised_verified else "needs_runtime_evidence"),
            }
            self.state.architect_contract_locked = revised_verified
        return json.dumps(
            {
                "ok": True,
                "contract": contract,
                "advisories": problems,
                "next_action": "Write the complete program and call run_analysis.",
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _clean_contract_items(items: list[str]) -> list[str]:
        return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))

    def _contract_problems(self, contract: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        question = self.question.casefold()
        contract_text = json.dumps(contract, ensure_ascii=False).casefold()
        for year in self._requested_filter_years(question):
            if year not in contract_text:
                problems.append(f"missing requested year {year}")
        if (
            re.search(r"\b(?:distinct|different|unique)\b", question)
            and not contract["distinct_counts"]
        ):
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
        edition_years = set(
            re.findall(
                r"\b(?:using|from|in)\s+(?:the\s+)?((?:19|20)\d{2})\b"
                r"[^?.]{0,80}\b(?:dataset|data\s+set|release|edition|inventory)\b",
                question,
                flags=re.IGNORECASE,
            )
        )
        edition_years.update(
            re.findall(
                r"\b(?:according\s+to|using|from)\s+(?:the\s+)?"
                r"(?:[A-Za-z]+\s+)?((?:19|20)\d{2})\b[^?.]{0,60}"
                r"\b(?:boundar(?:y|ies)|snapshot|organogram|list|report)\b",
                question,
                flags=re.IGNORECASE,
            )
        )
        return years - edition_years

    def _validate_dataset_edition_filter(self, code: str) -> dict[str, Any] | None:
        """Block treating an edition year embedded in an ID label as row data."""
        all_years = set(re.findall(r"\b(?:19|20)\d{2}\b", self.question))
        edition_years = all_years - self._requested_filter_years(self.question)
        if not edition_years or self._requested_filter_years(self.question):
            return None
        identifier_columns = [
            column
            for column in self._all_columns()
            if re.search(r"(?:cd|code|id)$", column, re.IGNORECASE)
            and any(year[-2:] in column for year in edition_years)
        ]
        offending: list[str] = []
        for column in identifier_columns:
            for year in edition_years:
                short_year = re.escape(year[-2:])
                column_pattern = re.escape(column)
                patterns = (
                    rf"\[['\"]{column_pattern}['\"]\][^\n]{{0,120}}"
                    rf"\.str\.contains\(\s*['\"]{short_year}['\"]",
                    rf"\[['\"]{column_pattern}['\"]\][^\n]{{0,100}}"
                    rf"==\s*['\"]?(?:{re.escape(year)}|{short_year})['\"]?",
                )
                if any(re.search(pattern, code) for pattern in patterns):
                    offending.append(column)
        if not offending:
            return None
        return {
            "stage": "preflight",
            "category": "dataset_edition_row_filter",
            "message": (
                "Generated code treats a dataset-edition year as a row-level "
                "filter on an identifier column."
            ),
            "columns": sorted(set(offending)),
            "edition_years": sorted(edition_years),
            "retryable": True,
            "repair_hint": (
                "Remove the identifier-column year filter. The year identifies "
                "the selected dataset edition; compute over all rows."
            ),
        }

    @staticmethod
    def _expected_top_n(question: str) -> int | None:
        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", question)
        if match:
            return int(match.group(1))
        match = re.search(rf"\b(?:top|bottom)\s+({'|'.join(number_words)})\b", question)
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
        distinct = (
            ["requested entity"] if re.search(r"\b(?:distinct|different|unique)\b", lowered) else []
        )
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
