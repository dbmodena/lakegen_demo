"""Checking a result against the contract: its coverage, its evidence, its shape."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from lakegen.coding.tools.state import CoderLifecycle


class ResultChecksMixin:
    """Part of ``Phase3ToolsManager``; the manager supplies the attributes it reads."""

    def _validate_contract_code(self, code: str) -> list[str]:
        contract = self.state.analysis_contract
        lowered = code.casefold()
        warnings: list[str] = []
        if re.search(r"if\s+[^:\n]*(?:empty|len\s*\([^)]*\)\s*==\s*0)[^:\n]*:[\s\S]{0,180}(?:=\s*df\b|\.copy\s*\()", code, re.IGNORECASE):
            warnings.append("fallback_to_all_rows_detected: empty filter restores unfiltered rows")
        qualifier_terms = {
            term for term in re.findall(
                r"\b(?:with|containing|contains|named|called|category|type|status)\s+([a-z][a-z-]{2,})",
                self.question.casefold(),
            )
        }
        nonempty_filter = bool(
            re.search(r"\.notna\s*\(\)|\.notnull\s*\(\)|\.str\.strip\s*\(\)\s*!=\s*['\"]{2}", code)
        )
        if qualifier_terms and nonempty_filter and not any(term in lowered for term in qualifier_terms):
            warnings.append(
                "semantic_filter_weakened_to_nonempty: requested qualifier "
                + ", ".join(sorted(qualifier_terms))
            )
        contract_years = set(re.findall(
            r"\b(?:19|20)\d{2}\b",
            json.dumps(contract.get("filters", []), ensure_ascii=False, default=str),
        ))
        snapshot_years = self._snapshot_years() if contract_years else set()
        for year in contract_years:
            if year in snapshot_years:
                # Every row of a selected table is stamped with this date.
                continue
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
        measures = json.dumps(
            contract.get("measures", []), ensure_ascii=False, default=str
        ).casefold()
        has_mean_aggregation = bool(re.search(
            r"(?:\.mean\s*\(|\b(?:np|numpy)\.mean\s*\(|['\"]mean['\"])",
            lowered,
        ))
        if ("average" in measures or "mean" in measures) and not has_mean_aggregation:
            warnings.append("contract_average_missing_in_code")
        if "count_rows" in measures and any(marker in lowered for marker in ("nunique(", "drop_duplicates(")):
            warnings.append("contract_count_rows_implemented_as_distinct")
        if "count_distinct" in measures and not any(marker in lowered for marker in ("nunique(", "drop_duplicates(")):
            warnings.append("contract_count_distinct_implemented_as_row_count")
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
        for join in contract.get("join_bindings", []):
            for key in (join.get("keys", {}) if isinstance(join, dict) else {}).values():
                if str(key).casefold() not in lowered:
                    warnings.append(f"contract_join_key_missing_in_code: {key}")
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
        if "pd.concat(" in lowered and strategy not in {"concat_partitions", "aggregate_separately"}:
            warnings.append(
                "table_concat_without_explicit_strategy: concatenation is not justified by the selection plan"
            )
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

    def _adapt_structured_result(self, value: Any) -> tuple[Any, list[str]]:
        """Apply only lossless or contract-explicit output normalizations."""
        adaptations: list[str] = []
        if isinstance(value, dict) and value:
            if all(isinstance(item, list) for item in value.values()):
                lengths = {len(item) for item in value.values()}
                if len(lengths) == 1:
                    keys = list(value)
                    value = [dict(zip(keys, row)) for row in zip(*(value[key] for key in keys))]
                    adaptations.append("columnar_mapping_to_records")

        contract = self.state.analysis_contract
        limit = contract.get("limit") or self._expected_top_n(self.question)
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
            ordering = str(contract.get("ordering") or "").casefold()
            fields = list(value[0])
            numeric_fields = [
                field for field in fields
                if all(
                    isinstance(row.get(field), (int, float))
                    and not isinstance(row.get(field), bool)
                    for row in value
                )
            ]
            ordered_fields = [
                field for field in numeric_fields
                if self._label_is_evident(ordering, [field])
            ]
            if len(ordered_fields) == 1 and any(
                token in ordering for token in ("asc", "desc", "top", "bottom", "highest", "lowest", "most", "least")
            ):
                reverse = any(token in ordering for token in ("desc", "top", "highest", "most", "largest"))
                value = sorted(value, key=lambda row: row[ordered_fields[0]], reverse=reverse)
                adaptations.append("applied_explicit_ordering")
            if isinstance(limit, int) and limit > 0 and len(value) > limit:
                value = value[:limit]
                adaptations.append(f"applied_explicit_limit:{limit}")
        return value, adaptations

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

        def contract_text(item: Any) -> str:
            return item if isinstance(item, str) else json.dumps(
                item, ensure_ascii=False, sort_keys=True, default=str
            )

        years = set(re.findall(
            r"\b(?:19|20)\d{2}\b",
            " ".join(map(contract_text, contract.get("filters", []))),
        ))
        for item in contract.get("filters", []):
            item_text = contract_text(item)
            item_years = set(re.findall(r"\b(?:19|20)\d{2}\b", item_text))
            evident = (
                all(self._code_represents_year(code, year, years) for year in item_years)
                if item_years else self._label_is_evident(item_text, [code])
            )
            add("filter", item_text, ["filter expression found in code"] if evident else [], ["filter affects computed structured result"] if evident else [])

        measures = contract.get("measures", [])
        for item in measures:
            item_text = contract_text(item)
            item_lower = item_text.casefold()
            operation = ""
            if any(term in item_lower for term in ("average", "mean")) and re.search(r"\.mean\s*\(|['\"]mean['\"]", lowered):
                operation = "mean aggregation found"
            elif any(term in item_lower for term in ("count", "number", "total")) and any(marker in lowered for marker in (".count(", ".size(", "nunique(", "len(")):
                operation = "count aggregation found"
            elif self._label_is_evident(item_text, [code]):
                operation = "measure terms found in code"
            result_match = [field for field in result_fields if self._label_is_evident(item_text, [field])]
            if not result_match and len(measures) == 1 and value is not None and not result_fields:
                result_match = ["structured scalar result"]
            add("measure", item_text, [operation] if operation else [], result_match)

        for item in contract.get("group_by", []):
            item_text = contract_text(item)
            grouped = any(marker in lowered for marker in (".groupby(", ".pivot(", ".pivot_table("))
            label_in_code = self._label_is_evident(item_text, [code])
            result_match = [field for field in result_fields if self._label_is_evident(item_text, [field])]
            add("grouping", item_text, ["grouping operation and dimension found"] if grouped and label_in_code else [], result_match)

        for item in contract.get("distinct_counts", []):
            item_text = contract_text(item)
            distinct = any(marker in lowered for marker in ("nunique(", "drop_duplicates(", ".unique("))
            result_match = [field for field in result_fields if self._label_is_evident(item_text, [field])]
            add("distinct count", item_text, ["distinct-count operation found"] if distinct else [], result_match or (["structured result"] if distinct and value is not None else []))

        for item in contract.get("joins", []):
            joined = any(marker in lowered for marker in (".merge(", ".join(", "pd.merge("))
            add("join", contract_text(item), ["table join operation found"] if joined else [], ["joined structured result"] if joined and value is not None else [])

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
            *self._validate_final_result_contract(value),
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
            "warning_diagnostics": [
                {
                    "requirement": warning.split(":", 1)[0],
                    "evidence": warning,
                    "correction_required": "Revise the code/result to satisfy the analysis contract, then run and inspect again.",
                }
                for warning in warnings
            ],
            "semantic_plan_validation": self.state.plan_validation,
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
            "required_action": "run_code_or_reject_tables" if warnings else "automatic_finalization",
            "allowed_actions": ["run_code", "reject_tables"] if warnings else [],
            "profile": profile,
        }, ensure_ascii=False, default=str)

    def _blocking_contract_code_warnings(self) -> list[str]:
        """Promote only objective code/contract violations to retry blockers."""
        blocking_prefixes = (
            "contract_distinct_count_missing_in_code",
            "count_semantics_check:",
            "contract_average_missing_in_code",
            "unsupported_bucket_assumption:",
            "contract_join_missing_in_code",
            "contract_grouping_missing_in_code",
            "contract_limit_missing_in_code:",
            "contract_ordering_missing_in_code",
            "top_n_check:",
            "fallback_to_all_rows_detected:",
            "contract_count_rows_implemented_as_distinct",
            "contract_count_distinct_implemented_as_row_count",
            "contract_join_key_missing_in_code:",
            "semantic_filter_weakened_to_nonempty:",
            "table_concat_without_explicit_strategy:",
        )
        return [
            warning for warning in self.state.contract_code_warnings
            if warning.startswith(blocking_prefixes)
        ]

    @staticmethod
    def _is_empty_result(value: Any) -> bool:
        """True only for explicit empty containers; None means no payload."""
        if isinstance(value, (list, tuple)):
            return all(
                isinstance(item, (list, tuple, dict))
                and ResultChecksMixin._is_empty_result(item)
                for item in value
            )
        if isinstance(value, dict):
            return not value or all(
                isinstance(item, (list, tuple, dict)) and ResultChecksMixin._is_empty_result(item)
                for item in value.values()
            )
        return False

    def _validate_contract_result(self, value: Any) -> list[str]:
        contract = self.state.analysis_contract
        warnings: list[str] = []
        if self._is_empty_result(value):
            # A filter or dropna that removed every row yields `[]`, which
            # still reads as a non-empty answer string downstream.
            warnings.append(
                "empty_result: the program produced no rows or values; check "
                "filters, joins and dropna on columns the answer does not need"
            )
            return warnings
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
            str(field_name) for row in value for field_name in row
        ))

        def matching_fields(label: str) -> list[str]:
            normalized = re.sub(r"[^a-z0-9]+", " ", label.casefold()).strip()
            if normalized in {"", "result", "requested result", "value", "requested value"}:
                return []
            return [field_name for field_name in fields if self._label_is_evident(label, [field_name])]

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
            keys = [tuple(row.get(field_name) for field_name in group_fields) for row in value]
            if len(set(keys)) != len(keys):
                warnings.append(
                    "contract_result_duplicate_group_keys: " + ", ".join(group_fields)
                )

        measure_fields: list[str] = []
        for measure in [*contract.get("measures", []), *contract.get("output_columns", [])]:
            for field_name in matching_fields(str(measure)):
                if field_name not in measure_fields:
                    measure_fields.append(field_name)
        # Output aliases need not repeat the brief's wording. Inspect every
        # non-dimension numeric field_name so an all-NaN measure cannot evade checks
        # simply by being renamed by the coder.
        for field_name in dict.fromkeys([*measure_fields, *fields]):
            if field_name in group_fields:
                continue
            numeric = [
                float(row[field_name]) for row in value
                if field_name in row and isinstance(row[field_name], (int, float))
                and not isinstance(row[field_name], bool)
            ]
            if numeric and not any(math.isfinite(number) for number in numeric):
                warnings.append(f"contract_result_all_non_finite: {field_name}")

        ordering = str(contract.get("ordering") or "none").casefold()
        if ordering not in {"", "none", "not applicable", "n/a"}:
            ordering_matches = [
                field_name for field_name in measure_fields
                if self._label_is_evident(ordering, [field_name])
            ]
            candidates = ordering_matches or measure_fields
            numeric_candidates = []
            for field_name in candidates:
                values = [row.get(field_name) for row in value]
                if values and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in values
                ):
                    numeric_candidates.append(field_name)
            if len(numeric_candidates) == 1:
                field_name = numeric_candidates[0]
                numbers = [float(row[field_name]) for row in value]
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
                        f"contract_result_order_mismatch: {field_name} is not {direction}"
                    )
        return warnings

    def _validate_final_result_contract(self, value: Any) -> list[str]:
        """Validate unmistakable final-answer shape contradictions without gold data."""
        final_contract = self._final_result_contract()
        operations = set(final_contract["derived_operations"])
        candidate = value
        if isinstance(candidate, list) and len(candidate) == 1:
            candidate = candidate[0]
        if isinstance(candidate, dict) and len(candidate) == 1:
            candidate = next(iter(candidate.values()))

        warnings: list[str] = []
        result_fields = self._result_field_names(value)
        if any(field in self.tables for field in result_fields) and not re.search(
            r"\b(?:schema|columns|available years)\b", self.question, re.IGNORECASE
        ):
            warnings.append("diagnostic_output: table profiles are not the requested analysis")
        diagnostic_fields = {
            "available_years", "available_columns", "columns", "dtypes", "schema",
            "sample_values", "unique_values", "value_counts_preview",
        }
        normalized_fields = {
            re.sub(r"[^a-z0-9]+", "_", str(field).casefold()).strip("_")
            for field in result_fields
        }
        if normalized_fields & diagnostic_fields and not re.search(
            r"\b(?:schema|columns|available years|data types|sample values)\b",
            self.question, re.IGNORECASE,
        ):
            warnings.append(
                "diagnostic_output: metadata fields "
                f"{sorted(normalized_fields & diagnostic_fields)} are not the requested analysis"
            )
        if "correlation" in operations:
            if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
                warnings.append("final_contract_correlation_requires_one_numeric_value")
            elif not math.isfinite(float(candidate)) or not -1.0 <= float(candidate) <= 1.0:
                warnings.append("final_contract_correlation_out_of_range")

        if "geographic_center" in operations:
            row = value[0] if isinstance(value, list) and len(value) == 1 else value
            if not isinstance(row, dict):
                warnings.append("final_contract_geographic_center_requires_coordinate_pair")
            else:
                numeric = {
                    str(key).casefold(): float(item)
                    for key, item in row.items()
                    if isinstance(item, (int, float)) and not isinstance(item, bool)
                }
                latitudes = [item for key, item in numeric.items() if "lat" in key]
                longitudes = [item for key, item in numeric.items() if "lon" in key or "lng" in key]
                if not latitudes or not longitudes:
                    warnings.append("final_contract_geographic_center_requires_coordinate_pair")
                elif not (-90 <= latitudes[0] <= 90 and -180 <= longitudes[0] <= 180):
                    warnings.append("final_contract_geographic_center_invalid_coordinates")

        measures = list(self.state.analysis_contract.get("measures", []))
        if (
            len(measures) >= 2
            and isinstance(value, list) and value
            and all(isinstance(row, dict) for row in value)
        ):
            fields = list(dict.fromkeys(str(field) for row in value for field in row))
            matched = {
                field for measure in measures for field in fields
                if self._label_is_evident(str(measure), [field])
            }
            numeric_fields = {
                field for field in fields
                if any(
                    isinstance(row.get(field), (int, float))
                    and not isinstance(row.get(field), bool)
                    for row in value
                )
            }
            # Semantic aliases are preferred, but an equivalent set of numeric
            # outputs is sufficient when labels are different.
            if len(matched | numeric_fields) < len(measures):
                warnings.append(
                    "final_contract_missing_measures: expected_at_least_"
                    f"{len(measures)}_numeric_measure_fields"
                )
        return warnings
