"""The coder's plan: the brief the architect wrote (or a fallback), checked against the tables' real columns."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pandas as pd

from lakegen.core.table_io import read_table


class PlanGroundingMixin:
    """Part of ``Phase3ToolsManager``; the manager supplies the attributes it reads."""

    def _runtime_fallback_coder_brief(self) -> dict[str, Any]:
        """Build a safe minimal brief when selection produced tables but no plan."""
        requirements = self.selection_plan.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {}
        return {
            "tables": list(self.tables),
            "selected_columns": {
                table: [str(column) for column in read_table(
                    self.csv_dir / table.strip(), nrows=0
                ).columns]
                for table in self.tables
            },
            "filters": list(requirements.get("filters") or []),
            "temporal_filters": list(requirements.get("temporal_filters") or []),
            "dimensions": [
                {"column": str(column), "output": str(column)}
                for column in requirements.get("grouping", [])
            ],
            "measures": list(requirements.get("measures") or []),
            "joins": list(requirements.get("joins") or []),
            "ordering": requirements.get("ordering") or [],
            "limit": requirements.get("limit"),
            "output_columns": [], "null_policy": "",
            "table_roles": dict(self.selection_plan.get("table_roles") or {}),
            "normalization_errors": [],
            "source": "runtime_fallback",
            "task": copy.deepcopy(requirements),
            "runtime_obligations": [
                "Derive filters, grouping, measures, joins, units, and output labels "
                "from the question and verify them against the selected runtime tables."
            ],
        }

    def _runtime_fallback_requirement_ledger(self) -> list[dict[str, Any]]:
        """Keep legacy/recovery paths compatible with the compact ledger view."""
        requirements = self.selection_plan.get("requirements")
        requirements = requirements if isinstance(requirements, dict) else {}
        ledger: list[dict[str, Any]] = []

        def readable(value: Any) -> str:
            if isinstance(value, dict):
                return " ".join(str(part) for part in (
                    value.get("output") or value.get("alias"),
                    value.get("operation") or value.get("type"),
                    value.get("column"),
                ) if part)
            return " ".join(str(value).split())

        for value in self.selection_plan.get("uncovered_requirements", []) or []:
            text = readable(value)
            if text:
                ledger.append({
                    "kind": "data_requirement", "request": text,
                    "status": "unresolved", "evidence": [],
                })
        for key, kind in (
            ("filters", "filter"), ("temporal_filters", "temporal_filter"),
            ("grouping", "dimension"), ("measures", "measure"),
            ("output_columns", "output"),
        ):
            values = requirements.get(key, [])
            for value in values if isinstance(values, list) else []:
                text = readable(value)
                if text:
                    ledger.append({
                        "kind": kind, "request": text,
                        "status": "computational", "evidence": [],
                    })
        for key in ("ordering", "limit"):
            value = requirements.get(key)
            if value not in (None, "", []):
                ledger.append({
                    "kind": key, "request": readable(value),
                    "status": "computational", "evidence": [],
                })
        for item in ledger:
            if item["kind"] == "dimension":
                item["role"] = "final"
            elif item["kind"] == "measure":
                item["role"] = "final"
        return ledger[:10]

    def coder_plan_view(self) -> dict[str, Any]:
        """Return grounded data constraints plus an extensible computation brief."""
        validation = self.state.plan_validation
        view: dict[str, Any] = {
            "semantic_plan_present": bool(self.selection_plan.get("semantic_plan")),
            "semantic_plan_valid": bool(validation.get("valid")),
            "semantic_plan_locked": bool(self.state.architect_contract_locked),
            "semantic_plan_revised": bool(validation.get("revised_after_runtime_inspection")),
            "semantic_plan_rejected": bool(validation.get("rejected")),
            "semantic_plan_missing": validation.get("status") == "missing",
            "status": validation.get("status", "unknown"),
            "evidence": validation.get("evidence", []),
            "diagnostics": validation.get("diagnostics", []),
            "warnings": validation.get("warnings", []),
            "runtime_obligations": validation.get("runtime_obligations", []),
            "required_action": validation.get("required_action"),
            "contract_type": validation.get("contract_type"),
            "coder_brief": copy.deepcopy(self.selection_plan.get("coder_brief", {})),
            "requirement_ledger": copy.deepcopy(
                self.selection_plan.get("requirement_ledger")
                or (self.selection_plan.get("coder_brief") or {}).get("requirement_ledger")
                or []
            ),
            "runtime_checklist": self._runtime_checklist(),
            "selection_brief_status": self.original_coder_brief_status,
            "effective_coder_brief_status": validation.get("status", "unknown"),
            "effective_coder_brief_source": str(
                (self.selection_plan.get("coder_brief") or {}).get("source")
                or ("legacy_semantic_plan" if self.original_coder_brief_status == "legacy_plan" else "selection")
            ),
            "temporal_requirement_classification": self._temporal_requirement_classification(),
            "plan_completion_policy": self._plan_completion_policy(),
            "final_result_contract": self._final_result_contract(),
        }
        if not self.state.architect_contract_locked:
            return view
        view["validated_selection_plan"] = copy.deepcopy(
            self.selection_plan.get("coder_brief")
            or self.selection_plan.get("semantic_plan", {})
        )
        return view

    def _plan_completion_policy(self) -> dict[str, Any]:
        """Describe what the coder may complete without weakening grounded evidence."""
        brief = self.selection_plan.get("coder_brief") or {}
        dimensions = brief.get("dimensions", [])
        task = brief.get("task") if isinstance(brief.get("task"), dict) else {}
        grouping = [*dimensions, *task.get("grouping", [])]
        question = self.question.casefold()
        obligations = [
            "Compare the complete question with the brief before writing code and add "
            "any missing computational step needed for the final answer."
        ]
        if "correlat" in question:
            obligations.append(
                "Compute and return the requested correlation; grouped source totals are "
                "only an intermediate result."
            )
        if any(token in question for token in (" ratio ", "ratio of", " rapporto ")):
            obligations.append(
                "Identify an evidence-backed numerator and denominator and compute the "
                "requested ratio."
            )
        if re.search(r"\btop\s+\d+\b", question):
            obligations.append(
                "Apply the requested ranking and top-N limit even if ordering or limit is "
                "missing from the brief."
            )
        if str(brief.get("source") or "") == "runtime_fallback":
            obligations.append(
                "The selection supplied no computational brief: inspect the selected data, "
                "derive the full calculation from the question, and reject the data if the "
                "required facts are not evidenced."
            )
        return {
            "mode": "soft_computational_completion",
            "immutable_evidence": [
                "selected tables", "observed join keys and inspected values",
            ] if self.brief_advisory else [
                "selected tables", "runtime column names", "explicit question filters",
                "observed join keys and inspected values",
            ],
            **({"verify_against_data": [
                "brief column choices", "brief row filters", "brief null policy",
            ]} if self.brief_advisory else {}),
            "coder_may_add_or_refine": [
                "derived computations", "final aggregations", "ratios", "correlations",
                "ordering", "limits", "output formatting",
            ],
            "coder_must_not": [
                "invent tables or columns", "invent filter values or join keys",
                "drop an explicit question constraint", "claim missing data without evidence",
            ],
            "obligations": obligations,
            "completion_is_inferred_from_code": True,
        }

    def _final_result_contract(self) -> dict[str, Any]:
        """Describe only robust final-shape invariants; aliases remain acceptable."""
        contract = self.state.analysis_contract or self._contract_from_coder_brief(
            self.selection_plan.get("coder_brief") or {}
        )
        question = self.question.casefold()
        operations: list[str] = []
        if "correlat" in question:
            operations.append("correlation")
        if "geographic center" in question or "geographical center" in question:
            operations.append("geographic_center")
        return {
            "mode": "soft_semantic_shape",
            "dimensions": list(contract.get("group_by", [])),
            "measure_count": len(contract.get("measures", [])),
            "derived_operations": operations,
            "alias_policy": "semantic aliases and lossless scalar/table wrappers are accepted",
            "blocking_policy": (
                "block only clear shape contradictions; ambiguous labels remain advisory"
            ),
        }

    def _temporal_requirement_classification(self) -> dict[str, Any]:
        all_years = set(re.findall(r"\b(?:19|20)\d{2}\b", self.question))
        filter_years = self._requested_filter_years(self.question)
        edition_years = all_years - filter_years
        metadata = self._metadata_text()
        temporal_columns = [
            column for column in self._all_columns()
            if re.search(r"(?:year|date|time|period|fiscal|school|^fy|^sy)", column, re.IGNORECASE)
        ]
        if edition_years and all(year in metadata for year in edition_years):
            status = "metadata_period_match"
        elif filter_years:
            status = "runtime_resolution_needed"
        elif all_years:
            status = "runtime_resolution_needed"
        else:
            status = "not_applicable"
        return {
            "status": status,
            "filter_years": sorted(filter_years),
            "edition_years": sorted(edition_years),
            "temporal_columns": temporal_columns[:20],
        }

    def _runtime_checklist(self) -> dict[str, Any]:
        """Compile objective, locally checkable requirements from the sole contract."""
        contract = self.state.analysis_contract
        requested_years = sorted(self._requested_filter_years(self.question))
        exact_rows = contract.get("limit") or self._expected_top_n(self.question)
        return {
            "required_years": requested_years,
            "required_columns": list(contract.get("output_columns", [])),
            "exact_row_count": exact_rows,
            "ordering": contract.get("ordering"),
            "aggregations": list(contract.get("measures", [])),
            "group_by": list(contract.get("group_by", [])),
            "joins": list(contract.get("join_bindings", [])),
        }

    def _infer_unique_join(self, brief: dict[str, Any]) -> None:
        """Fill a join only when schema names and sampled values identify one pair."""
        if len(self.tables) != 2 or brief.get("joins"):
            return
        if brief.get("source") == "runtime_fallback" or brief.get("combination_strategy") in {
            "concat_partitions", "aggregate_separately", "compare",
        }:
            return
        left, right = self.tables
        try:
            left_frame = read_table(self.csv_dir / left.strip(), nrows=500)
            right_frame = read_table(self.csv_dir / right.strip(), nrows=500)
        except Exception:
            return

        def canonical(value: Any) -> str:
            return re.sub(r"[^a-z0-9]", "", str(value).casefold())

        candidates: list[tuple[str, str]] = []
        for left_column in left_frame.columns:
            for right_column in right_frame.columns:
                if canonical(left_column) != canonical(right_column):
                    continue
                left_values = set(left_frame[left_column].dropna().astype(str).str.strip())
                right_values = set(right_frame[right_column].dropna().astype(str).str.strip())
                if left_values & right_values:
                    candidates.append((str(left_column), str(right_column)))
        if len(candidates) == 1:
            left_column, right_column = candidates[0]
            brief["joins"] = [{
                "tables": [left, right],
                "keys": {left: left_column, right: right_column},
                "how": "inner", "source": "unique_schema_and_sample_overlap",
            }]
            brief["combination_strategy"] = "join"
            for table, column in ((left, left_column), (right, right_column)):
                selected = brief.setdefault("selected_columns", {}).setdefault(table, [])
                if column not in selected:
                    selected.append(column)
        elif len(candidates) != 1:
            brief.setdefault("runtime_obligations", []).append(
                "Resolve the relationship between the selected tables; no unique "
                "join key was established from schema names and sampled overlap."
            )

    def _validate_coder_brief_grounding(self, brief: Any) -> dict[str, Any]:
        """Verify only the data references needed to let the coder start."""
        status = (
            "invalid" if (
                "coder_brief" in self.selection_plan
                or "semantic_plan" in self.selection_plan
            ) else "missing"
        )
        if not isinstance(brief, dict) or not brief:
            return {
                "valid": False, "locked": False, "status": status,
                "diagnostics": [], "evidence": [],
                "required_action": "confirm tables and columns",
            }
        declared_tables = [str(item) for item in brief.get("tables", [])]
        selected_columns = brief.get("selected_columns", {})
        diagnostics: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        if not declared_tables or set(declared_tables) != set(self.tables):
            diagnostics.append({
                "status": "contradicted", "category": "unknown_table",
                "evidence": {"selected": self.tables, "declared": declared_tables},
            })
        schemas: dict[str, set[str]] = {}
        for table in self.tables:
            try:
                columns = {str(column) for column in read_table(
                    self.csv_dir / table.strip(), nrows=0
                ).columns}
                schemas[table] = columns
                evidence.append({"table": table, "columns": sorted(columns)})
            except Exception as exc:
                diagnostics.append({
                    "status": "contradicted", "category": "table_unreadable",
                    "table": table, "evidence": str(exc),
                })
        if not isinstance(selected_columns, dict):
            selected_columns = {}
            diagnostics.append({
                "status": "contradicted", "category": "invalid_selected_columns",
                "evidence": "selected_columns must map each table to exact columns",
            })
        declared_columns = [
            (str(table), str(column))
            for table, columns in selected_columns.items()
            if isinstance(columns, list)
            for column in columns
        ]
        if not declared_columns:
            warnings.append({
                "status": "unknown", "category": "missing_columns",
                "evidence": "coder must choose columns from the runtime schema",
            })
        unknown_columns = sorted(
            f"{table}.{column}" for table, column in declared_columns
            if table not in self.tables or column not in schemas.get(table, set())
        )
        if unknown_columns:
            diagnostics.append({
                "status": "contradicted", "category": "unknown_column",
                "evidence": {
                    "unknown": unknown_columns,
                    "available_by_table": {
                        table: sorted(columns) for table, columns in schemas.items()
                    },
                },
            })
        for error in brief.get("normalization_errors", []):
            warnings.append({
                "status": "unknown", "category": "ambiguous_column",
                "evidence": str(error),
            })
        joins = brief.get("joins", [])
        strategy = str(brief.get("combination_strategy") or "single_table")
        if len(self.tables) > 1 and strategy in {"join", "lookup"} and not joins:
            warnings.append({
                "status": "unknown", "category": "missing_join",
                "evidence": "multi-table briefs require explicit join keys",
            })
        for join in joins if isinstance(joins, list) else []:
            if not isinstance(join, dict):
                warnings.append({
                    "status": "unknown", "category": "invalid_join",
                    "evidence": "join must be a structured object",
                })
                continue
            if isinstance(join.get("tables"), list) and isinstance(join.get("keys"), dict):
                for table in join["tables"]:
                    key = str(join["keys"].get(table) or "")
                    if table not in self.tables or key not in schemas.get(str(table), set()):
                        diagnostics.append({
                            "status": "contradicted", "category": "unknown_join_column",
                            "evidence": {"table": table, "columns": [key]},
                        })
                continue
            for side in ("left", "right"):
                table = str(join.get(f"{side}_table") or "")
                columns = join.get(f"{side}_columns", [])
                if table not in self.tables or not isinstance(columns, list) or not columns:
                    diagnostics.append({
                        "status": "contradicted", "category": "invalid_join",
                        "evidence": {"side": side, "table": table, "columns": columns},
                    })
                    continue
                missing = [str(column) for column in columns if str(column) not in schemas.get(table, set())]
                if missing:
                    diagnostics.append({
                        "status": "contradicted", "category": "unknown_join_column",
                        "evidence": {"table": table, "columns": missing},
                    })
        valid = not diagnostics
        return {
            "valid": valid, "locked": valid,
            "status": (
                "executable_with_obligations" if valid and warnings else
                "verified" if valid else status
            ),
            "diagnostics": diagnostics, "evidence": evidence,
            "warnings": warnings,
            "required_action": None if valid else "revise coder brief data references",
            "contract_type": "coder_brief",
        }

    @staticmethod
    def _normalize_coder_brief(brief: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize equivalent syntax without inventing semantic choices."""
        normalized = copy.deepcopy(brief)
        operation_aliases = {"count": "count_rows", "avg": "mean", "average": "mean"}
        for key in ("filters", "temporal_filters", "measures"):
            canonical_items: list[Any] = []
            for raw in normalized.get(key, []):
                if not isinstance(raw, dict):
                    canonical_items.append(raw)
                    continue
                item = dict(raw)
                if key in {"filters", "temporal_filters"} and not item.get("operator"):
                    item["operator"] = str(item.get("type") or "").casefold()
                if key == "measures":
                    operation = str(
                        item.get("operation") or item.get("aggregation") or item.get("type") or ""
                    ).casefold()
                    item["operation"] = operation_aliases.get(operation, operation)
                    if "columns" not in item and item.get("column"):
                        item["columns"] = [item["column"]]
                canonical_items.append(item)
            normalized[key] = canonical_items
        joins: list[Any] = []
        for raw in normalized.get("joins", []):
            if not isinstance(raw, dict):
                joins.append(raw)
                continue
            join = dict(raw)
            if not join.get("tables") and join.get("left_table") and join.get("right_table"):
                join["tables"] = [join["left_table"], join["right_table"]]
            tables = [str(item) for item in join.get("tables", [])]
            if not join.get("keys"):
                left = join.get("left_columns") or (
                    [join.get("left_key")] if join.get("left_key") else []
                )
                right = join.get("right_columns") or (
                    [join.get("right_key")] if join.get("right_key") else []
                )
                if len(tables) == 2 and len(left) == len(right) == 1:
                    join["keys"] = {
                        tables[0]: str(left[0]), tables[1]: str(right[0]),
                    }
            joins.append(join)
        normalized["joins"] = joins
        return normalized

    @staticmethod
    def _legacy_semantic_plan_to_coder_brief(plan: dict[str, Any]) -> dict[str, Any]:
        """Compile the legacy architect artifact into the sole runtime contract."""
        tables = list(dict.fromkeys([
            *map(str, (plan.get("table_roles") or {}).keys()),
            *(
                str(binding.get("table"))
                for kind in ("filters", "temporal_filters", "dimensions", "measures")
                for binding in plan.get(kind, [])
                if isinstance(binding, dict) and binding.get("table")
            ),
        ]))
        selected_columns: dict[str, list[str]] = {table: [] for table in tables}
        for kind in ("filters", "temporal_filters", "dimensions", "measures"):
            for binding in plan.get(kind, []):
                if not isinstance(binding, dict):
                    continue
                table = str(binding.get("table") or (tables[0] if len(tables) == 1 else ""))
                columns = binding.get("columns") or [binding.get("column")]
                for column in columns:
                    if column and table in selected_columns and str(column) not in selected_columns[table]:
                        selected_columns[table].append(str(column))
        for join in plan.get("joins", []):
            if not isinstance(join, dict):
                continue
            for table, column in (join.get("keys") or {}).items():
                table, column = str(table), str(column)
                if table in selected_columns and column not in selected_columns[table]:
                    selected_columns[table].append(column)
        return {
            "tables": tables,
            "selected_columns": selected_columns,
            "filters": list(plan.get("filters", [])),
            "temporal_filters": list(plan.get("temporal_filters", [])),
            "dimensions": list(plan.get("dimensions", [])),
            "measures": list(plan.get("measures", [])),
            "joins": list(plan.get("joins", [])),
            "ordering": plan.get("ordering") or [],
            "limit": plan.get("limit"),
            "output_columns": list(plan.get("output_columns", [])),
            "null_policy": str(plan.get("null_policy") or ""),
            "table_roles": dict(plan.get("table_roles") or {}),
            "normalization_errors": [],
        }

    @staticmethod
    def _contract_from_coder_brief(brief: dict[str, Any]) -> dict[str, Any]:
        canonical = dict(brief)
        task = brief.get("task") if isinstance(brief.get("task"), dict) else {}
        canonical.setdefault("measures", list(brief.get("operations", [])))
        canonical.setdefault("dimensions", [
            {"output": item, "column": item}
            for item in task.get("grouping", [])
        ])
        canonical["ordering"] = canonical.get("ordering") or []
        contract = PlanGroundingMixin._contract_from_semantic_plan(canonical)
        contract.update({
            "tables": list(brief.get("tables", [])),
            "columns": dict(brief.get("selected_columns", {})),
            "result_type": "auto",
            "source": "validated_coder_brief",
        })
        return contract

    def _validate_semantic_plan_grounding(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Validate architect semantics using only selected runtime tables."""
        grounding = _Grounding(self.tables, self.csv_dir, self.question)
        grounding.check(plan)
        return grounding.verdict()

    @staticmethod
    def _contract_from_semantic_plan(plan: dict[str, Any]) -> dict[str, Any]:
        filters = []
        for binding in [*plan.get("filters", []), *plan.get("temporal_filters", [])]:
            if isinstance(binding, str) and binding.strip():
                filters.append(binding.strip())
                continue
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
            if isinstance(binding, str) and binding.strip():
                measures.append(binding.strip())
                continue
            if not isinstance(binding, dict):
                continue
            operation = str(binding.get("operation") or "").strip()
            columns = ", ".join(map(str, binding.get("columns", [])))
            output = str(binding.get("output") or "").strip() or "_".join(
                filter(None, [operation, *map(str, binding.get("columns", []))])
            )
            measures.append(" ".join(filter(None, [operation, columns, "as", output])))
            if operation == "count_distinct":
                distinct_counts.extend(map(str, binding.get("columns", [])))
        dimensions = [
            (
                binding.strip() if isinstance(binding, str) else
                str(binding.get("output") or binding.get("column") or "").strip()
            )
            for binding in plan.get("dimensions", [])
            if isinstance(binding, (dict, str))
        ]
        ordering = "; ".join(
            " ".join(filter(None, [
                "requested measure",
                str(item.get("direction") or "").strip(),
            ]))
            for item in plan.get("ordering", []) if isinstance(item, dict)
        ) or "none"
        joins = []
        join_bindings = []
        for item in plan.get("joins", []):
            if isinstance(item, dict):
                tables = list(map(str, item.get("tables", [])))
                keys = {str(k): str(v) for k, v in item.get("keys", {}).items()}
                joins.append(f"{tables} on {keys}")
                join_bindings.append(item)
            elif str(item).strip():
                joins.append(str(item).strip())
        return {
            "filters": list(dict.fromkeys(filter(None, filters))),
            "measures": list(dict.fromkeys(filter(None, measures))),
            "group_by": list(dict.fromkeys(filter(None, dimensions))),
            "distinct_counts": list(dict.fromkeys(filter(None, distinct_counts))),
            "joins": joins,
            "join_bindings": join_bindings,
            "ordering": ordering,
            "limit": plan.get("limit"),
            "output_columns": (
                [str(item).strip() for item in plan.get("output_columns", []) if str(item).strip()]
                or list(dict.fromkeys(filter(None, [
                    *dimensions,
                    *(str(binding.get("output") or "").strip() or "_".join(filter(None, [
                        str(binding.get("operation") or "measure"),
                        *map(str, binding.get("columns", [])),
                    ])) for binding in plan.get("measures", []) if isinstance(binding, dict)),
                ])))
            ),
            "null_policy": str(plan.get("null_policy") or ""),
            "table_roles": dict(plan.get("table_roles") or {}),
            "source": "architect_semantic_plan",
        }


BINDING_KINDS = ("filters", "temporal_filters", "dimensions", "measures")
QUALIFIED_FILTER = re.compile(r"\b(?:with|containing|contains|named|called|category|type|status)\s+[a-z][a-z-]{2,}")
"""A question that qualifies a category by name: a not-null filter is too weak for it."""


class _Grounding:
    """The architect's plan checked against the selected tables' real schemas and values.

    What contradicts the data is a diagnostic that blocks (``contradicted``);
    what the data cannot confirm is a warning the coder must resolve
    (``unknown``); what holds is evidence. ``check`` runs the obligations,
    the tables, the bindings, the joins and the result shape in that order;
    ``verdict`` is the status the coder starts from.
    """

    def __init__(self, tables: list[str], csv_dir: Path, question: str) -> None:
        self.tables = tables
        self.csv_dir = csv_dir
        self.question = question
        self.diagnostics: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.frames: dict[str, pd.DataFrame] = {}
        self.schemas: dict[str, set[str]] = {}
        self._profiled: dict[str, pd.DataFrame] = {}

    # --- the steps ----------------------------------------------------------

    def check(self, plan: dict[str, Any]) -> None:
        for obligation in plan.get("runtime_obligations", []):
            self.diagnostics.append({
                "requirement": "runtime plan completion",
                "status": "unknown", "category": "runtime_plan_obligation",
                "evidence": str(obligation),
                "correction_required": str(obligation),
            })
        self._read_tables()
        for kind in BINDING_KINDS:
            for binding in plan.get(kind, []):
                self._check_binding(kind, binding)
        for join in plan.get("joins", []):
            self._check_join(join)
        self._check_shape(plan)

    def _read_tables(self) -> None:
        for table in self.tables:
            try:
                frame = read_table(self.csv_dir / table.strip(), nrows=500)
                self.frames[table] = frame
                self.schemas[table] = {str(column) for column in frame.columns}
            except Exception as exc:
                self.diagnostics.append({
                    "requirement": "selected table is readable",
                    "status": "contradicted", "table": table, "category": "table_unreadable",
                    "evidence": str(exc), "correction_required": "inspect_data or reject_data",
                })

    def _table_of(self, binding: dict[str, Any]) -> str:
        table = str(binding.get("table") or "")
        return table or (self.tables[0] if len(self.tables) == 1 else "")

    def profiled(self, table: str) -> pd.DataFrame:
        """A larger sample of the table, read once, for the checks a small sample cannot settle."""
        if table not in self._profiled:
            self._profiled[table] = read_table(self.csv_dir / table.strip(), nrows=100_000)
        return self._profiled[table]

    # --- the bindings -------------------------------------------------------

    def _bound(self, kind: str, binding: Any) -> tuple[str, list[str]] | None:
        """The binding's table and columns when they exist in the selection; else the diagnostic that says why not."""
        if not isinstance(binding, dict):
            self.diagnostics.append({
                "requirement": kind, "status": "unknown",
                "category": "unstructured_binding",
                "evidence": str(binding),
                "correction_required": (
                    "Coder must resolve this requirement against the exact "
                    "selected columns; runtime references remain gated."
                ),
            })
            return None
        table = self._table_of(binding)
        columns = [str(column) for column in (binding.get("columns") or [binding.get("column")]) if column]
        if not columns:
            self.diagnostics.append({
                "requirement": str(binding.get("requirement") or binding.get("type") or kind),
                "status": "unknown", "category": "missing_binding_columns",
                "table": table if table in self.tables else None,
                "evidence": {"binding": binding},
                "correction_required": (
                    "Resolve this requirement using selected columns or "
                    "prove that it is dataset-level metadata."
                ),
            })
            return None
        missing = [column for column in columns if column not in self.schemas.get(table, set())]
        if table not in self.tables or missing:
            self.diagnostics.append({
                "requirement": kind, "status": "contradicted",
                "table": table if table in self.tables else None,
                "column": [],
                "category": "unknown_table_or_column",
                "evidence": {
                    "selected_tables": list(self.tables),
                    "available_columns": sorted(self.schemas.get(table, set())),
                },
                "correction_required": "inspect_data and revise the binding",
            })
            return None
        return table, columns

    def _filter_compatible(self, table: str, column: str, operator: str, value: str, observed: list[Any]) -> bool:
        """Whether the filter's value occurs in the column: in the sample, in a date range, or in a larger profile."""
        normalized = [str(item).strip().casefold() for item in observed]
        target = value.casefold()
        compatible = any(target in item for item in normalized) if operator == "contains" else target in normalized
        if operator == "range":
            series = pd.to_datetime(self.profiled(table)[column], errors="coerce")
            requested_years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", value)]
            observed_years = series.dropna().dt.year
            return bool(observed_years.size) and (
                not requested_years or (
                    min(requested_years) >= int(observed_years.min())
                    and max(requested_years) <= int(observed_years.max())
                )
            )
        if compatible:
            return True
        expanded = self.profiled(table)[column].dropna().drop_duplicates().head(5000).tolist()
        expanded_normalized = [str(item).strip().casefold() for item in expanded]
        return any(target in item for item in expanded_normalized) if operator == "contains" else target in expanded_normalized

    def _filter_diagnostic(self, kind: str, binding: dict[str, Any], table: str, columns: list[str], observed: list[Any]) -> bool:
        """Whether a filter binding earned a diagnostic (too weak, or a value never observed)."""
        operator = str(binding.get("operator") or "")
        value = str(binding.get("value") or "").strip()
        if kind == "filters" and operator == "not_null" and QUALIFIED_FILTER.search(self.question.casefold()):
            self.diagnostics.append({
                "requirement": "qualified categorical filter", "status": "contradicted",
                "table": table, "column": columns[0] if columns else None,
                "category": "semantic_filter_too_weak",
                "evidence": {"question": self.question, "operator": operator},
                "correction_required": "inspect categorical values and bind the requested category explicitly",
            })
            return True
        if kind in {"filters", "temporal_filters"} and value:
            if not self._filter_compatible(table, columns[0], operator, value, observed):
                self.diagnostics.append({
                    "requirement": f"runtime filter on {columns[0]}",
                    "status": "unknown",
                    "table": table, "column": columns[0],
                    "category": "filter_value_not_observed",
                    "evidence": {
                        "observed_values": observed,
                        "profiled_rows": len(self.profiled(table)),
                    },
                    "correction_required": "perform targeted inspect_data profiling; absence is not proof of contradiction",
                })
                return True
        return False

    def _measure_diagnostic(self, binding: dict[str, Any], table: str, columns: list[str]) -> bool:
        """Whether a measure binding earned a diagnostic (a non-numeric column, or a distinct row count)."""
        if str(binding.get("operation")) in {"sum", "mean", "min", "max"}:
            if columns and not pd.api.types.is_numeric_dtype(self.frames[table][columns[0]]):
                self.diagnostics.append({
                    "requirement": "numeric measure", "status": "contradicted",
                    "table": table, "column": columns[0],
                    "category": "measure_type_incompatible",
                    "evidence": str(self.frames[table][columns[0]].dtype),
                    "correction_required": "choose a numeric measure or prove a safe conversion",
                })
                return True
        operation = str(binding.get("operation") or "")
        distinct = bool(binding.get("distinct"))
        if operation == "count_rows" and distinct:
            self.diagnostics.append({
                "requirement": "count measure", "status": "contradicted",
                "table": table, "column": columns,
                "category": "count_distinctness_conflict",
                "evidence": {"operation": operation, "distinct": distinct},
                "correction_required": "make operation and distinct semantics consistent",
            })
            return True
        return False

    def _check_binding(self, kind: str, binding: Any) -> None:
        bound = self._bound(kind, binding)
        if bound is None:
            return
        table, columns = bound
        observed: list[Any] = []
        if columns and table in self.frames:
            observed = self.frames[table][columns[0]].dropna().drop_duplicates().head(25).tolist()
        if self._filter_diagnostic(kind, binding, table, columns, observed):
            return
        if kind == "measures" and self._measure_diagnostic(binding, table, columns):
            return
        self.evidence.append({
            "requirement": f"runtime {kind} binding",
            "status": "verified",
            "table": table, "column": columns[0] if len(columns) == 1 else columns,
            "observed_values": observed[:12],
            "observed_range": None,
            "evidence": "verified against runtime schema/profile",
            "origin": "runtime_table_profile",
            "correction_required": "",
        })

    # --- the joins ----------------------------------------------------------

    def _check_join(self, join: Any) -> None:
        if not isinstance(join, dict):
            self.diagnostics.append({
                "requirement": "join", "status": "unknown",
                "category": "unstructured_join",
                "evidence": str(join),
                "correction_required": "resolve against selected tables before execution",
            })
            return
        keys = join.get("keys", {})
        tables = list(map(str, join.get("tables", [])))
        if len(tables) < 2 or not isinstance(keys, dict) or any(table not in self.frames for table in tables):
            self.diagnostics.append({
                "requirement": "join", "status": "contradicted", "category": "join_tables_or_keys_missing",
                "evidence": {
                    "selected_tables": list(self.tables),
                    "available_columns": {table: sorted(self.schemas.get(table, set())) for table in self.tables},
                },
                "correction_required": "inspect both tables and provide observable keys",
            })
            return
        invalid, overlap = self._join_overlap(tables, keys)
        if invalid or not overlap:
            self.diagnostics.append({
                "requirement": "join",
                "status": "contradicted" if invalid else "unknown",
                "category": "join_key_not_verifiable",
                "evidence": {
                    "verified_key_columns": keys if not invalid else {},
                    "sample_overlap": list(overlap)[:10],
                },
                "correction_required": "inspect join cardinality/type/overlap; no observed overlap alone is not proof of impossibility",
            })
            return
        self.evidence.append({
            "requirement": "join selected tables", "status": "verified",
            "table": tables, "columns": keys, "observed_values": list(overlap)[:10],
            "observed_range": None,
            "evidence": "sample key overlap observed", "origin": "runtime_join_profile",
            "correction_required": "",
        })

    def _join_overlap(self, tables: list[str], keys: dict[str, Any]) -> tuple[bool, set[str]]:
        """Whether a key is missing, and the key values the tables share (from the sample, else the profile)."""
        series = []
        for table in tables:
            key = str(keys.get(table) or "")
            if key not in self.schemas.get(table, set()):
                return True, set()
            series.append(set(self.frames[table][key].dropna().astype(str).str.strip().head(500)))
        overlap = set.intersection(*series) if series else set()
        if not overlap:
            expanded = [
                set(self.profiled(table)[str(keys[table])].dropna().astype(str).str.strip()) for table in tables
            ]
            overlap = set.intersection(*expanded) if expanded else set()
        return False, overlap

    # --- the result shape ---------------------------------------------------

    def _check_shape(self, plan: dict[str, Any]) -> None:
        if plan.get("ordering"):
            self.evidence.append({
                "requirement": "result ordering", "status": "verified",
                "table": None, "columns": [], "observed_values": [],
                "observed_range": None, "origin": "structured_semantic_plan",
                "evidence": "ordering direction is structurally representable",
                "correction_required": "",
            })
        if plan.get("limit") is not None:
            limit = plan.get("limit")
            if isinstance(limit, int) and limit > 0:
                self.evidence.append({
                    "requirement": "result limit", "status": "verified",
                    "table": None, "columns": [], "observed_values": [limit],
                    "observed_range": None, "origin": "question_semantic_plan",
                    "evidence": "positive integer limit is representable",
                    "correction_required": "",
                })
            else:
                self.diagnostics.append({
                    "requirement": "result limit", "status": "contradicted",
                    "category": "invalid_limit", "evidence": type(limit).__name__,
                    "correction_required": "provide a positive integer limit or null",
                })
        if plan.get("null_policy"):
            relevant_columns = list(dict.fromkeys(
                str(column)
                for kind in BINDING_KINDS
                for binding in plan.get(kind, []) if isinstance(binding, dict)
                for column in (binding.get("columns") or [binding.get("column")]) if column
            ))
            self.evidence.append({
                "requirement": "null policy", "status": "verified",
                "table": None, "columns": relevant_columns,
                "observed_values": [], "observed_range": None,
                "origin": "runtime_table_profile",
                "evidence": "null counts are available through runtime profiling",
                "correction_required": "",
            })

    def verdict(self) -> dict[str, Any]:
        blockers = [item for item in self.diagnostics if item.get("status") == "contradicted"]
        warnings = [item for item in self.diagnostics if item.get("status") != "contradicted"]
        status = "rejected" if blockers else "executable_with_obligations" if warnings else "verified"
        return {
            "valid": not blockers,
            "locked": not blockers,
            "status": status,
            "diagnostics": blockers,
            "warnings": warnings,
            "evidence": self.evidence,
            "required_action": (
                "resolve_runtime_obligations" if warnings and not blockers else
                None if not blockers else "inspect_data_or_reject_data"
            ),
            "runtime_obligations": warnings,
        }
