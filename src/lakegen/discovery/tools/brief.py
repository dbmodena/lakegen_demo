"""The brief the unified agent writes for the coder from a confirmed selection."""

import re

from lakegen.core.table_io import read_table
from lakegen.discovery.tools.plan_models import _normalize_semantic_plan


class BriefMixin:
    """The coder-brief method of ``Phase12ToolsManager``."""

    def _build_coder_brief(
        self,
        tables: list[str],
        coverage: dict[str, dict[str, object]],
        requirements: dict[str, object],
        combination_strategy: str,
        table_roles: dict[str, str],
        semantic_plan: dict[str, object] | None,
    ) -> dict[str, object]:
        """Normalize the agent's explicit choices without making new choices."""
        schemas: dict[str, list[str]] = {}
        for table in tables:
            try:
                schemas[table] = [str(column) for column in read_table(
                    self.csv_dir / table, nrows=0
                ).columns]
            except Exception:
                # Phase 3 remains the authoritative readability gate. Discovery
                # tests and remote adapters may expose inspection text only.
                schemas[table] = []
        selected_columns: dict[str, list[str]] = {table: [] for table in tables}
        normalization_errors: list[str] = []

        def canonical(value: str) -> str:
            return re.sub(r"[^a-z0-9]", "", value.casefold())

        def resolve(table: str, raw_column: object) -> str | None:
            raw = str(raw_column).strip()
            if not raw or table not in schemas:
                return None
            annotated = re.fullmatch(r"(.+?)\s*\(([^()]+\.(?:parquet|csv))\)\s*", raw)
            if annotated:
                raw = annotated.group(1).strip()
                annotated_table = annotated.group(2).strip()
                if annotated_table != table:
                    return None
            exact = [column for column in schemas[table] if column == raw]
            if len(exact) == 1:
                return exact[0]
            folded = [column for column in schemas[table] if column.casefold() == raw.casefold()]
            if len(folded) == 1:
                return folded[0]
            mechanical = [column for column in schemas[table] if canonical(column) == canonical(raw)]
            return mechanical[0] if len(mechanical) == 1 else None

        def resolve_unique(raw_column: object) -> tuple[str, str] | None:
            # Prefer a globally unique exact spelling before trying harmless
            # aliases. Trip_distance and trip_distance can belong to different
            # yearly tables and must not become an ambiguous folded match.
            exact = [(table, str(raw_column)) for table, names in schemas.items()
                     if raw_column in names]
            if exact:
                return exact[0] if len(exact) == 1 else None
            matches = [
                (table, column)
                for table in tables
                if (column := resolve(table, raw_column)) is not None
            ]
            return matches[0] if len(matches) == 1 else None

        annotated_join_columns: list[tuple[str, str]] = []
        for requirement, evidence in coverage.items():
            if not isinstance(evidence, dict):
                continue
            evidence_table = str(evidence.get("table", "")).strip()
            for raw_column in evidence.get("columns", []):
                annotation = re.fullmatch(
                    r"(.+?)\s*\(([^()]+\.(?:parquet|csv))\)\s*",
                    str(raw_column).strip(),
                )
                table = annotation.group(2).strip() if annotation else evidence_table
                column = resolve(table, raw_column)
                if column and table in selected_columns:
                    if column not in selected_columns[table]:
                        selected_columns[table].append(column)
                    if evidence_table not in tables or "join" in requirement.casefold():
                        annotated_join_columns.append((table, column))
                else:
                    normalization_errors.append(
                        f"{requirement}: {raw_column!s} is not one unambiguous column of {table or 'a selected table'}"
                    )

        joins = list(requirements.get("joins", [])) if isinstance(
            requirements.get("joins"), list
        ) else []
        if not joins and len({table for table, _ in annotated_join_columns}) == 2:
            left, right = annotated_join_columns[:2]
            if left[0] != right[0]:
                joins = [{
                    "left_table": left[0], "left_columns": [left[1]],
                    "right_table": right[0], "right_columns": [right[1]],
                    "how": "inner",
                }]
        dimensions: list[object] = []
        raw_grouping = requirements.get("grouping", [])
        if isinstance(raw_grouping, list):
            for grouping in raw_grouping:
                if not isinstance(grouping, str):
                    dimensions.append(grouping)
                    continue
                match = resolve_unique(grouping)
                dimensions.append(
                    {"table": match[0], "column": match[1], "output": grouping}
                    if match else grouping
                )

        brief: dict[str, object] = {
            "tables": list(tables),
            "selected_columns": selected_columns,
            "filters": list(requirements.get("filters", []))
            if isinstance(requirements.get("filters"), list) else [],
            "temporal_filters": [],
            "dimensions": dimensions,
            "measures": list(requirements.get("measures", []))
            if isinstance(requirements.get("measures"), list) else [],
            "ordering": requirements.get("ordering"),
            "limit": requirements.get("limit"),
            "joins": joins,
            "output_columns": list(requirements.get("output_columns", []))
            if isinstance(requirements.get("output_columns"), list) else [],
            "null_policy": str(requirements.get("null_policy") or ""),
            "table_roles": dict(table_roles),
            "normalization_errors": normalization_errors,
        }
        if semantic_plan:
            brief["filters"] = list(semantic_plan.get("filters", []))
            brief["temporal_filters"] = list(semantic_plan.get("temporal_filters", []))
            brief["dimensions"] = list(semantic_plan.get("dimensions", []))
            brief["measures"] = list(semantic_plan.get("measures", []))
            brief["ordering"] = semantic_plan.get("ordering")
            brief["limit"] = semantic_plan.get("limit")
            brief["joins"] = list(semantic_plan.get("joins", []))
            brief["output_columns"] = list(semantic_plan.get("output_columns", []))
            brief["null_policy"] = str(semantic_plan.get("null_policy") or "")
            brief["table_roles"] = dict(semantic_plan.get("table_roles") or {})
            for group in ("dimensions", "measures", "filters", "temporal_filters"):
                for binding in semantic_plan.get(group, []):
                    if not isinstance(binding, dict):
                        continue
                    table = str(binding.get("table") or (tables[0] if len(tables) == 1 else ""))
                    for raw_column in binding.get("columns", [binding.get("column")]):
                        column = resolve(table, raw_column)
                        if column and column not in selected_columns.get(table, []):
                            selected_columns[table].append(column)
        if len(tables) > 1 and combination_strategy != "single_table":
            brief["combination_strategy"] = combination_strategy
        for kind in ("filters", "temporal_filters", "dimensions", "measures"):
            for binding in brief[kind]:
                if not isinstance(binding, dict):
                    continue
                columns = binding.get("columns") or [binding.get("column")]
                columns = [column for column in columns if column]
                table = str(binding.get("table") or "")
                if not table and columns:
                    matches = [resolve_unique(column) for column in columns]
                    if all(matches) and len({match[0] for match in matches}) == 1:
                        table = matches[0][0]
                        binding["table"] = table
                if table:
                    resolved = [resolve(table, column) for column in columns]
                    if columns and all(resolved):
                        if "column" in binding:
                            binding["column"] = resolved[0]
                        if "columns" in binding:
                            binding["columns"] = resolved
                        for column in resolved:
                            if column not in selected_columns[table]:
                                selected_columns[table].append(column)
        # Canonicalize harmless join-shape variants before the brief crosses
        # the Phase 2/3 boundary. Semantic strings are preserved verbatim.
        brief = _normalize_semantic_plan(brief, tables)
        # Temporary read compatibility for consumers of the old brief shape.
        # Phase 3 uses only the canonical fields above.
        brief["operations"] = list(brief["measures"])
        brief["task"] = {
            **dict(requirements),
            "grouping": ([
                str(item.get("output") or item.get("column") or "")
                for item in brief["dimensions"] if isinstance(item, dict)
            ] or list(requirements.get("grouping", []))),
        }
        return brief
