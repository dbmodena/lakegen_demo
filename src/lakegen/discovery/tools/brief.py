"""The brief the unified agent writes for the coder from a confirmed selection.

The brief normalises the agent's explicit choices without making new ones:
every column it names is resolved against the selected tables' real
schemas (exact spelling first, then case, then a mechanical match), the
joins and dimensions the requirements imply are bound, and a semantic plan,
when the confirmation carried one, supplies the bindings. Phase 3 reads
the canonical fields; ``operations`` and ``task`` keep the old shape
readable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lakegen.core.table_io import read_table
from lakegen.discovery.tools.plan_models import _normalize_semantic_plan

ANNOTATED_COLUMN = re.compile(r"(.+?)\s*\(([^()]+\.(?:parquet|csv))\)\s*")
"""A column the agent wrote with its table in brackets: ``year (sales.parquet)``."""


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


class ColumnResolver:
    """The real column a name refers to, in one table or in one of the selected tables."""

    def __init__(self, schemas: dict[str, list[str]], tables: list[str]) -> None:
        self.schemas = schemas
        self.tables = tables

    def resolve(self, table: str, raw_column: object) -> str | None:
        raw = str(raw_column).strip()
        if not raw or table not in self.schemas:
            return None
        annotated = ANNOTATED_COLUMN.fullmatch(raw)
        if annotated:
            raw = annotated.group(1).strip()
            if annotated.group(2).strip() != table:
                return None
        columns = self.schemas[table]
        exact = [column for column in columns if column == raw]
        if len(exact) == 1:
            return exact[0]
        folded = [column for column in columns if column.casefold() == raw.casefold()]
        if len(folded) == 1:
            return folded[0]
        mechanical = [column for column in columns if _canonical(column) == _canonical(raw)]
        return mechanical[0] if len(mechanical) == 1 else None

    def resolve_unique(self, raw_column: object) -> tuple[str, str] | None:
        """The one selected table and column a bare name means, or None when ambiguous.

        A globally unique exact spelling wins before the harmless aliases are
        tried: Trip_distance and trip_distance can belong to different yearly
        tables and must not become an ambiguous folded match.
        """
        exact = [(table, str(raw_column)) for table, names in self.schemas.items() if raw_column in names]
        if exact:
            return exact[0] if len(exact) == 1 else None
        matches = [
            (table, column)
            for table in self.tables
            if (column := self.resolve(table, raw_column)) is not None
        ]
        return matches[0] if len(matches) == 1 else None


def _schemas(csv_dir: Path, tables: list[str]) -> dict[str, list[str]]:
    schemas: dict[str, list[str]] = {}
    for table in tables:
        try:
            schemas[table] = [str(column) for column in read_table(csv_dir / table, nrows=0).columns]
        except Exception:
            # Phase 3 remains the authoritative readability gate. Discovery
            # tests and remote adapters may expose inspection text only.
            schemas[table] = []
    return schemas


class CoderBriefBuilder:
    """One brief from one confirmation; ``build`` is the only entry."""

    def __init__(self, csv_dir: Path, tables: list[str]) -> None:
        self.tables = list(tables)
        self.resolver = ColumnResolver(_schemas(csv_dir, self.tables), self.tables)
        self.selected_columns: dict[str, list[str]] = {table: [] for table in self.tables}
        self.errors: list[str] = []

    def _select(self, table: str, column: str) -> None:
        if column not in self.selected_columns.setdefault(table, []):
            self.selected_columns[table].append(column)

    def _coverage_columns(self, coverage: dict[str, Any]) -> list[tuple[str, str]]:
        """Select the columns the coverage names; the ones that look like join keys."""
        join_columns: list[tuple[str, str]] = []
        for requirement, evidence in coverage.items():
            if not isinstance(evidence, dict):
                continue
            evidence_table = str(evidence.get("table", "")).strip()
            for raw_column in evidence.get("columns", []):
                annotation = ANNOTATED_COLUMN.fullmatch(str(raw_column).strip())
                table = annotation.group(2).strip() if annotation else evidence_table
                column = self.resolver.resolve(table, raw_column)
                if column and table in self.selected_columns:
                    self._select(table, column)
                    if evidence_table not in self.tables or "join" in requirement.casefold():
                        join_columns.append((table, column))
                else:
                    self.errors.append(
                        f"{requirement}: {raw_column!s} is not one unambiguous column of "
                        f"{table or 'a selected table'}"
                    )
        return join_columns

    @staticmethod
    def _joins(requirements: dict[str, Any], join_columns: list[tuple[str, str]]) -> list[Any]:
        """The requirements' joins, or the one join two annotated columns imply."""
        joins = list(requirements.get("joins", [])) if isinstance(requirements.get("joins"), list) else []
        if not joins and len({table for table, _ in join_columns}) == 2:
            left, right = join_columns[:2]
            if left[0] != right[0]:
                joins = [{
                    "left_table": left[0], "left_columns": [left[1]],
                    "right_table": right[0], "right_columns": [right[1]],
                    "how": "inner",
                }]
        return joins

    def _dimensions(self, requirements: dict[str, Any]) -> list[Any]:
        """The grouping columns bound to their tables where the name is unambiguous."""
        dimensions: list[Any] = []
        raw_grouping = requirements.get("grouping", [])
        if not isinstance(raw_grouping, list):
            return dimensions
        for grouping in raw_grouping:
            if not isinstance(grouping, str):
                dimensions.append(grouping)
                continue
            match = self.resolver.resolve_unique(grouping)
            dimensions.append({"table": match[0], "column": match[1], "output": grouping} if match else grouping)
        return dimensions

    def _base(self, requirements: dict[str, Any], table_roles: dict[str, str], joins: list[Any], dimensions: list[Any]) -> dict[str, Any]:
        def listed(key: str) -> list[Any]:
            return list(requirements.get(key, [])) if isinstance(requirements.get(key), list) else []

        return {
            "tables": list(self.tables),
            "selected_columns": self.selected_columns,
            "filters": listed("filters"),
            "temporal_filters": [],
            "dimensions": dimensions,
            "measures": listed("measures"),
            "ordering": requirements.get("ordering"),
            "limit": requirements.get("limit"),
            "joins": joins,
            "output_columns": listed("output_columns"),
            "null_policy": str(requirements.get("null_policy") or ""),
            "table_roles": dict(table_roles),
            "normalization_errors": self.errors,
        }

    def _apply_plan(self, brief: dict[str, Any], plan: dict[str, Any]) -> None:
        """A semantic plan supplies the bindings; its columns are selected."""
        for key in ("filters", "temporal_filters", "dimensions", "measures", "joins", "output_columns"):
            brief[key] = list(plan.get(key, []))
        brief["ordering"] = plan.get("ordering")
        brief["limit"] = plan.get("limit")
        brief["null_policy"] = str(plan.get("null_policy") or "")
        brief["table_roles"] = dict(plan.get("table_roles") or {})
        default_table = self.tables[0] if len(self.tables) == 1 else ""
        for group in ("dimensions", "measures", "filters", "temporal_filters"):
            for binding in plan.get(group, []):
                if not isinstance(binding, dict):
                    continue
                table = str(binding.get("table") or default_table)
                for raw_column in binding.get("columns", [binding.get("column")]):
                    column = self.resolver.resolve(table, raw_column)
                    if column and column not in self.selected_columns.get(table, []):
                        self.selected_columns[table].append(column)

    def _bind(self, binding: dict[str, Any]) -> None:
        """A binding's columns in their real spelling, its table inferred when unambiguous."""
        columns = [column for column in (binding.get("columns") or [binding.get("column")]) if column]
        table = str(binding.get("table") or "")
        if not table and columns:
            matches = [self.resolver.resolve_unique(column) for column in columns]
            if all(matches) and len({match[0] for match in matches}) == 1:
                table = matches[0][0]
                binding["table"] = table
        if not table:
            return
        resolved = [self.resolver.resolve(table, column) for column in columns]
        if not columns or not all(resolved):
            return
        if "column" in binding:
            binding["column"] = resolved[0]
        if "columns" in binding:
            binding["columns"] = resolved
        for column in resolved:
            self._select(table, column)

    def build(
        self,
        coverage: dict[str, Any],
        requirements: dict[str, Any],
        combination_strategy: str,
        table_roles: dict[str, str],
        semantic_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        join_columns = self._coverage_columns(coverage)
        brief = self._base(requirements, table_roles, self._joins(requirements, join_columns), self._dimensions(requirements))
        if semantic_plan:
            self._apply_plan(brief, semantic_plan)
        if len(self.tables) > 1 and combination_strategy != "single_table":
            brief["combination_strategy"] = combination_strategy
        for kind in ("filters", "temporal_filters", "dimensions", "measures"):
            for binding in brief[kind]:
                if isinstance(binding, dict):
                    self._bind(binding)
        # Canonicalize harmless join-shape variants before the brief crosses
        # the Phase 2/3 boundary. Semantic strings are preserved verbatim.
        brief = _normalize_semantic_plan(brief, self.tables)
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
        return CoderBriefBuilder(self.csv_dir, tables).build(
            coverage, requirements, combination_strategy, table_roles, semantic_plan,
        )
