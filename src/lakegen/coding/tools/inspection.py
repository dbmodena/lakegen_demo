"""The inspection tools of the coder: a table's columns and sample rows."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from lakegen.integrations.table_io import read_table, table_load_command


class InspectionMixin:
    """Part of ``Phase3ToolsManager``; the manager supplies the attributes it reads."""

    def inspect_table(self, file_name: str, columns: str = "") -> str:
        """Inspect a selected table from one cached sample. Up to eight columns may be profiled progressively."""
        table = file_name.strip()
        supplied = Path(table).expanduser()
        if table not in self.tables and supplied.is_absolute():
            supplied_resolved = supplied.resolve()
            table = next(
                (
                    selected
                    for selected in self.tables
                    if (self.csv_dir / selected).resolve() == supplied_resolved
                ),
                table,
            )
        if table not in self.tables:
            return json.dumps({"ok": False, "error": "Only selected tables may be inspected.", "selected_tables": self.tables})
        path = self.csv_dir / table
        cached = table in self.state.table_samples
        if cached:
            frame = self.state.table_samples[table]
        else:
            try:
                frame = read_table(path, nrows=5000)
            except Exception as exc:
                return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            self.state.inspected_tables.add(table)
            self.state.table_samples[table] = frame
            self.state.table_profiled_columns[table] = set()
        requested = [item.strip() for item in columns.split(",") if item.strip()]
        exact = {str(column): column for column in frame.columns}
        details: dict[str, Any] = {}
        resolved_requests: list[tuple[str, str]] = []
        for requested_column in requested:
            matches = difflib.get_close_matches(requested_column.strip(), list(exact), n=1, cutoff=0.55)
            if not matches:
                details[requested_column] = {"error": "column_not_found"}
                continue
            name = matches[0]
            resolved_requests.append((requested_column, name))
        already_profiled = self.state.table_profiled_columns[table]
        new_names = {name for _requested, name in resolved_requests} - already_profiled
        if len(already_profiled | new_names) > 8:
            return json.dumps({
                "ok": False,
                "error": "Column profile limit exceeded: at most 8 distinct columns per table.",
                "already_profiled": sorted(already_profiled),
                "remaining_profile_slots": max(0, 8 - len(already_profiled)),
                "next_action": "Use the existing evidence and choose run_analysis or reject_data.",
            })
        already_profiled.update(new_names)
        for requested_column, name in resolved_requests:
            series = frame[exact[name]]
            non_null = series.dropna()
            detail: dict[str, Any] = {
                "exact_name": name,
                "dtype": str(series.dtype),
                "null_count_in_sample": int(series.isna().sum()),
                "distinct_count_in_sample": int(non_null.nunique()),
                "sample_values": [str(value)[:100] for value in non_null.drop_duplicates().head(12)],
            }
            metadata_description = self._column_description(table, name)
            if metadata_description:
                detail["metadata_description"] = metadata_description
            if not non_null.empty and ("date" in name.casefold() or "year" in name.casefold()):
                parsed = pd.to_datetime(non_null, errors="coerce").dropna()
                if not parsed.empty:
                    detail["temporal_min"] = str(parsed.min())
                    detail["temporal_max"] = str(parsed.max())
            details[requested_column] = detail
        temporal_candidates = [
            str(column) for column in frame.columns
            if re.search(
                r"(?:^|[_\W])(year|date|time|period|fy|sy)(?:$|[_\W])",
                str(column), re.IGNORECASE,
            )
            or re.search(r"(?:year|date)$", str(column), re.IGNORECASE)
        ]
        response = json.dumps({
            "ok": True,
            "cached_sample": cached,
            "table": table,
            "load_command": table_load_command(path),
            "sampled_rows": len(frame),
            "columns": [str(column) for column in frame.columns],
            "resource_metadata": self._resource_metadata_preview(table),
            "semantic_ambiguities": ({
                "temporal_columns": temporal_candidates[:8],
                "guidance": "Choose using resource/column meaning and inspected values, not the column name alone.",
            } if len(temporal_candidates) > 1 else {}),
            "requested_column_profiles": details,
            "profiled_columns_total": sorted(already_profiled),
            "remaining_profile_slots": 8 - len(already_profiled),
            "next_action": (
                "Choose one: run_analysis if the evidence is sufficient, or "
                "reject_data if required data is missing."
            ),
        }, ensure_ascii=False, default=str)
        return response

    def inspect_data(
        self,
        tables: list[str],
        columns: dict[str, list[str]] | None = None,
        include: list[str] | None = None,
    ) -> str:
        """Inspect selected tables in one bounded, consolidated tool call."""
        if self.state.data_inspection_used:
            return json.dumps({
                "ok": False, "status": "inspection_limit_reached",
                "message": "inspect_data is available once; use the cached evidence and run_analysis.",
            })
        if not isinstance(tables, list) or not all(
            isinstance(table, str) for table in tables
        ):
            return json.dumps({
                "ok": False, "status": "invalid_arguments",
                "message": "tables must be a list of selected table names",
            })
        if columns is not None and not isinstance(columns, dict):
            return json.dumps({
                "ok": False, "status": "invalid_arguments",
                "message": (
                    "columns must map each selected table name to a list of exact "
                    "runtime column names"
                ),
                "example": {table: [] for table in tables},
            })
        if isinstance(columns, dict) and any(
            not isinstance(table, str) or not isinstance(names, list)
            or not all(isinstance(name, str) for name in names)
            for table, names in columns.items()
        ):
            return json.dumps({
                "ok": False, "status": "invalid_arguments",
                "message": "every columns entry must be a list of strings",
            })
        if include is not None and (
            not isinstance(include, list)
            or not all(isinstance(item, str) for item in include)
        ):
            return json.dumps({
                "ok": False, "status": "invalid_arguments",
                "message": "include must be a list of strings",
            })
        requested = list(dict.fromkeys(map(str, tables)))
        invalid = [table for table in requested if table not in self.tables]
        if invalid:
            return json.dumps({
                "ok": False, "status": "invalid_table", "tables": invalid,
                "available_tables": self.tables,
            })
        self.state.data_inspection_used = True
        profiles = []
        for table in requested:
            requested_columns = (columns or {}).get(table, [])
            profiles.append(json.loads(self.inspect_table(
                table, ",".join(map(str, requested_columns))
            )))
        return json.dumps({
            "ok": True, "status": "inspected", "profiles": profiles,
            "include": include or ["schema", "sample_values", "null_counts", "temporal_range"],
            "next_actions": ["run_analysis", "reject_data"],
        }, ensure_ascii=False, default=str)
