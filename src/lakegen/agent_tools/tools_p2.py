import sys
import json
import re
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool
from valentine import valentine_match
from valentine.algorithms import ComaPy

from lakegen.core.config import CSV_DIR

try:
    try:
        sloth_dir = Path(__file__).resolve().parent / "lakegen" / "data_integration_tools" / "sloth"
        if str(sloth_dir) not in sys.path:
            sys.path.append(str(sloth_dir))
        from lakegen.data_integration_tools.sloth.sloth import sloth
    except ImportError:
        from lakegen.data_integration_tools.sloth import sloth
except ImportError as e:
    print(f"❌ Critical error: impossible to import sloth: {e}")
    # sys.exit(1)

# ==========================================
# TOOLS
# ==========================================
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_SCHEMA_SAMPLE_ROWS = 500
MAX_SCHEMA_COLUMNS = 80
MAX_UNIQUE_VALUES = 8
MAX_PREVIEW_COLUMNS = 20
MAX_JOINABILITY_MATCHES = 5
JOINABILITY_SAMPLE_ROWS = 5000
MAX_TEMPORAL_PROFILE_COLUMNS = 4
PROFILE_CHUNK_ROWS = 100_000

_TEMPORAL_COLUMN_PATTERN = re.compile(
    r"(^|_)(date|datetime|timestamp|time|year)($|_)",
    re.IGNORECASE,
)


class ConfirmSelectionSchema(BaseModel):
    reasoning: str = Field(description="MANDATORY. Write a brief explanation IN ENGLISH explaining why these specific tables were selected and how they answer the question. Do NOT use quotes, apostrophes, or special characters.")
    tables: list[str] = Field(description="A list of the exact file names needed (e.g., ['2016.csv']). Do not omit any table you need!")

class RejectSelectionSchema(BaseModel):
    reasoning: str = Field(description="Explain step-by-step why the current tables are not good.")
    suggestion: str = Field(description="Suggest better keywords to search for.")


def _compact_tool_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = str(text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _compact_value(value, max_chars: int = 40) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _csv_path(csv_dir: Path, file_name: str) -> Path:
    return Path(csv_dir) / file_name.strip()


def _temporal_profile(path: Path, columns: list[str]) -> tuple[int, list[str]]:
    """Return row count and compact full-file coverage for likely temporal columns."""
    temporal_columns = [
        col for col in columns if _TEMPORAL_COLUMN_PATTERN.search(str(col))
    ][:MAX_TEMPORAL_PROFILE_COLUMNS]
    usecols = temporal_columns or columns[:1]
    total_rows = 0
    stats = {
        col: {"valid": 0, "min": None, "max": None}
        for col in temporal_columns
    }

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=PROFILE_CHUNK_ROWS,
        low_memory=False,
    ):
        total_rows += len(chunk)
        for col in temporal_columns:
            series = chunk[col]
            col_stats = stats[col]

            if re.search(r"(^|_)year($|_)", str(col), re.IGNORECASE):
                parsed = pd.to_numeric(series, errors="coerce")
                parsed = parsed[(parsed >= 1000) & (parsed <= 3000)]
            else:
                parsed = pd.to_datetime(
                    series,
                    errors="coerce",
                    utc=True,
                    format="mixed",
                )

            parsed = parsed.dropna()
            if parsed.empty:
                continue
            col_stats["valid"] += len(parsed)
            chunk_min = parsed.min()
            chunk_max = parsed.max()
            if col_stats["min"] is None or chunk_min < col_stats["min"]:
                col_stats["min"] = chunk_min
            if col_stats["max"] is None or chunk_max > col_stats["max"]:
                col_stats["max"] = chunk_max

    coverage_lines = []
    for col, col_stats in stats.items():
        if not col_stats["valid"]:
            continue
        unavailable_pct = (
            (total_rows - col_stats["valid"]) / total_rows * 100
            if total_rows
            else 0.0
        )
        min_value = col_stats["min"]
        max_value = col_stats["max"]
        if isinstance(min_value, pd.Timestamp):
            min_value = min_value.strftime("%Y-%m-%d")
            max_value = max_value.strftime("%Y-%m-%d")
        coverage_lines.append(
            f"- {col}: {min_value} to {max_value} "
            f"(missing/unparseable {unavailable_pct:.1f}%)"
        )

    return total_rows, coverage_lines


def _inspect_columns(csv_dir: Path, file_name: str) -> str:
    path = _csv_path(csv_dir, file_name)
    if not path.exists():
        return f"Error: File missing in active dataset: {file_name}"

    try:
        df = pd.read_csv(path, nrows=MAX_SCHEMA_SAMPLE_ROWS, low_memory=False)
        schema_info = []
        columns = list(df.columns)
        total_rows, temporal_coverage = _temporal_profile(path, columns)

        for col in columns[:MAX_SCHEMA_COLUMNS]:
            dtype = str(df[col].dtype)
            if (
                pd.api.types.is_string_dtype(df[col].dtype)
                or isinstance(df[col].dtype, pd.CategoricalDtype)
            ):
                unique_vals = df[col].dropna().astype(str).unique().tolist()
                if 0 < len(unique_vals) <= MAX_UNIQUE_VALUES:
                    values = [_compact_value(value) for value in unique_vals]
                    schema_info.append(f"- {col} (Category sample): {values}")
                    continue
            schema_info.append(f"- {col} ({dtype})")

        if len(columns) > MAX_SCHEMA_COLUMNS:
            schema_info.append(
                f"- ... {len(columns) - MAX_SCHEMA_COLUMNS} more columns omitted"
            )

        header_lines = [f"Schema for {file_name}:", f"Rows: {total_rows:,}"]
        if temporal_coverage:
            header_lines.extend(["Temporal coverage:", *temporal_coverage])
        header_lines.append(
            f"Columns (types and categories sampled from first {MAX_SCHEMA_SAMPLE_ROWS} rows):"
        )
        output = "\n".join(header_lines + schema_info)
        return _compact_tool_output(output)
    except Exception as e:
        return f"Error: {str(e)}"


def _preview_data(csv_dir: Path, file_name: str, n_rows: int = 3) -> str:
    path = _csv_path(csv_dir, file_name)
    if not path.exists():
        return f"Error: File missing in active dataset: {file_name}"

    try:
        n_rows = int(n_rows)
    except (TypeError, ValueError):
        n_rows = 3
    n_rows = max(1, min(n_rows, 5))
    try:
        df = pd.read_csv(path, nrows=n_rows)
        omitted = ""
        if len(df.columns) > MAX_PREVIEW_COLUMNS:
            omitted = f"\n... {len(df.columns) - MAX_PREVIEW_COLUMNS} more columns omitted"
            df = df.iloc[:, :MAX_PREVIEW_COLUMNS]
        output = f"Preview of {file_name}:\n{df.to_string(index=False)}{omitted}"
        return _compact_tool_output(output, max_chars=3000)
    except Exception as e:
        return f"Error: {str(e)}"


def _find_exact_overlaps(csv_dir: Path, file_name_1: str, file_name_2: str) -> str:
    path_1 = str(_csv_path(csv_dir, file_name_1))
    path_2 = str(_csv_path(csv_dir, file_name_2))
    try:
        df1 = pd.read_csv(path_1, nrows=5000).astype(str)
        df2 = pd.read_csv(path_2, nrows=5000).astype(str)
        r_tab = [df1[col].tolist() for col in df1.columns]
        s_tab = [df2[col].tolist() for col in df2.columns]
        results = sloth(
            r_tab=r_tab,
            s_tab=s_tab,
            min_a=10,
            min_w=1,
            max_w=min(len(df1.columns), len(df2.columns)),
            min_h=5,
            max_h=min(len(df1), len(df2)),
            complete=False,
            verbose=False,
        )
        if not results:
            return "No exact overlap found."
        return "Exact overlap found!"
    except Exception as e:
        return f"Error in SLOTH when comparing '{file_name_1}' and '{file_name_2}': {e}. Try different tables."


def _normalize_join_values(series: pd.Series) -> pd.Series:
    """Normalize non-null values before measuring exact equi-join overlap."""
    values = series.dropna()
    if values.empty:
        return pd.Series(dtype="string")

    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return numeric.map(lambda value: format(float(value), ".15g"))

    normalized = values.astype("string").str.strip().str.casefold()
    return normalized[normalized != ""]


def _joinability_metrics(left: pd.Series, right: pd.Series) -> dict[str, object]:
    left_values = _normalize_join_values(left)
    right_values = _normalize_join_values(right)
    left_counts = left_values.value_counts()
    right_counts = right_values.value_counts()
    common_values = left_counts.index.intersection(right_counts.index)

    left_distinct = len(left_counts)
    right_distinct = len(right_counts)
    common_distinct = len(common_values)
    union_distinct = left_distinct + right_distinct - common_distinct

    left_common_rows = int(left_counts.loc[common_values].sum()) if common_distinct else 0
    right_common_rows = int(right_counts.loc[common_values].sum()) if common_distinct else 0
    estimated_inner_rows = (
        int((left_counts.loc[common_values] * right_counts.loc[common_values]).sum())
        if common_distinct
        else 0
    )

    left_unique = (
        bool((left_counts.loc[common_values] == 1).all())
        if common_distinct
        else False
    )
    right_unique = (
        bool((right_counts.loc[common_values] == 1).all())
        if common_distinct
        else False
    )
    if not common_distinct:
        relationship = "undetermined"
    elif left_unique and right_unique:
        relationship = "one-to-one"
    elif left_unique:
        relationship = "one-to-many"
    elif right_unique:
        relationship = "many-to-one"
    else:
        relationship = "many-to-many"

    left_distinct_coverage = common_distinct / left_distinct if left_distinct else 0.0
    right_distinct_coverage = common_distinct / right_distinct if right_distinct else 0.0
    containment = max(left_distinct_coverage, right_distinct_coverage)

    if not common_distinct:
        verdict = "not joinable: no common values in the sample"
    elif relationship == "many-to-many" and containment >= 0.5:
        verdict = "risky: value overlap exists but the join is many-to-many"
    elif containment >= 0.8:
        verdict = "joinable candidate"
    elif containment >= 0.4:
        verdict = "partially joinable"
    else:
        verdict = "weak join candidate"

    matched_baseline = max(left_common_rows, right_common_rows, 1)
    return {
        "common_distinct": common_distinct,
        "left_distinct_coverage": left_distinct_coverage,
        "right_distinct_coverage": right_distinct_coverage,
        "jaccard": common_distinct / union_distinct if union_distinct else 0.0,
        "left_row_coverage": left_common_rows / len(left_values) if len(left_values) else 0.0,
        "right_row_coverage": right_common_rows / len(right_values) if len(right_values) else 0.0,
        "left_uniqueness": left_distinct / len(left_values) if len(left_values) else 0.0,
        "right_uniqueness": right_distinct / len(right_values) if len(right_values) else 0.0,
        "relationship": relationship,
        "estimated_inner_rows": estimated_inner_rows,
        "expansion_factor": estimated_inner_rows / matched_baseline,
        "verdict": verdict,
    }


def _find_schema_matches(csv_dir: Path, file_name_1: str, file_name_2: str) -> str:

    path_1 = str(_csv_path(csv_dir, file_name_1))
    path_2 = str(_csv_path(csv_dir, file_name_2))

    try:
        df1 = pd.read_csv(path_1, nrows=JOINABILITY_SAMPLE_ROWS, low_memory=False)
        df2 = pd.read_csv(path_2, nrows=JOINABILITY_SAMPLE_ROWS, low_memory=False)

        matcher = ComaPy(use_instances=True)
        matches = valentine_match(
            df1.astype("string"),
            df2.astype("string"),
            matcher,
        )

        if not matches:
            return "No schema matches found."

        ranked_matches = [
            (col1, col2, score)
            for ((_, col1), (_, col2)), score in sorted(
                matches.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0.0 and col1 in df1.columns and col2 in df2.columns
        ]
        if not ranked_matches:
            return "No schema matches found."

        lines = [
            f"Valentine joinability analysis between '{file_name_1}' and '{file_name_2}':",
            f"Sample: first {JOINABILITY_SAMPLE_ROWS} rows per table; comparisons use normalized exact values.",
        ]
        for index, (col1, col2, score) in enumerate(
            ranked_matches[:MAX_JOINABILITY_MATCHES],
            start=1,
        ):
            metrics = _joinability_metrics(df1[col1], df2[col2])
            lines.extend(
                [
                    "",
                    f"{index}. {col1} <-> {col2}",
                    f"   Valentine similarity: {score:.4f}",
                    (
                        f"   Distinct overlap: {metrics['common_distinct']} "
                        f"(left {metrics['left_distinct_coverage']:.1%}, "
                        f"right {metrics['right_distinct_coverage']:.1%}, "
                        f"Jaccard {metrics['jaccard']:.1%})"
                    ),
                    (
                        f"   Row coverage: left {metrics['left_row_coverage']:.1%}, "
                        f"right {metrics['right_row_coverage']:.1%}"
                    ),
                    (
                        f"   Key uniqueness: left {metrics['left_uniqueness']:.1%}, "
                        f"right {metrics['right_uniqueness']:.1%}"
                    ),
                    (
                        f"   Cardinality: {metrics['relationship']}; "
                        f"estimated inner-join rows {metrics['estimated_inner_rows']:,}; "
                        f"expansion {metrics['expansion_factor']:.2f}x"
                    ),
                    f"   Verdict: {metrics['verdict']}",
                ]
            )

        omitted = len(ranked_matches) - MAX_JOINABILITY_MATCHES
        if omitted > 0:
            lines.extend(["", f"... {omitted} lower-scoring matches omitted"])

        return _compact_tool_output("\n".join(lines))

    except Exception as e:
        return f"Error in Valentine matcher for '{file_name_1}' and '{file_name_2}': {e}. Try different tables."


class Phase2JudgeToolsManager:
    """Manager for Phase 2 judge tools to avoid closures and improve testability."""
    
    def __init__(self, candidates: list[str], csv_dir: Path):
        self.candidates = candidates
        self.csv_dir = Path(csv_dir)

    def inspect_columns(self, file_name: str) -> str:
        """
        Returns a compact profile for one CSV in the active dataset.
        Shows row count, full-file min/max coverage for temporal columns, column
        types, and sample values for low-cardinality categorical columns.
        Use this to understand what data a table contains.
        """
        return _inspect_columns(self.csv_dir, file_name)

    def find_schema_matches(self, file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine to identify matching columns, then verify their practical
        joinability through value overlap, row coverage, key uniqueness,
        cardinality, and estimated join expansion.
        """
        return _find_schema_matches(self.csv_dir, file_name_1, file_name_2)

    def confirm_table_selection(self, reasoning: str, tables: list[str]) -> str:
        """
        CRITICAL: Use this tool ONLY when you have identified the required files.
        Calling this tool terminates execution and confirms the selection.
        """
        final_tables = ", ".join(str(t) for t in tables)

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    def reject_selection(self, reasoning: str, suggestion: str) -> str:
        """
        CRITICAL: Use this tool ONLY when NONE of the candidate tables are relevant to the user's question.
        Calling this tool means you have successfully finished the task by rejecting the candidates.
        """
        return f"REJECT_KEYWORDS: {reasoning}\nSuggestion: {suggestion}"

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.inspect_columns),
            FunctionTool.from_defaults(fn=self.find_schema_matches),
            FunctionTool.from_defaults(fn=self.confirm_table_selection, fn_schema=ConfirmSelectionSchema, return_direct=True),
            FunctionTool.from_defaults(fn=self.reject_selection, fn_schema=RejectSelectionSchema, return_direct=True),
        ]


def make_p2_judge_tools(
    candidates: list[str],
    csv_dir: Path,
) -> list:
    """
    Build tools for the Phase 2 *judge-only* agent.
    Does NOT include search_solr — the Solr query is done programmatically
    before the agent runs, and candidates are provided in the prompt.

    Tools: inspect_columns, find_schema_matches (Valentine),
           confirm_table_selection.
    """
    manager = Phase2JudgeToolsManager(candidates, csv_dir)
    return manager.get_tools()
