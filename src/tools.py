import sys
import json
import pandas as pd
import polars as pl
from pathlib import Path
from llama_index.core.tools import FunctionTool
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher

from utils import CSV_DIR

try:
    from blend.blend import BLEND
except ImportError as e:
    print(f"❌ Critical error: impossible to import BLEND.\nDettaglio: {e}")
    sys.exit(1)

try:
    try:
        from Sloth.sloth import sloth
    except ImportError:
        from sloth import sloth
except ImportError as e:
    print(f"❌ Critical error: impossible to import sloth.\nDettaglio: {e}")
    sys.exit(1)

# ==========================================
# TOOLS
# ==========================================
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_SCHEMA_SAMPLE_ROWS = 500
MAX_SCHEMA_COLUMNS = 80
MAX_UNIQUE_VALUES = 8
MAX_PREVIEW_COLUMNS = 20
MAX_SCHEMA_MATCHES = 12


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


def _inspect_columns(csv_dir: Path, file_name: str) -> str:
    path = _csv_path(csv_dir, file_name)
    if not path.exists():
        return f"Error: File missing in active dataset: {file_name}"

    try:
        df = pd.read_csv(path, nrows=MAX_SCHEMA_SAMPLE_ROWS, low_memory=False)
        schema_info = []
        columns = list(df.columns)

        for col in columns[:MAX_SCHEMA_COLUMNS]:
            dtype = str(df[col].dtype)
            if dtype in {"object", "string", "category"}:
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

        output = (
            f"Schema for {file_name} "
            f"(sampled first {MAX_SCHEMA_SAMPLE_ROWS} rows):\n"
            + "\n".join(schema_info)
        )
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
        return f"Error SLOTH: {e}"


def _find_schema_matches(csv_dir: Path, file_name_1: str, file_name_2: str) -> str:
    path_1 = str(_csv_path(csv_dir, file_name_1))
    path_2 = str(_csv_path(csv_dir, file_name_2))
    try:
        df1 = pd.read_csv(path_1, nrows=5000).astype(str)
        df2 = pd.read_csv(path_2, nrows=5000).astype(str)

        matcher = JaccardDistanceMatcher()
        matches = valentine_match(df1, df2, matcher)

        if not matches:
            return "No schema matches found."

        output = f"Valentine matches between '{file_name_1}' and '{file_name_2}':\n"
        found_match = False
        shown = 0
        sorted_matches = sorted(
            matches.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for ((_, col1), (_, col2)), score in sorted_matches:
            if score <= 0.0:
                continue
            output += f"-> Column '{col1}' matches Column '{col2}' (Score: {score:.3f})\n"
            found_match = True
            shown += 1
            if shown >= MAX_SCHEMA_MATCHES:
                break

        if not found_match:
            return "No schema matches found."
        if len(matches) > shown:
            output += f"... {len(matches) - shown} lower-scoring matches omitted\n"

        return _compact_tool_output(output)
    except Exception as e:
        return f"Error Valentine: {e}"


def inspect_columns(file_name: str) -> str:
    """
    Returns the exact list of column names in a CSV file. 
    If a column is categorical (low cardinality), shows its unique values.
    """
    return _inspect_columns(CSV_DIR, file_name)

def preview_data(file_name: str, n_rows: int = 3) -> str:
    """
    Returns the first 'n_rows' of a CSV file to check data TYPES and FORMATS (e.g., if a date is YYYY-MM-DD).
    CRITICAL RULE: DO NOT use this tool to search for specific rows, names, or values (like 'Chicago' or '2017'). 
    Trust the column names. The Python script will do the filtering later.
    """
    return _preview_data(CSV_DIR, file_name, n_rows)

def find_exact_overlaps(file_name_1: str, file_name_2: str) -> str:
    """
    Use the SLOTH engine to find structural overlaps between two files.
    This tool confirms which columns can be used for a pd.merge() by analyzing data content.
    """
    return _find_exact_overlaps(CSV_DIR, file_name_1, file_name_2)

def find_schema_matches(file_name_1: str, file_name_2: str) -> str:
    """
    Use Valentine with JaccardDistanceMatcher to find matching columns between two files based on data content and schema.
    This tool helps identify overlapping columns that can be used for JOIN operations.
    """
    return _find_schema_matches(CSV_DIR, file_name_1, file_name_2)

def confirm_table_selection(selected_files: str, reasoning: str) -> str:
    """
    CRITICAL: Use this tool ONLY when you have identified the required files.
    - selected_files: A comma-separated string of the exact file names needed (e.g., "2016.csv").
    - reasoning: Write a brief explanation IN ENGLISH. Do NOT use quotes, apostrophes, or special characters.
    Calling this tool means you have successfully finished the task.
    """
    dati_uscita = {
        "tables": selected_files,
        "reasoning": reasoning
    }
    return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"


def make_agent_tools(blend_db_path: Path, csv_dir: Path | None = None) -> list:
    """
    Builds the list of LlamaIndex FunctionTools for the Data Architect agent.
    The `find_joinable_tables` closure is bound to the pre-built BLEND index at
    *blend_db_path*, which is constructed once in `select_tables` over the
    fuzzy-matched top-10 candidate files.
    """
    active_csv_dir = Path(csv_dir) if csv_dir is not None else CSV_DIR

    def inspect_columns_tool(file_name: str) -> str:
        """
        Returns a compact schema for one CSV in the active dataset.
        Prefer this over preview_data. Call it at most once per candidate file.
        """
        return _inspect_columns(active_csv_dir, file_name)

    inspect_columns_tool.__name__ = "inspect_columns"

    def preview_data_tool(file_name: str) -> str:
        """
        Returns a compact preview of one CSV in the active dataset.
        Use only when column names are insufficient to decide relevance.
        """
        return _preview_data(active_csv_dir, file_name, 2)

    preview_data_tool.__name__ = "preview_data"

    def find_exact_overlaps_tool(file_name_1: str, file_name_2: str) -> str:
        """
        Use the SLOTH engine to check whether two active-dataset files have an exact structural overlap.
        """
        return _find_exact_overlaps(active_csv_dir, file_name_1, file_name_2)

    find_exact_overlaps_tool.__name__ = "find_exact_overlaps"

    def find_schema_matches_tool(file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine with JaccardDistanceMatcher to find compact matching-column evidence between two active-dataset files.
        """
        return _find_schema_matches(active_csv_dir, file_name_1, file_name_2)

    find_schema_matches_tool.__name__ = "find_schema_matches"

    def find_joinable_tables(file_name: str, target_columns: list[str]) -> str:
        """
        Use the BLEND engine to find which other tables in the Data Lake can be joined with the specified file.

        Args:
            file_name: The name of the file to search for joins.
            target_columns: A list of strings representing the specific columns of interest to use for the join search. Do NOT use all columns, only those relevant to the user's query.

        PAY ATTENTION TO SCORE RULES:
        The score is an AVERAGE across all columns. A low score (e.g., 0.05 - 0.20) is actually EXCELLENT
        and indicates that the two files share the exact key column (Primary Key) for the merge,
        but have different data elsewhere in the table (which is the purpose of a JOIN!).
        Consider valid and recommend all files with scores > 0.05.
        """
        file_name = file_name.strip()
        path_file = active_csv_dir / file_name
        if not path_file.exists():
            return f"Error: Target file missing in active dataset: {file_name}"
        if not blend_db_path.exists():
            return "Error: BLEND index not found. The index should have been built before the agent started."

        try:
            blend_engine = BLEND(db_path=blend_db_path)
            df_target = pl.read_csv(str(path_file), n_rows=2000, ignore_errors=True)

            valid_cols = [col for col in target_columns if col in df_target.columns]
            if not valid_cols:
                blend_engine.close()
                return f"Error: None of the specified target_columns {target_columns} exist in {file_name}."

            df_target = df_target.select(valid_cols)

            results = blend_engine.multi_column_join_search(table=df_target, k=5, clean=True)
            blend_engine.close()

            if not results:
                return "No compatible table found."
            output = f"BLEND Results for '{file_name}' using columns {valid_cols}:\n"
            for t_id, _, score in results:
                if t_id != file_name:
                    output += f"-> {t_id} (Score: {score:.3f})\n"
            return _compact_tool_output(output)
        except Exception as e:
            return f"Error BLEND: {e}"

    return [
        FunctionTool.from_defaults(fn=inspect_columns_tool),
        FunctionTool.from_defaults(fn=preview_data_tool),
        FunctionTool.from_defaults(fn=find_joinable_tables),
        FunctionTool.from_defaults(fn=find_exact_overlaps_tool),
        FunctionTool.from_defaults(fn=find_schema_matches_tool),
        FunctionTool.from_defaults(fn=confirm_table_selection, return_direct=True),
    ]
