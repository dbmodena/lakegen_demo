import sys
import json
import pandas as pd
import polars as pl
from pathlib import Path
from llama_index.core.tools import FunctionTool
from valentine import valentine_match
from valentine.algorithms import ComaPy

from lakegen.utils import CSV_DIR

try:
    from blend import BLEND
except ImportError as e:
    print(f"❌ Critical error: impossible to import BLEND: {e}")
    sys.exit(1)

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
    def format_matches_table(res, max_matches=MAX_SCHEMA_MATCHES):
        rows = [
            (col1, col2, score)
            for ((_, col1), (_, col2)), score in sorted(
                res.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0.0
        ]

        if not rows:
            return ""

        rows = rows[:max_matches]

        lines = []
        header = f"{'table_1_column':30} | {'table_2_column':30} | score"
        lines.append(header)
        lines.append("-" * len(header))

        for col1, col2, score in rows:
            lines.append(f"{col1:30} | {col2:30} | {score:.4f}")

        omitted = len([score for score in res.values() if score > 0.0]) - len(rows)
        if omitted > 0:
            lines.append(f"... {omitted} lower-scoring matches omitted")

        return "\n".join(lines)

    path_1 = str(_csv_path(csv_dir, file_name_1))
    path_2 = str(_csv_path(csv_dir, file_name_2))

    try:
        df1 = pd.read_csv(path_1, nrows=5000).astype(str)
        df2 = pd.read_csv(path_2, nrows=5000).astype(str)

        matcher = ComaPy(use_instances=True)
        matches = valentine_match(df1, df2, matcher)

        if not matches:
            return "No schema matches found."

        table = format_matches_table(matches)

        if not table:
            return "No schema matches found."

        output = (
            f"Valentine matches between '{file_name_1}' and '{file_name_2}':\n\n"
            f"{table}"
        )

        return _compact_tool_output(output)

    except Exception as e:
        return f"Error Valentine: {e}"





def make_p2_judge_tools(
    candidates: list[str],
    csv_dir: Path,
) -> list:
    """
    Build tools for the Phase 2 *judge-only* agent.
    Does NOT include search_solr — the Solr query is done programmatically
    before the agent runs, and candidates are provided in the prompt.

    Tools: inspect_columns, find_joinable_tables (BLEND on-demand),
           find_schema_matches (Valentine), confirm_table_selection.
    """
    import os
    import uuid
    import shutil
    import blend

    active_csv_dir = Path(csv_dir)

    def inspect_columns_tool(file_name: str) -> str:
        """
        Returns a compact schema for one CSV in the active dataset.
        Shows column names, data types, and sample values for categorical columns.
        Use this to understand what data a table contains.
        """
        return _inspect_columns(active_csv_dir, file_name)

    inspect_columns_tool.__name__ = "inspect_columns"

    def find_joinable_tables(file_name: str, target_columns: list[str]) -> str:
        """
        Use the BLEND engine to find which other tables among the candidates can be joined with the specified file.

        Args:
            file_name: The name of the file to search for joins.
            target_columns: A list of strings representing the specific columns of interest. Do NOT use all columns.

        PAY ATTENTION TO SCORE RULES:
        A low score (0.05 - 0.20) is EXCELLENT and means the tables share a key column.
        Consider valid all files with scores > 0.05.
        """
        file_name = file_name.strip()
        path_file = active_csv_dir / file_name
        if not path_file.exists():
            return f"Error: Target file missing in active dataset: {file_name}"

        if not candidates:
            return "Error: No candidates available."

        tmp_folder = active_csv_dir.parent / f".tmp_blend_{uuid.uuid4().hex}"
        tmp_folder.mkdir(exist_ok=True)
        try:
            for cand in candidates:
                cand_path = active_csv_dir / cand
                if cand_path.exists():
                    os.symlink(cand_path, tmp_folder / cand)

            tmp_db = tmp_folder / "temp_blend.db"
            indexer = blend.BLEND(db_path=tmp_db)
            _blend_load_opts = {"ignore_errors": True, "infer_schema_length": 0, "n_rows": 5000}
            blend.index_tables_seq(indexer, tmp_folder, load_opts=_blend_load_opts, log_stdout=False)

            df_target = pl.read_csv(str(path_file), n_rows=2000, ignore_errors=True)
            valid_cols = [col for col in target_columns if col in df_target.columns]
            if not valid_cols:
                indexer.close()
                return f"Error: None of the specified target_columns {target_columns} exist in {file_name}."

            df_target = df_target.select(valid_cols)
            results = indexer.multi_column_join_search(table=df_target, k=5, clean=True)
            indexer.close()

            if not results:
                return "No compatible table found among the candidates."
            output = f"BLEND Results for '{file_name}' using columns {valid_cols}:\n"
            for t_id, _, score in results:
                if t_id != file_name:
                    output += f"-> {t_id} (Score: {score:.3f})\n"
            return _compact_tool_output(output)
        except Exception as e:
            return f"Error BLEND: {e}"
        finally:
            if tmp_folder.exists():
                shutil.rmtree(tmp_folder, ignore_errors=True)

    def find_schema_matches_tool(file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine to find matching columns between two files based on data content and schema.
        This tool helps identify overlapping columns that can be used for JOIN operations.
        """
        return _find_schema_matches(active_csv_dir, file_name_1, file_name_2)

    find_schema_matches_tool.__name__ = "find_schema_matches"

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

    def reject_selection(reasoning: str, suggestion: str) -> str:
        """
        CRITICAL: Use this tool ONLY when NONE of the candidate tables are relevant to the user's question.
        - reasoning: Explain step-by-step why the current tables are not good.
        - suggestion: Suggest better keywords to search for.
        Calling this tool means you have successfully finished the task by rejecting the candidates.
        """
        return f"REJECT_KEYWORDS: {reasoning}\nSuggestion: {suggestion}"

    return [
        FunctionTool.from_defaults(fn=inspect_columns_tool),
        FunctionTool.from_defaults(fn=find_joinable_tables),
        FunctionTool.from_defaults(fn=find_schema_matches_tool),
        FunctionTool.from_defaults(fn=confirm_table_selection, return_direct=True),
        FunctionTool.from_defaults(fn=reject_selection, return_direct=True),
    ]

