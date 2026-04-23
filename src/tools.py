import sys
import os
import json
import pandas as pd
import polars as pl
from pathlib import Path
from llama_index.core.tools import FunctionTool

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
def inspect_columns(file_name: str) -> str:
    """
    Reads a CSV file from the Data Lake and returns the exact list of column names.
    Use this tool to understand what data a table contains before deciding whether or not to use it.
    """
    path = CSV_DIR / file_name.strip()
    if not path.exists(): 
        return f"Error: The file {file_name} does not exist."
    return f"Columns in {file_name}: {list(pd.read_csv(path, nrows=0).columns)}"

def preview_data(file_name: str, n_rows: int = 3) -> str:
    """
    Returns the first 'n_rows' of a CSV file to check data TYPES and FORMATS (e.g., if a date is YYYY-MM-DD).
    CRITICAL RULE: DO NOT use this tool to search for specific rows, names, or values (like 'Chicago' or '2017'). 
    Trust the column names. The Python script will do the filtering later.
    """
    path = CSV_DIR / file_name.strip()
    if not path.exists(): 
        return "Error: File missing."
    return f"Preview of {file_name}:\n{pd.read_csv(path, nrows=n_rows).to_string(index=False)}"

def find_exact_overlaps(file_name_1: str, file_name_2: str) -> str:
    """
    Use the SLOTH engine to find structural overlaps between two files.
    This tool confirms which columns can be used for a pd.merge() by analyzing data content.
    """
    path_1 = str(CSV_DIR / file_name_1.strip())
    path_2 = str(CSV_DIR / file_name_2.strip())
    try:
        df1, df2 = pd.read_csv(path_1, nrows=5000).astype(str), pd.read_csv(path_2, nrows=5000).astype(str)
        r_tab = [df1[col].tolist() for col in df1.columns]
        s_tab = [df2[col].tolist() for col in df2.columns]
        results = sloth(r_tab=r_tab, s_tab=s_tab, min_a=10, min_w=1, max_w=min(len(df1.columns), len(df2.columns)), min_h=5, max_h=min(len(df1), len(df2)), complete=False, verbose=False)
        if not results: 
            return "No exact overlap found."
        return "Exact overlap found!"
    except Exception as e:
        return f"Error SLOTH: {e}"

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


def make_agent_tools(blend_db_path: Path) -> list:
    """
    Builds the list of LlamaIndex FunctionTools for the Data Architect agent.
    The `find_joinable_tables` closure is bound to the pre-built BLEND index at
    *blend_db_path*, which is constructed once in `select_tables` over the
    fuzzy-matched top-10 candidate files.
    """
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
        path_file = CSV_DIR / file_name
        if not path_file.exists():
            return "Error: Target file missing."
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
            return output
        except Exception as e:
            return f"Error BLEND: {e}"

    return [
        FunctionTool.from_defaults(fn=inspect_columns),
        FunctionTool.from_defaults(fn=preview_data),
        FunctionTool.from_defaults(fn=find_joinable_tables),
        FunctionTool.from_defaults(fn=find_exact_overlaps),
        FunctionTool.from_defaults(fn=confirm_table_selection, return_direct=True),
    ]

