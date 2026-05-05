import sys
import os
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
def inspect_columns(file_name: str) -> str:
    """
    Returns the exact list of column names in a CSV file. 
    If a column is categorical (low cardinality), shows its unique values.
    """
    file_path = os.path.join(CSV_DIR, file_name)
    try:
        df = pd.read_csv(file_path)
        schema_info = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            # Se la colonna è testo/stringa e ha pochi valori univoci (es. meno di 15)
            if dtype == 'object' and df[col].nunique() < 15:
                unique_vals = df[col].dropna().unique().tolist()
                schema_info.append(f"- {col} (Category): {unique_vals}")
            else:
                schema_info.append(f"- {col} ({dtype})")
                
        return f"Schema for {file_name}:\n" + "\n".join(schema_info)
    except Exception as e:
        return f"Error: {str(e)}"

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

def find_schema_matches(file_name_1: str, file_name_2: str) -> str:
    """
    Use Valentine with JaccardDistanceMatcher to find matching columns between two files based on data content and schema.
    This tool helps identify overlapping columns that can be used for JOIN operations.
    """
    path_1 = str(CSV_DIR / file_name_1.strip())
    path_2 = str(CSV_DIR / file_name_2.strip())
    try:
        df1 = pd.read_csv(path_1, nrows=5000).astype(str)
        df2 = pd.read_csv(path_2, nrows=5000).astype(str)
        
        matcher = JaccardDistanceMatcher()
        matches = valentine_match(df1, df2, matcher)
        
        if not matches:
            return "No schema matches found."
            
        output = f"Valentine matches between '{file_name_1}' and '{file_name_2}':\n"
        found_match = False
        for ((_, col1), (_, col2)), score in matches.items():
            if score > 0.0:
                output += f"-> Column '{col1}' matches Column '{col2}' (Score: {score:.3f})\n"
                found_match = True
                
        if not found_match:
            return "No schema matches found."
            
        return output
    except Exception as e:
        return f"Error Valentine: {e}"

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
        FunctionTool.from_defaults(fn=find_schema_matches),
        FunctionTool.from_defaults(fn=confirm_table_selection, return_direct=True),
    ]

