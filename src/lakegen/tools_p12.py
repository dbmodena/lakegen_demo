import json
import os
import shutil
import uuid
from pathlib import Path

import polars as pl
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping

from lakegen.types import SolrMetadata
from lakegen.tools import _inspect_columns, _find_schema_matches
from src.client_solr import LocalSolrClient
from lakegen.phases.utils import match_local_csv, solr_metadata_from_doc, format_candidate_context


class P12State:
    """State tracker for the Phase 1 & 2 unified agent."""
    def __init__(self):
        self.all_candidates: list[str] = []
        self.solr_meta: SolrMetadata = {}
        self.used_keywords: list[str] = []


def make_p12_tools(
    state: P12State,
    solr_client: LocalSolrClient,
    all_files: list[str],
    csv_dir: Path,
):
    """
    Build the tools for the unified Phase 1 & 2 agent and return an ObjectRetriever.
    The retriever will dynamically fetch the top relevant tools based on the agent's intent.
    """

    def search_solr(keywords_str: str) -> str:
        """
        Search for relevant tables in Solr using a space-separated string of keywords.
        ATTENZIONE: Le keyword devono TASSATIVAMENTE essere mantenute nella lingua nativa 
        del portale Open Data che si sta interrogando (es. francese per Parigi, italiano per Bologna).
        Because this uses AND logic, use ONLY 2-3 essential keywords at most to avoid getting zero results.
        Example: "sales 2024" (or equivalent in target language).
        Returns the top matching table names and their schema descriptions.
        """
        try:
            keywords = [k.strip() for k in keywords_str.split(" ") if k.strip()]
            state.used_keywords = keywords

            solr_response = solr_client.select(tokens=keywords, q_op="AND", rows=15)
            docs = solr_response.get("response", {}).get("docs", [])

            candidates: list[str] = []
            for doc in docs:
                matched = match_local_csv(doc, all_files)
                if matched is None or matched in candidates:
                    continue
                candidates.append(matched)
                state.solr_meta[matched] = solr_metadata_from_doc(doc)
                if matched not in state.all_candidates:
                    state.all_candidates.append(matched)
                if len(candidates) >= 10:
                    break

            if not candidates:
                return f"Keywords used: {keywords}\nNo tables found. Try with fewer or different keywords."

            return f"Keywords used: {keywords}\n\n" + format_candidate_context(candidates, state.solr_meta)
        except Exception as e:
            return f"Error querying Solr: {str(e)}"

    def inspect_columns_tool(file_name: str | None = None, filename: str | None = None) -> str:
        """
        Returns a compact schema for one CSV in the active dataset.
        Shows column names, data types, and sample values for categorical columns.
        Use this SOLO DOPO aver identificato un csv_path valido con search_solr.
        """
        name = file_name or filename
        if not name:
            return "Error: file_name or filename parameter is required."
        return _inspect_columns(csv_dir, name)

    inspect_columns_tool.__name__ = "inspect_columns"

    def find_joinable_tables(file_name: str | None = None, target_columns: list[str] = [], filename: str | None = None) -> str:
        """
        Use the BLEND engine to find which other tables among the discovered candidates can be joined with the specified file.
        DA USARE NELLA FASE DI INTEGRAZIONE, quando devi incrociare i dati di una tabella già ispezionata.
        Args:
            file_name: The name of the file to search for joins.
            target_columns: A list of strings representing the specific columns of interest. Do NOT use all columns.
        PAY ATTENTION TO SCORE RULES:
        A low score (0.05 - 0.20) is EXCELLENT and means the tables share a key column.
        Consider valid all files with scores > 0.05.
        """
        import blend

        name = file_name or filename
        if not name:
            return "Error: file_name or filename parameter is required."
        name = name.strip()
        path_file = csv_dir / name
        if not path_file.exists():
            return f"Error: Target file missing in active dataset: {name}"

        if not state.all_candidates:
            return "Error: No candidates found yet. Please use search_solr first."

        tmp_folder = csv_dir.parent / f".tmp_blend_{uuid.uuid4().hex}"
        tmp_folder.mkdir(exist_ok=True)
        try:
            for cand in state.all_candidates:
                cand_path = csv_dir / cand
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
                return f"Error: None of the specified target_columns {target_columns} exist in {name}."

            df_target = df_target.select(valid_cols)
            results = indexer.multi_column_join_search(table=df_target, k=5, clean=True)
            indexer.close()

            if not results:
                return "No compatible table found among the candidates."
            output = f"BLEND Results for '{name}' using columns {valid_cols}:\n"
            for t_id, _, score in results:
                if t_id != name:
                    output += f"-> {t_id} (Score: {score:.3f})\n"
            return output
        except Exception as e:
            return f"Error BLEND: {e}"
        finally:
            if tmp_folder.exists():
                shutil.rmtree(tmp_folder, ignore_errors=True)

    def find_schema_matches_tool(file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine to find matching columns between two files based on data content and schema.
        Da usare ESCLUSIVAMENTE nella fase di integrazione, quando possiedi già due schemi estratti.
        """
        return _find_schema_matches(csv_dir, file_name_1, file_name_2)

    find_schema_matches_tool.__name__ = "find_schema_matches"

    def confirm_unified_selection(selected_tables: str, reasoning: str) -> str:
        """
        CRITICAL: Use this tool ONLY when you have identified the required files after searching solr and inspecting them.
        - selected_tables: A comma-separated string of ALL the exact file names needed (e.g., "sales.csv", or "sales.csv, dates.csv", or "sales.csv, dates.csv, lookup.csv"). Do not omit any table you need!
        - reasoning: Write a brief explanation IN ENGLISH.
        Calling this tool means you have successfully finished the task.
        """
        dati_uscita = {
            "tables": selected_tables,
            "reasoning": reasoning
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    # 1. Define the list of tools
    tools_list = [
        FunctionTool.from_defaults(fn=search_solr),
        FunctionTool.from_defaults(fn=inspect_columns_tool),
        FunctionTool.from_defaults(fn=find_joinable_tables),
        FunctionTool.from_defaults(fn=find_schema_matches_tool),
        FunctionTool.from_defaults(fn=confirm_unified_selection, return_direct=True),
    ]

    # 2. Define the tool mapping
    tool_mapping = SimpleToolNodeMapping.from_objects(tools_list)
    
    # 3. Create the vector index (this will create the embeddings of the docstrings in background)
    obj_index = ObjectIndex.from_objects(
        tools_list,
        tool_mapping,
        VectorStoreIndex,
    )

    # 4. Return the retriever
    # similarity_top_k=3 ensures that the LLM sees at most the 3 tools most relevant to the action it wants to perform.
    # Always include confirm_unified_selection so that it can stop when it has finished.
    return obj_index.as_retriever(similarity_top_k=3)