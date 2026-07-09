import json
import os
import shutil
import uuid
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field

import polars as pl
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping

from lakegen.core.types import SolrMetadata
from lakegen.agent_tools.tools_p2 import _inspect_columns, _find_schema_matches
from src.client_solr import LocalSolrClient
from lakegen.phases.utils import match_local_csv, solr_metadata_from_doc, format_candidate_context


class ConfirmUnifiedSelectionSchema(BaseModel):
    reasoning: str = Field(description="MANDATORY. Write a brief explanation IN ENGLISH explaining why these specific tables were selected and how they answer the question.")
    tables: list[str] = Field(description="A list of ALL the exact file names needed (e.g., ['sales.csv', 'dates.csv']). Do not omit any table you need!")


class P12State:
    """State tracker for the Phase 1 & 2 unified agent."""
    def __init__(self):
        self.all_candidates: list[str] = []
        self.solr_meta: SolrMetadata = {}
        self.used_keywords: list[str] = []


class Phase12ToolsManager:
    """Manager for Phase 1 & 2 unified tools to avoid closures and improve testability."""
    
    def __init__(
        self,
        state: P12State,
        solr_client: LocalSolrClient,
        all_files: list[str],
        csv_dir: Path,
    ):
        self.state = state
        self.solr_client = solr_client
        self.all_files = all_files
        self.csv_dir = Path(csv_dir)

    def search_solr(self, keywords_str: str) -> str:
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
            self.state.used_keywords = keywords

            solr_response = self.solr_client.select(tokens=keywords, q_op="AND", rows=15)
            docs = solr_response.get("response", {}).get("docs", [])

            candidates: list[str] = []
            for doc in docs:
                matched = match_local_csv(doc, self.all_files)
                if matched is None or matched in candidates:
                    continue
                candidates.append(matched)
                self.state.solr_meta[matched] = solr_metadata_from_doc(doc)
                if matched not in self.state.all_candidates:
                    self.state.all_candidates.append(matched)
                if len(candidates) >= 10:
                    break

            if not candidates:
                return f"Keywords used: {keywords}\nNo tables found. Try with fewer or different keywords."

            return f"Keywords used: {keywords}\n\n" + format_candidate_context(candidates, self.state.solr_meta)
        except Exception as e:
            return f"Error querying Solr: {str(e)}. Try different keywords."

    def inspect_columns(self, file_name: str | None = None, filename: str | None = None) -> str:
        """
        Returns a compact schema for one CSV in the active dataset.
        Shows column names, data types, and sample values for categorical columns.
        Use this SOLO DOPO aver identificato un csv_path valido con search_solr.
        """
        name = file_name or filename
        if not name:
            return "Error: file_name or filename parameter is required."
        return _inspect_columns(self.csv_dir, name)

    def find_joinable_tables(self, file_name: str | None = None, target_columns: list[str] = [], filename: str | None = None) -> str:
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
            return "Error: file_name or filename parameter is required. Please pass a valid file name."
        name = name.strip()
        path_file = self.csv_dir / name
        if not path_file.exists():
            return f"Error: Target file '{name}' missing. You must provide a valid file name that exists in the dataset."

        if not self.state.all_candidates:
            return "Error: No candidates found yet. Please use search_solr first."

        try:
            with tempfile.TemporaryDirectory(dir=self.csv_dir.parent, prefix=".tmp_blend_") as tmp_dir:
                tmp_folder = Path(tmp_dir)
                for cand in self.state.all_candidates:
                    cand_path = self.csv_dir / cand
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
                    return f"Error: None of the specified target_columns {target_columns} exist in '{name}'. Please inspect the columns first."

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
            return f"Error in BLEND tool for '{name}': {str(e)}. This might be due to incompatible data types or memory issues. Try a different table."

    def find_schema_matches(self, file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine to find matching columns between two files based on data content and schema.
        Da usare ESCLUSIVAMENTE nella fase di integrazione, quando possiedi già due schemi estratti.
        """
        return _find_schema_matches(self.csv_dir, file_name_1, file_name_2)

    def confirm_unified_selection(self, reasoning: str, tables: list[str]) -> str:
        """
        CRITICAL: Use this tool ONLY when you have identified the required files after searching solr and inspecting them.
        Calling this tool terminates execution and confirms the selection.
        """
        final_tables = ", ".join(str(t) for t in tables)

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.search_solr),
            FunctionTool.from_defaults(fn=self.inspect_columns),
            FunctionTool.from_defaults(fn=self.find_joinable_tables),
            FunctionTool.from_defaults(fn=self.find_schema_matches),
            FunctionTool.from_defaults(fn=self.confirm_unified_selection, fn_schema=ConfirmUnifiedSelectionSchema, return_direct=True),
        ]


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
    manager = Phase12ToolsManager(state, solr_client, all_files, csv_dir)
    return manager.get_tools()