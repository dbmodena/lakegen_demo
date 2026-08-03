import json
from pathlib import Path
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping

from lakegen.core.types import SolrMetadata
from lakegen.agent_tools.tools_p2 import _inspect_columns, _find_schema_matches
from src.client_solr import LocalSolrClient
from lakegen.phases.utils import match_local_csv, solr_metadata_from_doc, format_candidate_context


class ConfirmUnifiedSelectionSchema(BaseModel):
    reasoning: str = Field(description="MANDATORY. Write a brief explanation IN ENGLISH explaining why these specific tables were selected and how they answer the question.")
    tables: list[str] = Field(description="A list of ALL the exact file names needed (e.g., ['sales.parquet', 'dates.parquet']). Do not omit any table you need!")


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
        IMPORTANT: Keywords must be written in the native language of the Open Data
        portal being queried (for example, French for Paris or Italian for Bologna).
        Search for DATASET CONCEPTS that appear in metadata, not concrete row-filter values.
        Use ONLY 1-2 essential keywords. The tool tries strict AND first and automatically
        falls back to broader OR matching when a multi-keyword AND search returns no results.
        Example: use "language interpretation", not row values such as "Mandarin Intake".
        Returns the top matching table names and their schema descriptions.
        """
        try:
            keywords = [k.strip() for k in keywords_str.split(" ") if k.strip()]
            if not keywords:
                return "No keywords provided. Search with one or two dataset concepts."
            self.state.used_keywords = keywords

            solr_response = self.solr_client.select(tokens=keywords, q_op="AND", rows=15)
            docs = solr_response.get("response", {}).get("docs", [])
            search_mode = "AND"

            if not docs and len(keywords) > 1:
                solr_response = self.solr_client.select(tokens=keywords, q_op="OR", rows=15)
                docs = solr_response.get("response", {}).get("docs", [])
                search_mode = "OR fallback"

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

            return (
                f"Keywords used: {keywords}\n"
                f"Search mode: {search_mode}\n\n"
                + format_candidate_context(candidates, self.state.solr_meta)
            )
        except Exception as e:
            return f"Error querying Solr: {str(e)}. Try different keywords."

    def inspect_columns(self, file_name: str | None = None, filename: str | None = None) -> str:
        """
        Returns a compact profile for one table in the active dataset.
        Shows row count, full-file min/max coverage for temporal columns, column
        types, and sample values for low-cardinality categorical columns.
        If the question has a date or time range, compare it with the reported
        temporal coverage before selecting the table.
        Use this only after identifying a valid table file with search_solr.
        """
        name = file_name or filename
        if not name:
            return "Error: file_name or filename parameter is required."
        return _inspect_columns(self.csv_dir, name)

    def find_schema_matches(self, file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine to identify matching columns, then verify their practical
        joinability through overlap, coverage, uniqueness, cardinality, and
        estimated join expansion.
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
