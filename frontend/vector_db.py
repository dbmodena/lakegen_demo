# -*- coding: utf-8 -*-
"""
A client for interacting with a Milvus vector database, supporting hybrid search
with local or remote embedding generation.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Final

import pymilvus
from dotenv import load_dotenv
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from together import Together

# --- Environment & Constants ---
# Load environment variables from a .env file
load_dotenv()

EMBEDDING_TYPE_LOCAL: Final = "local"
EMBEDDING_TYPE_TOGETHER: Final = "together"

LOCAL_MODEL: Final = "BAAI/bge-m3"
TOGETHER_MODEL: Final = "BAAI/bge-large-en-v1.5"
TOGETHER_EMBEDDING_DIM: Final = 1024

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MilvusDB:
    """
    Manages connections, schema, and hybrid search operations for a Milvus collection.

    This class encapsulates the logic for creating and managing a Milvus
    collection with dense and sparse vectors, supporting both local BGE-M3
    and remote TogetherAI embeddings.

    Attributes:
        client (MilvusClient): The Milvus client instance.
        collection_name (str): The name of the Milvus collection.
    """

    def __init__(
        self,
        db_path: str = "./milvus_embeddings.db",
        collection_name: str = "packages",
        embedding_type: str = EMBEDDING_TYPE_TOGETHER,
        together_api_key: str | None = None,
        device: str = "cuda:0",
    ):
        """
        Initializes the MilvusDB instance and embedding functions.

        Args:
            db_path: Path to the local Milvus database file.
            collection_name: Name of the collection to manage.
            embedding_type: Type of embedding ('local' or 'together').
            together_api_key: Optional API key for TogetherAI. If not provided,
                              it's read from the TOGETHER_API_KEY environment variable.
            device: Device for local embeddings ('cpu' or 'cuda:0').

        Raises:
            ValueError: If 'together' is chosen without an API key being available,
                        or if an invalid embedding_type is provided.
        """
        self.client = MilvusClient(uri=db_path)
        self.collection_name = collection_name
        self._embedding_type = embedding_type
        self._embedding_dim: int
        self._ef: BGEM3EmbeddingFunction | None = None
        self._together_client: Together | None = None

        if self._embedding_type == EMBEDDING_TYPE_LOCAL:
            self._ef = BGEM3EmbeddingFunction(
                model_name=LOCAL_MODEL, device=device, use_fp16=(device != "cpu")
            )
            self._embedding_dim = self._ef.dim["dense"]
        elif self._embedding_type == EMBEDDING_TYPE_TOGETHER:
            # Prioritize the direct argument, otherwise use the environment variable.
            api_key = together_api_key or os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError(
                    "For 'together' embedding type, pass the 'together_api_key' "
                    "argument or set TOGETHER_API_KEY in your .env file."
                )
            self._together_client = Together(api_key=api_key)
            self._embedding_dim = TOGETHER_EMBEDDING_DIM
        else:
            raise ValueError(
                f"Invalid embedding_type: '{self._embedding_type}'. "
                f"Choose from '{EMBEDDING_TYPE_LOCAL}' or '{EMBEDDING_TYPE_TOGETHER}'."
            )
        logger.info(
            "MilvusDB initialized for collection '%s' with '%s' embeddings (dim: %d).",
            self.collection_name,
            self._embedding_type,
            self._embedding_dim,
        )

    # ... all other methods remain the same ...

    def create_collection(self, overwrite: bool = True):
        """
        Creates the Milvus collection with the defined schema and indexes.

        Internal helper methods `_define_schema` and `_create_index_params` are
        used to configure the collection.

        Args:
            overwrite: If True, drops the collection if it already exists before creating.
        """
        if overwrite and self.collection_name in self.client.list_collections():
            self.client.drop_collection(self.collection_name)
            logger.info("Dropped existing collection: '%s'", self.collection_name)

        schema = self._define_schema()
        index_params = self._create_index_params()

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Successfully created collection: '%s'", self.collection_name)

    def insert_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Embeds and inserts a list of documents into the collection.

        Each document is a dictionary expected to have 'id', 'title', and 'notes'.
        Documents that fail to generate an embedding are skipped.

        Args:
            documents: A list of document dictionaries to insert.

        Returns:
            The results from the Milvus insert operation.
        """
        docs_to_insert = []
        for doc in documents:
            text_to_embed = (
                f"{doc.get('title', '')} {doc.get('notes', '')[:1024]}".strip()
            )
            if not text_to_embed:
                logger.warning(
                    "Skipping document with id '%s' due to empty content.", doc.get("id")
                )
                continue

            embedding = self._get_embedding(text_to_embed, item_id=doc.get("id"))
            if embedding:
                docs_to_insert.append(
                    {
                        "id": doc["id"],
                        "text": text_to_embed,
                        "text_dense": embedding,
                    }
                )

        if not docs_to_insert:
            logger.warning("No valid documents to insert.")
            return {}

        results = self.client.insert(
            collection_name=self.collection_name, data=docs_to_insert
        )
        logger.info(
            "Inserted %d documents into '%s'.",
            len(docs_to_insert),
            self.collection_name,
        )
        return results

    def hybrid_search(
        self,
        query: str,
        limit: int = 4,
        rerank_k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Performs a hybrid search using dense and sparse vectors with RRF ranking.

        Args:
            query: The search query string.
            limit: The final number of results to return after reranking.
            rerank_k: The reranking factor for the RRFRanker.

        Returns:
            A list of search result dictionaries, ranked by the RRFRanker.
        """
        dense_vector = self._get_embedding(query, "search_query")
        if not dense_vector:
            logger.error("Failed to generate embedding for the search query.")
            return []

        # 1. Dense Search Request
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="text_dense",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit,
        )

        # 2. Sparse Search Request (BM25)
        # The query string itself is used for the sparse search.
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="text_sparse",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=limit,
        )

        # 3. Rerank and Execute
        reranker = RRFRanker(k=rerank_k)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=reranker,
            limit=limit,
        )
        # The result of a single query is the first element in the returned list.
        return results[0] if results else []

    def _define_schema(self) -> pymilvus.CollectionSchema:
        """
        Defines and returns the collection schema.

        The schema includes fields for a primary key, the original text, a dense vector,
        and a sparse vector. It also integrates a BM25 function to auto-generate
        the sparse vector from the text field.

        Returns:
            The configured `CollectionSchema` object.
        """
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=1000, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=2048, enable_analyzer=True)
        schema.add_field("text_dense", DataType.FLOAT_VECTOR, dim=self._embedding_dim)
        schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)

        # Automatically generate sparse embeddings from 'text' field using BM25
        bm25_function = Function(
            "text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["text_sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)
        return schema

    def _create_index_params(self) -> pymilvus.IndexParams:
        """
        Defines and returns the index parameters for dense and sparse fields.

        Returns:
            The configured `IndexParams` object.
        """
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="text_dense",
            index_name="text_dense_idx",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="text_sparse",
            index_name="text_sparse_idx",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        return index_params

    def _get_embedding(
        self, text: str, item_id: str | None = None
    ) -> list[float] | None:
        """
        Dispatches to the appropriate embedding function based on initialization.

        This is an internal method that centralizes embedding generation.

        Args:
            text: The input text to embed.
            item_id: An optional identifier for logging purposes.

        Returns:
            The generated embedding as a list of floats, or None on failure.
        """
        if self._embedding_type == EMBEDDING_TYPE_LOCAL:
            return self._bge_local_embedding(text)
        return self._together_api_embedding(text, item_id)

    def _bge_local_embedding(self, text: str) -> list[float] | None:
        """Generates an embedding using the local BGE-M3 model."""
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            logger.warning("Skipping embedding for empty or invalid text.")
            return None
        try:
            # The model expects a list of documents and returns a dict.
            embedding = self._ef.encode_documents([cleaned_text])
            return embedding["dense"][0]
        except Exception:
            logger.error(
                "Failed to generate local BGE embedding for text: '%s...'",
                cleaned_text[:100],
                exc_info=True,
            )
            return None

    def _together_api_embedding(
        self, text: str, item_id: str | None
    ) -> list[float] | None:
        """Generates an embedding using the TogetherAI API."""
        cleaned_text = self._clean_text(text)
        if not cleaned_text or not self._together_client:
            logger.warning("Skipping TogetherAI embedding for empty or invalid text.")
            return None
        try:
            response = self._together_client.embeddings.create(
                model=TOGETHER_MODEL, input=cleaned_text
            )
            return response.data[0].embedding
        except Exception:
            logger.error(
                "Failed to generate TogetherAI embedding for item '%s'.",
                item_id or "Unknown",
                exc_info=True,
            )
            return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Removes problematic characters from text before embedding."""
        if not isinstance(text, str):
            return ""
        # Remove ASCII control characters
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text).strip()

    def get_collection_stats(self) -> dict[str, Any]:
        """Retrieves statistics about the collection, such as row count."""
        try:
            return self.client.get_collection_stats(
                collection_name=self.collection_name
            )
        except Exception:
            logger.error(
                "Failed to retrieve stats for collection '%s'.",
                self.collection_name,
                exc_info=True,
            )
            return {}

    def close(self):
        """Closes the Milvus client connection."""
        self.client.close()
        logger.info("Milvus client connection closed.")