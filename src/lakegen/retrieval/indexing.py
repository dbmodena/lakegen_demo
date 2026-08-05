"""Offline indexing of versioned table metadata embeddings in Solr."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.client_solr import LocalSolrClient

from lakegen.retrieval.config import RetrievalConfig
from lakegen.retrieval.embeddings import EmbeddingModel
from lakegen.retrieval.representation import represent_table


def _batches(items: Iterator[dict[str, Any]], size: int):
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class IndexingSummary:
    indexed_documents: int
    skipped_documents: int
    vector_dimension: int | None
    representation_version: str
    embedding_model: str
    dry_run: bool


def ensure_vector_schema(
    solr: LocalSolrClient,
    *,
    vector_field: str,
    dimension: int,
) -> None:
    """Create or validate the Solr cosine vector and provenance fields."""
    schema = solr.schema().get("schema", {})
    fields = {field.get("name"): field for field in schema.get("fields", [])}
    field_types = {
        field_type.get("name"): field_type
        for field_type in schema.get("fieldTypes", [])
    }
    vector_type_name = f"lakegen_vector_{dimension}"

    existing_vector = fields.get(vector_field)
    if existing_vector is not None:
        existing_type = field_types.get(existing_vector.get("type"), {})
        actual_dimension = int(existing_type.get("vectorDimension", 0))
        similarity = existing_type.get("similarityFunction")
        if actual_dimension != dimension or similarity != "cosine":
            raise RuntimeError(
                f"Existing {vector_field!r} is incompatible: expected "
                f"dimension={dimension}, similarityFunction='cosine'"
            )
    else:
        if vector_type_name not in field_types:
            solr.update_schema(
                {
                    "add-field-type": {
                        "name": vector_type_name,
                        "class": "solr.DenseVectorField",
                        "vectorDimension": dimension,
                        "similarityFunction": "cosine",
                    }
                }
            )
        solr.update_schema(
            {
                "add-field": {
                    "name": vector_field,
                    "type": vector_type_name,
                    "indexed": True,
                    "stored": True,
                }
            }
        )

    for field_name in ("representation_version", "embedding_model"):
        if field_name not in fields:
            solr.update_schema(
                {
                    "add-field": {
                        "name": field_name,
                        "type": "string",
                        "indexed": True,
                        "stored": True,
                        "multiValued": False,
                    }
                }
            )


class SolrEmbeddingIndexer:
    """Build metadata-v1 embeddings with the same model used at query time."""

    def __init__(
        self,
        solr: LocalSolrClient,
        config: RetrievalConfig,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.solr = solr
        self.config = config
        self.embedding_model = embedding_model

    def run(
        self,
        *,
        batch_size: int = 16,
        create_schema: bool = False,
        dry_run: bool = False,
    ) -> IndexingSummary:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        schema = self.solr.schema().get("schema", {})
        unique_key = schema.get("uniqueKey")
        if not unique_key:
            raise RuntimeError("Solr schema does not define a uniqueKey")

        # DenseVectorField does not support Solr's atomic ``set`` syntax.
        # Fetch every stored field without transforming dotted schema fields,
        # then replace the document while adding vector provenance.
        documents = self.solr.iter_documents(
            fields=("*",),
            batch_size=max(batch_size, 100),
            sort_field=unique_key,
            restore_columns=False,
        )
        indexed = 0
        skipped = 0
        dimension: int | None = None
        schema_checked = False

        for batch in _batches(documents, batch_size):
            prepared = [
                (document, represent_table(document, self.config.representation_version))
                for document in batch
            ]
            prepared = [item for item in prepared if item[1].strip()]
            skipped += len(batch) - len(prepared)
            if not prepared:
                continue
            vectors = self.embedding_model.encode_documents(
                [representation for _, representation in prepared]
            )
            if len(vectors) != len(prepared):
                raise RuntimeError("Embedding model returned an unexpected batch size")
            current_dimensions = {len(vector) for vector in vectors}
            if len(current_dimensions) != 1:
                raise RuntimeError("Embedding dimensions changed within a batch")
            current_dimension = current_dimensions.pop()
            if dimension is not None and dimension != current_dimension:
                raise RuntimeError("Embedding dimensions changed during indexing")
            dimension = current_dimension

            if create_schema and not schema_checked and not dry_run:
                ensure_vector_schema(
                    self.solr,
                    vector_field=self.config.vector_field,
                    dimension=dimension,
                )
                schema_checked = True

            updates = []
            for (document, _), vector in zip(prepared, vectors, strict=True):
                key_value = document.get(unique_key)
                if key_value is None:
                    skipped += 1
                    continue
                replacement = {
                    key: value
                    for key, value in document.items()
                    if key != "_version_"
                }
                replacement[self.config.vector_field] = vector
                replacement["representation_version"] = (
                    self.config.representation_version
                )
                replacement["embedding_model"] = self.config.embedding_model
                updates.append(replacement)
            if updates and not dry_run:
                self.solr.update_documents(updates)
            indexed += len(updates)

        if indexed and not dry_run:
            self.solr.commit()
        return IndexingSummary(
            indexed_documents=indexed,
            skipped_documents=skipped,
            vector_dimension=dimension,
            representation_version=self.config.representation_version,
            embedding_model=self.config.embedding_model,
            dry_run=dry_run,
        )
