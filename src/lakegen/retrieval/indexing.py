"""Offline indexing of versioned table metadata embeddings in Solr."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
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


def validate_stored_replacement_schema(
    schema: dict[str, Any], *, vector_field: str
) -> None:
    """Reject unsafe read-modify-replace indexing from ``fl=*``.

    A full Solr replacement deletes any populated field absent from the JSON
    update.  ``fl=*`` cannot recover indexed-only fields.  Copy-field targets
    are safe because Solr rebuilds them from their source fields, and the
    vector itself is replaced deliberately; every other indexed/non-stored
    field makes this strategy unsafe.
    """
    copy_destinations = {
        str(item.get("dest"))
        for item in schema.get("copyFields", [])
        if isinstance(item, dict) and item.get("dest")
    }
    unsafe_fields = []
    for field in (*schema.get("fields", []), *schema.get("dynamicFields", [])):
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "")
        if not name or name == vector_field or name in copy_destinations:
            continue
        if field.get("indexed") is True and field.get("stored") is not True:
            unsafe_fields.append(name)
    if unsafe_fields:
        names = ", ".join(sorted(set(unsafe_fields)))
        raise RuntimeError(
            "Unsafe full-document replacement: these indexed fields are not "
            f"stored and cannot be recovered through fl=*: {names}. Rebuild "
            "documents from the original Solr-ready metadata source instead."
        )


def validate_vector_schema(
    schema: dict[str, Any],
    *,
    vector_field: str,
    dimension: int,
    allow_missing: bool = False,
) -> None:
    """Validate vector/provenance fields without changing the Solr schema."""
    fields = {field.get("name"): field for field in schema.get("fields", [])}
    field_types = {
        field_type.get("name"): field_type
        for field_type in schema.get("fieldTypes", [])
    }
    existing_vector = fields.get(vector_field)
    if existing_vector is None:
        if allow_missing:
            return
        raise RuntimeError(
            f"Solr field {vector_field!r} is missing; run with create_schema/"
            "--ensure-schema after a successful dry-run"
        )
    existing_type = field_types.get(existing_vector.get("type"), {})
    actual_dimension = int(existing_type.get("vectorDimension", 0))
    similarity = existing_type.get("similarityFunction")
    if actual_dimension != dimension or similarity != "cosine":
        raise RuntimeError(
            f"Existing {vector_field!r} is incompatible: expected "
            f"dimension={dimension}, similarityFunction='cosine'"
        )
    for field_name in ("representation_version", "embedding_model"):
        field = fields.get(field_name)
        if field is None:
            if allow_missing:
                continue
            raise RuntimeError(f"Solr provenance field {field_name!r} is missing")
        if field.get("indexed") is False or field.get("stored") is False:
            raise RuntimeError(
                f"Solr provenance field {field_name!r} must be indexed and stored"
            )


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
        existing_field = fields.get(field_name)
        if existing_field is not None:
            if (
                existing_field.get("indexed") is False
                or existing_field.get("stored") is False
            ):
                raise RuntimeError(
                    f"Existing provenance field {field_name!r} must be indexed and stored"
                )
        else:
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
    """Build versioned embeddings with the same model used at query time."""

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
        source_documents: Iterable[dict[str, Any]] | None = None,
    ) -> IndexingSummary:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        schema = self.solr.schema().get("schema", {})
        unique_key = schema.get("uniqueKey")
        if not unique_key:
            raise RuntimeError("Solr schema does not define a uniqueKey")

        if source_documents is None:
            # DenseVectorField cannot be safely updated via a partial document
            # on schemas with unrecoverable indexed-only fields. Validate the
            # complete schema before reading or writing any document.
            validate_stored_replacement_schema(
                schema, vector_field=self.config.vector_field
            )
            documents = self.solr.iter_documents(
                fields=("*",),
                batch_size=max(batch_size, 100),
                sort_field=unique_key,
                restore_columns=False,
            )
        else:
            # Preferred path: callers reconstruct complete Solr documents from
            # the same canonical metadata source used for initial ingestion.
            documents = iter(source_documents)
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

            if not schema_checked:
                if create_schema and not dry_run:
                    ensure_vector_schema(
                        self.solr,
                        vector_field=self.config.vector_field,
                        dimension=dimension,
                    )
                else:
                    validate_vector_schema(
                        schema,
                        vector_field=self.config.vector_field,
                        dimension=dimension,
                        allow_missing=create_schema and dry_run,
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
