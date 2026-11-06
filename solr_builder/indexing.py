"""Offline indexing of versioned table metadata embeddings in Solr."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from lakegen.config.retrieval import RetrievalConfig
from lakegen.llm.embeddings import EmbeddingModel
from lakegen.retrieval.solr.client import SolrClient
from solr_builder.representation import represent_table


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
    """What an indexing run did.

    Attributes:
        indexed_documents: Documents given a vector.
        skipped_documents: Documents with an empty representation or no unique key.
        vector_dimension: The vectors' dimension; None when nothing was embedded.
        representation_version: The representation embedded.
        embedding_model: The model that embedded it.
        dry_run: Whether nothing was written.
    """

    indexed_documents: int
    skipped_documents: int
    vector_dimension: int | None
    representation_version: str
    embedding_model: str
    dry_run: bool


def validate_stored_replacement_schema(schema: dict[str, Any], *, vector_field: str) -> None:
    """Reject unsafe read-modify-replace indexing from ``fl=*``.

    A full Solr replacement deletes any populated field absent from the JSON
    update.  ``fl=*`` cannot recover indexed-only fields.  Copy-field targets
    are safe because Solr rebuilds them from their source fields, and the
    vector itself is replaced deliberately; every other indexed/non-stored
    field makes this strategy unsafe.

    Args:
        schema: The core's schema.
        vector_field: The vector field, which is replaced on purpose.

    Raises:
        RuntimeError: An indexed field is not stored.
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
        field_type.get("name"): field_type for field_type in schema.get("fieldTypes", [])
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
            raise RuntimeError(f"Solr provenance field {field_name!r} must be indexed and stored")


def ensure_vector_schema(
    solr: SolrClient,
    *,
    vector_field: str,
    dimension: int,
) -> None:
    """Create or validate the Solr cosine vector and provenance fields."""
    schema = solr.schema().get("schema", {})
    fields = {field.get("name"): field for field in schema.get("fields", [])}
    field_types = {
        field_type.get("name"): field_type for field_type in schema.get("fieldTypes", [])
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
            if existing_field.get("indexed") is False or existing_field.get("stored") is False:
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
    """Build versioned embeddings with the same model used at query time.

    Attributes:
        solr: The core's client.
        config: The retrieval configuration (representation, model, vector field).
        embedding_model: The model that embeds the representations.

    Args:
        solr: The core's client.
        config: The retrieval configuration (representation, model, vector field).
        embedding_model: The model that embeds the representations.
    """

    def __init__(
        self,
        solr: SolrClient,
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
        """Embed every document's representation and write the vectors back with their provenance.

        A document whose representation is empty, or that has no unique key,
        is skipped. The vector schema is created (``create_schema``) or
        validated once the first batch tells the dimension.

        Args:
            batch_size: Documents per embedding call and per update.
            create_schema: Create the vector and provenance fields when missing.
            dry_run: Embed and validate, but write nothing.
            source_documents: Complete Solr documents rebuilt from the metadata source;
                None reads the core's own stored documents (``fl=*``).

        Returns:
            The counts, the vector dimension and the provenance.

        Raises:
            ValueError: ``batch_size`` is not positive.
            RuntimeError: The schema has no unique key.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        schema = self.solr.schema().get("schema", {})
        unique_key = schema.get("uniqueKey")
        if not unique_key:
            raise RuntimeError("Solr schema does not define a uniqueKey")

        documents = self._documents(schema, unique_key, batch_size, source_documents)
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
            vectors, dimension = self._embed(prepared, dimension)

            if not schema_checked:
                self._check_schema(schema, dimension, create_schema=create_schema, dry_run=dry_run)
                schema_checked = True

            updates, missing_keys = self._replacements(prepared, vectors, unique_key)
            skipped += missing_keys
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

    def _documents(
        self,
        schema: dict[str, Any],
        unique_key: str,
        batch_size: int,
        source_documents: Iterable[dict[str, Any]] | None,
    ) -> Iterator[dict[str, Any]]:
        """The documents to index: the caller's, else the core's own, after a schema check.

        Args:
            schema: The core's schema.
            unique_key: The schema's unique key, the cursor's sort field.
            batch_size: The indexing batch size; the cursor reads at least 100 at a time.
            source_documents: The caller's documents, or None.

        Returns:
            An iterator over the documents.
        """
        if source_documents is not None:
            # Preferred path: callers reconstruct complete Solr documents from
            # the same canonical metadata source used for initial ingestion.
            return iter(source_documents)
        # DenseVectorField cannot be safely updated via a partial document
        # on schemas with unrecoverable indexed-only fields. Validate the
        # complete schema before reading or writing any document.
        validate_stored_replacement_schema(schema, vector_field=self.config.vector_field)
        return self.solr.iter_documents(
            fields=("*",),
            batch_size=max(batch_size, 100),
            sort_field=unique_key,
            restore_columns=False,
        )

    def _embed(
        self, prepared: list[tuple[dict[str, Any], str]], dimension: int | None
    ) -> tuple[list[list[float]], int]:
        """Embed a batch's representations and check the vectors' dimension.

        Args:
            prepared: (document, representation) pairs.
            dimension: The dimension of the earlier batches; None for the first.

        Returns:
            The vectors, in order, and their dimension.

        Raises:
            RuntimeError: The model returned a different number of vectors, or the
                dimension changed within the batch or since the earlier ones.
        """
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
        return vectors, current_dimension

    def _check_schema(
        self, schema: dict[str, Any], dimension: int, *, create_schema: bool, dry_run: bool
    ) -> None:
        """Create the vector schema when asked to write it, else validate the one there is."""
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

    def _replacements(
        self,
        prepared: list[tuple[dict[str, Any], str]],
        vectors: list[list[float]],
        unique_key: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """The full replacement documents of a batch, each with its vector and provenance.

        Args:
            prepared: (document, representation) pairs.
            vectors: Their vectors, in the same order.
            unique_key: The schema's unique key.

        Returns:
            The replacements, without ``_version_``, and how many documents had no key.
        """
        updates = []
        missing_keys = 0
        for (document, _), vector in zip(prepared, vectors, strict=True):
            key_value = document.get(unique_key)
            if key_value is None:
                missing_keys += 1
                continue
            replacement = {key: value for key, value in document.items() if key != "_version_"}
            replacement[self.config.vector_field] = vector
            replacement["representation_version"] = self.config.representation_version
            replacement["embedding_model"] = self.config.embedding_model
            updates.append(replacement)
        return updates, missing_keys
