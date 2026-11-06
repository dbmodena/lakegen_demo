import json

import pytest

from lakegen.config.retrieval import RetrievalConfig
from solr_builder.index import validate_source_coverage
from solr_builder.indexing import (
    SolrEmbeddingIndexer,
    ensure_vector_schema,
    validate_stored_replacement_schema,
)
from solr_builder.representation import represent_table


class FakeEmbedding:
    model_name = "test-multilingual"

    def __init__(self):
        self.queries = []
        self.documents = []

    def encode_query(self, text):
        self.queries.append(text)
        return [0.25, 0.75]

    def encode_documents(self, texts):
        self.documents.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts, 1)]


def test_metadata_v1_is_stable_and_includes_requested_table_metadata():
    representation = represent_table(
        {
            "title": "Incidenti",
            "description": "Serie comunale",
            "tags": ["mobilità", "sicurezza"],
            "columns": [
                {"name": "quartiere", "description": "Nome del quartiere"},
                {"name": "totale", "description": "Numero di incidenti"},
            ],
        },
        "metadata-v1",
    )

    assert representation.splitlines() == [
        "Represent this table metadata for information retrieval.",
        "Title: Incidenti",
        "Description: Serie comunale",
        "Tags: mobilità | sicurezza",
        "Column names: quartiere | totale",
        "Column descriptions: Nome del quartiere | Numero di incidenti",
    ]
    with pytest.raises(ValueError, match="Unknown table representation"):
        represent_table({}, "future-version")


class FakeIndexSolr:
    def __init__(self):
        self.schema_commands = []
        self.updates = []
        self.commits = 0

    def schema(self):
        return {"schema": {"uniqueKey": "resource_id", "fields": [], "fieldTypes": []}}

    def iter_documents(self, **params):
        assert params["fields"] == ("*",)
        assert params["restore_columns"] is False
        yield {
            "resource_id": "table-1",
            "title": "Incidenti",
            "columns.name": ["anno", "totale"],
        }

    def update_schema(self, command):
        self.schema_commands.append(command)
        return {}

    def update_documents(self, documents):
        self.updates.append(documents)
        return {}

    def commit(self):
        self.commits += 1
        return {}


def test_embedding_indexer_stores_vector_and_representation_provenance():
    solr = FakeIndexSolr()
    embedding = FakeEmbedding()
    config = RetrievalConfig(embedding_model="test-multilingual")

    summary = SolrEmbeddingIndexer(solr, config, embedding).run(batch_size=4, create_schema=True)

    assert summary.indexed_documents == 1
    assert summary.vector_dimension == 2
    assert embedding.documents == [
        [
            "Represent this table metadata for information retrieval.\n"
            "Title: Incidenti\nColumn names: anno | totale"
        ]
    ]
    assert solr.updates == [
        [
            {
                "resource_id": "table-1",
                "title": "Incidenti",
                "columns.name": ["anno", "totale"],
                "table_embedding": [1.0, 1.0],
                "representation_version": "metadata-v1",
                "embedding_model": "test-multilingual",
            }
        ]
    ]
    assert solr.commits == 1


def test_vector_schema_creation_uses_cosine_and_provenance_fields():
    solr = FakeIndexSolr()

    ensure_vector_schema(solr, vector_field="table_embedding", dimension=1024)

    serialized = json.dumps(solr.schema_commands)
    assert "solr.DenseVectorField" in serialized
    assert '"vectorDimension": 1024' in serialized
    assert '"similarityFunction": "cosine"' in serialized
    assert '"name": "representation_version"' in serialized
    assert '"name": "embedding_model"' in serialized


def test_indexing_dry_run_validates_existing_vector_schema_without_writes():
    class IncompatibleVectorSolr(FakeIndexSolr):
        def schema(self):
            return {
                "schema": {
                    "uniqueKey": "resource_id",
                    "fields": [
                        {
                            "name": "table_embedding",
                            "type": "wrong_vector",
                            "indexed": True,
                            "stored": True,
                        },
                        {
                            "name": "representation_version",
                            "indexed": True,
                            "stored": True,
                        },
                        {
                            "name": "embedding_model",
                            "indexed": True,
                            "stored": True,
                        },
                    ],
                    "fieldTypes": [
                        {
                            "name": "wrong_vector",
                            "vectorDimension": 3,
                            "similarityFunction": "cosine",
                        }
                    ],
                }
            }

    solr = IncompatibleVectorSolr()
    with pytest.raises(RuntimeError, match="dimension=2"):
        SolrEmbeddingIndexer(
            solr,
            RetrievalConfig(embedding_model="test-multilingual"),
            FakeEmbedding(),
        ).run(create_schema=True, dry_run=True)

    assert solr.schema_commands == []
    assert solr.updates == []
    assert solr.commits == 0


def test_indexer_aborts_before_reading_when_indexed_fields_are_not_stored():
    schema = {
        "uniqueKey": "resource_id",
        "fields": [
            {"name": "resource_id", "indexed": True, "stored": True},
            {"name": "private_sort", "indexed": True, "stored": False},
        ],
        "fieldTypes": [],
    }

    with pytest.raises(RuntimeError, match="private_sort"):
        validate_stored_replacement_schema(schema, vector_field="table_embedding")


def test_source_reindex_requires_all_ids_and_preserves_every_stored_field():
    current = [
        {
            "resource_id": "table-1",
            "title": "Incidenti",
            "stored_provenance": "portal-a",
            "table_embedding": [0.1, 0.2],
            "representation_version": "metadata-v1",
        }
    ]
    incomplete_source = [{"resource_id": "table-1", "title": "Incidenti"}]

    with pytest.raises(RuntimeError, match="stored_provenance"):
        validate_source_coverage(
            incomplete_source,
            current,
            unique_key="resource_id",
            vector_field="table_embedding",
        )

    validate_source_coverage(
        [{**incomplete_source[0], "stored_provenance": "portal-a"}],
        current,
        unique_key="resource_id",
        vector_field="table_embedding",
    )
