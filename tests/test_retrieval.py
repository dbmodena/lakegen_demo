import json
import math

import pytest

from src.client_solr import LocalSolrClient
from lakegen.retrieval import (
    HybridRetriever,
    KeywordRetriever,
    MissingSignalPolicy,
    RetrievalConfig,
    RetrievalHit,
    RetrievalMode,
    SemanticRetriever,
    SolrEmbeddingIndexer,
    ensure_vector_schema,
    evaluate_ranking,
    min_max_normalize,
    represent_table,
)


class FakeSolr:
    def __init__(self, *, select_docs=(), knn_docs=()):
        self.select_docs = list(select_docs)
        self.knn_docs = list(knn_docs)
        self.select_calls = []
        self.knn_calls = []

    def select(self, tokens, **params):
        self.select_calls.append((list(tokens), params))
        return {"response": {"docs": list(self.select_docs)}}

    def knn_select(self, vector, **params):
        self.knn_calls.append((list(vector), params))
        return {"response": {"docs": list(self.knn_docs)}}


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


class StubBranch:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, value, *, top_k):
        self.calls.append((value, top_k))
        return list(self.hits)


def hit(resource_id, score, rank):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id},
        score=score,
        rank=rank,
    )


def test_min_max_normalization_handles_empty_constant_and_non_finite_scores():
    assert min_max_normalize({}) == {}
    assert min_max_normalize({"a": 4.0, "b": 4.0}) == {"a": 1.0, "b": 1.0}
    assert min_max_normalize({"a": 2.0, "b": 4.0}) == {"a": 0.0, "b": 1.0}
    assert min_max_normalize({"bad": math.nan, "also_bad": math.inf}) == {}


def test_keyword_retrieval_uses_phase1_keywords_and_preserves_default_solr_fields():
    solr = FakeSolr(
        select_docs=[
            {"resource_id": "a", "score": 3.0},
            {"resource_id": "a", "score": 2.0},
            {"resource_id": "bad", "score": "NaN"},
            {"resource_id": "b", "score": 1.0},
        ]
    )

    results = KeywordRetriever(solr).retrieve(
        ["mobilità", "incidenti"], top_k=10, q_op="AND"
    )

    assert [result.document["resource_id"] for result in results] == ["a", "b"]
    assert solr.select_calls == [
        (
            ["mobilità", "incidenti"],
            {"q_op": "AND", "rows": 10, "fl": "*,score"},
        )
    ]
    assert all(result.lexical_score == result.score for result in results)


def test_keyword_field_weights_are_opt_in():
    solr = FakeSolr()
    KeywordRetriever(solr, query_fields="title^3 description tags").retrieve(
        ["acqua"], top_k=5
    )

    assert solr.select_calls[0][1]["qf"] == "title^3 description tags"


def test_semantic_retrieval_embeds_complete_question_and_filters_index_provenance():
    solr = FakeSolr(knn_docs=[{"resource_id": "a", "score": 0.8}])
    embedding = FakeEmbedding()
    config = RetrievalConfig(
        mode=RetrievalMode.SEMANTIC,
        embedding_model="test-multilingual",
        representation_version="metadata-v1",
    )

    results = SemanticRetriever(solr, config, embedding).retrieve(
        "Quali quartieri hanno più incidenti?", top_k=7
    )

    assert embedding.queries == ["Quali quartieri hanno più incidenti?"]
    vector, params = solr.knn_calls[0]
    assert vector == [0.25, 0.75]
    assert params["top_k"] == params["rows"] == 7
    assert 'representation_version:"metadata-v1"' in params["filters"]
    assert 'embedding_model:"test-multilingual"' in params["filters"]
    assert results[0].semantic_score == 0.8


def test_hybrid_expands_each_branch_normalizes_and_fuses_union():
    lexical = StubBranch([hit("a", 10, 1), hit("b", 5, 2), hit("d", 0, 3)])
    semantic = StubBranch([hit("b", 10, 1), hit("c", 5, 2), hit("e", 0, 3)])
    config = RetrievalConfig(mode="hybrid", top_k=10, candidate_multiplier=5)

    results = HybridRetriever(lexical, semantic, config).retrieve(
        "domanda completa", ["parola"], top_k=10
    )

    assert lexical.calls == [(["parola"], 50)]
    assert semantic.calls == [("domanda completa", 50)]
    assert [result.document["resource_id"] for result in results] == [
        "b",
        "a",
        "c",
        "d",
        "e",
    ]
    assert results[0].score == pytest.approx(0.75)
    assert results[1].normalized_semantic_score == 0.0
    assert results[2].normalized_lexical_score == 0.0


def test_hybrid_handles_empty_branch_duplicates_and_non_finite_candidates():
    lexical = StubBranch(
        [
            hit("a", 1, 1),
            hit("a", 9, 2),
            hit("bad", math.inf, 3),
            hit("b", 9, 4),
        ]
    )
    semantic = StubBranch([])
    config = RetrievalConfig(mode="hybrid")

    results = HybridRetriever(lexical, semantic, config).retrieve(
        "question", ["keyword"], top_k=10
    )

    assert {result.document["resource_id"] for result in results} == {"a", "b"}
    assert all(result.score == pytest.approx(0.5) for result in results)
    assert all(result.semantic_score is None for result in results)


def test_hybrid_rescore_policy_requires_and_uses_reliable_resolver():
    lexical = StubBranch([hit("a", 8, 1)])
    semantic = StubBranch([hit("b", 0.9, 1)])
    config = RetrievalConfig(
        mode="hybrid", missing_signal_policy=MissingSignalPolicy.RESCORE
    )

    with pytest.raises(RuntimeError, match="missing_score_resolver"):
        HybridRetriever(lexical, semantic, config).retrieve(
            "question", ["keyword"], top_k=10
        )

    calls = []

    def resolver(branch, document, question, keywords):
        calls.append((branch, document["resource_id"], question, list(keywords)))
        return {("semantic", "a"): 0.1, ("lexical", "b"): 2.0}[branch, document["resource_id"]]

    results = HybridRetriever(
        lexical, semantic, config, missing_score_resolver=resolver
    ).retrieve("question", ["keyword"], top_k=10)

    assert len(calls) == 2
    assert {result.document["resource_id"] for result in results} == {"a", "b"}
    assert all(result.lexical_score is not None for result in results)
    assert all(result.semantic_score is not None for result in results)


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
        }
    )

    assert representation.splitlines() == [
        "Title: Incidenti",
        "Description: Serie comunale",
        "Tags: mobilità | sicurezza",
        "Column names: quartiere | totale",
        "Column descriptions: Nome del quartiere | Numero di incidenti",
    ]
    with pytest.raises(ValueError, match="Unknown table representation"):
        represent_table({}, "future-version")


def test_target_metrics_include_hit_recall_mrr_and_graded_ndcg():
    metrics = evaluate_ranking(
        ["noise", "rel-high", "rel-low", "rel-high"],
        {"rel-high": 3.0, "rel-low": 1.0},
        k_values=(1, 3),
    )

    assert metrics["Hit@1"] == 0.0
    assert metrics["Recall@1"] == 0.0
    assert metrics["Hit@3"] == 1.0
    assert metrics["Recall@3"] == 1.0
    assert metrics["MRR"] == 0.5
    assert 0.0 < metrics["nDCG@3"] < 1.0


class FakeIndexSolr:
    def __init__(self):
        self.schema_commands = []
        self.updates = []
        self.commits = 0

    def schema(self):
        return {"schema": {"uniqueKey": "resource_id", "fields": [], "fieldTypes": []}}

    def iter_documents(self, **params):
        del params
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

    summary = SolrEmbeddingIndexer(solr, config, embedding).run(batch_size=4)

    assert summary.indexed_documents == 1
    assert summary.vector_dimension == 2
    assert embedding.documents == [["Title: Incidenti\nColumn names: anno | totale"]]
    assert solr.updates == [
        [
            {
                "resource_id": "table-1",
                "table_embedding": {"set": [1.0, 1.0]},
                "representation_version": {"set": "metadata-v1"},
                "embedding_model": {"set": "test-multilingual"},
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


def test_solr_knn_query_serializes_finite_vector_and_filters(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": {
                    "docs": [
                        {
                            "resource_id": "a",
                            "columns.name": ["anno"],
                            "score": 0.7,
                        }
                    ]
                }
            }

    def fake_get(url, *, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return Response()

    monkeypatch.setattr("src.client_solr.requests.get", fake_get)
    client = LocalSolrClient("bologna", timeout=4)

    response = client.knn_select(
        [0.25, -0.5],
        vector_field="table_embedding",
        top_k=50,
        rows=10,
        filters=['representation_version:"metadata-v1"'],
    )

    assert captured["url"].endswith("/bologna/select")
    assert captured["params"]["q"] == (
        "{!knn f=table_embedding topK=50}[0.25,-0.5]"
    )
    assert captured["params"]["fq"] == [
        'representation_version:"metadata-v1"'
    ]
    assert response["response"]["docs"][0]["columns"] == [
        {"name": "anno", "description": None}
    ]

    with pytest.raises(ValueError, match="finite"):
        client.knn_select(
            [math.nan], vector_field="table_embedding", top_k=10
        )
