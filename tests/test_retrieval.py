import json
import math
import csv

import pytest

from index_retrieval import validate_source_coverage
from src.client_solr import LocalSolrClient
from lakegen.retrieval import (
    HybridRetriever,
    FusionMethod,
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
    validate_stored_replacement_schema,
)
from lakegen.retrieval.benchmark import (
    BenchmarkCase,
    append_benchmark_metrics_log,
    run_retriever_benchmark,
)
from lakegen.retrieval.embeddings import (
    EmbeddingGenerationError,
    OllamaMultilingualEmbedding,
    _validate,
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


def test_embedding_validation_rejects_zero_norm_vector():
    with pytest.raises(ValueError, match="zero-norm"):
        _validate([0.0, 0.0])


def test_query_embedding_retries_transient_provider_failure():
    class FlakyModel:
        def __init__(self):
            self.calls = 0

        def get_query_embedding(self, _text):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary NaN response")
            return [0.25, 0.75]

    embedding = OllamaMultilingualEmbedding.__new__(OllamaMultilingualEmbedding)
    embedding._model = FlakyModel()

    assert embedding.encode_query("valid question") == [0.25, 0.75]
    assert embedding._model.calls == 3


def test_query_embedding_reports_provider_failure_after_three_attempts():
    class BrokenModel:
        def __init__(self):
            self.calls = 0

        def get_query_embedding(self, _text):
            self.calls += 1
            raise RuntimeError("unsupported value: NaN")

    embedding = OllamaMultilingualEmbedding.__new__(OllamaMultilingualEmbedding)
    embedding._model = BrokenModel()

    with pytest.raises(EmbeddingGenerationError) as error:
        embedding.encode_query("valid question")
    assert embedding._model.calls == 3
    assert "after 3 attempts" in str(error.value)


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


def test_hybrid_rrf_is_a_configurable_rank_fusion_baseline():
    lexical = StubBranch([hit("a", 100, 1), hit("b", 1, 2)])
    semantic = StubBranch([hit("b", 0.99, 1), hit("c", 0.98, 2)])
    config = RetrievalConfig(
        mode="hybrid", fusion_method=FusionMethod.RRF, rrf_k=60
    )

    results = HybridRetriever(lexical, semantic, config).retrieve(
        "question", ["keyword"], top_k=3
    )

    assert [item.document["resource_id"] for item in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)


@pytest.mark.parametrize(
    ("alpha", "expected_ids", "expected_ranks", "expected_branch"),
    [
        (1.0, ["bm25-a", "bm25-b"], [1, 2], "lexical"),
        (0.0, ["dense-a", "dense-b"], [1, 2], "semantic"),
    ],
)
def test_weighted_hybrid_endpoints_exactly_match_the_selected_branch(
    alpha, expected_ids, expected_ranks, expected_branch
):
    lexical = StubBranch(
        [hit("bm25-a", 10.0, 1), hit("bm25-b", 9.0, 2), hit("bm25-c", 8.0, 3)]
    )
    semantic = StubBranch(
        [hit("dense-a", 0.9, 1), hit("dense-b", 0.8, 2), hit("dense-c", 0.7, 3)]
    )
    config = RetrievalConfig(
        mode="hybrid",
        alpha=alpha,
        fusion_method=FusionMethod.WEIGHTED,
        candidate_multiplier=4,
    )

    results = HybridRetriever(lexical, semantic, config).retrieve(
        "complete question", ["generated", "keywords"], top_k=2
    )

    assert [item.document["resource_id"] for item in results] == expected_ids
    assert [item.rank for item in results] == expected_ranks
    assert len(results) == 2
    if expected_branch == "lexical":
        assert lexical.calls == [(["generated", "keywords"], 2)]
        assert semantic.calls == []
    else:
        assert semantic.calls == [("complete question", 2)]
        assert lexical.calls == []


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


def test_retriever_only_benchmark_runs_once_per_case_without_pipeline_stages():
    solr = FakeSolr(
        select_docs=[
            {"resource_id": "gold", "title": "Gold", "score": 2.0},
            {"resource_id": "other", "title": "Other", "score": 1.0},
        ]
    )
    cases = [
        BenchmarkCase("q1", "Question one?", ("gold",), ("gold",)),
        BenchmarkCase("q2", "Question two?", ("other",), ("gold",)),
    ]

    report = run_retriever_benchmark(
        solr,
        cases,
        base_config=RetrievalConfig(top_k=10),
        modes=(RetrievalMode.KEYWORD,),
    )

    assert len(solr.select_calls) == 2
    keyword = report["experiments"]["keyword"]
    assert keyword["mean_metrics"]["Hit@1"] == 1.0
    assert keyword["cases"][0]["ranking"][:2] == ["gold", "other"]
    assert "table_selection" not in report
    assert "code_execution" not in report


def test_retriever_benchmark_uses_identical_cases_and_top_k_for_all_modes():
    solr = FakeSolr(
        select_docs=[{"resource_id": "gold", "score": 2.0}],
        knn_docs=[{"resource_id": "gold", "score": 0.9}],
    )
    cases = [
        BenchmarkCase("q1", "Complete question one?", ("keyword-one",), ("gold",)),
        BenchmarkCase("q2", "Complete question two?", ("keyword-two",), ("gold",)),
    ]
    embedding = FakeEmbedding()

    report = run_retriever_benchmark(
        solr,
        cases,
        base_config=RetrievalConfig(top_k=3, candidate_multiplier=2),
        modes=tuple(RetrievalMode),
        alphas=(0.5,),
        include_rrf=False,
        k_values=(1, 3),
        embedding_model=embedding,
    )

    assert set(report["experiments"]) == {
        "keyword",
        "semantic",
        "hybrid-weighted-a0.5",
    }
    for experiment in report["experiments"].values():
        assert experiment["config"]["top_k"] == 3
        assert [case["case_id"] for case in experiment["cases"]] == ["q1", "q2"]
        assert [case["relevant_table_ids"] for case in experiment["cases"]] == [
            ["gold"],
            ["gold"],
        ]
        assert "table_selection" not in experiment
        assert "code_execution" not in experiment
    assert embedding.queries == [
        "Complete question one?",
        "Complete question two?",
        "Complete question one?",
        "Complete question two?",
    ]


def test_benchmark_metrics_csv_uses_variable_case_count_and_reuses_job_id(tmp_path):
    solr = FakeSolr(select_docs=[{"resource_id": "gold", "score": 2.0}])
    cases = [
        BenchmarkCase(str(index), f"Question {index}?", ("gold",), ("gold",))
        for index in range(3)
    ]
    report = run_retriever_benchmark(
        solr,
        cases,
        base_config=RetrievalConfig(top_k=10),
        modes=(RetrievalMode.KEYWORD,),
    )
    path = tmp_path / "retrieval_benchmarks_log.csv"

    append_benchmark_metrics_log(
        report,
        path,
        run_id="run-variable",
        core="nyc",
        source_path="queries.json",
        source_job_ids={"keyword": "existing-job-id"},
    )

    with path.open(newline="", encoding="utf-8") as input_file:
        row = next(csv.DictReader(input_file))
    assert row["JOB_ID"] == "existing-job-id"
    assert row["EXPERIMENT_ID"] == "keyword"
    assert row["RETRIEVAL_MODE"] == "keyword"
    assert row["HYBRID_ALPHA"] == "0.5"
    assert row["EMBEDDING_BASE_URL"] == "http://localhost:11434"
    assert row["VECTOR_FIELD"] == "table_embedding"
    assert row["MISSING_SIGNAL_POLICY"] == "zero"
    assert row["QUESTION_COUNT"] == "3"
    assert row["HIT_AT_1"] == "1.0"
    assert row["RECALL_AT_10"] == "1.0"
    assert row["MRR"] == "1.0"
    assert "CASE_METRICS_JSON" not in row
    assert "MEAN_METRICS_JSON" not in row
    assert "FULL_TRACE" not in row


def test_benchmark_log_evolves_legacy_header_without_backfilling_old_rows(tmp_path):
    path = tmp_path / "retrieval_benchmarks_log.csv"
    path.write_text(
        "ID,TIMESTAMP,MODE,ALPHA\n1,old,keyword,0.5\n",
        encoding="utf-8",
    )
    report = run_retriever_benchmark(
        FakeSolr(select_docs=[{"resource_id": "gold", "score": 1.0}]),
        [BenchmarkCase("q1", "Question?", ("gold",), ("gold",))],
        base_config=RetrievalConfig(top_k=10),
        modes=(RetrievalMode.KEYWORD,),
    )

    append_benchmark_metrics_log(
        report,
        path,
        run_id="new-run",
        core="nyc",
        source_path="questions.json",
        source_job_ids={"keyword": "new-job"},
        model="model-name",
        architecture="unified",
        portal_name="NYC Open Data",
    )

    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert "MODE" not in rows[0]
    assert rows[0]["RETRIEVAL_MODE"] == "keyword"
    assert rows[0]["HYBRID_ALPHA"] == "0.5"
    assert rows[1]["RETRIEVAL_MODE"] == "keyword"
    assert rows[1]["MODEL"] == "model-name"
    assert rows[1]["ARCHITECTURE"] == "unified"
    assert rows[1]["PORTAL_NAME"] == "NYC Open Data"


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

    summary = SolrEmbeddingIndexer(solr, config, embedding).run(
        batch_size=4, create_schema=True
    )

    assert summary.indexed_documents == 1
    assert summary.vector_dimension == 2
    assert embedding.documents == [[
        "Represent this table metadata for information retrieval.\n"
        "Title: Incidenti\nColumn names: anno | totale"
    ]]
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


def test_direct_solr_client_uses_shared_environment_base_url(monkeypatch):
    monkeypatch.setenv("SOLR_BASE_URL", "http://127.0.0.1:8993/solr")

    client = LocalSolrClient("nyc")

    assert client.base_url == "http://127.0.0.1:8993/solr"


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
                            "table_embedding": [0.1, 0.2],
                            "score": 0.7,
                        }
                    ]
                }
            }

    def fake_post(url, *, data, timeout):
        captured.update(url=url, params=data, timeout=timeout)
        return Response()

    monkeypatch.setattr("src.client_solr.requests.post", fake_post)
    client = LocalSolrClient("bologna", timeout=4)

    response = client.knn_select(
        [0.25, -0.5],
        vector_field="table_embedding",
        top_k=50,
        rows=10,
        filters=['representation_version:"metadata-v1"'],
    )

    assert captured["url"].endswith("/bologna/query")
    assert captured["params"]["q"] == (
        "{!knn f=table_embedding topK=50}[0.25,-0.5]"
    )
    assert captured["params"]["fq"] == [
        'representation_version:"metadata-v1"'
    ]
    assert "*" not in captured["params"]["fl"]
    assert "table_embedding" not in captured["params"]["fl"]
    assert "title" in captured["params"]["fl"]
    assert "score" in captured["params"]["fl"]
    assert "table_embedding" not in response["response"]["docs"][0]
    assert response["response"]["docs"][0]["columns"] == [
        {"name": "anno", "description": None}
    ]

    with pytest.raises(ValueError, match="finite"):
        client.knn_select(
            [math.nan], vector_field="table_embedding", top_k=10
        )
