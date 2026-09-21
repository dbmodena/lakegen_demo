import json
import math
import csv
import time

import pytest

from index_retrieval import validate_source_coverage
from src.client_solr import LocalSolrClient
from scripts_pneuma.pneuma_judge import (
    JudgeError,
    RelevanceJudgment,
    StructuredRelevanceJudge,
    order_by_relevance,
    structured_relevance_prompt,
)
from lakegen.retrieval import (
    HybridRetriever,
    FusionMethod,
    KeywordRetriever,
    MissingSignalPolicy,
    PneumaRetriever,
    RetrievalConfig,
    RetrievalHit,
    RetrievalMode,
    SemanticRetriever,
    SolrEmbeddingIndexer,
    SolrPneumaDocumentResolver,
    ensure_vector_schema,
    evaluate_ranking,
    min_max_normalize,
    represent_table,
    validate_stored_replacement_schema,
)
from lakegen.retrieval import benchmark as benchmark_module
from lakegen.retrieval.benchmark import (
    BenchmarkCase,
    append_benchmark_metrics_log,
    load_benchmark_cases,
    run_retriever_benchmark,
)
from lakegen.retrieval.embeddings import (
    EmbeddingGenerationError,
    OllamaMultilingualEmbedding,
    _normalize_query,
    _validate,
)


def test_solr_select_keeps_strict_and_for_words_in_multiword_concepts(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": {"docs": []}}

    def fake_get(_url, *, params, timeout):
        captured.update(params)
        return Response()

    monkeypatch.setattr("src.client_solr.requests.get", fake_get)
    LocalSolrClient("uk").select(
        ["Transport for Greater Manchester", "invoices"], q_op="AND"
    )

    assert captured["q"] == "Transport for Greater Manchester invoices"
    assert captured["q.op"] == "AND"


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


class FakePneuma:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def query_index(self, index_name, queries, *, k, n, alpha):
        self.calls.append((index_name, queries, k, n, alpha))
        return self.response


def test_pneuma_retriever_preserves_order_and_maps_documents():
    client = FakePneuma(
        {
            "status": "SUCCESS",
            "data": [
                {
                    "query": "Which tables?",
                    "retrieved_tables": ["/tables/b.csv", "/tables/a.csv"],
                }
            ],
        }
    )
    documents = {
        "/tables/a.csv": {"resource_id": "a", "title": "A"},
        "/tables/b.csv": {"resource_id": "b", "title": "B"},
    }
    config = RetrievalConfig(
        mode="pneuma",
        top_k=2,
        alpha=0.25,
        candidate_multiplier=7,
        pneuma_index_name="test-index",
    )

    hits = PneumaRetriever(config, documents.get, client=client).retrieve(
        "Which tables?", top_k=2
    )

    assert [item.document["resource_id"] for item in hits] == ["b", "a"]
    assert [item.rank for item in hits] == [1, 2]
    assert [item.score for item in hits] == [1.0, 0.5]
    assert client.calls == [("test-index", "Which tables?", 2, 7, 0.25)]


def test_pneuma_retriever_rejects_failed_response():
    client = FakePneuma({"status": "ERROR", "message": "missing index"})
    retriever = PneumaRetriever(
        RetrievalConfig(mode="pneuma"), lambda _table_id: None, client=client
    )

    with pytest.raises(RuntimeError, match="missing index"):
        retriever.retrieve("question", top_k=1)


class FakeCatalogSolr:
    def __init__(self, documents):
        self.documents = list(documents)

    def iter_documents(self, *, fields, sort_field):
        return iter(self.documents)


def test_solr_pneuma_resolver_maps_dataset_resource_file_names():
    # UK files are named <dataset_id>___<resource_id>; NYC files by resource id.
    first = {"dataset_id": "DS-1", "resource_id": "res-2", "title": "First"}
    sibling = {"dataset_id": "DS-1", "resource_id": "res-3", "title": "Sibling"}
    nyc = {"dataset_id": "abcd-1234", "resource_id": "abcd-1234", "title": "NYC"}
    resolver = SolrPneumaDocumentResolver(FakeCatalogSolr([first, sibling, nyc]))

    # The pair picks the resource, not the dataset's first document.
    assert resolver("/lake/uk/parquet/ds-1___RES-3.parquet") is sibling
    assert resolver("/lake/uk/parquet/DS-1___res-2.parquet") is first
    assert resolver("/lake/nyc/parquet/abcd-1234.parquet") is nyc
    assert resolver("/lake/uk/parquet/DS-1___res-9.parquet") is None


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


def test_embedding_validation_rejects_unexpected_dimension_and_implausible_norm():
    with pytest.raises(ValueError, match="dimension 2; expected 3"):
        _validate([0.25, 0.75], expected_dimension=3)
    with pytest.raises(ValueError, match="implausible vector norm"):
        _validate([2.0, 0.0], max_norm=1.0)


def test_query_normalization_is_conservative_and_rejects_oversized_input():
    assert _normalize_query("  full\u2011width\x00  ２０２４\nNYC  ") == (
        "full‐width 2024 NYC"
    )
    with pytest.raises(EmbeddingGenerationError, match="character limit of 3"):
        _normalize_query("four", max_chars=3)


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
    embedding.retry_delays = (0.25, 1.0)
    delays = []
    embedding._sleep = delays.append

    assert embedding.encode_query("valid question") == [0.25, 0.75]
    assert embedding._model.calls == 3
    assert delays == [0.25, 1.0]


def test_query_embedding_uses_retrieval_prefix_after_deterministic_nan():
    class NanSensitiveModel:
        def __init__(self):
            self.queries = []

        def get_query_embedding(self, text):
            self.queries.append(text)
            if text.startswith("Represent this question for dataset retrieval. "):
                return [0.25, 0.75]
            raise RuntimeError("unsupported value: NaN")

    embedding = OllamaMultilingualEmbedding.__new__(OllamaMultilingualEmbedding)
    embedding._model = NanSensitiveModel()
    embedding.retry_delays = (0.0, 0.0)
    embedding._sleep = lambda _delay: None

    assert embedding.encode_query("valid question") == [0.25, 0.75]
    assert embedding._model.queries == [
        "valid question",
        "Represent this question for dataset retrieval. valid question",
    ]


def test_query_embedding_reports_provider_failure_after_nan_fallback():
    class BrokenModel:
        def __init__(self):
            self.calls = 0

        def get_query_embedding(self, _text):
            self.calls += 1
            raise RuntimeError("unsupported value: NaN")

    embedding = OllamaMultilingualEmbedding.__new__(OllamaMultilingualEmbedding)
    embedding._model = BrokenModel()
    embedding.retry_delays = (0.0, 0.0)
    embedding._sleep = lambda _delay: None

    with pytest.raises(EmbeddingGenerationError) as error:
        embedding.encode_query("valid question")
    assert embedding._model.calls == 2
    assert "after 2 attempts" in str(error.value)
    assert "non-finite fallback used: True" in str(error.value)
    assert "RuntimeError: unsupported value: NaN" in str(error.value)


def test_query_embedding_still_retries_transient_non_nan_failure():
    class BrokenModel:
        def __init__(self):
            self.calls = 0

        def get_query_embedding(self, _text):
            self.calls += 1
            raise RuntimeError("connection reset")

    embedding = OllamaMultilingualEmbedding.__new__(OllamaMultilingualEmbedding)
    embedding._model = BrokenModel()
    embedding.retry_delays = (0.0, 0.0)
    embedding._sleep = lambda _delay: None

    with pytest.raises(EmbeddingGenerationError) as error:
        embedding.encode_query("valid question")
    assert embedding._model.calls == 3
    assert "after 3 attempts" in str(error.value)
    assert "non-finite fallback used: False" in str(error.value)


def test_query_embedding_normalizes_input_and_checks_known_model_dimension():
    class RecordingModel:
        def __init__(self):
            self.queries = []

        def get_query_embedding(self, text):
            self.queries.append(text)
            return [0.25, 0.75]

    embedding = OllamaMultilingualEmbedding.__new__(OllamaMultilingualEmbedding)
    embedding._model = RecordingModel()
    embedding.expected_dimension = 3
    embedding.retry_delays = ()

    with pytest.raises(EmbeddingGenerationError, match="dimension 2; expected 3"):
        embedding.encode_query("  NYC\x00  permits\n")
    assert embedding._model.queries == ["NYC permits"]


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


class FakeSolrByOp:
    """Returns different documents depending on the q_op param, so tests can
    assert the AND-then-OR fallback without a real Solr/Lucene query engine."""

    def __init__(self, *, docs_by_op):
        self.docs_by_op = docs_by_op
        self.select_calls = []

    def select(self, tokens, **params):
        self.select_calls.append((list(tokens), params))
        return {"response": {"docs": list(self.docs_by_op.get(params["q_op"], []))}}


def test_keyword_retrieval_or_fallback_is_off_by_default():
    solr = FakeSolrByOp(docs_by_op={"OR": [{"resource_id": "a", "score": 1.0}]})

    results = KeywordRetriever(solr).retrieve(
        ["Transport for Greater Manchester", "invoice spending"],
        top_k=10,
        q_op="AND",
    )

    assert results == []
    assert [params["q_op"] for _, params in solr.select_calls] == ["AND"]


def test_keyword_retrieval_falls_back_to_or_when_and_returns_nothing_and_enabled():
    solr = FakeSolrByOp(
        docs_by_op={"OR": [{"resource_id": "a", "score": 1.0}]}
    )

    results = KeywordRetriever(solr, or_fallback=True).retrieve(
        ["Transport for Greater Manchester", "invoice spending"],
        top_k=10,
        q_op="AND",
    )

    assert [result.document["resource_id"] for result in results] == ["a"]
    assert [params["q_op"] for _, params in solr.select_calls] == ["AND", "OR"]


def test_keyword_retrieval_does_not_fall_back_when_and_already_has_hits():
    solr = FakeSolrByOp(docs_by_op={"AND": [{"resource_id": "a", "score": 1.0}]})

    results = KeywordRetriever(solr, or_fallback=True).retrieve(
        ["acqua"], top_k=10, q_op="AND"
    )

    assert [result.document["resource_id"] for result in results] == ["a"]
    assert len(solr.select_calls) == 1


def test_keyword_retrieval_does_not_fall_back_when_already_or():
    solr = FakeSolrByOp(docs_by_op={})

    results = KeywordRetriever(solr, or_fallback=True).retrieve(
        ["acqua"], top_k=10, q_op="OR"
    )

    assert results == []
    assert len(solr.select_calls) == 1


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
        modes=(
            RetrievalMode.KEYWORD,
            RetrievalMode.SEMANTIC,
            RetrievalMode.HYBRID,
        ),
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


def _generated_queries_payload():
    bus_lanes = {
        "question": "How many bus lanes are in Brooklyn?",
        "question_keywords": ["bus", "lane", "Brooklyn"],
        "tables": [{"name": "Table_0"}],
        "client_id": "q-bus",
        "status": "success",
    }
    return {
        "PANDAS": {
            "single_table": {
                "st_0": {
                    "_meta": {"tables": {"Table_0": "bus-lanes"}},
                    "0": bus_lanes,
                    "1": {
                        **bus_lanes,
                        "question": "Rejected question?",
                        "client_id": "q-rejected",
                        "status": "failure",
                    },
                },
            },
            "multi_table": {
                "mt_0": {
                    "_meta": {
                        "tables": {
                            "Table_0": "fares-2017",
                            "Table_1": "fares-2018",
                            "Table_2": "unused",
                        }
                    },
                    "0": {
                        "question": "Total fare in 2017 and 2018?",
                        "question_keywords": ["fare", "2017", "2018"],
                        "tables": ["Table_1", "Table_0", "Table_1"],
                        "client_id": "q-fares",
                        "status": "success",
                    },
                },
            },
        },
        # The same question over the same table, generated again for SQL.
        "SQL": {
            "single_table": {
                "st_0": {
                    "_meta": {"tables": {"Table_0": "bus-lanes"}},
                    "0": {**bus_lanes, "client_id": "q-bus-sql"},
                },
            },
        },
    }


def test_benchmark_loads_each_successful_generated_query_once(tmp_path):
    path = tmp_path / "generated_queries_semantic.json"
    path.write_text(json.dumps(_generated_queries_payload()), encoding="utf-8")

    assert load_benchmark_cases(path) == [
        BenchmarkCase(
            "q-bus",
            "How many bus lanes are in Brooklyn?",
            ("bus", "lane", "Brooklyn"),
            ("bus-lanes",),
        ),
        BenchmarkCase(
            "q-fares",
            "Total fare in 2017 and 2018?",
            ("fare", "2017", "2018"),
            ("fares-2018", "fares-2017"),
        ),
    ]


def test_benchmark_rejects_generated_query_with_unresolved_table_alias(tmp_path):
    payload = _generated_queries_payload()
    payload["PANDAS"]["multi_table"]["mt_0"]["0"]["tables"] = ["Table_9"]
    path = tmp_path / "generated_queries_semantic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="PANDAS/multi_table/mt_0/0"):
        load_benchmark_cases(path)


def test_benchmark_cli_takes_input_core_and_retrieval_settings_from_config(
    tmp_path, monkeypatch
):
    generated = tmp_path / "generated_queries_semantic.json"
    generated.write_text(json.dumps(_generated_queries_payload()), encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"core: uk\nbenchmark:\n  path: {generated}\n"
        "retrieval:\n  top_k: 20\n  alpha: 0.25\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(solr, cases, *, base_config, **kwargs):
        captured.update(core=solr.core, cases=cases, base_config=base_config)
        return {"created_at": "2026-01-01T00:00:00+00:00", "experiments": {}}

    monkeypatch.setattr(benchmark_module, "run_retriever_benchmark", fake_run)

    exit_code = benchmark_module.main(
        [
            "--config",
            str(config),
            "--candidate-multiplier",
            "3",
            "--output",
            str(tmp_path / "report.json"),
            "--metrics-log",
            str(tmp_path / "retrieval_benchmarks_log.csv"),
        ]
    )

    assert exit_code == 0
    assert captured["core"] == "uk"
    assert [case.case_id for case in captured["cases"]] == ["q-bus", "q-fares"]
    assert captured["base_config"].top_k == 20
    assert captured["base_config"].alpha == 0.25
    assert captured["base_config"].candidate_multiplier == 3


def test_benchmark_cli_requires_input_or_configured_benchmark_path(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text("core: nyc\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        benchmark_module.main(
            ["--config", str(config), "--output", str(tmp_path / "report.json")]
        )


def test_only_question_ranking_modes_leave_discovery_keywords_unsearched():
    assert {mode for mode in RetrievalMode if mode.ranks_question_only} == {
        RetrievalMode.SEMANTIC,
        RetrievalMode.PNEUMA,
    }
    assert RetrievalMode.PNEUMA.split_keywords(["Home", "Office"]) == (
        [],
        ["Home", "Office"],
    )
    assert RetrievalMode.HYBRID.split_keywords(["Home"]) == (["Home"], [])
    assert RetrievalMode.KEYWORD.split_keywords(None) == ([], [])


class ScriptedTransport:
    """A judge transport replaying canned responses and recording each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, messages, **params):
        self.requests.append({"messages": [dict(message) for message in messages], **params})
        return self.responses.pop(0)


def test_structured_judge_decodes_greedily_against_the_verdict_schema():
    transport = ScriptedTransport('{"relevant": true, "reason": "It has FTE posts per region."}')

    judgment = StructuredRelevanceJudge(transport, retry_delay=0).judge("Is it relevant?")

    assert judgment == RelevanceJudgment(relevant=True, reason="It has FTE posts per region.")
    (request,) = transport.requests
    assert request["temperature"] == 0.0
    assert request["schema"] == RelevanceJudgment.model_json_schema()
    assert request["schema"]["additionalProperties"] is False
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert request["messages"][1]["content"] == "Is it relevant?"
    with pytest.raises(ValueError):
        RelevanceJudgment(relevant=True, reason="   ")


def test_structured_judge_feeds_errors_back_until_the_verdict_validates():
    transport = ScriptedTransport(
        "**Yes**, the table is relevant.",
        '{"relevant": "maybe", "reason": "Unsure."}',
        '```json\n{"relevant": false, "reason": "No Asylum rows."}\n```',
    )

    judgment = StructuredRelevanceJudge(transport, retry_delay=0).judge("Is it relevant?")

    assert judgment.relevant is False
    first, after_json_error, after_validation_error = (
        request["messages"] for request in transport.requests
    )
    # Rebuilt on every attempt, never appended to; the system message is fixed.
    assert len(after_json_error) == len(after_validation_error) == 2
    assert after_json_error[0] == after_validation_error[0] == first[0]
    assert "JSON PARSING ERROR" in after_json_error[1]["content"]
    assert "**Yes**, the table is relevant." in after_json_error[1]["content"]
    assert "SCHEMA VALIDATION ERROR" in after_validation_error[1]["content"]
    assert "  - relevant:" in after_validation_error[1]["content"]


def test_structured_judge_raises_rather_than_guessing_a_verdict():
    transport = ScriptedTransport("Yes", "Yes", "Yes")

    with pytest.raises(JudgeError, match="no valid verdict after 3 attempt"):
        StructuredRelevanceJudge(transport, retry_delay=0).judge("Is it relevant?")


def test_judge_all_returns_verdicts_in_document_order_across_workers():
    def transport(messages, **_params):
        prompt = messages[1]["content"]
        index = int(prompt[-1])
        time.sleep(0.01 * (5 - index))  # later documents answer first
        return json.dumps({"relevant": index % 2 == 0, "reason": prompt})

    judge = StructuredRelevanceJudge(transport, workers=4, retry_delay=0)
    judgments = judge.judge_all([f"document {index}" for index in range(5)])

    assert [judgment.reason for judgment in judgments] == [
        f"document {index}" for index in range(5)
    ]
    assert [judgment.relevant for judgment in judgments] == [True, False, True, False, True]


def test_order_by_relevance_moves_irrelevant_documents_behind_in_fused_order():
    def verdict(relevant):
        return RelevanceJudgment(relevant=relevant, reason="r")

    nodes = ["a", "b", "c", "d"]
    verdicts = [verdict(False), verdict(True), verdict(False), verdict(True)]

    assert order_by_relevance(nodes, verdicts) == ["b", "d", "a", "c"]
    with pytest.raises(ValueError):
        order_by_relevance(nodes, verdicts[:1])


def test_structured_relevance_prompt_changes_only_the_answer_instruction():
    pneuma_prompt = (
        "and this question:\n/*\nWhich?\n*/\n"
        "Is the table relevant to answer the question? Begin your answer with yes/no."
    )

    prompt = structured_relevance_prompt(pneuma_prompt)

    assert prompt == (
        "and this question:\n/*\nWhich?\n*/\n"
        "Is the table relevant to answer the question? "
        "Answer with the JSON verdict described in the system message."
    )
