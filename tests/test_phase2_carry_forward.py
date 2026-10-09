from lakegen.config.retrieval import RetrievalConfig
from lakegen.discovery.divided import _solr_and_search
from lakegen.domain.retrieval import RetrievalHit


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.retrieve_kwargs: dict | None = None

    def retrieve(self, **kwargs):
        self.retrieve_kwargs = kwargs
        return self._hits


def test_solr_and_search_drops_excluded_and_carries_kept_tables(monkeypatch):
    hits = [
        RetrievalHit(
            document={"resource_id": "new", "dataset_id": "new", "title": "New"},
            score=1.0,
            rank=1,
        ),
        RetrievalHit(
            document={"resource_id": "bad", "dataset_id": "bad", "title": "Bad"},
            score=0.9,
            rank=2,
        ),
    ]
    monkeypatch.setattr(
        "lakegen.discovery.divided.get_table_retrieval_service",
        lambda *_a, **_kw: _FakeRetriever(hits),
    )

    candidates, metadata = _solr_and_search(
        query="question",
        keywords=["kw"],
        solr_client=object(),
        all_files=["new.parquet", "bad.parquet"],
        retrieval_config=RetrievalConfig(top_k=5),
        carried_tables=["good.parquet"],
        carried_metadata={"good.parquet": {"title": "Good"}},
        excluded_tables=["bad.parquet"],
    )

    assert "bad.parquet" not in candidates
    assert candidates[0] == "good.parquet"
    assert "new.parquet" in candidates
    assert metadata["good.parquet"]["title"] == "Good"


def test_solr_and_search_respects_top_k_after_prepending_carried_tables(monkeypatch):
    hits = [
        RetrievalHit(
            document={"resource_id": "one", "dataset_id": "one", "title": "One"},
            score=1.0,
            rank=1,
        ),
        RetrievalHit(
            document={"resource_id": "two", "dataset_id": "two", "title": "Two"},
            score=0.9,
            rank=2,
        ),
    ]
    monkeypatch.setattr(
        "lakegen.discovery.divided.get_table_retrieval_service",
        lambda *_a, **_kw: _FakeRetriever(hits),
    )

    candidates, _metadata = _solr_and_search(
        query="question",
        keywords=["kw"],
        solr_client=object(),
        all_files=["one.parquet", "two.parquet"],
        retrieval_config=RetrievalConfig(top_k=2),
        carried_tables=["good.parquet"],
    )

    assert candidates[0] == "good.parquet"
    assert len(candidates) == 2


def test_solr_and_search_backfills_from_beyond_the_original_cutoff(monkeypatch):
    # Three ranked hits: only the first two would normally fit in top_k=2, but
    # "a" is skip-listed, so "c" (rank 3) must backfill its slot instead of the
    # final pool shrinking to just "b".
    hits = [
        RetrievalHit(
            document={"resource_id": "a", "dataset_id": "a", "title": "A"},
            score=1.0, rank=1,
        ),
        RetrievalHit(
            document={"resource_id": "b", "dataset_id": "b", "title": "B"},
            score=0.9, rank=2,
        ),
        RetrievalHit(
            document={"resource_id": "c", "dataset_id": "c", "title": "C"},
            score=0.8, rank=3,
        ),
    ]
    retriever = _FakeRetriever(hits)
    monkeypatch.setattr(
        "lakegen.discovery.divided.get_table_retrieval_service",
        lambda *_a, **_kw: retriever,
    )

    candidates, _metadata = _solr_and_search(
        query="question",
        keywords=["kw"],
        solr_client=object(),
        all_files=["a.parquet", "b.parquet", "c.parquet"],
        retrieval_config=RetrievalConfig(top_k=2),
        excluded_tables=["a.parquet"],
    )

    assert candidates == ["b.parquet", "c.parquet"]
    # top_k was widened by the excluded-table count so a lower-ranked hit
    # could still be fetched to fill the gap left by the excluded one.
    assert retriever.retrieve_kwargs["top_k"] == 3
