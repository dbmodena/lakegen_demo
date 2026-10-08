"""Which retriever ran is reported to the operator, never to the agent.

The agent must pick concepts on the merits of the question. If it could read
which modality is configured it could adapt to the retriever, which would make
an A/B comparison between modalities measure the wrong thing.
"""

import pandas as pd
import pytest

from lakegen.agent_tools import tools_p12
from lakegen.agent_tools.tools_p12 import P12State, Phase12ToolsManager
from lakegen.retrieval import RetrievalConfig, RetrievalHit
from lakegen.tracing import summarize_tool_calls


def _hit(resource_id, rank):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id, "columns": []},
        score=1.0 / rank,
        rank=rank,
        lexical_rank=rank,
    )


@pytest.fixture
def lake(monkeypatch, tmp_path):
    names = ["alpha", "beta"]
    for name in names:
        pd.DataFrame({"value": [1]}).to_parquet(tmp_path / f"{name}.parquet")

    class FakeService:
        def retrieve(self, **kwargs):
            return [_hit(name, index) for index, name in enumerate(names, 1)]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    return tmp_path, [f"{name}.parquet" for name in names]


def _manager(lake, mode, notice=None):
    tmp_path, files = lake
    return Phase12ToolsManager(
        P12State(), object(), files, tmp_path,
        question="anything",
        retrieval_config=RetrievalConfig(mode=mode),
        notice_callback=notice,
    )


# The grep retrieval modes exist only on the testing-grounds branch.
NEEDS_TESTING_GROUNDS = pytest.mark.xfail(
    strict=True, reason="needs testing-grounds: the grep retrieval modes"
)


@pytest.mark.parametrize(
    "mode",
    [
        "keyword",
        pytest.param("grep", marks=NEEDS_TESTING_GROUNDS),
        pytest.param("grep_values", marks=NEEDS_TESTING_GROUNDS),
        "duckdb_agentic",
        "hybrid",
    ],
)
def test_the_configured_modality_is_named_on_the_notice_channel(lake, mode):
    notices = []
    _manager(lake, mode, notices.append).search_tables("some concept")

    assert len(notices) == 1
    assert f"`{mode}`" in notices[0]
    assert "Retrieval:" in notices[0]


@pytest.mark.parametrize(
    "mode",
    [
        "keyword",
        pytest.param("grep", marks=NEEDS_TESTING_GROUNDS),
        pytest.param("grep_values", marks=NEEDS_TESTING_GROUNDS),
    ],
)
def test_the_agent_is_never_told_which_retriever_ran(lake, mode):
    notices = []
    result = _manager(lake, mode, notices.append).search_tables("some concept")

    assert mode not in result
    assert "Retrieval:" not in result and "ranked hits" not in result


@NEEDS_TESTING_GROUNDS
def test_the_notice_does_not_count_as_a_tool_call(lake):
    notices = []
    _manager(lake, "grep", notices.append).search_tables("some concept")

    # The activity log is parsed for tool telemetry; the notice must not register.
    assert summarize_tool_calls(notices[0]) == []


@NEEDS_TESTING_GROUNDS
def test_a_standalone_run_still_reports_on_stdout(lake, capsys):
    _manager(lake, "grep", None).search_tables("some concept")

    assert "`grep`" in capsys.readouterr().out


@NEEDS_TESTING_GROUNDS
def test_no_notice_when_no_search_actually_runs(lake):
    notices = []
    manager = _manager(lake, "grep", notices.append)
    manager.search_tables("some concept")
    assert len(notices) == 1

    # Repeated concepts are served from cache, and the budget then blocks further
    # searches; neither reaches a retriever, so neither should claim one ran.
    manager.search_tables("some concept")
    manager.search_tables("a different concept")
    assert len(notices) == 1
