"""Functional checks for automatic candidate inspection (parallel_inspection_enabled).

With the flag on, every table the agent can see is inspected by a parallel
mini-agent, inside ``search_tables`` for the initial window and inside
``expand_candidates`` for what it reveals. The inspector reports facts and
passes no verdict; the report is printed beneath the candidate. The agent has
no inspection tool.
"""

import pandas as pd
import pytest

from lakegen.discovery import tools as tools_p12
from lakegen.discovery.tools import P12State, Phase12ToolsManager
from lakegen.config.experiment import DiscoveryConfig
from lakegen.retrieval import RetrievalConfig, RetrievalHit


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeLLM:
    """Scripted responses: pick_columns gets a JSON column list, the
    observation prompt gets fact lines. Distinguished by prompt content,
    matching how the two prompts in candidate_inspection.py differ. The
    observation names the table's title so a test can see whose report it is."""

    def __init__(self, fail_observations=False):
        self.calls = 0
        self.fail_observations = fail_observations
        self.prompts = []

    def chat(self, messages):
        self.calls += 1
        prompt = messages[0].content
        if "Respond with ONLY a JSON list" in prompt:
            return FakeResponse('["value"]')
        self.prompts.append(prompt)
        if self.fail_observations:
            raise RuntimeError("model unavailable")
        title = prompt.split("Table title: ")[1].split("\n")[0]
        return FakeResponse(f"- value: {title} holds 3 rows, values 1 to 3")


def _hit(resource_id, rank):
    return RetrievalHit(
        document={
            "resource_id": resource_id,
            "title": resource_id,
            "columns": [{"name": "value"}],
        },
        score=1.0 / rank,
        rank=rank,
        lexical_rank=rank,
    )


def _manager(
    tmp_path, monkeypatch, *, count=3, llm=None, flag=True, question="what is the value?",
    frame=None, **discovery,
):
    names = [f"t{i:02d}" for i in range(1, count + 1)]
    for name in names:
        (frame if frame is not None else pd.DataFrame({"value": [1, 2, 3]})).to_parquet(
            tmp_path / f"{name}.parquet"
        )

    class FakeService:
        def retrieve(self, **kwargs):
            return [_hit(name, rank) for rank, name in enumerate(names, start=1)]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_: FakeService())
    state = P12State()
    return Phase12ToolsManager(
        state, object(), [f"{n}.parquet" for n in names], tmp_path,
        question=question,
        retrieval_config=RetrievalConfig(),
        discovery_config=DiscoveryConfig(parallel_inspection_enabled=flag, **discovery),
        llm=llm,
        notice_callback=lambda _text: None,
    )


def test_search_inspects_every_visible_table_and_prints_reports(tmp_path, monkeypatch):
    llm = FakeLLM()
    manager = _manager(tmp_path, monkeypatch, count=12, llm=llm)

    response = manager.search_tables("value")

    state = manager.state
    # Retrieval fetched 12, the agent sees the first initial_candidates (10),
    # and only those are inspected: the hidden two wait for expand_candidates.
    assert len(state.all_candidates) == 12
    assert state.visible_candidate_count == 10
    assert response.count("Independent inspection:") == 10
    assert "2 additional ranked candidates are available through expand_candidates" in response
    assert llm.calls == 20  # one column pick and one observation call per table
    assert len(state.inspected_candidates()) == 10
    assert "Rows: 3" in response and "Columns examined: value" in response
    assert "- value: t01 holds 3 rows, values 1 to 3" in response


def test_reports_carry_facts_and_no_verdict(tmp_path, monkeypatch):
    llm = FakeLLM()
    manager = _manager(tmp_path, monkeypatch, llm=llm)

    response = manager.search_tables("value")

    assert "VERDICT" not in response
    assert manager.state.verdict_cache == {}
    # The inspector is told outright that the decision is not its to make.
    assert llm.prompts and "Do not say whether the table is" in llm.prompts[0]


def test_expansion_inspects_the_candidates_it_reveals(tmp_path, monkeypatch):
    llm = FakeLLM()
    manager = _manager(tmp_path, monkeypatch, count=12, llm=llm)
    manager.search_tables("value")
    assert llm.calls == 20

    response = manager.expand_candidates("t11")

    state = manager.state
    assert "t11.parquet" in response
    assert "Independent inspection:" in response
    assert "t11 holds 3 rows" in response
    assert llm.calls == 22  # only the revealed table was inspected
    assert "t11.parquet" in state.inspected_candidates()
    assert "t12.parquet" not in state.inspected_candidates()  # still hidden
    assert state.inspection_counts == {}


def test_automatic_inspection_spends_no_budget_and_leaves_search_open(tmp_path, monkeypatch):
    manager = _manager(
        tmp_path, monkeypatch, llm=FakeLLM(), max_search_attempts=2,
    )
    manager.search_tables("value")

    assert len(manager.state.inspected_candidates()) == 3  # inspected, yet no budget spent
    assert manager.state.inspection_counts == {}
    # Serial inspection forecloses another search; the mini-agents must not.
    assert "Search refinement blocked" not in manager.search_tables("other")


def test_carried_table_is_not_reinspected(tmp_path, monkeypatch):
    llm = FakeLLM()
    manager = _manager(tmp_path, monkeypatch, llm=llm)
    manager.state.inspection_cache["t01.parquet"] = "Rows: 9\nObservations:\n- carried"

    response = manager.search_tables("value")

    assert llm.calls == 4  # only t02 and t03
    assert "- carried" in response  # still shown, from the carried text


def test_failed_model_call_keeps_the_deterministic_head(tmp_path, monkeypatch):
    llm = FakeLLM(fail_observations=True)
    manager = _manager(tmp_path, monkeypatch, count=1, llm=llm)
    monkeypatch.setattr(tools_p12.time, "sleep", lambda _s: None)

    response = manager.search_tables("value")

    # The row count still reaches the agent, and the failure is stated as a
    # failure, not dressed up as a judgement about the table.
    assert "Rows: 3" in response
    assert "unavailable: inspector error: model unavailable" in response
    assert "t01.parquet" in manager.state.inspected_candidates()


def test_flag_off_keeps_serial_inspection_and_the_hidden_pool(tmp_path, monkeypatch):
    llm = FakeLLM()
    manager = _manager(tmp_path, monkeypatch, count=12, llm=llm, flag=False)

    response = manager.search_tables("value")

    assert llm.calls == 0
    assert manager.state.visible_candidate_count == 10
    assert "Independent inspection:" not in response
    assert "expand_candidates" in response


def test_tools_offered_depend_on_automatic_inspection(tmp_path, monkeypatch):
    def tools(**kwargs):
        return {t.metadata.name: t for t in _manager(tmp_path, monkeypatch, **kwargs).get_tools()}

    automatic = tools(llm=FakeLLM())
    # No inspection tool, but expansion stays: it is the only way to reach
    # the hidden candidates.
    assert list(automatic) == [
        "search_tables", "expand_candidates", "check_join_union",
        "confirm_unified_selection", "reject_unified_selection",
    ]
    for serial in (tools(llm=FakeLLM(), flag=False), tools(llm=None)):
        assert "inspect_columns" in serial and "expand_candidates" in serial
    # Only the automatic mode asks the agent which tables to carry over.
    assert "keep_tables" in automatic["reject_unified_selection"].metadata.fn_schema.model_fields
    serial_reject = tools(llm=None)["reject_unified_selection"]
    assert "keep_tables" not in serial_reject.metadata.fn_schema.model_fields


def test_reject_carries_over_only_the_tables_the_agent_lists(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch, llm=FakeLLM())
    manager.search_tables("value")

    manager.reject_unified_selection(
        "nothing covers it", "search wider",
        ban_tables={"t03.parquet": "publisher is a different body entirely"},
        keep_tables=["t01.parquet", "t03.parquet", "t99.parquet"],
    )

    # Every visible table counts as inspected; carrying all of them forward
    # would fill the next attempt's pool with this search's results. Unbanned
    # and listed only: a banned table is not kept and an unknown one is ignored.
    assert manager.state.rejection_keep_tables == ["t01.parquet"]
    assert manager.state.rejection_skip_tables == ["t03.parquet"]


def test_serial_reject_still_keeps_every_inspected_table(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch, llm=None)
    manager.state.inspection_cache.update({
        "t01.parquet": "Schema for t01", "t02.parquet": "Schema for t02",
    })
    manager.search_tables("value")

    manager.reject_unified_selection("nothing covers it", "search wider")

    assert manager.state.rejection_keep_tables == ["t01.parquet", "t02.parquet"]


def test_selection_needs_no_serial_inspection(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch, llm=FakeLLM())
    manager.search_tables("value")

    result = manager.confirm_unified_selection(
        "The table holds the value.", ["t01.parquet"],
        requirement_coverage={"value": {"table": "t01.parquet", "columns": ["value"]}},
        table_roles={"t01.parquet": "fact records"},
    )

    assert "FINAL_PAYLOAD" in result
    assert manager.state.inspection_counts == {}


def test_year_validation_reads_the_coverage_in_the_report(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "event_date": pd.to_datetime(["2020-01-05", "2020-06-30"]),
        "value": [1, 2],
    })
    manager = _manager(
        tmp_path, monkeypatch, count=1, llm=FakeLLM(), frame=frame,
        question="What is the value in 2023?",
    )

    response = manager.search_tables("value")

    assert "Temporal coverage:" in response
    # A verdict never carried the coverage lines, so this check was idle for
    # automatically inspected tables; the report's head restores it.
    with pytest.raises(ValueError, match="temporal validation"):
        manager.confirm_unified_selection(
            "The table holds the value.", ["t01.parquet"],
            requirement_coverage={"value": {"table": "t01.parquet", "columns": ["value"]}},
            table_roles={"t01.parquet": "fact records"},
        )
