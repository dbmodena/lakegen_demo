import json

import pandas as pd
import pytest

from lakegen.discovery.selectors import jev as jev_selectors
from lakegen.experiment_config import DiscoveryConfig, ExperimentConfig
from lakegen.orchestrated_context import PreparedCandidate, PreparedDiscoveryContext
from lakegen.phases import jev_selector, orchestrated_discovery, phase2
from lakegen.phases.jev_selector import (
    JevSelection,
    JevSelectorError,
    build_request,
    choose_tables,
    profile_table,
    select_with_jev,
)

JEV = DiscoveryConfig(selector="jev")


def _meta(title, **extra):
    return {"title": title, "description": f"About {title}", "tags": ["t"],
            "columns": [{"name": "year", "type": "int", "description": ""}], **extra}


def _answers(scores, primary):
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 123},
        "answers": {
            "primary_table": {"type": "choice", "choice": primary},
            **{f"needs::{key}": {"type": "noul", "noul": value} for key, value in scores.items()},
        },
    }


def test_choose_tables_puts_primary_first_and_keeps_every_table_over_threshold():
    scores = {"a": 0.9, "b": 0.6, "c": 0.2}
    assert choose_tables(["a", "b", "c"], scores, "b", 0.5, 0.15) == ["b", "a"]


def test_choose_tables_falls_back_to_primary_when_nothing_passes_threshold():
    scores = {"a": 0.3, "b": 0.4}
    assert choose_tables(["a", "b"], scores, "a", 0.5, 0.15) == ["a"]


def test_choose_tables_rejects_when_every_score_is_below_the_floor():
    assert choose_tables(["a", "b"], {"a": 0.05, "b": 0.1}, "a", 0.5, 0.15) == []


def test_build_request_asks_one_question_per_candidate_with_readable_keys():
    keys = {"budget_2024": "budget_2024.parquet", "budget_2025": "budget_2025.parquet"}
    metadata = {
        "budget_2024.parquet": _meta("Budget 2024", resource_name="Budget 2024 (English)"),
        "budget_2025.parquet": _meta("Budget 2025"),
    }
    payload = build_request("How much in 2025?", keys, metadata, {}, "jev-1.13.0")

    assert set(payload["questions"]) == {"primary_table", "needs::budget_2024", "needs::budget_2025"}
    assert payload["questions"]["primary_table"]["criteria"] == {
        "budget_2024": "Budget 2024", "budget_2025": "Budget 2025"}
    assert payload["questions"]["needs::budget_2025"]["criteria"] == jev_selector.NEEDED_CRITERIA
    tables = payload["state"]["candidate_tables"]
    assert tables["budget_2024"]["resource_name"] == "Budget 2024 (English)"
    assert "resource_name" not in tables["budget_2025"]
    assert tables["budget_2025"]["years_covered"] == "2025"


def test_build_request_shrinks_descriptions_to_fit_the_size_limit():
    keys = {f"t{i}": f"t{i}.parquet" for i in range(8)}
    metadata = {name: _meta(key, description="word " * 20_000) for key, name in keys.items()}
    payload = build_request("q", keys, metadata, {}, "jev-1.13.0")
    assert len(json.dumps(payload).encode()) <= jev_selector.MAX_REQUEST_BYTES


def test_profile_table_reads_years_values_and_ranges(tmp_path):
    path = tmp_path / "t.parquet"
    pd.DataFrame({"annee": [2019, 2020, 2021], "district": ["A", "A", "B"],
                  "amount": [1.5, 2.0, 9.0]}).to_parquet(path)
    profile = profile_table(path)
    assert profile["row_count"] == 3
    assert profile["years"] == [2019, 2020, 2021]
    assert profile["samples"]["district"].startswith("A")
    assert profile["samples"]["amount"] == "range 1.5 .. 9.0"


def test_profile_table_survives_an_unreadable_file(tmp_path):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not a parquet file")
    assert "error" in profile_table(path)


def test_select_with_jev_maps_keys_back_to_candidate_files():
    candidates = ["a.parquet", "b.parquet", "c.parquet"]
    metadata = {name: _meta(name) for name in candidates}
    sent = []

    def fake_post(payload):
        sent.append(payload)
        return _answers({"a": 0.2, "b": 0.8, "c": 0.7}, "c")

    result = select_with_jev("q", candidates, metadata, None, JEV, post=fake_post)

    assert result.selected == ["c.parquet", "b.parquet"]
    assert result.scores["a.parquet"] == 0.2
    assert result.input_tokens == 123
    assert sent[0]["model"] == "jev-1.13.0"
    assert result.trace.startswith(jev_selector.TRACE_HEADER)


def test_select_with_jev_turns_low_scores_into_a_keyword_rejection():
    candidates = ["a.parquet", "b.parquet"]
    result = select_with_jev(
        "q", candidates, {}, None, JEV,
        post=lambda _payload: _answers({"a": 0.05, "b": 0.1}, "b"),
    )
    assert result.selected == []
    assert result.reasoning.startswith("REJECT_KEYWORDS:")


def test_select_with_jev_limits_the_pool_to_max_candidates():
    candidates = [f"t{i}.parquet" for i in range(12)]
    sent = []

    def fake_post(payload):
        sent.append(payload)
        return _answers({f"t{i}": 0.9 for i in range(8)}, "t0")

    select_with_jev("q", candidates, {}, None, JEV, post=fake_post)
    assert len(sent[0]["state"]["candidate_tables"]) == 8


def test_select_with_jev_rejects_a_malformed_response():
    with pytest.raises(JevSelectorError):
        select_with_jev("q", ["a.parquet"], {}, None, JEV, post=lambda _p: {"answers": {}})


def test_missing_api_key_is_a_selector_error(monkeypatch):
    for name in jev_selector.KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(JevSelectorError):
        jev_selector.post_request({"questions": {}})


def test_config_accepts_jev_only_in_the_divided_architecture():
    ExperimentConfig(discovery_architecture="divided", discovery={"selector": "jev"})
    with pytest.raises(ValueError, match="divided"):
        ExperimentConfig(discovery_architecture="unified", discovery={"selector": "jev"})


def test_config_rejects_a_reject_floor_above_the_threshold():
    with pytest.raises(ValueError, match="jev_reject_floor"):
        DiscoveryConfig(selector="jev", jev_threshold=0.3, jev_reject_floor=0.4)


class _StopBeforeLLMJudge(Exception):
    pass


def _phase2(monkeypatch, discovery):
    monkeypatch.setattr(
        phase2, "_solr_and_search",
        lambda *_a, **_k: (["a.parquet", "b.parquet"], {"a.parquet": _meta("A"), "b.parquet": _meta("B")}),
    )

    def llm_judge_entered(*_a, **_k):
        raise _StopBeforeLLMJudge

    monkeypatch.setattr(phase2, "Phase2JudgeToolsManager", llm_judge_entered)
    return phase2.phase2_select_tables(
        query="q", llm=object(), pm=object(), all_files=["a.parquet", "b.parquet"],
        keywords=["kw"], solr_client=object(), csv_dir=None, discovery=discovery,
    )


def test_phase2_uses_jev_instead_of_the_llm_judge(monkeypatch):
    monkeypatch.setattr(jev_selectors, "select_with_jev",
        lambda *_a, **_k: JevSelection(["b.parquet"], "Jev rated as needed: b", "trace", {"b.parquet": 0.9}, 10),
    )
    selected, candidates, _meta_out, reasoning, trace, tokens = _phase2(monkeypatch, JEV)
    assert selected == ["b.parquet"]
    assert candidates == ["a.parquet", "b.parquet"]
    assert reasoning.startswith("Jev")
    assert tokens == 0


def test_phase2_falls_back_to_the_llm_judge_when_jev_fails(monkeypatch):
    def failing(*_a, **_k):
        raise JevSelectorError("HTTP 502")

    monkeypatch.setattr(jev_selectors, "select_with_jev", failing)
    with pytest.raises(_StopBeforeLLMJudge):
        _phase2(monkeypatch, JEV)


def test_phase2_keeps_the_llm_judge_by_default(monkeypatch):
    def unexpected(*_a, **_k):
        raise AssertionError("Jev must not run with the default selector")

    monkeypatch.setattr(jev_selectors, "select_with_jev", unexpected)
    with pytest.raises(_StopBeforeLLMJudge):
        _phase2(monkeypatch, DiscoveryConfig())


def _context():
    return PreparedDiscoveryContext(
        query="q", retrieval_mode="hybrid", retrieved_hit_count=1, prepared_candidate_count=1,
        candidates=[PreparedCandidate(retrieval_rank=1, prepared_position=1, dataset="a.parquet",
                                      scores={}, missing_signals=[], metadata=_meta("A"))],
    )


def test_orchestrated_selector_uses_jev(monkeypatch):
    monkeypatch.setattr(jev_selectors, "select_with_jev",
        lambda *_a, **_k: JevSelection(["a.parquet"], "Jev rated as needed: a", "trace", {"a.parquet": 0.9}, 10),
    )
    selected, reasoning, _trace, tokens = orchestrated_discovery.select_from_prepared_context(
        query="q", llm=object(), context=_context(), all_files=["a.parquet"],
        architecture="divided", discovery=JEV,
    )
    assert selected == ["a.parquet"] and reasoning.startswith("Jev") and tokens == 0


def test_orchestrated_selector_falls_back_to_the_llm_when_jev_fails(monkeypatch):
    def failing(*_a, **_k):
        raise JevSelectorError("HTTP 502")

    monkeypatch.setattr(jev_selectors, "select_with_jev", failing)
    monkeypatch.setattr(
        orchestrated_discovery, "_run_tool_free_turn",
        lambda **_k: ('FINAL_PAYLOAD: {"tables": ["a.parquet"], "reasoning": "llm"}', "stream", 5),
    )
    selected, reasoning, trace, tokens = orchestrated_discovery.select_from_prepared_context(
        query="q", llm=object(), context=_context(), all_files=["a.parquet"],
        architecture="divided", discovery=JEV,
    )
    assert selected == ["a.parquet"] and reasoning == "llm" and tokens == 5
    assert "jev selector unavailable" in trace
