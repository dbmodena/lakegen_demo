"""The divided keyword round: a proposal that matches no table goes back to the writer, banned and explained.

The phases are the golden harness's scripted fakes; Solr is a strict-AND count
over a two-table catalogue, so the round's own check decides what reaches
retrieval.
"""

from __future__ import annotations

import lakegen.pipeline.service as service
from golden.harness import (
    SERVICE_NAMES,
    Case,
    Script,
    experiment,
    make_runtime,
    phase1_result,
    phase3_result,
    prepare_run,
)
from lakegen.pipeline.engine import Pipeline
from lakegen.pipeline.run.context import RunContext

CATALOGUE = {
    "network.csv": {"network", "contribution", "belfast", "ni"},
    "parks.csv": {"parks", "district"},
}
NETWORK = (
    ["network.csv"],
    ["network.csv"],
    {"network.csv": {"title": "network"}},
    "It covers the question.",
    "trace",
    5,
)
PARKS = (
    ["parks.csv"],
    ["parks.csv"],
    {"parks.csv": {"title": "parks"}},
    "It covers the question.",
    "trace",
    5,
)
NOTHING = (
    [],
    [],
    {},
    "REJECT_KEYWORDS: No tables found with keyword retrieval. Try broader or different terms.",
    "",
    0,
)
ANSWER = ("There are 42.", 3)


class FakeSolr:
    """How many CATALOGUE entries contain every word of a query; records each query."""

    def __init__(self) -> None:
        self.queries: list[list[str]] = []

    def select(self, tokens, **_params):
        words = {word.casefold() for token in tokens for word in token.split()}
        self.queries.append(sorted(words))
        return {
            "response": {
                "numFound": sum(words <= entry for entry in CATALOGUE.values()),
                "docs": [],
            }
        }


def _run(monkeypatch, tmp_path, script: Script, *, mode: str = "keyword"):
    case = Case(
        experiment(discovery_architecture="divided", retrieval={"mode": mode}),
        script,
        files=tuple(CATALOGUE),
    )
    _replacements, log, _capture = prepare_run(monkeypatch, tmp_path, service, case, SERVICE_NAMES)
    solr = FakeSolr()
    monkeypatch.setattr(service, "get_solr", lambda _core: solr)
    ctx = RunContext.from_runtime(make_runtime(case.experiment, tmp_path))
    result = Pipeline(ctx, service.pipeline_deps()).run(case.question)
    writer = [call["kwargs"] for call in log.calls if call["call"] == "write_keywords"]
    retrieved = [
        call["kwargs"]["keywords"] for call in log.calls if call["call"] == "select_tables"
    ]
    return result, writer, retrieved, solr


def _outcomes(result, round_index: int = 0) -> list[str]:
    return [
        entry["outcome"]
        for entry in result.discovery["selection_attempts"][round_index]["keyword_searches"]
    ]


def test_a_proposal_matching_no_table_is_rewritten_in_the_same_conversation(monkeypatch, tmp_path):
    script = Script(
        phase1=[
            phase1_result(("network contribution", "belfast 2007")),
            phase1_result(("network contribution", "belfast")),
        ],
        phase2=[NETWORK],
        phase3=[phase3_result()],
        phase4=[ANSWER],
    )
    result, writer, retrieved, _ = _run(monkeypatch, tmp_path, script)
    assert (
        writer[0]["follow_ups"] == []
        and writer[1]["zero_result_sets"] == writer[0]["zero_result_sets"] == []
    )
    ((answer, feedback),) = writer[1]["follow_ups"]
    assert answer == "raw keyword output"
    assert (
        "Smallest word sets that already match nothing: {2007}." in feedback
        and "Searches left: 2." in feedback
    )
    assert retrieved == [["network contribution", "belfast"]]
    assert _outcomes(result) == ["empty", "matched"]
    assert result.status == "completed" and len(result.discovery["selection_attempts"]) == 1


def test_a_proposal_containing_a_failing_set_never_reaches_solr(monkeypatch, tmp_path):
    script = Script(
        phase1=[
            phase1_result(("network 2007",)),
            phase1_result(("belfast 2007",)),
            phase1_result(("belfast",)),
        ],
        phase2=[NETWORK],
        phase3=[phase3_result()],
        phase4=[ANSWER],
    )
    result, writer, retrieved, solr = _run(monkeypatch, tmp_path, script)
    assert _outcomes(result) == ["empty", "refused", "matched"]
    assert writer[2]["follow_ups"][1][1].startswith(
        "Refused before searching: {2007, belfast} contains {2007}"
    )
    assert not any({"belfast", "2007"} <= set(query) for query in solr.queries)
    assert retrieved == [["belfast"]] and result.status == "completed"


def test_keyword_mode_stops_once_the_retries_are_spent(monkeypatch, tmp_path):
    script = Script(
        phase1=[
            phase1_result(("network 1999",)),
            phase1_result(("network 2001",)),
            phase1_result(("network 2003",)),
        ],
        phase2=[NOTHING],
    )
    result, writer, retrieved, _ = _run(monkeypatch, tmp_path, script)
    assert len(writer) == 3 and retrieved == [["network 2003"]]
    assert _outcomes(result) == ["empty", "empty", "empty"]
    assert len(result.discovery["selection_attempts"]) == 1
    assert result.status == "failed" and "retries are spent" in result.error


def test_hybrid_rewrites_an_empty_keyword_side_and_then_selects_from_what_it_found(
    monkeypatch, tmp_path
):
    script = Script(
        phase1=[
            phase1_result(("network 1999",)),
            phase1_result(("network 2001",)),
            phase1_result(("network 2003",)),
        ],
        phase2=[NETWORK],
        phase3=[phase3_result()],
        phase4=[ANSWER],
    )
    result, writer, retrieved, _ = _run(monkeypatch, tmp_path, script, mode="hybrid")
    assert len(writer) == 3 and retrieved == [["network 2003"]]
    assert _outcomes(result) == ["empty", "empty", "empty"]
    assert result.status == "completed"


def test_the_word_sets_one_round_proved_empty_reach_the_next_rounds_first_message(
    monkeypatch, tmp_path
):
    rejection = phase3_result(
        rejected_reason="network.csv has no parks.",
        raw_result=None,
        structured_result=None,
        rejection_skip_tables=["network.csv"],
    )
    script = Script(
        phase1=[
            phase1_result(("network 2007",)),
            phase1_result(("network",)),
            phase1_result(("parks",)),
        ],
        phase2=[NETWORK, PARKS],
        phase3=[rejection, phase3_result()],
        phase4=[ANSWER],
    )
    result, writer, retrieved, _ = _run(monkeypatch, tmp_path, script)
    assert [call["zero_result_sets"] for call in writer] == [[], [], [["2007"]]]
    assert writer[2]["follow_ups"] == []
    assert retrieved == [["network"], ["parks"]]
    assert _outcomes(result, 1) == ["matched"] and result.status == "completed"
