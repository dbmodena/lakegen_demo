"""A replay starts from what an earlier run recorded: its keywords, or its selection."""

from __future__ import annotations

import json
from pathlib import Path

from golden.harness import (
    SERVICE_NAMES,
    Case,
    Script,
    experiment,
    make_runtime,
    phase3_result,
    prepare_run,
)
from lakegen.domain.outcomes import SelectorKind
from lakegen.workflow.pipeline import service
from lakegen.workflow.pipeline.engine import Pipeline
from lakegen.workflow.pipeline.model.context import RunContext
from lakegen.workflow.pipeline.replay import (
    Replay,
    StoredKeywords,
    StoredSelection,
    catalogue_metadata,
)

ANSWER = ("There are 42 parks.", 3)
FILES = ("gold.csv", "other.csv")
METADATA = {"gold.csv": {"title": "gold", "columns": []}}
SELECTED = (["gold.csv"], ["parks"], METADATA, "The table covers the question.", "trace", 5)
REJECTION = phase3_result(
    rejected_reason="gold.csv has no 2024 rows.", raw_result=None, structured_result=None,
    rejection_skip_tables=["gold.csv"],
)


def _run(monkeypatch, tmp_path, case: Case, replay: Replay):
    _replacements, log, _capture = prepare_run(monkeypatch, tmp_path, service, case, SERVICE_NAMES)
    ctx = RunContext.from_runtime(make_runtime(case.experiment, tmp_path))
    result = Pipeline(ctx, service.pipeline_deps()).run(case.question, replay=replay)
    return result, [call["call"] for call in log.calls], log


def test_a_stored_selection_goes_straight_to_the_coder(monkeypatch, tmp_path):
    case = Case(experiment(), Script(phase3=[phase3_result()], phase4=[ANSWER]), files=FILES)
    selection = StoredSelection(
        tables=("gold.csv",), keywords=("parks",), metadata=METADATA, selection_plan={"table_roles": {"gold.csv": "primary"}},
    )
    result, calls, log = _run(monkeypatch, tmp_path, case, Replay(selection=selection))
    assert result.status == "completed" and result.tables == ["gold.csv"] and result.keywords == ["parks"]
    assert calls == ["phase3_generate_and_execute", "phase4_synthesize"]
    phase3 = log.calls[0]["args"]
    assert phase3["tables"] == ["gold.csv"] and phase3["solr_meta"] == METADATA
    attempt = result.discovery["selection_attempts"][0]
    assert attempt["selection_plan"] == {"table_roles": {"gold.csv": "primary"}}
    assert attempt["selection_plan_source"] == "stored"
    assert result.tokens["p1_p2"] == 0


def test_a_stored_selection_the_coder_rejects_is_not_rediscovered(monkeypatch, tmp_path):
    case = Case(experiment(), Script(phase3=[REJECTION]), files=FILES)
    result, calls, _ = _run(monkeypatch, tmp_path, case, Replay(selection=StoredSelection(("gold.csv",), metadata=METADATA)))
    assert result.status == "failed"
    assert calls.count("phase3_generate_and_execute") == 1 and "phase12_agent" not in calls
    assert [attempt["outcome"] for attempt in result.discovery["selection_attempts"]][-1] != "discovery_started"


def test_stored_keywords_skip_the_keyword_writer(monkeypatch, tmp_path):
    case = Case(
        experiment(discovery_architecture="divided"),
        Script(phase2=[SELECTED], phase3=[phase3_result()], phase4=[ANSWER]), files=FILES,
    )
    replay = Replay(keywords=StoredKeywords(("parks", "district"), raw="raw", reasoning="stored"))
    result, calls, log = _run(monkeypatch, tmp_path, case, replay)
    assert result.status == "completed"
    assert "phase1_generate_keywords" not in calls
    assert log.calls[0]["call"] == "phase2_select_tables"
    assert log.calls[0]["kwargs"]["keywords"] == ["parks", "district"]


def test_a_stored_selection_reads_a_recorded_result():
    results = Path("tests/golden/fixtures/jobs/sample/3687ab4f842a4532a0e665fd61c08c5a.results.jsonl")
    entry = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    selection = StoredSelection.from_result(entry["result"])
    assert selection.tables == tuple(entry["result"]["tables"])
    assert selection.keywords == tuple(entry["result"]["keywords"])
    assert selection.selection_plan is None
    kept = StoredSelection.from_result({
        "discovery": {
            "selected_datasets": ["a.parquet"], "keywords": ["k"],
            "selection_attempts": [{"selection_plan": {"x": 1}}, {"selection_plan": {}}],
        }
    })
    assert kept.tables == ("a.parquet",) and kept.keywords == ("k",) and kept.selection_plan == {"x": 1}


def test_catalogue_metadata_reads_each_tables_document_from_solr():
    queries = []

    class Solr:
        def select(self, tokens, **params):
            queries.append((list(tokens), params))
            rid = tokens[0].split('"')[1]
            docs = {"r1": [{"resource_id": "r1", "title": "First", "columns": [{"name": "year", "type": "int"}]}]}
            return {"response": {"docs": docs.get(rid, [])}}

    metadata = catalogue_metadata(Solr(), ["ds___r1.parquet", "missing.csv"], ["ds___r1.parquet", "missing.csv"])
    assert queries == [(['resource_id:"r1"'], {"rows": 5}), (['resource_id:"missing"'], {"rows": 5})]
    assert metadata["ds___r1.parquet"]["title"] == "First" and metadata["ds___r1.parquet"]["columns.name"] == ["year"]
    assert metadata["missing.csv"] == {}
    assert SelectorKind.STORED.value == "stored"


def test_the_generated_code_lives_under_the_manifests_run_id(monkeypatch, tmp_path):
    """The manifest file and the coding directory of a run carry the same id, so one leads to the other."""
    from lakegen.workflow.observability.manifest import create_manifest

    case = Case(experiment(), Script(phase12=[SELECTED], phase3=[phase3_result()], phase4=[ANSWER]), files=FILES)
    _replacements, log, capture = prepare_run(monkeypatch, tmp_path, service, case, SERVICE_NAMES)
    ctx = RunContext.from_runtime(make_runtime(case.experiment, tmp_path))
    manifest = create_manifest(case.experiment, base_dir=tmp_path, question=case.question, run_id="run-1")
    result = Pipeline(ctx, service.pipeline_deps()).run(case.question, manifest=manifest)
    assert result.status == "completed" and result.manifest["run_id"] == "run-1"
    phase3 = next(call for call in log.calls if call["call"] == "phase3_generate_and_execute")
    assert phase3["kwargs"]["run_dir"].endswith("/coding/run-1")
