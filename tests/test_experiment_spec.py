"""An experiment spec expands to resolved cells; the store remembers what ran."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from lakegen.experiments.spec import parse_slice, select_cases
from lakegen.experiments.spec import ExperimentSpec, compose, expand
from lakegen.experiments.store import RunStore

ROOT = Path(__file__).resolve().parents[1]

SPEC = """
    name: smoke
    description: two lakes x three modes, two passes
    base: [config/base.yaml, config/presets/model/ollama_qwen3_27b.yaml]
    set: {discovery.architecture: divided}
    grid:
      lake: [nyc, uk]
      retrieval.mode: [keyword, pneuma_seeker, hybrid]
    per_mode:
      retrieval.mode:
        pneuma_seeker: {retrieval.seeker.content_fusion: fill, retrieval.seeker.enumerate_tables: false}
        hybrid: {retrieval.hybrid.fusion: rrf}
    per_lake:
      uk: {retrieval.pneuma.base_url: http://localhost:8767}
    questions: {set: "100q_{lake}", passes: [{name: a, slice: ":20"}, {name: b, slice: "20:"}]}
    experiment_id: "{name}-{pass}-{lake}-{retrieval.mode}"
    execution: {backend: api, api_url: http://127.0.0.1:8006, workers: 2, stagger_seconds: 240}
    requires: ["pneuma:{lake}"]
"""


def _spec(tmp_path: Path, text: str = SPEC) -> ExperimentSpec:
    path = tmp_path / "spec.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return ExperimentSpec.load(path)


def test_the_grid_expands_to_lakes_times_modes_times_passes(tmp_path):
    cells = expand(_spec(tmp_path))
    assert len(cells) == 2 * 3 * 2
    first = cells[0]
    assert first.experiment_id == "smoke-a-nyc-keyword"
    assert first.axes == {"core": "nyc", "retrieval.mode": "keyword"}
    assert first.questions_file == ROOT / "benchmark" / "100q_nyc.json" and first.questions_slice == ":20"
    assert first.config.model == "ollama.qwen3.8-27b" and first.config.discovery_architecture.value == "divided"
    assert first.config.benchmark.path == str(ROOT / "benchmark" / "100q_nyc.json")
    assert len({cell.cell_id for cell in cells}) == len(cells)


def test_per_value_settings_apply_only_to_their_grid_value(tmp_path):
    cells = {cell.experiment_id: cell for cell in expand(_spec(tmp_path))}
    seeker, keyword = cells["smoke-b-uk-pneuma_seeker"], cells["smoke-b-uk-keyword"]
    assert seeker.config.retrieval.pneuma_content_fusion.value == "fill"
    assert seeker.config.retrieval.pneuma_enumerate_tables is False
    assert keyword.config.retrieval.pneuma_content_fusion.value == "weighted"
    assert cells["smoke-a-nyc-hybrid"].config.retrieval.fusion_method.value == "rrf"
    assert seeker.config.retrieval.pneuma_base_url == "http://localhost:8767"
    assert cells["smoke-a-nyc-pneuma_seeker"].config.retrieval.pneuma_base_url == "http://localhost:8774"


def test_compose_is_the_document_before_validation(tmp_path):
    document = compose(_spec(tmp_path), {"lake": "uk", "retrieval.mode": "keyword"})
    assert document["core"] == "uk" and document["discovery_architecture"] == "divided"
    assert document["retrieval"]["pneuma_base_url"] == "http://localhost:8767"


def test_an_unknown_template_key_is_an_error(tmp_path):
    spec = _spec(tmp_path, SPEC.replace('experiment_id: "{name}-{pass}-{lake}-{retrieval.mode}"', 'experiment_id: "{name}-{portal}"'))
    with pytest.raises(KeyError, match="'portal'"):
        expand(spec)


def test_a_spec_without_a_grid_is_one_cell(tmp_path):
    spec = _spec(tmp_path, """
        name: single
        base: [config/base.yaml]
        set: {lake: paris, retrieval.mode: semantic}
        questions: {set: "100q_{lake}", slice: ":5"}
        """)
    (cell,) = expand(spec)
    assert cell.experiment_id == "single" and cell.config.core == "paris" and cell.questions_slice == ":5"


def test_question_slices():
    assert parse_slice(None) == slice(None) and parse_slice(":20") == slice(None, 20)
    assert parse_slice("20:") == slice(20, None) and parse_slice("5:10") == slice(5, 10)
    cases = select_cases(ROOT / "benchmark" / "20q_uk.json", "3:5")
    assert len(cases) == 2 and all("question" in case for case in cases)


def test_the_store_records_cells_and_their_jobs(tmp_path):
    spec = _spec(tmp_path)
    cells = expand(spec)
    store = RunStore(tmp_path / "runs")
    store.write_spec(spec)
    record = store.record(cells[0], backend="api", job_dir=Path("/jobs"), job_id="a" * 32)
    assert record.experiment_id == "smoke-a-nyc-keyword" and record.job_id == "a" * 32
    assert (tmp_path / "runs" / "smoke" / "spec.yaml").is_file()
    cell_dir = tmp_path / "runs" / "smoke" / "cells" / cells[0].cell_id
    assert (cell_dir / "config.yaml").is_file()
    assert json.loads((cell_dir / "cell.json").read_text())["pass"] == "a"
    assert store.submitted("smoke") == {cells[0].cell_id}
    assert store.load_index("smoke")[cells[0].cell_id].questions_slice == ":20"
    assert store.submitted("other") == set()
