"""Configuration files: the v2 layout resolves to the flat schema and back."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lakegen.config.experiment import ExperimentConfig, load_experiment_config
from lakegen.config.files import (
    V2_PATHS,
    ConfigFileError,
    flatten_document,
    load_document,
    parse_set_values,
    to_v2,
    translate_override,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_every_v2_path_is_unique():
    assert len(set(V2_PATHS.values())) == len(V2_PATHS)


def test_v2_document_resolves_to_the_flat_schema():
    flat = flatten_document({
        "schema_version": 2,
        "experiment_id": "x",
        "lake": "chicago",
        "discovery": {"architecture": "divided", "selector": "jev", "budget": {"initial_candidates": 12}},
        "retrieval": {"mode": "hybrid", "top_k": 20, "hybrid": {"fusion": "rrf", "alpha": 0.3}, "rerank": {"depth": 10}},
        "coding": {"context_level": "schema_only"},
        "judge": {"model": "together.gpt-oss-120b"},
        "interaction": {"mode": "autonomous"},
    })
    assert flat == {
        "experiment_id": "x",
        "core": "chicago",
        "discovery_architecture": "divided",
        "discovery": {"selector": "jev", "initial_candidates": 12},
        "retrieval": {"mode": "hybrid", "top_k": 20, "fusion_method": "rrf", "alpha": 0.3, "rerank_depth": 10},
        "coder_context_level": "schema_only",
        "semantic_code_judge_model": "together.gpt-oss-120b",
        "interaction_mode": "autonomous",
    }
    ExperimentConfig.model_validate(flat)  # the flat form is what the schema validates


def test_a_mode_block_is_rejected_unless_the_mode_reads_it():
    with pytest.raises(ConfigFileError, match="retrieval.seeker applies to pneuma_seeker, not to mode 'hybrid'"):
        flatten_document({"schema_version": 2, "retrieval": {"mode": "hybrid", "seeker": {"content_weight": 0.5}}})
    flatten_document({"schema_version": 2, "retrieval": {"mode": "pneuma_seeker", "seeker": {"content_weight": 0.5}, "pneuma": {"timeout_seconds": 60}}})
    flatten_document({"schema_version": 2, "retrieval": {"mode": "hybrid_duckdb_semantic", "hybrid": {"fusion": "rrf"}, "duckdb": {"max_files": 10}}})


def test_flat_documents_pass_through_and_unknown_versions_fail():
    assert flatten_document({"core": "nyc", "retrieval": {"top_k": 5}}) == {"core": "nyc", "retrieval": {"top_k": 5}}
    with pytest.raises(ConfigFileError, match="unknown schema_version"):
        flatten_document({"schema_version": 3})


def test_extends_composes_files_in_order_and_refuses_cycles(tmp_path):
    _write(tmp_path / "base.yaml", """
        schema_version: 2
        experiment_id: base
        retrieval: {mode: keyword, top_k: 10}
        """)
    _write(tmp_path / "preset.yaml", """
        schema_version: 2
        retrieval: {top_k: 30}
        """)
    child = _write(tmp_path / "child.yaml", """
        schema_version: 2
        extends: [base.yaml, preset.yaml]
        lake: uk
        """)
    assert load_document(child) == {"experiment_id": "base", "retrieval": {"mode": "keyword", "top_k": 30}, "core": "uk"}
    _write(tmp_path / "a.yaml", "extends: [b.yaml]\n")
    _write(tmp_path / "b.yaml", "extends: [a.yaml]\n")
    with pytest.raises(ConfigFileError, match="extends cycle"):
        load_document(tmp_path / "a.yaml")


def test_set_values_accept_either_spelling_and_read_yaml():
    assert translate_override("retrieval.hybrid.alpha") == "retrieval.alpha"
    assert translate_override("retrieval.alpha") == "retrieval.alpha"
    assert translate_override("discovery.selector.kind") == "discovery.selector"
    assert parse_set_values(["retrieval.hybrid.alpha=0.3", 'model="together.gpt-oss-120b"', "coding.automatic_test_coder=false"]) == {
        "retrieval.alpha": 0.3, "model": "together.gpt-oss-120b", "automatic_test_coder": False,
    }
    with pytest.raises(ConfigFileError, match="expected PATH=VALUE"):
        parse_set_values(["nonsense"])


def test_v2_rendering_round_trips_and_names_what_it_drops():
    config = load_experiment_config(ROOT / "config" / "experiment.example_uk_pneuma_seeker.yaml")
    flat = config.model_dump(mode="json")
    document, dropped = to_v2(flat)
    assert document["lake"] == "uk" and document["retrieval"]["mode"] == "pneuma_seeker"
    assert "seeker" in document["retrieval"] and "hybrid" not in document["retrieval"]
    assert set(dropped) >= {"retrieval.alpha", "retrieval.duckdb_max_files"}
    reloaded = ExperimentConfig.model_validate(flatten_document(document)).model_dump(mode="json")
    for key, value in dropped.items():  # the dropped settings are the schema's defaults for this mode
        section, _, field = key.partition(".")
        reloaded.setdefault(section, {})[field] = value
    assert reloaded == flat
