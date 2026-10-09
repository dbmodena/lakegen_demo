"""The manifest records what the environment contributed and what the run used."""

from __future__ import annotations

from lakegen.config import locations as core_config
from lakegen.config.experiment import ExperimentConfig, load_experiment_config
from lakegen.observability.manifest import create_manifest, effective_settings


def test_manifest_records_set_variables_and_the_effective_route(monkeypatch, tmp_path):
    monkeypatch.setattr(core_config, "BASE_DIR", tmp_path)
    monkeypatch.setenv("LAKEGEN_RETRIEVAL_TOP_K", "7")
    monkeypatch.setenv("LAKEGEN_SOLR_CORE_UK", "uk_fixed")
    monkeypatch.setenv("LAKEGEN_PNEUMA_UK_BASE_URL", "http://localhost:8768")
    monkeypatch.setenv("LAKEGEN_GROQ_OFFLOAD", "1")

    config = load_experiment_config(overrides={"core": "uk", "interaction_mode": "autonomous"})
    manifest = create_manifest(config, base_dir=tmp_path, question="How many?", run_id="run-1")

    assert config.retrieval.top_k == 7
    assert manifest.env_overrides == {"retrieval.top_k": 7}
    assert manifest.effective == {
        "model": config.model,
        "solr_core": "uk_fixed",
        "tables_dir": str(tmp_path / "data" / "uk" / "clean_datasets" / "parquets"),
        "pneuma_base_url": "http://localhost:8768",
        "pneuma_index_name": "lakegen-cohere-v4-1024",
        "switches": {
            "selection_cardinality_experiment": False,
            "enable_tabpfn": False,
            "groq_offload": True,
            "together_parallel_tool_calls": True,
            "together_recover_leaks": True,
        },
    }
    assert manifest.model_dump(mode="json")["effective"]["solr_core"] == "uk_fixed"


def test_a_silent_environment_leaves_no_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(core_config, "BASE_DIR", tmp_path)
    manifest = create_manifest(ExperimentConfig(), base_dir=tmp_path, question="q")
    assert manifest.env_overrides == {}
    assert effective_settings(ExperimentConfig())["solr_core"] == "nyc"
    assert manifest.effective["pneuma_base_url"] == "http://localhost:8774"
