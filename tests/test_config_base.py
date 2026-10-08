"""config/base.yaml and the presets compose into valid configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from lakegen.config.files import to_v2
from lakegen.experiment_config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PRESETS = sorted((CONFIG / "presets").glob("*/*.yaml"))


def test_base_is_the_example_configuration_with_its_identity_left_open():
    base = load_experiment_config(CONFIG / "base.yaml").model_dump(mode="json")
    example = load_experiment_config(CONFIG / "experiment.example.yaml").model_dump(mode="json")
    assert base.pop("experiment_id") == "default" and example.pop("experiment_id") == "thesis-baseline-004"
    assert base["benchmark"].pop("path") is None and example["benchmark"].pop("path")
    # The example (mode hybrid) also sets Pneuma-Seeker values that only other
    # modes read; the nested layout refuses those under hybrid, so the base
    # leaves them at the schema's defaults and the mode presets carry them.
    _, dropped = to_v2(example)
    assert set(dropped) >= {"retrieval.pneuma_index_name", "retrieval.pneuma_base_url", "retrieval.scan_workers"}
    for key in dropped:
        section, _, field = key.partition(".")
        del base[section][field], example[section][field]
    assert base == example


@pytest.mark.parametrize("preset", PRESETS, ids=[f"{p.parent.name}/{p.stem}" for p in PRESETS])
def test_every_preset_composes_on_the_base(preset: Path, tmp_path):
    child = tmp_path / "experiment.yaml"
    child.write_text(
        f"schema_version: 2\nextends: [{CONFIG / 'base.yaml'}, {preset}]\nexperiment_id: preset-check\n",
        encoding="utf-8",
    )
    config = load_experiment_config(child)
    assert config.experiment_id == "preset-check"


def test_a_grid_cell_composes_several_presets(tmp_path):
    child = tmp_path / "cell.yaml"
    child.write_text(
        "schema_version: 2\n"
        f"extends: [{CONFIG / 'base.yaml'}, {CONFIG / 'presets/retrieval/pneuma_seeker_reference.yaml'}, "
        f"{CONFIG / 'presets/selector/jev.yaml'}, {CONFIG / 'presets/model/ollama_qwen3_27b.yaml'}]\n"
        "experiment_id: qwen-jev-uk-seeker\nlake: uk\n",
        encoding="utf-8",
    )
    config = load_experiment_config(child)
    assert (config.core, config.model, config.semantic_code_judge_model) == ("uk", "ollama.qwen3.8-27b", "openai.gpt-oss-120b")
    assert config.discovery_architecture.value == "divided" and config.discovery.selector.value == "jev"
    assert config.retrieval.mode.value == "pneuma_seeker" and config.retrieval.pneuma_content_fusion.value == "fill"
    assert config.retrieval.pneuma_entity_model == "openai.gpt-oss-120b" and config.retrieval.pneuma_enumerate_tables is False
