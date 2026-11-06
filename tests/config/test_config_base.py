"""config/base.yaml and the presets compose into valid configurations."""

from __future__ import annotations

from pathlib import Path

import pytest

from lakegen.config.experiment import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
PRESETS = sorted((CONFIG / "presets").glob("*/*.yaml"))


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
        f"{CONFIG / 'presets/selector/memgpt.yaml'}, {CONFIG / 'presets/model/ollama_qwen3_27b.yaml'}]\n"
        "experiment_id: qwen-memgpt-uk-seeker\nlake: uk\n",
        encoding="utf-8",
    )
    config = load_experiment_config(child)
    assert (config.core, config.model, config.semantic_code_judge_model) == (
        "uk",
        "ollama.qwen3.8-27b",
        "openai.gpt-oss-120b",
    )
    assert (
        config.discovery_architecture.value == "divided"
        and config.discovery.selector.value == "memgpt"
    )
    assert (
        config.retrieval.mode.value == "pneuma_seeker"
        and config.retrieval.pneuma_content_fusion.value == "fill"
    )
    assert (
        config.retrieval.pneuma_entity_model == "openai.gpt-oss-120b"
        and config.retrieval.pneuma_enumerate_tables is False
    )
