"""Goldens of configuration loading.

Every experiment YAML in the repository and in the (git-ignored) experiment
directories must keep resolving to the same ExperimentConfig, and every
resolved config stored with an API job must keep loading. The snapshot of the
stored configs records what loading changes about them (fields added with a
default, values normalised), which is what a later config migration must
preserve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lakegen.config.experiment import ExperimentConfig, load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "configs"
YAMLS = sorted(ROOT.glob("config/examples/experiment.example*.yaml")) + sorted(FIXTURES.glob("experiments/*/*.yaml"))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def _changes(stored: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    before, after = _flatten(stored), _flatten(loaded)
    return {
        key: {"stored": before.get(key, "<absent>"), "loaded": after.get(key, "<absent>")}
        for key in sorted(set(before) | set(after))
        if before.get(key, "<absent>") != after.get(key, "<absent>")
    }


def _group(path: Path) -> str:
    """The folder a snapshot is filed under: the examples keep their old name, ``config``."""
    return "config" if path.parent.name == "examples" else path.parent.name


@pytest.mark.parametrize("path", YAMLS, ids=[f"{_group(p)}/{p.name}" for p in YAMLS])
def test_config_file_resolves_the_same(path: Path, golden):
    config = load_experiment_config(path)
    golden(f"config/{_group(path)}__{path.stem}", config.model_dump(mode="json"))


def test_stored_job_configs_keep_loading(golden):
    stored = json.loads((FIXTURES / "stored_job_configs.json").read_text(encoding="utf-8"))
    report = {}
    for key, entry in stored.items():
        loaded = ExperimentConfig.model_validate(entry["config"]).model_dump(mode="json")
        report[key] = {"jobs": entry["jobs"], "changes": _changes(entry["config"], loaded)}
    golden("config/stored_job_configs", report)
