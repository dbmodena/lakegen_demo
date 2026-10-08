"""The two retrieval configuration classes and the three default sets stay in step.

``RetrievalConfig`` (a dataclass, used at run time) and
``RetrievalExperimentConfig`` (a pydantic model, used by the experiment
config) list the same fields. Their defaults differ on exactly three fields,
which are also exactly where ``RetrievalConfig.from_env`` differs from the
dataclass: the production defaults. This test pins all of that, so a field
added to one class without the other, or a changed default, fails here.
"""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import get_type_hints

from lakegen.experiment_config import RetrievalExperimentConfig
from lakegen.retrieval.config import RetrievalConfig

# field -> (dataclass default, experiment-config default)
EXPERIMENT_OVERRIDES = {
    "scan_workers": (8, 16),
    "rerank_model": (None, "cohere.rerank-v4.0-fast"),
    "pneuma_index_name": ("lakegen", "lakegen-cohere-v4-1024"),
}
# field -> (dataclass default, from_env default)
ENVIRONMENT_OVERRIDES = {
    "scan_workers": (8, 16),
    "rerank_model": (None, "cohere.rerank-v4.0-fast"),
    "pneuma_index_name": ("lakegen", "lakegen-cohere-v4-1024"),
}


def _dataclass_defaults() -> dict[str, object]:
    found = {}
    for item in fields(RetrievalConfig):
        found[item.name] = item.default if item.default is not MISSING else item.default_factory()
    return found


def test_same_fields_in_the_same_order():
    assert [item.name for item in fields(RetrievalConfig)] == list(RetrievalExperimentConfig.model_fields)


def test_same_types():
    hints = get_type_hints(RetrievalConfig)
    for name, info in RetrievalExperimentConfig.model_fields.items():
        assert info.annotation == hints[name], name


def test_defaults_differ_only_where_documented():
    runtime = _dataclass_defaults()
    experiment = {name: info.default for name, info in RetrievalExperimentConfig.model_fields.items()}
    differing = {name for name in runtime if runtime[name] != experiment[name]}
    assert differing == set(EXPERIMENT_OVERRIDES)
    for name, (runtime_value, experiment_value) in EXPERIMENT_OVERRIDES.items():
        assert (runtime[name], experiment[name]) == (runtime_value, experiment_value), name


def test_from_env_defaults_differ_only_where_documented():
    # The suite scrubs every LAKEGEN_* variable, so from_env yields its defaults.
    runtime = RetrievalConfig().__dict__
    production = RetrievalConfig.from_env().__dict__
    differing = {name for name in runtime if runtime[name] != production[name]}
    assert differing == set(ENVIRONMENT_OVERRIDES)
    for name, (runtime_value, production_value) in ENVIRONMENT_OVERRIDES.items():
        assert (runtime[name], production[name]) == (runtime_value, production_value), name


def test_round_trip_between_the_two_classes():
    production = RetrievalConfig.from_env()
    assert RetrievalExperimentConfig.from_runtime(production).to_runtime() == production
    assert RetrievalExperimentConfig().to_runtime() == RetrievalConfig(
        scan_workers=16, rerank_model="cohere.rerank-v4.0-fast", pneuma_index_name="lakegen-cohere-v4-1024",
    )
    # The experiment config's static defaults are the production defaults.
    assert RetrievalExperimentConfig().to_runtime() == RetrievalConfig.from_env()
