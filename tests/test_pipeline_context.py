"""The engine reads its settings from a RunContext and its phases from a PipelineDeps."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lakegen.experiment_config import ExperimentConfig, InteractionMode
from lakegen.pipeline.context import RunContext
from lakegen.pipeline.deps import DEPENDENCY_NAMES, PipelineDeps
from lakegen.retrieval import RetrievalConfig


def test_from_module_takes_the_names_as_the_module_has_them_now():
    module = ModuleType("surface")
    for name in DEPENDENCY_NAMES:
        setattr(module, name, lambda *a, _name=name, **k: _name)
    deps = PipelineDeps.from_module(module)
    assert deps.phase12_agent() == "phase12_agent" and deps.persist_manifest() == "persist_manifest"

    module.get_llm = lambda model: ("patched", model)  # a later patch is seen by a later from_module
    assert PipelineDeps.from_module(module).get_llm("m") == ("patched", "m")
    assert deps.get_llm("m") == "get_llm"

    del module.phase4_synthesize
    with pytest.raises(AttributeError):
        PipelineDeps.from_module(module)


def test_the_default_deps_are_the_real_phases_and_resources():
    import lakegen.service as service

    deps = PipelineDeps.default()
    for name in DEPENDENCY_NAMES:
        assert getattr(deps, name) is getattr(service, name), name


def test_run_context_from_a_runtime_and_from_a_lightweight_object(tmp_path: Path):
    config = ExperimentConfig(experiment_id="x", core="uk", model="openai.gpt-oss-120b")
    runtime = SimpleNamespace(
        model_name=config.model, solr_core="uk", csv_dir=str(tmp_path), portal_name="UK",
        retrieval=config.retrieval.to_runtime(), experiment=config,
    )
    ctx = RunContext.from_runtime(runtime)
    assert ctx.experiment is config and ctx.lake == "uk" and ctx.portal_name == "UK"
    assert ctx.tables_dir == tmp_path and isinstance(ctx.retrieval, RetrievalConfig)

    bare = SimpleNamespace(
        model_name="openai.gpt-oss-120b", solr_core="nyc", csv_dir=tmp_path, portal_name="NYC",
        retrieval=RetrievalConfig(),
    )
    ctx = RunContext.from_runtime(bare)
    assert ctx.experiment.model == "openai.gpt-oss-120b" and ctx.experiment.core == "nyc"
    assert ctx.experiment.interaction_mode == InteractionMode.AUTONOMOUS
