"""The engine reads its settings from a RunContext and its phases from a PipelineDeps."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from lakegen.config.experiment import ExperimentConfig, InteractionMode
from lakegen.pipeline.context import RunContext
from lakegen.pipeline.deps import PipelineDeps
from lakegen.retrieval import RetrievalConfig

FIELDS = tuple(field.name for field in fields(PipelineDeps))


def test_each_surface_binds_its_own_names_when_a_run_starts(monkeypatch):
    """A test that replaces a name on a surface replaces what the engine calls."""
    import lakegen.service as service
    from lakegen.interfaces.cli import ask
    from lakegen.ui import workflow

    for surface in (service, ask, workflow):
        deps = surface.pipeline_deps()
        for name in FIELDS:
            assert getattr(deps, name) is getattr(surface, name), (surface.__name__, name)
        original = surface.get_llm
        monkeypatch.setattr(surface, "get_llm", lambda model: ("patched", model))
        assert surface.pipeline_deps().get_llm("m") == ("patched", "m")
        assert deps.get_llm is original  # deps built earlier keep what they were given


def test_the_default_deps_are_the_real_phases_and_resources():
    import lakegen.service as service

    deps = PipelineDeps.default()
    for name in FIELDS:
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
