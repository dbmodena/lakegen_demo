"""The settings of one run, as the three surfaces build them today.

``RuntimeSettings`` carries the model, the lake (``solr_core``), the tables
directory, the retrieval configuration and the resolved experiment. It is
chainlit-free so that the service, the API and the CLI can import it; the UI
re-exports it from ``ui.state`` together with the option lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lakegen.config.registry import get_lake, lake_names, lakes, model_names
from lakegen.core.config import resolve_portal_tables_dir
from lakegen.experiment_config import (
    DiscoveryArchitecture,
    DiscoveryConfig,
    ExperimentConfig,
    InteractionMode,
    load_experiment_config,
)
from lakegen.retrieval import RetrievalConfig, RetrievalMode

# The option lists the interfaces show come from the registries.
MODEL_OPTIONS = list(model_names())
SOLR_CORE_OPTIONS = list(lake_names())
SOLR_CORE_PORTAL_NAMES = {lake.name: lake.title for lake in lakes()}
RETRIEVAL_MODE_OPTIONS = [mode.value for mode in RetrievalMode]
# Dropdown label -> mode value; the label is what people see.
RETRIEVAL_MODE_ITEMS = {mode.label: mode.value for mode in RetrievalMode}


@dataclass
class RuntimeSettings:
    model_name: str = MODEL_OPTIONS[0]
    solr_core: str = SOLR_CORE_OPTIONS[0]
    csv_dir: Path = field(default_factory=lambda: resolve_portal_tables_dir("nyc"))
    db_path: Path = field(default_factory=lambda: get_lake("nyc").blend_db_path)
    use_unified_agent: bool = True
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig.from_env)
    experiment: ExperimentConfig | None = None

    def __post_init__(self) -> None:
        self.retrieval = self.retrieval.for_portal(self.solr_core)
        if self.experiment is None:
            self.experiment = load_experiment_config(overrides={
                "model": self.model_name,
                "core": self.solr_core,
                "discovery_architecture": (
                    DiscoveryArchitecture.UNIFIED.value
                    if self.use_unified_agent
                    else DiscoveryArchitecture.DIVIDED.value
                ),
                "interaction_mode": InteractionMode.HUMAN_GATED.value,
                **{
                    f"retrieval.{key}": value
                    for key, value in self.retrieval.__dict__.items()
                },
            })

    @property
    def portal_name(self) -> str:
        return SOLR_CORE_PORTAL_NAMES.get(self.solr_core, self.solr_core)

    @property
    def discovery(self) -> DiscoveryConfig:
        """Discovery bounds for this run; ``__post_init__`` always resolves one."""
        assert self.experiment is not None
        return self.experiment.discovery

    @classmethod
    def default(cls) -> RuntimeSettings:
        return cls()

    @classmethod
    def from_chainlit_settings(
        cls,
        settings: dict[str, Any],
        *,
        solr_core: str | None = None,
    ) -> RuntimeSettings:
        default = cls()

        selected_solr_core = str(
            solr_core or settings.get("solr_core") or default.solr_core
        )
        if selected_solr_core not in SOLR_CORE_OPTIONS:
            selected_solr_core = default.solr_core

        model_name = str(settings.get("model_name") or default.model_name)
        if model_name not in MODEL_OPTIONS:
            model_name = default.model_name

        use_unified_agent = settings.get("use_unified_agent", default.use_unified_agent)
        retrieval_mode = str(
            settings.get("retrieval_mode") or default.retrieval.mode
        )
        if retrieval_mode not in RETRIEVAL_MODE_OPTIONS:
            retrieval_mode = default.retrieval.mode
        
        retrieval = RetrievalConfig.from_env(mode=retrieval_mode).for_portal(
            selected_solr_core
        )
        experiment = load_experiment_config(overrides={
            "model": model_name,
            "core": selected_solr_core,
            "discovery_architecture": (
                "unified" if bool(use_unified_agent) else "divided"
            ),
            "interaction_mode": "human_gated",
            **{f"retrieval.{key}": value for key, value in retrieval.__dict__.items()},
        })
        return cls(
            model_name=model_name,
            solr_core=selected_solr_core,
            csv_dir=resolve_portal_tables_dir(selected_solr_core),
            db_path=get_lake(selected_solr_core).blend_db_path,
            use_unified_agent=bool(use_unified_agent),
            retrieval=retrieval,
            experiment=experiment,
        )
