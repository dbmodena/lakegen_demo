"""The settings of one run, as the engine reads them.

``RunContext`` is what the engine knows about the run: the resolved
experiment, the model, the lake and its tables, the retrieval
configuration. The surfaces build it from their own settings objects
(``RunContext.from_runtime`` reads a ``RuntimeSettings`` or any object with
the same attributes); the engine reads nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lakegen.experiment_config import ExperimentConfig, InteractionMode
from lakegen.retrieval import RetrievalConfig


@dataclass(frozen=True)
class RunContext:
    experiment: ExperimentConfig
    model_name: str
    lake: str
    """The lake's name, which is its Solr core."""
    portal_name: str
    tables_dir: Path
    retrieval: RetrievalConfig

    @classmethod
    def from_runtime(cls, runtime: Any) -> RunContext:
        """From a ``RuntimeSettings`` (or a lighter object with its attributes).

        A runtime without an ``experiment`` (lightweight test objects) gets
        one from its model and core; the public constructors always resolve
        one.
        """
        experiment = getattr(runtime, "experiment", None)
        if experiment is None:
            experiment = ExperimentConfig().model_copy(update={
                "model": runtime.model_name,
                "core": runtime.solr_core,
                "interaction_mode": InteractionMode.AUTONOMOUS,
            })
        return cls(
            experiment=experiment,
            model_name=runtime.model_name,
            lake=runtime.solr_core,
            portal_name=runtime.portal_name,
            tables_dir=Path(runtime.csv_dir),
            retrieval=runtime.retrieval,
        )
