"""The settings of one run, as the engine reads them.

``RunContext`` is what the engine knows about the run: the resolved
experiment, the model, the lake and its tables, the retrieval
configuration. The surfaces build it from their own settings objects
(``RunContext.from_runtime`` reads a ``RuntimeSettings`` or any object with
the same attributes); the engine reads nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lakegen.config.experiment import ExperimentConfig, InteractionMode
from lakegen.retrieval import RetrievalConfig


@dataclass(frozen=True)
class SurfaceOptions:
    """How the surface driving the run wants it to behave, beyond the experiment.

    Attributes:
        human_gated: Ask the driver at the gates (keywords, tables, a coder rejection).
        answer_on_failure: Synthesise an answer from the failure when no code succeeded.
    """

    human_gated: bool = False
    answer_on_failure: bool = False


@dataclass(frozen=True)
class RunContext:
    """What the engine knows about the run.

    Attributes:
        experiment: The resolved experiment config.
        model_name: The model name.
        lake: The lake's name, which is its Solr core.
        portal_name: The portal's display name.
        tables_dir: The directory of the lake's tables.
        retrieval: The retrieval configuration.
        surface: How the surface driving the run wants it to behave.
    """

    experiment: ExperimentConfig
    model_name: str
    lake: str
    portal_name: str
    tables_dir: Path
    retrieval: RetrievalConfig
    surface: SurfaceOptions = field(default_factory=SurfaceOptions)

    @classmethod
    def from_runtime(cls, runtime: Any, surface: SurfaceOptions | None = None) -> RunContext:
        """From a ``RuntimeSettings`` (or a lighter object with its attributes).

        A runtime without an ``experiment`` (lightweight test objects) gets
        one from its model and core; the public constructors always resolve
        one.

        Args:
            runtime: The surface's settings object.
            surface: How the surface wants the run to behave; the defaults when None.

        Returns:
            The run's context.
        """
        experiment = getattr(runtime, "experiment", None)
        if experiment is None:
            experiment = ExperimentConfig().model_copy(
                update={
                    "model": runtime.model_name,
                    "core": runtime.solr_core,
                    "interaction_mode": InteractionMode.AUTONOMOUS,
                }
            )
        return cls(
            experiment=experiment,
            model_name=runtime.model_name,
            lake=runtime.solr_core,
            portal_name=runtime.portal_name,
            tables_dir=Path(runtime.csv_dir),
            retrieval=runtime.retrieval,
            surface=surface or SurfaceOptions(),
        )
