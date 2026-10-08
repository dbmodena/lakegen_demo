"""The settings of one run, as the engine reads them.

``RunContext`` is what the engine knows about the run: the resolved
experiment, the model, the lake and its tables, the retrieval
configuration. The surfaces build it from their own settings objects
(``RunContext.from_runtime`` reads a ``RuntimeSettings`` or any object with
the same attributes); the engine reads nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from lakegen.experiment_config import ExperimentConfig, InteractionMode
from lakegen.retrieval import RetrievalConfig


class MemoryScope(StrEnum):
    """Where the discovery agent's keyword memory lives for a run."""

    DURABLE = "durable"
    """The lake's memory file; what the batch service uses."""
    SESSION = "session"
    """Kept for the run only, as the Chainlit app does: what a round learns reaches the next, nothing is written."""
    NONE = "none"
    """No memory: each run starts blank, as the step-by-step CLI did."""


@dataclass(frozen=True)
class SurfaceOptions:
    """How the surface driving the run wants it to behave, beyond the experiment."""

    human_gated: bool = False
    """Ask the driver at the gates (keywords, tables, a coder rejection)."""
    memory_scope: MemoryScope = MemoryScope.DURABLE
    answer_on_failure: bool = False
    """Synthesise an answer from the failure when no code succeeded."""


@dataclass(frozen=True)
class RunContext:
    experiment: ExperimentConfig
    model_name: str
    lake: str
    """The lake's name, which is its Solr core."""
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
            surface=surface or SurfaceOptions(),
        )
