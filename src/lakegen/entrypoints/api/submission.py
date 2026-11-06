"""How a submission becomes a run: its configuration, and how many runs at once.

A direct query and a batch (``experiment_runner.batches``) resolve their
configuration the same way, and share one limit on the questions a process
answers at once.
"""

from __future__ import annotations

import threading
from typing import Any

from lakegen.config.experiment import (
    DiscoveryArchitecture,
    ExperimentConfig,
    load_experiment_config,
)
from lakegen.config.runtime import MODEL_OPTIONS, SOLR_CORE_OPTIONS
from lakegen.config.settings import positive_int_or_default, read_var
from lakegen.retrieval import RetrievalMode

DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_CORE = SOLR_CORE_OPTIONS[0]


def positive_int_from_env(name: str, default: int) -> int:
    """Read a positive worker limit without making API startup fragile."""
    return read_var(name, positive_int_or_default, default)


# A workflow may already parallelise table scans internally. Keep the safe
# historical default, while allowing a bounded number of questions/batches in flight.
WORKFLOW_WORKERS = positive_int_from_env("LAKEGEN_WORKFLOW_WORKERS", 1)
WORKFLOW_LOCK = threading.BoundedSemaphore(WORKFLOW_WORKERS)


def resolve_config(
    *,
    core: str,
    model: str,
    retrieval_mode: RetrievalMode | str,
    top_k: int,
    hybrid_alpha: float,
    candidate_multiplier: int,
    discovery_architecture: DiscoveryArchitecture | str = DiscoveryArchitecture.UNIFIED,
    experiment_id: str = "default",
    seed: int = 0,
    config_data: dict[str, Any] | None = None,
    explicit_fields: set[str] | None = None,
) -> ExperimentConfig:
    """The experiment configuration of a submission: its inline config under its explicit fields.

    Only the fields the submission set explicitly override the inline config
    (or the defaults), so a field left at its default does not mask the
    config's value. The interaction mode is always autonomous.

    Args:
        core: The Solr core to search.
        model: The pipeline's model.
        retrieval_mode: The retrieval mode (``retrieval.mode``).
        top_k: How many tables retrieval returns (``retrieval.top_k``).
        hybrid_alpha: The lexical weight of weighted hybrid fusion (``retrieval.alpha``).
        candidate_multiplier: The per-branch pool size as a multiple of ``top_k``.
        discovery_architecture: The discovery architecture.
        experiment_id: The experiment id the runs record.
        seed: The seed.
        config_data: The submission's inline config, if any.
        explicit_fields: The names of the fields the submission set; None means
            all of them.

    Returns:
        The validated experiment configuration.
    """
    explicit = (
        {
            "core",
            "model",
            "retrieval_mode",
            "top_k",
            "hybrid_alpha",
            "candidate_multiplier",
            "discovery_architecture",
            "experiment_id",
            "seed",
        }
        if explicit_fields is None
        else explicit_fields
    )
    possible = {
        "core": core,
        "model": model,
        "retrieval.mode": retrieval_mode,
        "retrieval.top_k": top_k,
        "retrieval.alpha": hybrid_alpha,
        "retrieval.candidate_multiplier": candidate_multiplier,
        "discovery_architecture": discovery_architecture,
        "experiment_id": experiment_id,
        "seed": seed,
        "interaction_mode": "autonomous",
    }
    aliases = {
        "retrieval_mode": "retrieval.mode",
        "top_k": "retrieval.top_k",
        "hybrid_alpha": "retrieval.alpha",
        "candidate_multiplier": "retrieval.candidate_multiplier",
    }
    override_keys = {aliases.get(name, name) for name in explicit}
    override_keys.add("interaction_mode")
    overrides = {key: value for key, value in possible.items() if key in override_keys}
    return load_experiment_config(
        data_override=config_data,
        overrides=overrides,
    )
