"""Per-run reproducibility controls that do not mutate process-global RNG state."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReproducibilityContext:
    """Local random generators and an honest description of their coverage."""

    configured_seed: int
    effective_seed: int
    python_random: random.Random
    numpy_rng: np.random.Generator

    def telemetry(
        self, *, generated_code_instructions_applied: bool = False
    ) -> dict[str, Any]:
        applied_to = [
            "python_random_local_generator",
            "numpy_local_generator",
        ]
        if generated_code_instructions_applied:
            applied_to.append("generated_code_instructions")
        return {
            "configured_seed": self.configured_seed,
            "effective_seed": self.effective_seed,
            "applied_to": applied_to,
            "llm_provider_seed_supported": False,
            "llm_provider_seed_applied": False,
            "deterministic_llm_generation": False,
            "uncontrolled_components": ["oci_llm_generation"],
        }


def initialize_reproducibility(seed: int) -> ReproducibilityContext:
    """Create independent RNGs for one run without calling global seed APIs."""

    return ReproducibilityContext(
        configured_seed=seed,
        effective_seed=seed,
        python_random=random.Random(seed),
        numpy_rng=np.random.default_rng(seed),
    )
