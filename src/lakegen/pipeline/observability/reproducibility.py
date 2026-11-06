"""Per-run reproducibility controls that do not mutate process-global RNG state."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReproducibilityContext:
    """Local random generators and an honest description of their coverage.

    Attributes:
        configured_seed: The seed the config asked for.
        effective_seed: The seed the generators were built with.
        python_random: The run's own ``random.Random``.
        numpy_rng: The run's own NumPy generator.
    """

    configured_seed: int
    effective_seed: int
    python_random: random.Random
    numpy_rng: np.random.Generator

    def telemetry(
        self, *, generated_code_seed_instruction_provided: bool = False
    ) -> dict[str, Any]:
        """What the seed controls in this run, and what it does not.

        Args:
            generated_code_seed_instruction_provided: Whether the code generator
                was told to use the seed.

        Returns:
            The telemetry record: the seeds, the components seeded, and the
            ones (the LLM's generation) no seed controls.
        """
        initialized_components = [
            "python_random_local_generator",
            "numpy_local_generator",
        ]
        instructions_provided_to = (
            ["code_generator"] if generated_code_seed_instruction_provided else []
        )
        return {
            "configured_seed": self.configured_seed,
            "effective_seed": self.effective_seed,
            "initialized_components": initialized_components,
            "seed_applied_to": [],
            "instructions_provided_to": instructions_provided_to,
            "generated_code_seed_instruction_provided": (generated_code_seed_instruction_provided),
            "generated_code_seed_usage_verified": False,
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
        # A seeded generator for reproducible sampling, not for security.
        python_random=random.Random(seed),  # nosec B311
        numpy_rng=np.random.default_rng(seed),
    )
