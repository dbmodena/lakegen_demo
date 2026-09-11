"""Public phase API without eager imports.

Importing a phase helper (for example ``lakegen.phases.utils``) must not load
every workflow phase.  In particular, Phase 12 imports its tools while those
tools also use the shared phase utilities.  Resolving the public functions on
first access keeps that dependency graph acyclic while preserving the existing
``from lakegen.phases import ...`` API.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "phase1_generate_keywords": ("lakegen.phases.phase1", "phase1_generate_keywords"),
    "phase2_select_tables": ("lakegen.phases.phase2", "phase2_select_tables"),
    "phase12_agent": ("lakegen.phases.phase12", "phase12_agent"),
    "Phase3Result": ("lakegen.phases.phase3", "Phase3Result"),
    "phase3_generate_and_execute": (
        "lakegen.phases.phase3",
        "phase3_generate_and_execute",
    ),
    "phase4_synthesize": ("lakegen.phases.phase4", "phase4_synthesize"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
