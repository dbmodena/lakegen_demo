"""The phase functions by their historical names, resolved on first access.

The phases live in their packages now: ``lakegen.discovery`` (keywords,
divided, unified, orchestrated), ``lakegen.coding.agent`` and
``lakegen.answer.synthesize``; the modules of this package alias them for
the scripts that still import ``lakegen.coding.agent`` and the like.
Resolving the functions on first access keeps importing one phase from
loading the others.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "phase1_generate_keywords": ("lakegen.discovery.keywords", "phase1_generate_keywords"),
    "phase2_select_tables": ("lakegen.discovery.divided", "phase2_select_tables"),
    "phase12_agent": ("lakegen.discovery.unified", "phase12_agent"),
    "Phase3Result": ("lakegen.coding.agent", "Phase3Result"),
    "phase3_generate_and_execute": (
        "lakegen.coding.agent",
        "phase3_generate_and_execute",
    ),
    "phase4_synthesize": ("lakegen.answer.synthesize", "phase4_synthesize"),
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
