"""Read a configuration a past run recorded, as it was recorded.

A manifest's ``resolved_config`` or an API job's stored config is a flat
document written by the schema of its day. Validating it with today's
``ExperimentConfig`` would reject keys the schema has since dropped (the 958
manifests with ``planner_enabled`` or ``discovery.parallel_inspection_enabled``)
and would fill fields the run never had with today's defaults, so an old run
could claim reranking or cascade search it never used. ``load_recorded_config``
keeps the recorded values, names the unknown keys and names the fields that
were not recorded; nothing is filled in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from pydantic import BaseModel

from lakegen.config.experiment import ExperimentConfig
from lakegen.config.files import nest


@dataclass(frozen=True)
class RecordedConfig:
    values: dict[str, Any] = field(default_factory=dict)
    """Dotted key -> value, for every key the schema knows and the run recorded."""
    unknown: dict[str, Any] = field(default_factory=dict)
    """Dotted key -> value, for keys today's schema does not have."""
    not_recorded: tuple[str, ...] = ()
    """Schema keys the run did not record; their values are unknown, not defaults."""

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def with_todays_defaults(self) -> ExperimentConfig:
        """Today's config for the recorded values, the missing ones at today's defaults.

        Use it to re-run a recorded setup; never to describe what ran, since
        the defaults may differ from what the run used.
        """
        return ExperimentConfig.model_validate(nest(self.values))


@cache
def schema_keys() -> tuple[str, ...]:
    """The dotted keys of ``ExperimentConfig``, nested blocks included."""

    def walk(model: type[BaseModel], prefix: str = ""):
        for name, info in model.model_fields.items():
            annotation = info.annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                yield from walk(annotation, f"{prefix}{name}.")
            else:
                yield f"{prefix}{name}"

    return tuple(walk(ExperimentConfig))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping) and not prefix.rstrip(".") in _LEAF_MAPPINGS:
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


# Schema keys whose value is itself a mapping, not a nested block.
_LEAF_MAPPINGS: frozenset[str] = frozenset()


def load_recorded_config(document: Mapping[str, Any]) -> RecordedConfig:
    """Split a stored flat config into known values, unknown keys and absent fields."""
    known = set(schema_keys())
    flat = _flatten(dict(document))
    values = {key: value for key, value in flat.items() if key in known}
    unknown = {key: value for key, value in flat.items() if key not in known}
    not_recorded = tuple(key for key in schema_keys() if key not in flat)
    return RecordedConfig(values=values, unknown=unknown, not_recorded=not_recorded)
