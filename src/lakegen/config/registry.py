"""What exists: the lakes and the models, read from config/lakes.yaml and config/models.yaml.

A ``Lake`` knows where its tables, metadata and blend database live, which
Solr core serves it and which Pneuma service holds its index; a ``ModelSpec``
knows a model's provider, the provider's own id and the limits the client is
built with. Adding a lake or a model is one entry in the YAML file; the
interfaces, the experiment config and the provider clients all read the
registry, so no list of names exists anywhere else.

The files are read once per process (``reset()`` clears that for tests).
Environment overrides (LAKEGEN_SOLR_CORE_<LAKE>, LAKEGEN_PNEUMA_<LAKE>_*)
are applied when a lake is asked for its core or route, never cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import yaml

from lakegen.config.settings import pneuma_override, solr_core_override
from lakegen.core.config import BASE_DIR

CONFIG_DIR = BASE_DIR / "config"
TABLE_SUFFIXES = {".parquet", ".pq", ".csv"}
PROVIDERS = ("oci", "ollama", "together")


@dataclass(frozen=True)
class Lake:
    name: str
    title: str
    pneuma_url: str
    pneuma_index: str
    dataset_roots: tuple[str, ...]
    table_dirs: tuple[str, ...]
    metadata_files: tuple[str, ...]
    statistics_file: str
    starters: str | None
    reference_questions: str | None

    @property
    def solr_core(self) -> str:
        """The core that serves the lake: its override, else its own name."""
        return solr_core_override(self.name) or self.name

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data" / self.name

    def pneuma_route(self) -> tuple[str, str]:
        """The lake's Pneuma (base URL, index name), with the environment's overrides."""
        url, index = pneuma_override(self.name)
        return (self.pneuma_url if url is None else url, self.pneuma_index if index is None else index)

    @property
    def tables_dir(self) -> Path:
        """The first candidate directory holding table files; else the first that exists; else the first."""
        candidates = tuple(
            self.data_dir / root / name for root in self.dataset_roots for name in self.table_dirs
        )
        for candidate in candidates:
            if candidate.is_dir() and any(
                entry.is_file() and entry.suffix.casefold() in TABLE_SUFFIXES
                for entry in candidate.iterdir()
            ):
                return candidate
        return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])

    @property
    def metadata_path(self) -> Path:
        """The first metadata catalogue that exists, else the preferred name."""
        metadata_dir = self.data_dir / "metadata"
        return next(
            (metadata_dir / name for name in self.metadata_files if (metadata_dir / name).is_file()),
            metadata_dir / self.metadata_files[0],
        )

    @property
    def statistics_path(self) -> Path:
        return self.data_dir / "metadata" / self.statistics_file

    @property
    def blend_db_path(self) -> Path:
        return BASE_DIR / "data" / f"blend_{self.name}.db"

    @property
    def starters_path(self) -> Path | None:
        return BASE_DIR / self.starters if self.starters else None

    @property
    def reference_questions_path(self) -> Path | None:
        return BASE_DIR / self.reference_questions if self.reference_questions else None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    provider_model: str
    context_window: int | None
    max_output_tokens: int
    temperature: float
    timeout_seconds: float | None
    groq_model: str | None


@cache
def _document(file_name: str) -> dict[str, Any]:
    path = CONFIG_DIR / file_name
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a mapping")
    return loaded


def reset() -> None:
    """Forget the loaded files, so a test can point CONFIG_DIR elsewhere."""
    _document.cache_clear()
    lakes.cache_clear()
    models.cache_clear()


def _known(kind: str, name: str, names: tuple[str, ...]) -> KeyError:
    return KeyError(f"unknown {kind} {name!r}; known: {', '.join(names)}")


@lru_cache(maxsize=1)
def lakes() -> tuple[Lake, ...]:
    document = _document("lakes.yaml")
    defaults = document.get("defaults", {})
    found = []
    for name, entry in document["lakes"].items():
        entry = {**defaults, **(entry or {})}
        found.append(Lake(
            name=name,
            title=entry["title"],
            pneuma_url=entry["pneuma_url"],
            pneuma_index=entry["pneuma_index"],
            dataset_roots=tuple(entry["dataset_roots"]),
            table_dirs=tuple(entry["table_dirs"]),
            metadata_files=tuple(entry["metadata_files"]),
            statistics_file=entry["statistics_file"],
            starters=entry.get("starters"),
            reference_questions=entry.get("reference_questions"),
        ))
    return tuple(found)


def lake_names() -> tuple[str, ...]:
    return tuple(lake.name for lake in lakes())


def get_lake(name: str) -> Lake:
    for lake in lakes():
        if lake.name == name:
            return lake
    raise _known("lake", name, lake_names())


def pneuma_routes() -> dict[str, tuple[str, str]]:
    """The built-in Pneuma routes by lake, without the environment's overrides."""
    return {lake.name: (lake.pneuma_url, lake.pneuma_index) for lake in lakes()}


@lru_cache(maxsize=1)
def models() -> tuple[ModelSpec, ...]:
    document = _document("models.yaml")
    defaults = document.get("defaults", {})
    found = []
    for name, entry in document["models"].items():
        entry = {**defaults, **(entry or {})}
        if entry["provider"] not in PROVIDERS:
            raise ValueError(f"model {name!r}: unknown provider {entry['provider']!r}; known: {', '.join(PROVIDERS)}")
        found.append(ModelSpec(
            name=name,
            provider=entry["provider"],
            provider_model=entry.get("provider_model", name),
            context_window=entry.get("context_window"),
            max_output_tokens=entry["max_output_tokens"],
            temperature=float(entry["temperature"]),
            timeout_seconds=entry.get("timeout_seconds"),
            groq_model=entry.get("groq_model"),
        ))
    return tuple(found)


def model_names() -> tuple[str, ...]:
    return tuple(model.name for model in models())


def get_model(name: str) -> ModelSpec:
    for model in models():
        if model.name == name:
            return model
    raise _known("model", name, model_names())


def models_by_provider(provider: str) -> dict[str, str]:
    """LakeGen model name -> the provider's own id, for one provider."""
    return {model.name: model.provider_model for model in models() if model.provider == provider}


def groq_models() -> dict[str, str]:
    """LakeGen model name -> Groq id, for the models the offload can serve."""
    return {model.name: model.groq_model for model in models() if model.groq_model}
