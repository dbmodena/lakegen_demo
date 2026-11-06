"""What exists: the lakes and the models, read from config/lakes.yaml and config/models.yaml.

A ``Lake`` knows where its tables, metadata and blend database live, which
Solr core serves it and which Pneuma service holds its index; a ``ModelSpec``
knows a model's provider, the provider's own id, the limits the client is
built with and, in ``options``, whatever else its entry says for its provider
(a dedicated endpoint's ``model_env``). Adding a lake or a model is one entry in the YAML file; the
interfaces, the experiment config and the provider clients all read the
registry, so no list of names exists anywhere else.

The files are read once per process (``reset()`` clears that for tests).
Environment overrides (LAKEGEN_SOLR_CORE_<LAKE>, LAKEGEN_PNEUMA_<LAKE>_*)
are applied when a lake is asked for its core or route, never cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import yaml

from lakegen.config import paths as core_config
from lakegen.config.settings import pneuma_override, solr_core_override

CONFIG_DIR = core_config.BASE_DIR / "config"
TABLE_SUFFIXES = {".parquet", ".pq", ".csv"}


@dataclass(frozen=True)
class Lake:
    """One lake of config/lakes.yaml: where its files live and which services hold its indexes.

    Attributes:
        name: The lake's name: its Solr core and its folder under data/.
        title: The portal's display name.
        pneuma_url: The built-in URL of the lake's Pneuma service.
        pneuma_index: The built-in name of the lake's Pneuma index.
        dataset_roots: Folders under data/<lake>/, searched in this order.
        table_dirs: Table folder names under each root; the first holding table files wins.
        metadata_files: Metadata catalogue names under data/<lake>/metadata/, preferred first.
        statistics_file: The datasets' statistics file under data/<lake>/metadata/.
        starters: The starter questions file, relative to the repository root.
        catalogue: The portal's note on its catalogue fields (what the titles are, who the
            publisher is), for the table selector.
    """

    name: str
    title: str
    pneuma_url: str
    pneuma_index: str
    dataset_roots: tuple[str, ...]
    table_dirs: tuple[str, ...]
    metadata_files: tuple[str, ...]
    statistics_file: str
    starters: str | None
    catalogue: str = ""

    @property
    def solr_core(self) -> str:
        """The core that serves the lake: its override, else its own name."""
        return solr_core_override(self.name) or self.name

    @property
    def data_dir(self) -> Path:
        """The lake's folder under data/."""
        return core_config.BASE_DIR / "data" / self.name

    def pneuma_route(self) -> tuple[str, str]:
        """The lake's Pneuma (base URL, index name), with the environment's overrides."""
        url, index = pneuma_override(self.name)
        return (
            self.pneuma_url if url is None else url,
            self.pneuma_index if index is None else index,
        )

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
            (
                metadata_dir / name
                for name in self.metadata_files
                if (metadata_dir / name).is_file()
            ),
            metadata_dir / self.metadata_files[0],
        )

    @property
    def statistics_path(self) -> Path:
        """The datasets' statistics file."""
        return self.data_dir / "metadata" / self.statistics_file

    @property
    def blend_db_path(self) -> Path:
        """The lake's Blend database, data/blend_<lake>.db."""
        return core_config.BASE_DIR / "data" / f"blend_{self.name}.db"

    @property
    def starters_path(self) -> Path | None:
        """The starter questions file; None when the lake has none."""
        return core_config.BASE_DIR / self.starters if self.starters else None


@dataclass(frozen=True)
class ModelSpec:
    """One chat model of config/models.yaml and the limits its client is built with.

    Attributes:
        name: The LakeGen name of the model.
        provider: oci, oci_dedicated, ollama or together.
        provider_model: The provider's own id; the name when the entry gives none.
        context_window: The context window, when the entry gives one.
        max_output_tokens: The most tokens one answer may have.
        temperature: The sampling temperature.
        timeout_seconds: The request timeout, when the entry gives one.
        groq_model: The id of the same weights on Groq, for the offload of small calls.
        options: The entry's other keys, read by its provider (``model_env`` of a
            dedicated endpoint).
        title: The label the interfaces show for it; the name unless the entry gives one.
    """

    name: str
    provider: str
    provider_model: str
    context_window: int | None
    max_output_tokens: int
    temperature: float
    timeout_seconds: float | None
    groq_model: str | None
    options: dict[str, Any] = field(default_factory=dict)
    title: str = ""

    def __post_init__(self) -> None:
        if not self.title:
            object.__setattr__(self, "title", self.name)


_MODEL_KEYS = frozenset(
    {
        "title",
        "provider",
        "provider_model",
        "context_window",
        "max_output_tokens",
        "temperature",
        "timeout_seconds",
        "groq_model",
    }
)


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


def _unknown_message(kind: str, name: str, names: tuple[str, ...]) -> str:
    return f"unknown {kind} {name!r}; known: {', '.join(names)}"


@lru_cache(maxsize=1)
def lakes() -> tuple[Lake, ...]:
    """Every lake of config/lakes.yaml, in file order, each entry over the file's defaults."""
    document = _document("lakes.yaml")
    defaults = document.get("defaults", {})
    found = []
    for name, entry in document["lakes"].items():
        entry = {**defaults, **(entry or {})}
        found.append(
            Lake(
                name=name,
                title=entry["title"],
                pneuma_url=entry["pneuma_url"],
                pneuma_index=entry["pneuma_index"],
                dataset_roots=tuple(entry["dataset_roots"]),
                table_dirs=tuple(entry["table_dirs"]),
                metadata_files=tuple(entry["metadata_files"]),
                statistics_file=entry["statistics_file"],
                starters=entry.get("starters"),
                catalogue=str(entry.get("catalogue") or "").strip(),
            )
        )
    return tuple(found)


def catalogue_note(lake: str) -> str:
    """The lake's catalogue note; "" for an unknown lake or one without a note."""
    try:
        return get_lake(lake).catalogue
    except KeyError:
        return ""


def lake_names() -> tuple[str, ...]:
    """The lakes' names, in file order."""
    return tuple(lake.name for lake in lakes())


def get_lake(name: str) -> Lake:
    """The lake of that name.

    Args:
        name: The lake's name.

    Returns:
        The lake.

    Raises:
        KeyError: No lake has that name; the message lists the known ones.
    """
    for lake in lakes():
        if lake.name == name:
            return lake
    raise KeyError(_unknown_message("lake", name, lake_names()))


def pneuma_routes() -> dict[str, tuple[str, str]]:
    """The built-in Pneuma routes by lake, without the environment's overrides."""
    return {lake.name: (lake.pneuma_url, lake.pneuma_index) for lake in lakes()}


@lru_cache(maxsize=1)
def models() -> tuple[ModelSpec, ...]:
    """Every model of config/models.yaml, in file order, each entry over the file's defaults."""
    document = _document("models.yaml")
    defaults = document.get("defaults", {})
    found = []
    for name, entry in document["models"].items():
        entry = {**defaults, **(entry or {})}
        found.append(
            ModelSpec(
                name=name,
                provider=entry["provider"],
                provider_model=entry.get("provider_model", name),
                context_window=entry.get("context_window"),
                max_output_tokens=entry["max_output_tokens"],
                temperature=float(entry["temperature"]),
                timeout_seconds=entry.get("timeout_seconds"),
                groq_model=entry.get("groq_model"),
                options={key: value for key, value in entry.items() if key not in _MODEL_KEYS},
                title=entry.get("title", ""),
            )
        )
    return tuple(found)


def model_names() -> tuple[str, ...]:
    """The models' names, in file order."""
    return tuple(model.name for model in models())


def get_model(name: str) -> ModelSpec:
    """The model of that name.

    Args:
        name: The model's LakeGen name.

    Returns:
        The model.

    Raises:
        KeyError: No model has that name; the message lists the known ones.
    """
    for model in models():
        if model.name == name:
            return model
    raise KeyError(_unknown_message("model", name, model_names()))


def models_by_provider(provider: str) -> dict[str, str]:
    """LakeGen model name -> the provider's own id, for one provider."""
    return {model.name: model.provider_model for model in models() if model.provider == provider}


def groq_models() -> dict[str, str]:
    """LakeGen model name -> Groq id, for the models the offload can serve."""
    return {model.name: model.groq_model for model in models() if model.groq_model}
