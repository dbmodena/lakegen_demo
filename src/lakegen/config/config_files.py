"""Experiment configuration files: one nested layout, composed with ``extends``.

A config file groups the settings by topic and composes others; every file
starts with ``schema_version: 2``, as ``config/base.yaml`` and the presets do::

    schema_version: 2
    extends: [base.yaml, presets/retrieval/hybrid_rrf.yaml]
    lake: chicago
    discovery:
      architecture: divided
      selector: memgpt
    retrieval:
      mode: hybrid
      hybrid: {fusion: rrf}

A mode block (``retrieval.hybrid``, ``pneuma``, ``seeker``, ``duckdb``, ``rerank``)
is accepted only when the selected retrieval mode uses it, so an experiment
cannot carry settings that do nothing. ``discovery.selector: memgpt`` is accepted
as shorthand for ``{kind: memgpt}``. A ``--set`` or a spec names its keys the same
way (``retrieval.hybrid.alpha=0.5``).

Inside, a configuration is flat: ``ExperimentConfig`` validates the flat form,
the API receives it and manifests record it. ``V2_PATHS`` maps between the two;
``load_document`` returns the flat form of a file, ``override_key`` the flat
key of a path someone wrote, and ``to_v2`` renders a flat config as a file for
``config show``. This module knows the key table, not the schema;
``lakegen.config.recorded`` reads stored configs against the schema.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# flat key -> v2 path. Keys missing here keep the same path in both layouts.
V2_PATHS: dict[str, str] = {
    "core": "lake",
    "discovery_architecture": "discovery.architecture",
    "discovery.selector": "discovery.selector.kind",
    "discovery.max_zero_result_retries": "discovery.budget.max_zero_result_retries",
    # The unified architect's dials, replaced on 2026-10-30: still read from
    # older files, and dropped on load (REMOVED_DISCOVERY_SETTINGS).
    "discovery.initial_candidates": "discovery.budget.initial_candidates",
    "discovery.expansion_size": "discovery.budget.expansion_size",
    "discovery.max_expansions": "discovery.budget.max_expansions",
    "discovery.max_search_attempts": "discovery.budget.max_search_attempts",
    "discovery.initial_shortlist_size": "discovery.budget.initial_shortlist_size",
    "discovery.max_inspected_candidates": "discovery.budget.max_inspected_candidates",
    "discovery.max_inspections_per_file": "discovery.budget.max_inspections_per_file",
    "discovery.search_after_inspection": "discovery.budget.search_after_inspection",
    "discovery.keyword_search": "discovery.keyword_search.strategy",
    "discovery.state_board": "discovery.context.state_board",
    "discovery.compact_history": "discovery.context.compact_history",
    "retrieval.alpha": "retrieval.hybrid.alpha",
    "retrieval.fusion_method": "retrieval.hybrid.fusion",
    "retrieval.rrf_k": "retrieval.hybrid.rrf_k",
    "retrieval.missing_signal_policy": "retrieval.hybrid.missing_signal",
    "retrieval.rerank_model": "retrieval.rerank.model",
    "retrieval.rerank_depth": "retrieval.rerank.depth",
    "retrieval.pneuma_index_name": "retrieval.pneuma.index_name",
    "retrieval.pneuma_base_url": "retrieval.pneuma.base_url",
    "retrieval.pneuma_timeout_seconds": "retrieval.pneuma.timeout_seconds",
    "retrieval.pneuma_entity_model": "retrieval.pneuma.entity_model",
    "retrieval.pneuma_value_scan_weight": "retrieval.pneuma.value_scan_weight",
    "retrieval.pneuma_content_weight": "retrieval.seeker.content_weight",
    "retrieval.pneuma_content_fusion": "retrieval.seeker.content_fusion",
    "retrieval.pneuma_enumerate_tables": "retrieval.seeker.enumerate_tables",
    "retrieval.pneuma_table_name_weight": "retrieval.seeker.table_name_weight",
    "retrieval.pneuma_column_name_weight": "retrieval.seeker.column_name_weight",
    "retrieval.pneuma_cell_weight": "retrieval.seeker.cell_weight",
    "retrieval.scan_workers": "retrieval.seeker.scan_workers",
    "retrieval.grep_max_files": "retrieval.seeker.max_files",
    "retrieval.grep_max_columns_per_file": "retrieval.seeker.max_columns_per_file",
    "retrieval.grep_max_scan_rows_per_file": "retrieval.seeker.max_scan_rows_per_file",
    "retrieval.grep_sample_rows": "retrieval.seeker.sample_rows",
    "retrieval.grep_probe_files": "retrieval.seeker.probe_files",
    "retrieval.grep_probe_rows_per_file": "retrieval.seeker.probe_rows_per_file",
    "retrieval.grep_value_weight": "retrieval.seeker.value_weight",
    "retrieval.grep_metadata_weight": "retrieval.seeker.metadata_weight",
    "retrieval.duckdb_max_files": "retrieval.duckdb.max_files",
    "retrieval.duckdb_max_columns_per_file": "retrieval.duckdb.max_columns_per_file",
    "retrieval.duckdb_sample_rows": "retrieval.duckdb.sample_rows",
    "retrieval.duckdb_max_scan_rows_per_file": "retrieval.duckdb.max_scan_rows_per_file",
    "retrieval.duckdb_probe_files": "retrieval.duckdb.probe_files",
    "retrieval.duckdb_probe_rows_per_file": "retrieval.duckdb.probe_rows_per_file",
    "coder_context_level": "coding.context_level",
    "coder_brief_mode": "coding.brief_mode",
    "automatic_test_coder": "coding.automatic_test_coder",
    "require_semantic_plan": "coding.require_semantic_plan",
    "revision_experiment_mode": "coding.revision_mode",
    "max_revision_rounds": "coding.max_revision_rounds",
    "semantic_code_judge_enabled": "judge.enabled",
    "semantic_code_judge_model": "judge.model",
    "interaction_mode": "interaction.mode",
    "gates.keywords": "interaction.gates.keywords",
    "gates.datasets": "interaction.gates.datasets",
    "gates.plan": "interaction.gates.plan",
    "gates.result": "interaction.gates.result",
}
FLAT_PATHS: dict[str, str] = {v2: flat for flat, v2 in V2_PATHS.items()}

# A retrieval mode block is accepted only for the modes that read it.
MODE_BLOCKS: dict[str, frozenset[str]] = {
    "retrieval.hybrid": frozenset({"hybrid", "hybrid_duckdb_semantic"}),
    "retrieval.rerank": frozenset({"keyword", "semantic", "hybrid", "hybrid_duckdb_semantic"}),
    "retrieval.pneuma": frozenset({"pneuma", "pneuma_seeker"}),
    "retrieval.seeker": frozenset({"pneuma_seeker"}),
    "retrieval.duckdb": frozenset({"duckdb_agentic", "hybrid_duckdb_semantic"}),
}
# Scalar shorthands: ``discovery.selector: memgpt`` means ``{kind: memgpt}``.
SHORTHANDS: dict[str, str] = {"discovery.selector": "kind"}
V2_VERSION = 2


class ConfigFileError(ValueError):
    """A configuration file or override that does not follow the nested layout."""


# --- documents ----------------------------------------------------------------


def parse_document(text: str, *, suffix: str) -> dict[str, Any]:
    """Parse YAML or JSON text into a mapping, without touching the filesystem."""
    normalized = suffix.casefold()
    try:
        if normalized == ".json":
            loaded = json.loads(text)
        elif normalized in {".yaml", ".yml"}:
            loaded = yaml.safe_load(text)
        else:
            raise ConfigFileError("experiment config must be a .yaml, .yml, or .json file")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigFileError(f"invalid experiment config: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigFileError("experiment config root must be an object")
    return dict(loaded)


def deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` updated with ``updates``, nested mappings merged key by key; inputs untouched."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_document(path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a config file, resolve its ``extends`` and return the flat form.

    Args:
        path: The file, in the nested layout.

    Returns:
        The flat form of the file and everything it extends.

    Raises:
        ConfigFileError: When the file, or one it extends, is not in the nested
            layout, or ``extends`` forms a cycle.
    """
    file = Path(path).resolve()
    if file in _seen:
        chain = " -> ".join(str(item) for item in (*_seen, file))
        raise ConfigFileError(f"extends cycle: {chain}")
    document = parse_document(file.read_text(encoding="utf-8"), suffix=file.suffix)
    if document.get("schema_version") != V2_VERSION:
        raise ConfigFileError(
            f"{file}: a config file uses the nested layout of config/base.yaml and "
            f"starts with 'schema_version: {V2_VERSION}'"
        )
    bases = document.pop("extends", None) or []
    if isinstance(bases, str):
        bases = [bases]
    resolved: dict[str, Any] = {}
    for base in bases:
        resolved = deep_merge(resolved, load_document(file.parent / base, _seen=(*_seen, file)))
    return deep_merge(resolved, flatten_document(document))


def flatten_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the flat form of an inline document.

    An inline document is either already flat (the API's payload, a spec's
    composed cell) or in the nested layout; ``schema_version`` tells which.

    Args:
        document: The document.

    Returns:
        Its flat form.

    Raises:
        ConfigFileError: When ``schema_version`` is neither absent, 1 nor 2.
    """
    body = dict(document)
    version = body.pop("schema_version", None)
    if version in (None, 1):
        return body
    if version != V2_VERSION:
        raise ConfigFileError(f"unknown schema_version {version!r}; this version reads 1 and 2")
    return _v2_to_flat(body)


def _v2_to_flat(document: Mapping[str, Any]) -> dict[str, Any]:
    flat_dotted: dict[str, Any] = {}
    for dotted, value in _walk(document):
        flat_dotted[translate_override(dotted)] = value
    _check_mode_blocks(document, flat_dotted.get("retrieval.mode"))
    return nest(flat_dotted)


def _walk(mapping: Mapping[str, Any], prefix: str = ""):
    """(dotted path, leaf value) pairs, with scalar shorthands expanded."""
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if dotted in SHORTHANDS and not isinstance(value, Mapping):
            value = {SHORTHANDS[dotted]: value}
        if isinstance(value, Mapping) and value:
            yield from _walk(value, f"{dotted}.")
        else:
            yield dotted, value


def _check_mode_blocks(document: Mapping[str, Any], mode: Any) -> None:
    retrieval = document.get("retrieval")
    if not isinstance(retrieval, Mapping):
        return
    for block, modes in MODE_BLOCKS.items():
        name = block.split(".", 1)[1]
        if name in retrieval and mode not in modes:
            raise ConfigFileError(
                f"retrieval.{name} applies to {', '.join(sorted(modes))}, not to mode {mode!r}"
            )


def nest(dotted: Mapping[str, Any]) -> dict[str, Any]:
    """``{"retrieval.top_k": 20}`` -> ``{"retrieval": {"top_k": 20}}``."""
    result: dict[str, Any] = {}
    for key, value in dotted.items():
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def translate_override(dotted: str) -> str:
    """Return the flat key of a dotted key, which may already be flat (internal callers)."""
    if dotted in FLAT_PATHS:
        return FLAT_PATHS[dotted]
    if dotted in SHORTHANDS:  # discovery.selector=memgpt is valid in both
        return dotted
    return dotted


def override_key(dotted: str) -> str:
    """Return the flat key of a path someone wrote, in the nested layout.

    Args:
        dotted: The path, e.g. ``retrieval.hybrid.alpha``.

    Returns:
        The flat key, e.g. ``retrieval.alpha``.

    Raises:
        ConfigFileError: When the path is a flat spelling, e.g. ``core`` for ``lake``.
    """
    if dotted in V2_PATHS and dotted not in SHORTHANDS:
        raise ConfigFileError(f"{dotted!r} is the flat spelling; write {V2_PATHS[dotted]!r}")
    return translate_override(dotted)


def parse_set_values(set_values: list[str] | None) -> dict[str, Any]:
    """Read ``PATH=VALUE`` overrides, each value as YAML and each path in the nested layout.

    Args:
        set_values: The ``--set`` arguments.

    Returns:
        The values by flat key.

    Raises:
        ConfigFileError: When an argument has no ``=`` or its path is a flat spelling.
    """
    overrides: dict[str, Any] = {}
    for raw in set_values or []:
        if "=" not in raw:
            raise ConfigFileError(f"invalid --set {raw!r}; expected PATH=VALUE")
        key, raw_value = raw.split("=", 1)
        overrides[override_key(key.strip())] = yaml.safe_load(raw_value)
    return overrides


# --- the v2 rendering of a flat configuration ---------------------------------


def to_v2(flat: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The nested layout of a flat config.

    Args:
        flat: The flat config.

    Returns:
        The v2 document and the dropped values: settings of retrieval mode
        blocks the selected mode does not read, which the layout cannot carry.
    """
    mode = (flat.get("retrieval") or {}).get("mode")
    document: dict[str, Any] = {"schema_version": V2_VERSION}
    dropped: dict[str, Any] = {}
    for dotted, value in _walk(flat):
        target = V2_PATHS.get(dotted, dotted)
        block = ".".join(target.split(".")[:2])
        if block in MODE_BLOCKS and mode not in MODE_BLOCKS[block]:
            dropped[dotted] = value
            continue
        document = deep_merge(document, nest({target: value}))
    return document, dropped
