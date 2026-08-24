"""Validated, serializable configuration for reproducible LakeGen runs."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lakegen.retrieval import (
    FusionMethod,
    MissingSignalPolicy,
    RetrievalConfig,
    RetrievalMode,
)


DEFAULT_MODEL = "openai.gpt-oss-120b"
DEFAULT_CORE = "nyc"
SUPPORTED_MODELS = (
    DEFAULT_MODEL,
    "meta.llama-3.3-70b-instruct",
)
SUPPORTED_CORES = ("nyc", "valencia", "bologna", "paris", "uk")


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class DiscoveryArchitecture(StrEnum):
    UNIFIED = "unified"
    DIVIDED = "divided"


class ToolAccess(StrEnum):
    AGENTIC = "agentic"
    ORCHESTRATED_CONTEXT = "orchestrated_context"


class InteractionMode(StrEnum):
    AUTONOMOUS = "autonomous"
    HUMAN_GATED = "human_gated"


class CoderContextLevel(StrEnum):
    FULL = "full"
    SCHEMA_ONLY = "schema_only"
    MINIMAL = "minimal"


class RetrievalExperimentConfig(FrozenModel):
    mode: RetrievalMode = RetrievalMode.KEYWORD
    top_k: int = Field(default=10, gt=0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(default=5, gt=0)
    representation_version: str = Field(default="metadata-v1", min_length=1)
    embedding_model: str = Field(default="bge-m3", min_length=1)
    embedding_base_url: str = "http://localhost:11434"
    vector_field: str = Field(default="table_embedding", min_length=1)
    lexical_query_fields: str | None = None
    missing_signal_policy: MissingSignalPolicy = MissingSignalPolicy.ZERO
    fusion_method: FusionMethod = FusionMethod.WEIGHTED
    rrf_k: int = Field(default=60, gt=0)
    pneuma_index_name: str = Field(default="lakegen", min_length=1)
    pneuma_base_url: str = Field(default="http://localhost:8765", min_length=1)
    pneuma_timeout_seconds: float = Field(default=120.0, gt=0)

    @classmethod
    def from_runtime(cls, value: RetrievalConfig) -> "RetrievalExperimentConfig":
        return cls(**value.__dict__)

    def to_runtime(self) -> RetrievalConfig:
        return RetrievalConfig(**self.model_dump())


class ReviewerConfig(FrozenModel):
    dataset: bool = False
    plan: bool = False
    code: bool = False
    result: bool = False


class GateConfig(FrozenModel):
    # These are the gates present in the existing CLI/Chainlit workflow.
    keywords: bool = True
    datasets: bool = True
    plan: bool = False
    result: bool = False


def _default_retrieval() -> RetrievalExperimentConfig:
    return RetrievalExperimentConfig.from_runtime(RetrievalConfig.from_env())


class ExperimentConfig(FrozenModel):
    """Complete experimental variable set.

    Validation deliberately rejects switches whose implementation belongs to a
    later experiment, so a manifest can never claim that an inactive feature ran.
    """

    experiment_id: str = Field(default="default", min_length=1)
    seed: int = Field(default=0, ge=0)
    core: str = DEFAULT_CORE
    model: str = DEFAULT_MODEL
    discovery_architecture: DiscoveryArchitecture = DiscoveryArchitecture.UNIFIED
    tool_access: ToolAccess = ToolAccess.AGENTIC
    retrieval: RetrievalExperimentConfig = Field(default_factory=_default_retrieval)
    planner_enabled: bool = False
    reviewers: ReviewerConfig = Field(default_factory=ReviewerConfig)
    max_revision_rounds: int = Field(default=3, ge=0)
    coder_context_level: CoderContextLevel = CoderContextLevel.FULL
    automatic_test_coder: bool = False
    semantic_code_judge_enabled: bool = True
    semantic_code_judge_model: str = DEFAULT_MODEL
    interaction_mode: InteractionMode = InteractionMode.HUMAN_GATED
    gates: GateConfig = Field(default_factory=GateConfig)

    @model_validator(mode="after")
    def validate_supported_configuration(self) -> "ExperimentConfig":
        if self.core not in SUPPORTED_CORES:
            raise ValueError(f"unsupported core {self.core!r}")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported model {self.model!r}")
        if self.semantic_code_judge_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"unsupported semantic_code_judge_model "
                f"{self.semantic_code_judge_model!r}"
            )
        if self.planner_enabled:
            raise ValueError("planner_enabled=true is not implemented")
        enabled_reviewers = [
            name for name, enabled in self.reviewers.model_dump().items() if enabled
        ]
        if enabled_reviewers:
            raise ValueError(
                "reviewers are not implemented: " + ", ".join(enabled_reviewers)
            )
        if self.max_revision_rounds != 3:
            raise ValueError("only max_revision_rounds=3 currently preserves the workflow")
        if self.gates.plan or self.gates.result:
            raise ValueError("plan and result gates are not implemented")
        if not self.gates.keywords or not self.gates.datasets:
            raise ValueError("disabling existing keywords/datasets gates is not implemented")
        return self

    @property
    def use_unified_agent(self) -> bool:
        return self.discovery_architecture == DiscoveryArchitecture.UNIFIED

    @property
    def architecture_name(self) -> str:
        """Canonical value used by manifests, traces, and legacy CSV logs."""

        return self.discovery_architecture.value


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def dotted_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ``retrieval.top_k=20`` style keys into a nested mapping."""

    result: dict[str, Any] = {}
    for dotted_key, value in values.items():
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def parse_experiment_config_document(
    text: str,
    *,
    suffix: str,
) -> dict[str, Any]:
    """Parse YAML/JSON configuration content without accessing the filesystem."""

    normalized_suffix = suffix.casefold()
    try:
        if normalized_suffix == ".json":
            loaded = json.loads(text)
        elif normalized_suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(text)
        else:
            raise ValueError("experiment config must be a .yaml, .yml, or .json file")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid experiment config: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("experiment config root must be an object")
    return dict(loaded)


def load_experiment_config(
    path: str | Path | None = None,
    *,
    data_override: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    base: ExperimentConfig | None = None,
) -> ExperimentConfig:
    """Resolve defaults, an optional YAML/JSON file, and final overrides."""

    data = (base or ExperimentConfig()).model_dump(mode="json")
    if path is not None:
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        loaded = parse_experiment_config_document(text, suffix=config_path.suffix)
        data = _deep_merge(data, loaded)
    if data_override:
        data = _deep_merge(data, data_override)
    if overrides:
        data = _deep_merge(data, dotted_overrides(overrides))
    return ExperimentConfig.model_validate(data)
