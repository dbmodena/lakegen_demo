"""Validated, serializable configuration for reproducible LakeGen runs."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lakegen.config.config_files import (
    deep_merge,
    flatten_document,
    load_document,
    nest,
    parse_document,
    translate_override,
)
from lakegen.config.registry import lake_names, model_names
from lakegen.config.retrieval import (
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_TOP_K,
    FusionMethod,
    MissingSignalPolicy,
    PneumaContentFusion,
    RetrievalConfig,
    RetrievalMode,
)
from lakegen.config.settings import retrieval_env_overrides

DEFAULT_MODEL = "openai.gpt-oss-120b"
DEFAULT_CORE = "nyc"
# What exists comes from config/models.yaml and config/lakes.yaml.
SUPPORTED_MODELS = model_names()
SUPPORTED_CORES = lake_names()


class FrozenModel(BaseModel):
    """An immutable config block that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class DiscoveryArchitecture(StrEnum):
    """Whether one agent searches and selects (unified) or two steps do (divided)."""

    UNIFIED = "unified"
    DIVIDED = "divided"


class TableSelector(StrEnum):
    """Who picks the tables among the retrieved candidates in the divided architecture."""

    MEMGPT = "memgpt"
    JEV = "jev"
    """TypeSafe's Jev decision model, removed on 2026-10-29. The value stays so the
    configurations of the runs made with it still load; a run that asks for it is refused."""
    LLM = "llm"
    """The divided architecture's judge agent, removed on 2026-10-21 for memgpt. The value stays
    so the configurations of the runs made with it still load; a run that asks for it is refused."""


class InteractionMode(StrEnum):
    """Whether a run asks a person at its gates (human_gated) or runs on its own."""

    AUTONOMOUS = "autonomous"
    HUMAN_GATED = "human_gated"


class CoderContextLevel(StrEnum):
    """How much of the tables' context the coder receives."""

    FULL = "full"
    SCHEMA_ONLY = "schema_only"
    MINIMAL = "minimal"


class RevisionExperimentMode(StrEnum):
    """Whether the coder revises a result with semantic coverage warnings (current) or not."""

    CURRENT = "current"
    NO_SEMANTIC_REVISION = "no_semantic_revision"


class CoderBriefMode(StrEnum):
    """Whether the coder must follow the brief (strict) or may override it on evidence."""

    STRICT = "strict"
    ADVISORY = "advisory"


class RetrievalExperimentConfig(FrozenModel):
    """The retrieval block of an experiment file.

    The same fields as ``RetrievalConfig`` (lakegen.config.retrieval), with the
    production defaults; tests/config/test_retrieval_config_sync.py keeps the two in
    step.
    """

    mode: RetrievalMode = RetrievalMode.KEYWORD
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(default=5, gt=0)
    representation_version: str = Field(default="metadata-v1", min_length=1)
    embedding_model: str = Field(default="cohere.embed-v4.0", min_length=1)
    embedding_base_url: str = "http://localhost:11434"
    vector_field: str = Field(default="table_embedding", min_length=1)
    lexical_query_fields: str | None = None
    keyword_or_fallback: bool = False
    missing_signal_policy: MissingSignalPolicy = MissingSignalPolicy.ZERO
    fusion_method: FusionMethod = FusionMethod.WEIGHTED
    rrf_k: int = Field(default=60, gt=0)
    # The production index name; RetrievalConfig itself defaults to "lakegen".
    pneuma_index_name: str = Field(default="lakegen-cohere-v4-1024", min_length=1)
    pneuma_base_url: str = Field(default="http://localhost:8767", min_length=1)
    pneuma_timeout_seconds: float = Field(default=120.0, gt=0)
    pneuma_content_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    pneuma_content_fusion: PneumaContentFusion = PneumaContentFusion.WEIGHTED
    pneuma_entity_model: str | None = None
    pneuma_enumerate_tables: bool = True
    pneuma_table_name_weight: float = Field(default=3.0, ge=0.0)
    pneuma_column_name_weight: float = Field(default=2.0, ge=0.0)
    pneuma_cell_weight: float = Field(default=1.0, ge=0.0)
    pneuma_value_scan_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    duckdb_max_files: int = Field(default=250, gt=0)
    duckdb_max_columns_per_file: int = Field(default=40, gt=0)
    duckdb_sample_rows: int = Field(default=3, gt=0)
    duckdb_max_scan_rows_per_file: int = Field(default=100_000, gt=0)
    duckdb_probe_files: int = Field(default=25, gt=0)
    duckdb_probe_rows_per_file: int = Field(default=1_000, gt=0)
    grep_max_files: int | None = Field(default=None, gt=0)
    grep_max_columns_per_file: int = Field(default=40, gt=0)
    grep_max_scan_rows_per_file: int = Field(default=100_000, gt=0)
    grep_sample_rows: int = Field(default=3, gt=0)
    grep_probe_files: int = Field(default=25, gt=0)
    grep_probe_rows_per_file: int = Field(default=1_000, gt=0)
    scan_workers: int = Field(default=16, gt=0)
    grep_value_weight: float = Field(default=1.0, ge=0.0)
    grep_metadata_weight: float = Field(default=1.0, ge=0.0)
    # Production retrieval reranks by default (see RetrievalConfig.rerank_model);
    # null turns it off.
    rerank_model: str | None = DEFAULT_RERANK_MODEL
    rerank_depth: int = Field(default=DEFAULT_RERANK_DEPTH, gt=0)

    @classmethod
    def from_runtime(cls, value: RetrievalConfig) -> RetrievalExperimentConfig:
        """The block for a runtime retrieval configuration."""
        return cls(**value.__dict__)

    def to_runtime(self) -> RetrievalConfig:
        """The runtime retrieval configuration of this block."""
        return RetrievalConfig(**self.model_dump())


# Discovery settings of removed features, which stored jobs still carry: the
# cascade's size, Jev's settings, and the dials of the unified architect the
# MemGPT agent replaced on 2026-10-30.
REMOVED_DISCOVERY_SETTINGS = frozenset(
    {
        "keyword_cascade_size",
        "jev_model",
        "jev_max_candidates",
        "jev_threshold",
        "jev_reject_floor",
        "initial_candidates",
        "expansion_size",
        "max_expansions",
        "max_search_attempts",
        "initial_shortlist_size",
        "max_inspected_candidates",
        "max_inspections_per_file",
        "search_after_inspection",
        "keyword_search",
        "state_board",
        "compact_history",
    }
)
# The settings of the removed orchestrated discovery mode, which stored jobs still carry.
REMOVED_TOOL_ACCESS_SETTINGS = frozenset({"tool_access", "orchestrated_memory_enabled"})


class DiscoveryConfig(FrozenModel):
    """The divided architecture's keyword budget and table selector.

    The agents' own bounds (candidates shown, inspections, expansions, searches,
    turns) are the constants of ``lakegen.pipeline.discovery.selectors.memgpt.agent``
    and ``lakegen.pipeline.discovery.unified``, where they were validated.

    Attributes:
        max_zero_result_retries: How many times the divided keyword writer rewrites its
            keywords after a strict-AND search of them (in hybrid mode, its keyword side)
            matched nothing. Rejected banned supersets never consume this budget.
        selector: Who picks tables among the retrieved candidates in the divided
            architecture: the MemGPT selector (lakegen.pipeline.discovery.selectors.memgpt),
            an agent with a system-kept working memory that may inspect the whole retrieved
            pool before it confirms. The unified architecture runs the same agent with a
            search tool. ``llm`` (the judge agent memgpt replaced) and ``jev`` (TypeSafe's
            removed decision model) only appear in the configurations of past runs.
    """

    max_zero_result_retries: int = Field(default=2, ge=0)
    selector: TableSelector = TableSelector.MEMGPT

    @model_validator(mode="before")
    @classmethod
    def drop_removed_settings(cls, data: Any) -> Any:
        """Drop the settings stored jobs carry for removed features; they set nothing now.

        The cascade's size (removed 2026-10-27), Jev's model, candidates,
        threshold and floor (removed 2026-10-29), and the unified architect's
        dials (replaced 2026-10-30).

        Args:
            data: The block as stored.

        Returns:
            The block without those settings.
        """
        if isinstance(data, dict) and REMOVED_DISCOVERY_SETTINGS & data.keys():
            return {
                key: value for key, value in data.items() if key not in REMOVED_DISCOVERY_SETTINGS
            }
        return data


class ReviewerConfig(FrozenModel):
    """Reviewers of each step; none is implemented, so the experiment rejects any enabled one.

    Attributes:
        dataset: Review the selected tables.
        plan: Review the plan.
        code: Review the code.
        result: Review the result.
    """

    dataset: bool = False
    plan: bool = False
    code: bool = False
    result: bool = False


class GateConfig(FrozenModel):
    """The points where a human-gated run waits for a person.

    These are the gates present in the existing CLI/Chainlit workflow; an
    experiment must keep keywords and datasets on and plan and result off.

    Attributes:
        keywords: Confirm the keywords.
        datasets: Confirm the selected tables.
        plan: Confirm the plan (not implemented).
        result: Confirm the result (not implemented).
    """

    keywords: bool = True
    datasets: bool = True
    plan: bool = False
    result: bool = False


class BenchmarkConfig(FrozenModel):
    """Input of the retriever-only benchmark (``python -m experiment_runner bench --config``).

    The interactive and batch workflows ignore it.

    Attributes:
        path: A curated case file such as benchmark/100q_nyc.json, or a
            generated-queries file such as generated_queries_semantic.json.
            Relative paths resolve against the working directory.
    """

    path: str | None = Field(default=None, min_length=1)


def _default_retrieval() -> RetrievalExperimentConfig:
    """The static production defaults; the environment applies in load_experiment_config."""
    return RetrievalExperimentConfig()


class ExperimentConfig(FrozenModel):
    """Complete experimental variable set.

    Validation deliberately rejects switches whose implementation belongs to a
    later experiment, so a manifest can never claim that an inactive feature ran.

    Attributes:
        experiment_id: The experiment's id; ``default`` outside an experiment.
        seed: The seed the run's reproducibility settings start from.
        core: The lake.
        model: The chat model.
        discovery_architecture: Unified or divided discovery.
        retrieval: The retrieval block.
        discovery: The discovery bounds and selector.
        benchmark: The retriever-only benchmark's input.
        reviewers: The step reviewers (none implemented).
        max_revision_rounds: The coder's revision rounds; only 3 is accepted.
        coder_context_level: How much table context the coder receives.
        automatic_test_coder: Run the coder-context sweep instead of one coding stage.
        revision_experiment_mode: Experimental ablation: accept the first inspected result
            with semantic coverage warnings, while preserving technical execution retries.
        coder_brief_mode: Experimental: ``advisory`` lets full-file data evidence override
            the brief's column choices, row filters and null policy.
        require_semantic_plan: Whether the coder starts only on a runnable semantic plan.
        semantic_code_judge_enabled: Whether the semantic code judge runs.
        semantic_code_judge_model: The semantic code judge's model.
        interaction_mode: Human-gated or autonomous.
        gates: The gates of a human-gated run.
    """

    experiment_id: str = Field(default="default", min_length=1)
    seed: int = Field(default=0, ge=0)
    core: str = DEFAULT_CORE
    model: str = DEFAULT_MODEL
    discovery_architecture: DiscoveryArchitecture = DiscoveryArchitecture.DIVIDED
    retrieval: RetrievalExperimentConfig = Field(default_factory=_default_retrieval)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    reviewers: ReviewerConfig = Field(default_factory=ReviewerConfig)
    max_revision_rounds: int = Field(default=3, ge=0)
    coder_context_level: CoderContextLevel = CoderContextLevel.FULL
    automatic_test_coder: bool = False
    revision_experiment_mode: RevisionExperimentMode = RevisionExperimentMode.CURRENT
    coder_brief_mode: CoderBriefMode = CoderBriefMode.ADVISORY
    require_semantic_plan: bool = True
    semantic_code_judge_enabled: bool = True
    semantic_code_judge_model: str = DEFAULT_MODEL
    interaction_mode: InteractionMode = InteractionMode.HUMAN_GATED
    gates: GateConfig = Field(default_factory=GateConfig)

    @model_validator(mode="before")
    @classmethod
    def drop_removed_settings(cls, data: Any) -> Any:
        """Drop the tool-access settings past runs recorded; refuse the orchestrated mode.

        Discovery always calls its own tools since the tool-free orchestrated
        mode was taken out on 2026-10-29. Every stored run recorded the
        settings at their agentic defaults, which are dropped; a configuration
        asking for the orchestrated mode is refused rather than run agentically.

        Args:
            data: The configuration as given.

        Returns:
            The configuration without those settings.

        Raises:
            ValueError: When it asks for the orchestrated mode or its memory.
        """
        if not isinstance(data, dict) or not REMOVED_TOOL_ACCESS_SETTINGS & data.keys():
            return data
        if str(data.get("tool_access", "agentic")) != "agentic" or data.get(
            "orchestrated_memory_enabled"
        ):
            raise ValueError(
                "tool_access=orchestrated_context (and orchestrated_memory_enabled) name the "
                "orchestrated discovery mode, removed on 2026-10-29; discovery is agentic"
            )
        return {
            key: value for key, value in data.items() if key not in REMOVED_TOOL_ACCESS_SETTINGS
        }

    @model_validator(mode="after")
    def validate_supported_configuration(self) -> ExperimentConfig:
        """Reject unknown lakes and models, and switches whose implementation does not exist.

        Returns:
            The config, unchanged.

        Raises:
            ValueError: The lake or a model is unknown, a reviewer or a plan/result gate is
                on, a keywords/datasets gate is off, ``max_revision_rounds`` is not 3,
                or ``no_semantic_revision`` is combined with the sweep or a reduced context.
        """
        if self.core not in SUPPORTED_CORES:
            raise ValueError(f"unsupported core {self.core!r}")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported model {self.model!r}")
        if self.semantic_code_judge_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"unsupported semantic_code_judge_model {self.semantic_code_judge_model!r}"
            )
        enabled_reviewers = [
            name for name, enabled in self.reviewers.model_dump().items() if enabled
        ]
        if enabled_reviewers:
            raise ValueError("reviewers are not implemented: " + ", ".join(enabled_reviewers))
        if self.max_revision_rounds != 3:
            raise ValueError("only max_revision_rounds=3 currently preserves the workflow")
        if self.revision_experiment_mode == RevisionExperimentMode.NO_SEMANTIC_REVISION:
            if self.automatic_test_coder:
                raise ValueError("no_semantic_revision requires automatic_test_coder=false")
            if self.coder_context_level != CoderContextLevel.FULL:
                raise ValueError(
                    "no_semantic_revision currently supports coder_context_level=full only"
                )
        if self.gates.plan or self.gates.result:
            raise ValueError("plan and result gates are not implemented")
        if not self.gates.keywords or not self.gates.datasets:
            raise ValueError("disabling existing keywords/datasets gates is not implemented")
        return self

    @property
    def use_unified_agent(self) -> bool:
        """Whether discovery runs the unified agent."""
        return self.discovery_architecture == DiscoveryArchitecture.UNIFIED

    @property
    def architecture_name(self) -> str:
        """Canonical value used by manifests, traces, and legacy CSV logs."""
        return self.discovery_architecture.value


_deep_merge = deep_merge


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


def parse_experiment_config_document(text: str, *, suffix: str) -> dict[str, Any]:
    """Parse YAML/JSON configuration content without accessing the filesystem."""
    return parse_document(text, suffix=suffix)


def load_experiment_config(
    path: str | Path | None = None,
    *,
    data_override: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    base: ExperimentConfig | None = None,
) -> ExperimentConfig:
    """Resolve defaults, an optional config file, and final overrides.

    The file is in the nested layout (lakegen.config.config_files): ``extends`` is
    resolved and the document translated to this flat schema.
    ``data_override`` is an inline document, flat or nested; ``overrides`` are
    dotted keys from code, flat or nested (a person's ``--set`` is checked
    before it gets here, by ``parse_set_values``).

    Args:
        path: A config file, applied over the defaults and the environment.
        data_override: An inline document, applied over the file.
        overrides: Dotted keys, applied last.
        base: The config to start from instead of the defaults.

    Returns:
        The validated config.
    """
    document = (base or ExperimentConfig()).model_dump(mode="json")
    # LAKEGEN_* retrieval variables that are set override the static defaults
    # and nothing else; the manifest records them as env_overrides.
    document = deep_merge(document, nest(retrieval_env_overrides()))
    if path is not None:
        document = deep_merge(document, load_document(path))
    if data_override:
        document = deep_merge(document, flatten_document(data_override))
    if overrides:
        translated = {translate_override(key): value for key, value in overrides.items()}
        document = deep_merge(document, dotted_overrides(translated))
    return ExperimentConfig.model_validate(document)
