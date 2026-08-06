from __future__ import annotations

import threading
import uuid
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chainlit as cl

from lakegen.ui.i18n import t
from lakegen.core.types import SolrMetadata
from lakegen.core.config import BASE_DIR, resolve_portal_tables_dir
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.experiment_config import (
    DiscoveryArchitecture,
    ExperimentConfig,
    InteractionMode,
    load_experiment_config,
)


MODEL_OPTIONS = [
    "openai.gpt-oss-120b",
    "meta.llama-3.3-70b-instruct",
]
SOLR_CORE_OPTIONS = ["nyc", "valencia", "bologna", "paris", "uk"]
SOLR_CORE_PORTAL_NAMES = {
    "nyc": "New York City Open Data portal",
    "valencia": "Valencia Open Data portal",
    "bologna": "Bologna Open Data portal",
    "paris": "Paris Open Data portal",
    "uk": "UK Open Data portal",
}
RETRIEVAL_MODE_OPTIONS = [mode.value for mode in RetrievalMode]


@dataclass
class RuntimeSettings:
    model_name: str = MODEL_OPTIONS[0]
    solr_core: str = SOLR_CORE_OPTIONS[0]
    csv_dir: Path = field(default_factory=lambda: resolve_portal_tables_dir("nyc"))
    db_path: Path = BASE_DIR / "data/blend_nyc.db"
    use_unified_agent: bool = True
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig.from_env)
    experiment: ExperimentConfig | None = None

    def __post_init__(self) -> None:
        if self.experiment is None:
            self.experiment = load_experiment_config(overrides={
                "model": self.model_name,
                "core": self.solr_core,
                "discovery_architecture": (
                    DiscoveryArchitecture.UNIFIED.value
                    if self.use_unified_agent
                    else DiscoveryArchitecture.DIVIDED.value
                ),
                "interaction_mode": InteractionMode.HUMAN_GATED.value,
                **{
                    f"retrieval.{key}": value
                    for key, value in self.retrieval.__dict__.items()
                },
            })

    @property
    def portal_name(self) -> str:
        return SOLR_CORE_PORTAL_NAMES.get(self.solr_core, self.solr_core)

    @classmethod
    def default(cls) -> "RuntimeSettings":
        return cls()

    @classmethod
    def from_chainlit_settings(
        cls,
        settings: dict[str, Any],
        *,
        solr_core: str | None = None,
    ) -> "RuntimeSettings":
        default = cls()

        selected_solr_core = str(
            solr_core or settings.get("solr_core") or default.solr_core
        )
        if selected_solr_core not in SOLR_CORE_OPTIONS:
            selected_solr_core = default.solr_core

        model_name = str(settings.get("model_name") or default.model_name)
        if model_name not in MODEL_OPTIONS:
            model_name = default.model_name

        use_unified_agent = settings.get("use_unified_agent", default.use_unified_agent)
        retrieval_mode = str(
            settings.get("retrieval_mode") or default.retrieval.mode
        )
        if retrieval_mode not in RETRIEVAL_MODE_OPTIONS:
            retrieval_mode = default.retrieval.mode
        
        retrieval = RetrievalConfig.from_env(mode=retrieval_mode)
        experiment = load_experiment_config(overrides={
            "model": model_name,
            "core": selected_solr_core,
            "discovery_architecture": (
                "unified" if bool(use_unified_agent) else "divided"
            ),
            "interaction_mode": "human_gated",
            **{f"retrieval.{key}": value for key, value in retrieval.__dict__.items()},
        })
        return cls(
            model_name=model_name,
            solr_core=selected_solr_core,
            csv_dir=resolve_portal_tables_dir(selected_solr_core),
            db_path=BASE_DIR / f"data/blend_{selected_solr_core}.db",
            use_unified_agent=bool(use_unified_agent),
            retrieval=retrieval,
            experiment=experiment,
        )


class WorkflowCancelled(Exception):
    """Raised when the user clicks Stop in the UI."""


@dataclass
class LakeGenSession:
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings.default)
    phase: str = "idle"
    query: str = ""
    keywords: list[str] = field(default_factory=list)
    raw_keywords: str = ""
    tables: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    phase1_runs: list[dict[str, Any]] = field(default_factory=list)
    architect_reasoning: str = ""
    full_trace: str = ""
    solr_metadata_map: SolrMetadata = field(default_factory=dict)
    fallback_reason: str = ""
    force_execution: bool = False
    tokens: dict[str, int] = field(
        default_factory=lambda: {"p1": 0, "p2": 0, "p3": 0, "p4": 0}
    )
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _cancelled: threading.Event = field(default_factory=threading.Event)
    workflow_task: asyncio.Task | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    human_interventions: list[dict[str, Any]] = field(default_factory=list)
    phase_seconds: dict[str, float] = field(
        default_factory=lambda: {"discovery": 0.0, "code": 0.0, "result": 0.0}
    )
    llm_call_counts: dict[str, int] = field(
        default_factory=lambda: {"discovery": 0, "code": 0, "result": 0}
    )

    @property
    def run_dir(self) -> Path:
        return BASE_DIR / "coding" / self.run_id

    def text(self, key: str, **kwargs: Any) -> str:
        return t(key, **kwargs)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def request_cancel(self) -> None:
        """Signal all running phases to stop."""
        self._cancelled.set()
        if self.workflow_task and not self.workflow_task.done():
            self.workflow_task.cancel()

    def check_cancelled(self) -> None:
        """Raise WorkflowCancelled if the stop button was pressed."""
        if self._cancelled.is_set():
            raise WorkflowCancelled("Workflow stopped by user.")

    def record_phase1_run(
        self,
        label: str,
        hint: str,
        keywords: list[str],
        raw_output: str,
        tokens: int,
        reasoning: str = "",
    ) -> None:
        self.phase1_runs.append({
            "label": label,
            "hint": hint,
            "keywords": keywords,
            "raw_output": raw_output,
            "tokens": tokens,
            "reasoning": reasoning,
        })


def get_session() -> LakeGenSession:
    session = cl.user_session.get("lakegen_session")
    if session is None:
        session = LakeGenSession()
        cl.user_session.set("lakegen_session", session)
    return session


def set_runtime_settings(runtime: RuntimeSettings) -> None:
    cl.user_session.set("runtime_settings", runtime)


def get_runtime_settings() -> RuntimeSettings:
    runtime = cl.user_session.get("runtime_settings")
    if runtime is None:
        runtime = RuntimeSettings.default()
        set_runtime_settings(runtime)
    return runtime


def get_keyword_rejection_reason(reasoning: str) -> str | None:
    if not reasoning.startswith("REJECT_KEYWORDS"):
        return None
    reason = reasoning.replace("REJECT_KEYWORDS:", "", 1)
    reason = reason.replace("REJECT_KEYWORDS", "", 1).strip()
    return reason or "The candidate tables did not match the generated keywords."


def apply_phase2_keyword_rejection(
    session: LakeGenSession,
    candidates: list[str],
    solr_metadata: SolrMetadata,
    reasoning: str,
    trace: str,
    tokens: int,
    accumulate_tokens: bool,
) -> bool:
    rejection_reason = get_keyword_rejection_reason(reasoning)
    if rejection_reason is None:
        return False

    session.tables = []
    session.candidates = candidates
    session.solr_metadata_map = solr_metadata
    session.architect_reasoning = reasoning
    session.full_trace = trace
    session.fallback_reason = rejection_reason
    if accumulate_tokens:
        session.tokens["p2"] += tokens
    else:
        session.tokens["p2"] = tokens
    session.phase = "fallback_approval_keywords"
    return True
