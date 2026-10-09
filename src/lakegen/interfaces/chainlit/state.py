from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chainlit as cl

from lakegen.config import paths
from lakegen.config.runtime import (
    MODEL_OPTIONS,
    RETRIEVAL_MODE_ITEMS,
    RETRIEVAL_MODE_OPTIONS,
    SOLR_CORE_OPTIONS,
    SOLR_CORE_PORTAL_NAMES,
    RuntimeSettings,
)

# Re-exported: the UI and older importers read these from here.
from lakegen.domain.errors import WorkflowCancelled, WorkflowTimedOut
from lakegen.domain.types import SolrMetadata
from lakegen.interfaces.chainlit.i18n import t
from lakegen.workflow.observability.tracing import HumanInterventionRecorder


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
    # Cross-round memory for the unified architecture (mirrors the divided
    # architecture's excluded_tables/carried_tables in service.py): tables
    # proven insufficient in an earlier round stay excluded from retrieval,
    # and tables already verified to satisfy part of the question carry
    # forward, across both "Recalculate" and "Re-evaluate tables" rounds.
    excluded_tables: set[str] = field(default_factory=set)
    carried_tables: list[str] = field(default_factory=list)
    carried_metadata: SolrMetadata = field(default_factory=dict)
    # Unified only: cached inspect_columns text per carried table, so a
    # table already proven good doesn't have to be re-inspected from a
    # blank P12State every round -- that wasted the round's own bounded
    # inspection budget on re-verifying what was already known.
    carried_inspection: dict[str, str] = field(default_factory=dict)
    # Retrieval evidence from earlier rounds of this question (terms tried,
    # rejection reasons, suggestions, ban evidence), rendered into each new
    # round's prompt. Kept for the session only: writing it to the batch
    # service's question memory would leak chat exploration into benchmarks.
    retrieval_memory_events: list[dict[str, object]] = field(default_factory=list)
    tokens: dict[str, int] = field(
        default_factory=lambda: {"p1": 0, "p2": 0, "p3": 0, "p4": 0}
    )
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _cancelled: threading.Event = field(default_factory=threading.Event)
    workflow_task: asyncio.Task | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    intervention_recorder: HumanInterventionRecorder = field(
        default_factory=HumanInterventionRecorder
    )
    started_at: float = field(default_factory=time.monotonic)
    finalized: bool = False
    final_code: str = ""
    raw_result: Any = None
    final_answer: str = ""
    retries: int = 0
    execution_error: str = ""
    generated_code_seed_instruction_provided: bool = False

    @property
    def run_dir(self) -> Path:
        return paths.coding_dir() / self.run_id

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
