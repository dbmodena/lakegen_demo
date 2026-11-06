"""The Chainlit app's per-chat state: the session of the question in progress and its settings."""

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
    MODEL_ITEMS,
    MODEL_OPTIONS,
    RETRIEVAL_MODE_ITEMS,
    RETRIEVAL_MODE_OPTIONS,
    SOLR_CORE_OPTIONS,
    SOLR_CORE_PORTAL_NAMES,
    RuntimeSettings,
)

# Re-exported: the UI and older importers read these from here.
from lakegen.datastructures.errors import WorkflowCancelled, WorkflowTimedOut
from lakegen.datastructures.types import SolrMetadata
from lakegen.entrypoints.webapp.i18n import t
from lakegen.pipeline.observability.tracing import HumanInterventionRecorder

# The runtime options and the workflow errors are re-exported: src/lakegen/entrypoints/webapp/app.py,
# the tests and older importers read them from here.
__all__ = [
    "MODEL_ITEMS",
    "MODEL_OPTIONS",
    "RETRIEVAL_MODE_ITEMS",
    "RETRIEVAL_MODE_OPTIONS",
    "SOLR_CORE_OPTIONS",
    "SOLR_CORE_PORTAL_NAMES",
    "LakeGenSession",
    "RuntimeSettings",
    "WorkflowCancelled",
    "WorkflowTimedOut",
    "get_runtime_settings",
    "get_session",
    "set_runtime_settings",
]


@dataclass
class LakeGenSession:
    """One question's run in the chat: its settings, what each phase found, and its stop switch.

    The Chainlit driver fills it as the engine's stages finish, and the phase
    summaries read it back.

    Attributes:
        runtime: The settings the question runs with.
        phase: The phase in progress (``"idle"`` until a run starts).
        query: The question.
        keywords: The keywords of the latest keyword round.
        raw_keywords: The keyword generator's raw output.
        tables: The tables selected.
        candidates: The candidate tables the selection chose from.
        phase1_runs: Each keyword generation round, as ``record_phase1_run`` stores it.
        architect_reasoning: The selector's reasoning for its tables.
        alternatives: The other selections the selector offers, as its plan records them (tables, note).
        full_trace: The selection trace (the unified agent's activity log in unified mode).
        solr_metadata_map: The catalogue metadata of the candidate tables, by table.
        fallback_reason: Why the selector or the coder rejected the last round.
        force_execution: Whether the person forced the code to run on rejected tables.
        excluded_tables: Tables proven insufficient in an earlier round (unified mode).
        carried_tables: Tables verified in an earlier round, carried forward (unified mode).
        carried_metadata: The catalogue metadata of the carried tables.
        tokens: Tokens spent per phase (``p1`` to ``p4``).
        run_id: The run's id, which names its coding directory.
        workflow_task: The asyncio task running the question, cancelled by Stop.
        manifest: The run's manifest, once the engine has written it.
        intervention_recorder: The record of the person's approvals and hints.
        started_at: When the session was created (``time.monotonic``).
        finalized: Whether the run has ended.
        final_code: The last generated script.
        raw_result: The last script's output.
        final_answer: The synthesized answer.
        retries: The number of retries.
        execution_error: The last script's error, or the run's.
        generated_code_seed_instruction_provided: Whether the coder was told which random seed to use.
    """

    runtime: RuntimeSettings = field(default_factory=RuntimeSettings.default)
    phase: str = "idle"
    query: str = ""
    keywords: list[str] = field(default_factory=list)
    raw_keywords: str = ""
    tables: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    phase1_runs: list[dict[str, Any]] = field(default_factory=list)
    architect_reasoning: str = ""
    alternatives: list[dict[str, Any]] = field(default_factory=list)
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
    tokens: dict[str, int] = field(default_factory=lambda: {"p1": 0, "p2": 0, "p3": 0, "p4": 0})
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
        """The run's directory under the coding directory."""
        return paths.coding_dir() / self.run_id

    def text(self, key: str, **kwargs: Any) -> str:
        """The app's message for ``key``, formatted with ``kwargs`` (``i18n.t``)."""
        return t(key, **kwargs)

    @property
    def cancelled(self) -> bool:
        """Whether Stop was pressed."""
        return self._cancelled.is_set()

    def request_cancel(self) -> None:
        """Signal all running phases to stop."""
        self._cancelled.set()
        if self.workflow_task and not self.workflow_task.done():
            self.workflow_task.cancel()

    def check_cancelled(self) -> None:
        """Raise WorkflowCancelled if the stop button was pressed.

        Raises:
            WorkflowCancelled: When Stop was pressed.
        """
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
        """Add a keyword generation round to ``phase1_runs``.

        Args:
            label: The round's name, such as "Initial generation".
            hint: The person's hint for the round.
            keywords: The keywords generated.
            raw_output: The generator's raw output.
            tokens: The tokens the round spent.
            reasoning: The model's reasoning.
        """
        self.phase1_runs.append(
            {
                "label": label,
                "hint": hint,
                "keywords": keywords,
                "raw_output": raw_output,
                "tokens": tokens,
                "reasoning": reasoning,
            }
        )


def get_session() -> LakeGenSession:
    """The chat's session, created on first use."""
    session = cl.user_session.get("lakegen_session")
    if session is None:
        session = LakeGenSession()
        cl.user_session.set("lakegen_session", session)
    return session


def set_runtime_settings(runtime: RuntimeSettings) -> None:
    """Store the chat's settings in its Chainlit user session."""
    cl.user_session.set("runtime_settings", runtime)


def get_runtime_settings() -> RuntimeSettings:
    """The chat's settings, the defaults on first use."""
    runtime = cl.user_session.get("runtime_settings")
    if runtime is None:
        runtime = RuntimeSettings.default()
        set_runtime_settings(runtime)
    return runtime
