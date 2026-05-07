from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chainlit as cl

from lakegen.ui.i18n import t
from lakegen.types import SolrMetadata
from src.utils import BASE_DIR


MODEL_OPTIONS = ["gemma4:26b", "qwen3.5:latest", "llama3.1:8b", "gpt-oss:20b"]
SOLR_CORE_OPTIONS = ["nyc", "valencia", "bologna", "paris"]
SOLR_CORE_PORTAL_NAMES = {
    "nyc": "New York City Open Data portal",
    "valencia": "Valencia Open Data portal",
    "bologna": "Bologna Open Data portal",
    "paris": "Paris Open Data portal",
}


@dataclass
class RuntimeSettings:
    ollama_url: str = "http://127.0.0.1:11434"
    model_name: str = MODEL_OPTIONS[0]
    solr_core: str = SOLR_CORE_OPTIONS[0]
    csv_dir: Path = BASE_DIR / "data/nyc/datasets/csv"
    db_path: Path = BASE_DIR / "data/blend_nyc.db"

    @property
    def portal_name(self) -> str:
        return SOLR_CORE_PORTAL_NAMES.get(self.solr_core, self.solr_core)

    @classmethod
    def default(cls) -> "RuntimeSettings":
        return cls()

    @classmethod
    def from_chainlit_settings(cls, settings: dict[str, Any]) -> "RuntimeSettings":
        default = cls()

        solr_core = str(settings.get("solr_core") or default.solr_core)
        if solr_core not in SOLR_CORE_OPTIONS:
            solr_core = default.solr_core

        model_name = str(settings.get("model_name") or default.model_name)
        if model_name not in MODEL_OPTIONS:
            model_name = default.model_name

        ollama_url = str(settings.get("ollama_url") or default.ollama_url).strip()
        return cls(
            ollama_url=ollama_url or default.ollama_url,
            model_name=model_name,
            solr_core=solr_core,
            csv_dir=BASE_DIR / f"data/{solr_core}/datasets/csv",
            db_path=BASE_DIR / f"data/blend_{solr_core}.db",
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
        default_factory=lambda: {"p1": 0, "p2": 0, "p3": 0, "p5": 0}
    )
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _cancelled: threading.Event = field(default_factory=threading.Event)

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

    def check_cancelled(self) -> None:
        """Raise WorkflowCancelled if the stop button was pressed."""
        if self._cancelled.is_set():
            raise WorkflowCancelled("Workflow stopped by user.")

    def reset_for_query(self, query: str) -> None:
        runtime = self.runtime
        self.__dict__.update(LakeGenSession(runtime=runtime, query=query).__dict__)

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
