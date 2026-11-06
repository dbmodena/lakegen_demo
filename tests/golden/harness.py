"""Golden snapshots of the orchestration: the contract the refactor must keep.

A golden test runs one surface (the service, the step-by-step CLI or the
Chainlit workflow) with every phase replaced by a scripted fake, records what
the orchestration did (which phase it called, with which hints, exclusions and
state; what it logged; what it returned) and compares that record with the
JSON under ``snapshots/``. ``pytest --update-golden`` rewrites the snapshots;
a refactoring commit may only do that when it is a listed behaviour change.

Two kinds of snapshot live side by side:

* interaction goldens: what a person sees (the CLI transcript, the Chainlit
  steps, messages and buttons). These never change.
* pipeline goldens: the call log, the result, the log rows, the manifest.
  These change only in listed commits.

Volatile values are masked before the comparison: run ids and other 32-hex
identifiers, timestamps, git versions, durations and temporary paths.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lakegen.config import paths as core_config
from lakegen.config.experiment import ExperimentConfig

_HEX32 = re.compile(r"\b[0-9a-f]{32}\b")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")
_MASKED_KEYS = {"git_version": "<git>"}


# --- normalisation -----------------------------------------------------------


def mask_text(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = _HEX32.sub("<hex32>", text)
    return _TIMESTAMP.sub("<timestamp>", text)


def normalise(value: Any, replacements: dict[str, str], key: str | None = None) -> Any:
    """A JSON-able, deterministic copy of ``value``."""
    if key in _MASKED_KEYS:
        return _MASKED_KEYS[key]
    if (
        key
        and key.endswith("seconds")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return 0
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return mask_text(value, replacements)
    if isinstance(value, Path):
        return mask_text(str(value), replacements)
    if isinstance(value, Enum):
        return normalise(value.value, replacements)
    if isinstance(value, dict):
        return {str(k): normalise(v, replacements, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v, replacements) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((normalise(v, replacements) for v in value), key=json.dumps)
    return describe(value, replacements)


def describe(obj: Any, replacements: dict[str, str]) -> Any:
    """How an object that is not plain data appears in a snapshot."""
    if hasattr(obj, "model_dump"):  # pydantic
        return normalise(obj.model_dump(mode="json"), replacements)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return normalise(dataclasses.asdict(obj), replacements)
    if isinstance(obj, SimpleNamespace):
        return normalise(vars(obj), replacements)
    if callable(obj):
        return f"<{getattr(obj, '__name__', type(obj).__name__)}>"
    state = getattr(obj, "__dict__", None)
    if state:
        return {"<type>": type(obj).__name__, **normalise(dict(state), replacements)}
    return f"<{type(obj).__name__}>"


# --- the call log and scripted fakes ----------------------------------------


class CallLog:
    """What the orchestration asked of each phase, in order, copied at call time."""

    def __init__(self, replacements: dict[str, str]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replacements = replacements

    def record(self, name: str, args=(), kwargs=None, arg_names=()) -> None:
        entry: dict[str, Any] = {"call": name}
        if args:
            if arg_names:
                entry["args"] = {
                    label: normalise(value, self.replacements)
                    for label, value in zip(arg_names, args, strict=False)
                }
            else:
                entry["args"] = normalise(list(args), self.replacements)
        if kwargs:
            entry["kwargs"] = normalise(dict(kwargs), self.replacements)
        self.calls.append(entry)

    def fake(self, name: str, responses: list, arg_names=()) -> Callable:
        """A fake that logs each call and answers from ``responses`` in order.

        A response may be a value or a callable taking the call's arguments;
        the last response repeats once the list is exhausted, so a scenario
        that always rejects needs one entry.
        """
        counter = itertools.count()

        def _fake(*args, **kwargs):
            self.record(name, args, kwargs, arg_names)
            if not responses:
                raise AssertionError(f"{name} was called but the scenario scripts no response")
            item = responses[min(next(counter), len(responses) - 1)]
            if callable(item):
                return item(*args, **kwargs)
            return copy.copy(item) if isinstance(item, SimpleNamespace) else item

        return _fake


PHASE3_ARGS = ("query", "tables", "candidates", "solr_meta", "reasoning", "llm", "pm", "csv_dir")
PHASE4_ARGS = ("query", "raw_result", "llm", "pm")
PHASE1_ARGS = ("query", "llm", "pm")


@dataclasses.dataclass
class Script:
    """The scripted answers of every phase for one scenario."""

    phase1: list = dataclasses.field(default_factory=list)
    phase2: list = dataclasses.field(default_factory=list)
    phase12: list = dataclasses.field(default_factory=list)
    phase3: list = dataclasses.field(default_factory=list)
    phase4: list = dataclasses.field(default_factory=list)
    judge: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Case:
    """One scenario: the configuration, the scripted phases and the question."""

    experiment: ExperimentConfig
    script: Script
    question: str = "How many parks are there per district?"
    log_context: dict[str, Any] | None = None
    files: tuple[str, ...] = ("gold.csv",)


def experiment(**overrides) -> ExperimentConfig:
    # The scenarios name the divided architecture when they use it; the others are unified,
    # which was the default when they were recorded.
    base: dict[str, Any] = dict(
        experiment_id="golden",
        core="nyc",
        model="openai.gpt-oss-120b",
        interaction_mode="autonomous",
        discovery_architecture="unified",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def phase3_result(**overrides) -> SimpleNamespace:
    """A CoderResult stand-in: a successful run unless overridden."""
    base = dict(
        tokens=6,
        clean_code="print(42)",
        code_raw="```python\nprint(42)\n```",
        rejected_reason="",
        rejection_details={},
        error=None,
        raw_result="42",
        structured_result=42,
        structured_result_error="",
        coder_runs=1,
        execution_error=None,
        coder_review=None,
        coder_context_audit=None,
        rejection_keep_tables=[],
        rejection_skip_tables=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def phase1_result(keywords=("parks", "district")) -> tuple:
    return (list(keywords), "raw keyword output", 2, "phase 1 reasoning")


def make_runtime(
    config: ExperimentConfig, tmp_path: Path, portal_name: str = "Test portal"
) -> SimpleNamespace:
    """The RuntimeSettings surface the three orchestrations read."""
    return SimpleNamespace(
        model_name=config.model,
        solr_core=config.core,
        csv_dir=tmp_path,
        portal_name=portal_name,
        retrieval=config.retrieval.to_runtime(),
        experiment=config,
        discovery=config.discovery,
        use_unified_agent=config.use_unified_agent,
    )


class Capture:
    """The side effects of a run: log rows, manifests, retrieval decisions."""

    def __init__(self, replacements: dict[str, str]) -> None:
        self.replacements = replacements
        self.log_rows: list[Any] = []
        self.manifests: list[Any] = []
        self.retrieval_decisions: list[Any] = []

    def save_experiment_log(self, **kwargs) -> None:
        self.log_rows.append(normalise(kwargs, self.replacements))

    def persist_manifest(self, manifest, directory=None, *args) -> Path:
        self.manifests.append(normalise(manifest, self.replacements))
        return Path("<manifest>")

    def log_retrieval_decision(self, **kwargs) -> None:
        self.retrieval_decisions.append(normalise(kwargs, self.replacements))

    def payload(self) -> dict[str, Any]:
        return {
            "log_rows": self.log_rows,
            "manifests": self.manifests,
            "retrieval_decisions": self.retrieval_decisions,
        }


class MatchingSolr:
    """The scripted runs' Solr: every strict-AND keyword count finds a table, so the scripted retrieval decides."""

    def select(self, tokens, **_params) -> dict[str, Any]:
        return {"response": {"numFound": 1, "docs": []}}


def install_fakes(
    monkeypatch, module, case: Case, log: CallLog, capture: Capture, names: dict[str, str]
) -> None:
    """Replace the module's phase and resource names with the scripted fakes.

    ``names`` maps the harness's role to the attribute name on ``module``: the
    service, the CLI and the UI import the same helpers, so the names agree,
    but not every surface imports every helper.
    """
    script = case.script
    fakes = {
        "get_llm": lambda _model: (object(), None),
        "get_solr": lambda _core: MatchingSolr(),
        "get_prompt_manager": object,
        "get_all_table_files": lambda _path: list(case.files),
        "phase1": log.fake("write_keywords", script.phase1, PHASE1_ARGS),
        "phase2": log.fake("select_tables", script.phase2),
        "phase12": log.fake("discover_tables", script.phase12),
        "phase3": log.fake("write_and_run_code", script.phase3, PHASE3_ARGS),
        "phase4": log.fake("phase4_synthesize", script.phase4, PHASE4_ARGS),
        "judge": log.fake("judge_semantic_code_result", script.judge),
        "save_experiment_log": capture.save_experiment_log,
        "persist_manifest": capture.persist_manifest,
        "log_retrieval_decision": capture.log_retrieval_decision,
    }
    for role, attribute in names.items():
        monkeypatch.setattr(module, attribute, fakes[role])


COMMON_NAMES = {
    "get_llm": "get_llm",
    "get_solr": "get_solr",
    "get_prompt_manager": "get_prompt_manager",
    "get_all_table_files": "get_all_table_files",
    "phase1": "write_keywords",
    "phase2": "select_tables",
    "phase12": "discover_tables",
    "phase3": "write_and_run_code",
    "phase4": "phase4_synthesize",
    "save_experiment_log": "save_experiment_log",
    "persist_manifest": "persist_manifest",
}
SERVICE_NAMES = {
    **COMMON_NAMES,
    "judge": "judge_semantic_code_result",
    "log_retrieval_decision": "log_retrieval_decision",
}


def prepare_run(monkeypatch, tmp_path: Path, module, case: Case, names: dict[str, str]):
    """Install the fakes for ``case`` on ``module`` and return the recorders."""
    replacements = {str(tmp_path): "<TMP>"}
    # The manifest records the lake's paths; under the temporary root they are
    # the same on every machine and masked with it.
    monkeypatch.setattr(core_config, "BASE_DIR", tmp_path)
    log = CallLog(replacements)
    capture = Capture(replacements)
    for name in case.files:
        (tmp_path / name).write_text("value\n42\n", encoding="utf-8")
    install_fakes(monkeypatch, module, case, log, capture, names)
    return replacements, log, capture
