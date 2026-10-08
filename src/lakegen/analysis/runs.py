"""One table of runs: a row per stored result, with what it ran and how it went.

``load_runs`` reads the jobs of an experiment through the run store's index,
or every job of the given directories, and returns a DataFrame with one row
per result: the configuration axes as recorded (lake, retrieval mode,
architecture, selector, model), the models the result actually ran on (the
split marks a result with its model; hand switches are undone by timestamp),
the outcome (``solved`` is ``code_evaluation.supported_correct``), the
selection, the gold tables, the final ranking, the tokens and the time.

A field the job never recorded is None, never a default: ``selector`` is
None for runs before the selector existed, which all used the LLM selector.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lakegen.analysis import results as stored
from lakegen.analysis.jobs import StoredJob, iter_jobs, read_job
from lakegen.analysis.switches import MODEL_FIELDS, parse_switches, values_at
from lakegen.domain.tables import resource_id
from lakegen.experiments.store import RunStore


@dataclass(frozen=True)
class Run:
    experiment: str | None
    cell_id: str | None
    experiment_id: str
    job_dir: str
    job_id: str
    job_status: str
    index: int
    question_id: str
    question: str
    lake: str | None
    mode: str | None
    architecture: str | None
    tool_access: str | None
    selector: str | None
    model: str | None
    effective_model: str | None
    judge_model: str | None
    entity_model: str | None
    status: str
    error: str
    solved: bool
    code_applicable: bool
    exact: bool
    tables: tuple[str, ...]
    gold: tuple[str, ...]
    ranking: tuple[str, ...]
    tokens_p1_p2: int
    tokens_p3: int
    tokens_p4: int
    tokens: int
    elapsed_seconds: float
    expansion_used: bool
    timestamp: str | None
    run_id: str | None
    result: dict[str, Any] | None = None
    """The stored result itself, for analyses that read more than the columns; see ``include_results``."""


def _effective_models(job: StoredJob, entry: dict[str, Any], timestamp: str | None) -> dict[str, str | None]:
    """The model fields a result ran with: the switches undone, then the split's model.

    A split result carries the model it ran on; the API had swapped that
    model for the job's one wherever the configuration named it, so the
    judge and the entity model follow when they were the job's model.
    """
    values = values_at(job.config.values, parse_switches(job.provider_switches), timestamp)
    own_model = entry.get("model")
    if own_model and own_model != values["model"]:
        job_model = values["model"]
        values = {key: (own_model if value == job_model else value) for key, value in values.items()}
    return values


def _gold(question: dict[str, Any] | None) -> tuple[str, ...]:
    gold = ((question or {}).get("log_fields") or {}).get("SOURCE_RELEVANT_TABLE_IDS") or []
    return tuple(dict.fromkeys(resource_id(value) for value in gold))


def runs_of_job(
    job: StoredJob,
    *,
    experiment: str | None = None,
    cell_id: str | None = None,
    include_results: bool = False,
) -> list[Run]:
    config = job.config
    runs = []
    for index, entry in enumerate(job.results):
        result = entry.get("result") or {}
        manifest = result.get("experiment_manifest") or {}
        timestamp = manifest.get("timestamp")
        models = _effective_models(job, entry, timestamp)
        spent = stored.tokens(result)
        runs.append(Run(
            experiment=experiment,
            cell_id=cell_id,
            experiment_id=job.experiment_id,
            job_dir=str(job.job_dir),
            job_id=job.job_id,
            job_status=job.status,
            index=index,
            question_id=str(entry.get("source_id") if entry.get("source_id") is not None else entry.get("source_path")),
            question=str(entry.get("question") or ""),
            lake=config.get("core"),
            mode=config.get("retrieval.mode"),
            architecture=config.get("discovery_architecture"),
            tool_access=config.get("tool_access"),
            selector=config.get("discovery.selector"),
            model=config.get("model"),
            effective_model=models[MODEL_FIELDS[0]],
            judge_model=models[MODEL_FIELDS[1]],
            entity_model=models[MODEL_FIELDS[2]],
            status=str(result.get("status") or ""),
            error=str(result.get("error") or ""),
            solved=stored.solved(result),
            code_applicable=stored.code_applicable(result),
            exact=bool(stored.code_evaluation(result).get("exact_result_match")),
            tables=tuple(stored.selected_tables(result)),
            gold=_gold(job.question(entry)),
            ranking=tuple(stored.final_ranking(result)),
            tokens_p1_p2=spent.p1_p2,
            tokens_p3=spent.p3,
            tokens_p4=spent.p4,
            tokens=spent.total,
            elapsed_seconds=float(result.get("elapsed_seconds") or 0.0),
            expansion_used=stored.expansion_used(result),
            timestamp=timestamp,
            run_id=manifest.get("run_id"),
            result=result if include_results else None,
        ))
    return runs


def iter_runs(
    experiment: str | None = None,
    *,
    job_dirs: Iterable[Path | str] = (),
    store: RunStore | None = None,
    include_results: bool = False,
) -> Iterator[Run]:
    """The runs of an experiment's indexed cells, then those of every job in ``job_dirs``."""
    if experiment is not None:
        index = (store or RunStore()).load_index(experiment)
        for record in index.values():
            if not record.job_dir or not record.job_id:
                continue
            path = Path(record.job_dir) / f"{record.job_id}.json"
            if path.is_file():
                yield from runs_of_job(
                    read_job(path), experiment=experiment, cell_id=record.cell_id, include_results=include_results,
                )
    for job_dir in job_dirs:
        for job in iter_jobs(Path(job_dir)):
            yield from runs_of_job(job, include_results=include_results)


COLUMNS = tuple(name for name in Run.__dataclass_fields__ if name != "result")


def load_runs(
    experiment: str | None = None,
    *,
    job_dirs: Iterable[Path | str] = (),
    store: RunStore | None = None,
    include_results: bool = False,
) -> pd.DataFrame:
    """One row per stored result; see the module docstring for the columns.

    With ``include_results`` the frame also carries each stored result as a
    ``result`` column of dicts, for analyses that read beyond the columns.
    """
    runs = iter_runs(experiment, job_dirs=job_dirs, store=store, include_results=include_results)
    rows = [asdict(run) for run in runs]
    columns = [*COLUMNS, "result"] if include_results else list(COLUMNS)
    return pd.DataFrame([{key: row[key] for key in columns} for row in rows], columns=columns)
