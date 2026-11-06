"""Summaries of the runs table: one row per cell, and the questions of a cell.

``cell_summaries`` reduces a runs frame (``load_runs``) to one row per cell
with its progress and its rates: solved (``supported_correct``), exact
selection, Hit@1, Recall@10 and in-context@20 (every gold table within the
first 20), with the mean time and tokens. ``cell_questions`` lists one
cell's results question by question, with where each gold table ranked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

IN_CONTEXT_K = 20


def _rate(values: pd.Series) -> float | None:
    return float(values.mean()) if len(values) else None


def _exact(tables: Sequence[str], gold: Sequence[str]) -> bool:
    return bool(gold) and set(tables) == set(gold)


def _hit_at(ranking: Sequence[str], gold: Sequence[str], k: int) -> bool:
    return bool(set(ranking[:k]) & set(gold))


def _recall_at(ranking: Sequence[str], gold: Sequence[str], k: int) -> float | None:
    return len(set(ranking[:k]) & set(gold)) / len(set(gold)) if gold else None


def _in_context(ranking: Sequence[str], gold: Sequence[str], k: int = IN_CONTEXT_K) -> bool:
    return bool(gold) and set(gold) <= set(ranking[:k])


def _scored(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["exact"] = [_exact(t, g) for t, g in zip(scored.tables, scored.gold, strict=True)]
    scored["hit1"] = [_hit_at(r, g, 1) for r, g in zip(scored.ranking, scored.gold, strict=True)]
    scored["recall10"] = [
        _recall_at(r, g, 10) for r, g in zip(scored.ranking, scored.gold, strict=True)
    ]
    scored["in_context"] = [
        _in_context(r, g) for r, g in zip(scored.ranking, scored.gold, strict=True)
    ]
    scored["completed"] = scored.status == "completed"
    return scored


def _first(values: pd.Series) -> Any:
    return values.iloc[0] if len(values) else None


def cell_summaries(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per cell of the frame, in order of first appearance."""
    if frame.empty:
        return []
    scored = _scored(frame)
    summaries = []
    for cell_id, group in scored.groupby("cell_id", sort=False, dropna=False):
        with_gold = group[group.gold.map(len) > 0]
        summaries.append(
            {
                "cell_id": cell_id,
                "experiment": _first(group.experiment),
                "experiment_id": _first(group.experiment_id),
                "job_id": _first(group.job_id),
                "job_status": _first(group.job_status),
                "lake": _first(group.lake),
                "mode": _first(group["mode"]),
                "selector": _first(group.selector),
                "architecture": _first(group.architecture),
                "model": _first(group.model),
                "effective_models": sorted({m for m in group.effective_model if m}),
                "n": int(len(group)),
                "completed": int(group.completed.sum()),
                "failed": int((~group.completed).sum()),
                "solved": _rate(group.solved),
                "exact": _rate(with_gold.exact),
                "hit1": _rate(with_gold.hit1),
                "recall10": _rate(with_gold.recall10.dropna()),
                "in_context": _rate(with_gold.in_context),
                "elapsed_mean": float(group.elapsed_seconds.mean()),
                "tokens_mean": float(group.tokens.mean()),
                "tokens_total": int(group.tokens.sum()),
            }
        )
    return summaries


def cell_questions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The results of a cell's frame, one per question, in result order."""
    rows = []
    for run in _scored(frame).itertuples(index=False):
        ranks = {table: position + 1 for position, table in enumerate(run.ranking)}
        rows.append(
            {
                "index": int(run.index),
                "question_id": run.question_id,
                "question": run.question,
                "status": run.status,
                "error": run.error,
                "solved": bool(run.solved),
                "exact": bool(run.exact),
                "in_context": bool(run.in_context),
                "gold": list(run.gold),
                "gold_ranks": {table: ranks.get(table) for table in run.gold},
                "tables": list(run.tables),
                "effective_model": run.effective_model,
                "elapsed_seconds": float(run.elapsed_seconds),
                "tokens": int(run.tokens),
                "timestamp": run.timestamp,
                "run_id": run.run_id,
            }
        )
    return rows
