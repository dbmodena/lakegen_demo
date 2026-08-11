"""TARGET-oriented ranking evaluation and trace logging."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
from typing import Any

from lakegen.retrieval.models import RetrievalRun


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items))


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    relevant: Iterable[str] | Mapping[str, float],
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Hit@k, Recall@k, MRR, and binary/graded nDCG@k."""
    ranking = _unique(retrieved_ids)
    if isinstance(relevant, Mapping):
        grades = {str(key): float(value) for key, value in relevant.items()}
    else:
        grades = {key: 1.0 for key in _unique(relevant)}
    grades = {
        key: value
        for key, value in grades.items()
        if math.isfinite(value) and value > 0
    }
    if not grades:
        raise ValueError("At least one relevant document is required")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    result: dict[str, float] = {}
    first_relevant = next(
        (rank for rank, doc_id in enumerate(ranking, 1) if doc_id in grades),
        None,
    )
    result["MRR"] = 1.0 / first_relevant if first_relevant is not None else 0.0

    ideal_grades = sorted(grades.values(), reverse=True)
    for k in k_values:
        prefix = ranking[:k]
        matches = sum(doc_id in grades for doc_id in prefix)
        result[f"Hit@{k}"] = float(matches > 0)
        result[f"Recall@{k}"] = matches / len(grades)
        dcg = sum(
            (2.0 ** grades.get(doc_id, 0.0) - 1.0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(prefix, 1)
        )
        ideal_dcg = sum(
            (2.0**grade - 1.0) / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_grades[:k], 1)
        )
        result[f"nDCG@{k}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return result


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set.intersection(*(set(row) for row in rows))
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in sorted(keys)
    }


class RetrievalRunLogger:
    """Append full rankings as JSONL for retriever-only evaluation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, run: RetrievalRun) -> None:
        payload: dict[str, Any] = run.to_log_dict()
        payload["event_type"] = "retrieval_run"
        self.log_payload(payload)

    def log_payload(self, payload: Mapping[str, Any]) -> None:
        payload = dict(payload)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
