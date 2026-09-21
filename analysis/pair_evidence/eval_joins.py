"""Measured join keys vs the name-only rule, on OrQa's labelled joins.

    PORTAL=uk  PYTHONPATH=src:. python analysis/pair_evidence/eval_joins.py
    PORTAL=nyc PYTHONPATH=src:. python analysis/pair_evidence/eval_joins.py     # ~12 min

Baseline = the previous tool: the best-scoring COMA pair IS the key. A labelled join "is reproduced"
by a key when an inner join on it produces the same matched row pairs (>= 95% overlap), which is
fairer than comparing column names: equivalent identifiers (BBL / BIN / Boro-Block-Lot) join alike.
Expected on the UK data: 42 of 56 labelled keys accepted, top-1 25/42 (baseline 20/42), best-of-3
37/42 (baseline 25/42).
"""

from __future__ import annotations

import pandas as pd

import orqa_labels
from common import PORTAL, coma_matches, table
from lakegen.agent_tools.join_keys import (
    JOIN, _composite, _normalise_pair, find_join_keys, measure_key,
)

MAX_PAIRS = 3_000_000   # never materialise a huge N:N join


def matched_pairs(q: pd.DataFrame, r: pd.DataFrame, left_cols, right_cols) -> set | None:
    """The (left_row, right_row) pairs an inner join on these keys produces (None if too large)."""
    left_parts, right_parts = [], []
    for left_col, right_col in zip(left_cols, right_cols):
        left_norm, right_norm, _ = _normalise_pair(q[left_col], r[right_col])
        left_parts.append(left_norm)
        right_parts.append(right_norm)
    left_key, right_key = _composite(left_parts), _composite(right_parts)
    left_counts, right_counts = left_key.value_counts(), right_key.value_counts()
    shared = left_counts.index.intersection(right_counts.index)
    size = int((left_counts[shared].to_numpy("int64") * right_counts[shared].to_numpy("int64")).sum())
    if size > MAX_PAIRS:
        return None
    left = pd.DataFrame({"k": left_key.to_numpy(), "li": left_key.index.to_numpy()})
    right = pd.DataFrame({"k": right_key.to_numpy(), "ri": right_key.index.to_numpy()})
    merged = left.merge(right, on="k")
    return set(zip(merged.li.tolist(), merged.ri.tolist()))


def overlap(a: set | None, b: set | None) -> float | None:
    if a is None or b is None:
        return None
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def evaluate(key: tuple, meta: dict) -> dict:
    left_id, right_id, left_on, right_on = key
    q, r = table(left_id), table(right_id)
    if any(c not in q.columns for c in left_on) or any(c not in r.columns for c in right_on):
        return {"source": meta["source"], "error": "label column not in table"}
    matches = coma_matches(q, r)
    ranked = find_join_keys(q, r, matches)
    labelled = measure_key(q, r, list(left_on), list(right_on), 1.0)
    truth = matched_pairs(q, r, left_on, right_on)

    def reproduces(pairs: list[tuple[str, str]]) -> float | None:
        return overlap(truth, matched_pairs(q, r, [p[0] for p in pairs], [p[1] for p in pairs]))

    def best_of(candidates: list[list[tuple[str, str]]]) -> float | None:
        scores = [s for s in (reproduces(c) for c in candidates) if s is not None]
        return max(scores, default=None)

    ours = [list(zip(e.left_cols, e.right_cols)) for e in ranked[:3]]
    baseline = [[(a, b)] for a, b, _ in matches[:3]]
    return {
        "source": meta["source"], "how": meta["how"], "verdict": labelled.verdict,
        "top1": reproduces(ours[0]) if ours else None, "top3": best_of(ours),
        "base1": reproduces(baseline[0]) if baseline else None, "base3": best_of(baseline),
    }


def main() -> None:
    labels = orqa_labels.join_labels(PORTAL)
    rows = []
    for number, (key, meta) in enumerate(labels.items(), 1):
        try:
            rows.append(evaluate(key, meta))
        except Exception as error:  # a broken table must not sink the run
            rows.append({"source": meta["source"], "error": f"{type(error).__name__}: {error}"[:80]})
        if number % 25 == 0:
            print(f"  {number}/{len(labels)}", flush=True)
    frame = pd.DataFrame(rows)
    ok = frame[frame.get("error", pd.Series(dtype=object)).isna()] if "error" in frame else frame
    accepted = ok[ok.verdict == JOIN]
    print(f"\n{PORTAL}: {len(labels)} labelled joins, {len(frame) - len(ok)} unusable, "
          f"{len(accepted)}/{len(ok)} labelled keys accepted (the rest share no values or are a constant)")
    for name, group in (("all", accepted), ("executed + judged", accepted[accepted.source != "cand"])):
        n = len(group)
        print(f"  [{name}, n={n}] a key reproduces the labelled join (>=95% matched-row overlap):")
        for label, column in (("measured top-1", "top1"), ("baseline top-1", "base1"),
                              ("measured best-of-3", "top3"), ("baseline best-of-3", "base3")):
            print(f"      {label:20s}{int((group[column] >= 0.95).sum()):4d}/{n}")


if __name__ == "__main__":
    main()
