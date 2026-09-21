"""Tiered union mapping vs the old average-score rule, on OrQa's union matches.

    PORTAL=uk  PYTHONPATH=src:. python analysis/pair_evidence/eval_unions.py
    PORTAL=nyc PYTHONPATH=src:. python analysis/pair_evidence/eval_unions.py     # ~5 min

Positives are OrQa's labelled unions plus tables that judge-approved queries actually concatenated.
Negatives are random pairs (a) from the whole lake and (b) among OrQa's candidate tables, always from
different source packages. Neither is guaranteed negative: same-schema files from different
publishers exist that OrQa never labelled (spend reports, organograms), so an accepted "negative" is
a prompt to LOOK at the pair, not a verdict. Recall against OrQa's own labels is partly circular
(OrQa used the same name matcher); the executed-only pairs are the independent evidence.

Rules compared:  old = average COMA score >= 0.5      tier = union_mapping (UNION / SUBSET / PARTIAL)
Expected on the UK data: old calls ~20% of random lake pairs a union; the UNION tier ~2%.
"""

from __future__ import annotations

import os
import random

import pandas as pd

import orqa_labels
from common import PORTAL, TABLE_DIR, coma_matches, table
from lakegen.agent_tools.union_mapping import NO_UNION, SUBSET_UNION, UNION, union_by_names

N_RANDOM = int(os.environ.get("N_RANDOM", "300"))
N_CANDIDATE_NEGATIVES = int(os.environ.get("N_CANDIDATE_NEGATIVES", "150"))


def _same_package(a: str, b: str) -> bool:
    return "___" in a and a.split("___")[0] == b.split("___")[0]


def _negatives(pool: list[str], n: int, avoid: set, seed: int) -> list[tuple[str, str]]:
    random.seed(seed)
    out: set[tuple[str, str]] = set()
    for _ in range(50_000):
        if len(out) >= n:
            break
        a, b = random.sample(pool, 2)
        pair = tuple(sorted((a, b)))
        if not _same_package(a, b) and pair not in avoid:
            out.add(pair)
    return sorted(out)


def measure(left: str, right: str) -> dict:
    q, r = table(left), table(right)
    matches = coma_matches(q, r)
    average = sum(m[2] for m in matches) / len(matches) if matches else 0.0
    mapping = union_by_names(q, r, matches)
    return {
        "old": average >= 0.5,
        "tier": mapping.verdict,
        "union": mapping.verdict == UNION,
        "union_or_subset": mapping.verdict in (UNION, SUBSET_UNION),
        "any": mapping.verdict != NO_UNION,
        "placeholders": mapping.placeholders,
    }


def main() -> None:
    positives, related, candidate_tables = orqa_labels.union_positives(PORTAL)
    avoid = set(positives) | related
    lake = [f[:-8] for f in os.listdir(TABLE_DIR) if f.endswith(".parquet")
            and (TABLE_DIR / f).stat().st_size < 3_000_000]
    jobs = [(v["L"], v["R"], v["source"]) for v in positives.values()]
    jobs += [(a, b, "random lake") for a, b in _negatives(lake, N_RANDOM, avoid, 11)]
    jobs += [(a, b, "candidate tables")
             for a, b in _negatives(candidate_tables, N_CANDIDATE_NEGATIVES, avoid, 7)]
    rows = []
    for number, (left, right, kind) in enumerate(jobs, 1):
        try:
            rows.append({"kind": kind, **measure(left, right)})
        except Exception as error:  # a broken table must not sink the run
            print("  skipped:", kind, f"{type(error).__name__}: {error}"[:70])
        if number % 100 == 0:
            print(f"  {number}/{len(jobs)}", flush=True)
    frame = pd.DataFrame(rows)
    order = ["cand", "cand+exec", "exec", "random lake", "candidate tables"]
    summary = frame.groupby("kind")[["old", "union", "union_or_subset", "any"]].mean().reindex(order)
    summary.insert(0, "pairs", frame.groupby("kind").size().reindex(order))
    print(f"\n{PORTAL}: share of pairs called a union (rows 1-3 are positives, 4-5 negatives)")
    print((summary * [1, 100, 100, 100, 100]).round(1).to_string())
    print(f"\npairs with placeholder header columns: {int((frame.placeholders > 0).sum())}/{len(frame)}")


if __name__ == "__main__":
    main()
