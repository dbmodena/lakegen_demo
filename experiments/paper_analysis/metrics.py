"""Retrieval, selection and coding results per lake x strategy x selector.

Writes results/metrics.json and results/metrics.md.

  PYTHONPATH=.:src .venv/bin/python experiments/paper_analysis/metrics.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import LAKES, SELECTORS, STRATEGIES, load  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    records = load()
    groups = defaultdict(list)
    for r in records:
        groups[(r.lake, r.strategy, r.selector)].append(r)
    rows = []
    for lake in LAKES:
        for strategy in STRATEGIES:
            row = {"lake": lake, "strategy": strategy}
            for selector in SELECTORS:
                rs = groups.get((lake, strategy, selector), [])
                if not rs:
                    continue
                row[f"n_{selector}"] = len(rs)
                row[f"retrieval_{selector}"] = {
                    "H@1": mean(r.hit_at(1) for r in rs),
                    "R@10": mean(r.recall_at(10) for r in rs),
                    "in_context": mean(float(r.in_context()) for r in rs),
                }
                row[f"selection_{selector}"] = mean(float(r.selection_correct) for r in rs)
                row[f"solved_{selector}"] = mean(r.solved for r in rs)
                row[f"coded_{selector}"] = sum(r.solved is not None for r in rs)
            # Retrieval is shared by both selectors on the portals (same context);
            # on the pipeline lakes each selector's run retrieved on its own, so
            # the reported retrieval is the mean of the two runs.
            parts = [row[f"retrieval_{s}"] for s in SELECTORS if f"retrieval_{s}" in row]
            if parts:
                row["retrieval"] = {k: mean(p[k] for p in parts) for k in parts[0]}
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))
    lines = ["| lake | strategy | H@1 | R@10 | in ctx | sel LLM | sel Jev | solved LLM (coded) | solved Jev (coded) |",
             "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{v:.2f}"
    for row in rows:
        ret = row.get("retrieval", {})
        lines.append(
            f"| {row['lake']} | {row['strategy']} | {f(ret.get('H@1'))} | {f(ret.get('R@10'))} | {f(ret.get('in_context'))} | "
            f"{f(row.get('selection_llm'))} | {f(row.get('selection_jev'))} | "
            f"{f(row.get('solved_llm'))} ({row.get('coded_llm', 0)}/{row.get('n_llm', 0)}) | "
            f"{f(row.get('solved_jev'))} ({row.get('coded_jev', 0)}/{row.get('n_jev', 0)}) |")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
