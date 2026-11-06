"""LaTeX rows for the paper's results and failure tables, plus the numbers its text cites.

Reads results/metrics.json and results/failures.jsonl; writes results/tables.tex and
results/numbers.json, and prints both.

  PYTHONPATH=.:src .venv/bin/python experiments/paper_analysis/latex.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failures import DOWNSTREAM, RETRIEVAL, SELECTION  # noqa: E402
from records import LAKES, SELECTORS, STRATEGIES, load  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
LAKE_NAME = {"valencia": "Valencia", "paris": "Paris", "bologna": "Bologna", "uk": "UK",
             "nyc": "NYC", "target_bird": "TARGET", "chicago": "Chicago"}
LAKE_SHORT = {"valencia": "VLC", "paris": "PAR", "bologna": "BLQ", "uk": "UK",
              "nyc": "NYC", "target_bird": "TGT", "chicago": "CHI"}
STRATEGY_NAME = {"keyword": "Keyword", "semantic": "Semantic", "hybrid": "Hybrid",
                 "duckdb": "Content", "pneuma": "Pneuma", "seeker": "Pneuma-Seeker"}
LABEL = {
    "missing_metadata": "Missing metadata", "wrong_year": "Wrong year or period",
    "wrong_geography": "Wrong geo./entity scope", "multilingual": "Multilingual mismatch",
    "value_in_cells": "Value only in cells", "and_restrictive": "Restrictive AND query",
    "terminology": "Terminology mismatch", "outranked": "Matched but outranked$^{*}$",
    "not_inspected": "Candidate not inspected", "join_missed": "Join not recognised",
    "extra_table": "Unnecessary table", "column_ambiguity": "Column-level ambiguity",
    "wrong_rejection": "Incorrect rejection",
    "coder_rejected": "Coder rejected tables$^{*}$", "execution_error": "Execution error",
    "cleaning": "Incorrect cleaning", "wrong_filter": "Wrong filter", "wrong_aggregation": "Wrong aggregation",
    "wrong_query": "Right tables, wrong query", "unverified": "Judge could not verify$^{*}$",
}


def f2(value) -> str:
    return "--" if value is None else f"{value:.2f}".replace("0.", ".", 1) if value < 1 else "1.00"


def results_rows(metrics: list[dict]) -> str:
    columns = [
        ("H@1", lambda r: (r.get("retrieval") or {}).get("H@1")),
        ("R@10", lambda r: (r.get("retrieval") or {}).get("R@10")),
        ("Ctx", lambda r: (r.get("retrieval") or {}).get("in_context")),
        ("SelL", lambda r: r.get("selection_llm")),
        ("SelJ", lambda r: r.get("selection_jev")),
        ("SolL", lambda r: r.get("solved_llm")),
        ("SolJ", lambda r: r.get("solved_jev")),
    ]
    lines = []
    for lake in LAKES:
        rows = [r for r in metrics if r["lake"] == lake]
        best = {name: max((round(get(r), 2) for r in rows if get(r) is not None), default=None) for name, get in columns}
        for i, r in enumerate(rows):
            cells = []
            for name, get in columns:
                value = get(r)
                text = f2(value)
                if value is not None and best[name] is not None and round(value, 2) == best[name]:
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            head = f"\\multirow{{6}}{{*}}{{{LAKE_NAME[lake]}}}" if i == 0 else ""
            lines.append(f"{head} & {STRATEGY_NAME[r['strategy']]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    return "\n".join(lines)


def failure_rows(failures: list[dict]) -> tuple[str, dict]:
    counts = defaultdict(Counter)
    for row in failures:
        counts[(row["stage"], row["label"])][row["lake"]] += 1
    lines, totals = [], {}
    for stage, labels, title in (("retrieval", RETRIEVAL, "Retrieval"), ("selection", SELECTION, "Selection"),
                                 ("downstream", DOWNSTREAM, "Downstream")):
        stage_total = Counter()
        for label in labels:
            stage_total.update(counts[(stage, label)])
        totals[stage] = dict(stage_total)
        lines.append(f"\\multicolumn{{{len(LAKES) + 2}}}{{l}}{{\\emph{{{title} failures}}}} \\\\")
        for label in labels:
            c = counts[(stage, label)]
            cells = []
            for lake in LAKES:
                n, total = c[lake], stage_total[lake]
                cells.append(f"{n}" if not total else f"{n} ({round(100 * n / total)})")
            lines.append(f"{LABEL[label]} & " + " & ".join(cells) + f" & {sum(c.values())} \\\\")
        lines.append(f"\\emph{{Total}} & " + " & ".join(str(stage_total[l]) for l in LAKES)
                     + f" & {sum(stage_total.values())} \\\\")
        lines.append("\\hline")
    return "\n".join(lines), totals


def funnel(records) -> dict:
    """Per lake, pooled over strategies and selectors: P(in context), P(selection right |
    in context), P(solved | selection right and coded), and P(solved) end to end."""
    out = {}
    for lake in LAKES:
        rs = [r for r in records if r.lake == lake]
        ctx = [r for r in rs if r.in_context()]
        right = [r for r in ctx if r.selection_correct]
        coded = [r for r in right if r.solved is not None]
        scored = [r for r in rs if r.solved is not None]
        out[lake] = {
            "runs": len(rs),
            "in_context": len(ctx) / len(rs),
            "selection_given_context": len(right) / len(ctx) if ctx else None,
            "solved_given_selection": sum(r.solved for r in coded) / len(coded) if coded else None,
            "solved": sum(r.solved for r in scored) / len(scored) if scored else None,
            "coded_share_of_right_selections": len(coded) / len(right) if right else None,
        }
    return out


def main() -> None:
    metrics = json.loads((OUT / "metrics.json").read_text())
    failures = [json.loads(line) for line in (OUT / "failures.jsonl").open()]
    records = load()
    results = results_rows(metrics)
    failure_table, totals = failure_rows(failures)
    numbers = {"funnel": funnel(records), "failure_totals": totals}
    # strategy and selector means across lakes, for the text
    by = defaultdict(list)
    for r in metrics:
        for key in ("selection_llm", "selection_jev", "solved_llm", "solved_jev"):
            if r.get(key) is not None:
                by[(r["strategy"], key)].append(r[key])
        if r.get("retrieval"):
            by[(r["strategy"], "H@1")].append(r["retrieval"]["H@1"])
            by[(r["strategy"], "R@10")].append(r["retrieval"]["R@10"])
            by[(r["strategy"], "in_context")].append(r["retrieval"]["in_context"])
    numbers["strategy_means"] = {f"{s}|{k}": round(sum(v) / len(v), 3) for (s, k), v in by.items()}
    wins = Counter()
    for r in metrics:
        for stage in ("selection", "solved"):
            a, b = r.get(f"{stage}_llm"), r.get(f"{stage}_jev")
            if a is None or b is None:
                continue
            wins[(r["lake"] in ("target_bird", "chicago"), stage, "jev" if round(b, 2) > round(a, 2) else "llm" if round(a, 2) > round(b, 2) else "tie")] += 1
    numbers["selector_wins"] = {f"{'pipeline' if p else 'portals'}|{s}|{w}": n for (p, s, w), n in wins.items()}
    (OUT / "tables.tex").write_text("% results rows\n" + results + "\n\n% failure rows\n" + failure_table + "\n")
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(results)
    print()
    print(failure_table)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
