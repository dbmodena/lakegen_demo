"""Every table, figure and number of the evaluation section (RQ1-RQ4).

Reads the records of records.py, results/rerank.jsonl (rerank_compare.py),
results/question_meta.json (question_meta.py) and results/failures.jsonl
(failures.py). Writes results/section_tables.json (LaTeX snippets by placeholder
name) and results/section_numbers.json (the numbers the text quotes).

  PYTHONPATH=.:src .venv/bin/python experiments/paper_analysis/section_tables.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import BENCH, PORTALS, STRATEGIES, load, solved_flag  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
NAME = {"valencia": "Valencia", "paris": "Paris", "bologna": "Bologna", "uk": "UK", "nyc": "NYC",
        "target_bird": "TARGET", "chicago": "Chicago"}
STRAT = {"keyword": "Keyword", "semantic": "Semantic", "hybrid": "Hybrid", "duckdb": "Content",
         "pneuma": "Pneuma", "seeker": "Pneuma-Seeker"}
PLATFORM = {"valencia": "CKAN", "paris": "OpenDataSoft", "bologna": "OpenDataSoft", "uk": "CKAN",
            "nyc": "Socrata", "target_bird": "BIRD/TARGET", "chicago": "Pneuma"}
LANG = {"valencia": "es, ca", "paris": "fr", "bologna": "it", "uk": "en", "nyc": "en",
        "target_bird": "en", "chicago": "en"}
SHORT = {"keyword": "Keyword", "semantic": "Semantic", "hybrid": "Hybrid", "duckdb": "Content",
         "pneuma": "Pneuma", "seeker": "Seeker"}
DEFAULT = "hybrid"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def f2(v) -> str:
    if v is None:
        return "--"
    return "1.00" if round(v, 2) >= 1 else f"{v:.2f}".replace("0.", ".", 1)


def pct(v) -> str:
    return "--" if v is None else f"{100 * v:.0f}"


# ------------------------------------------------------------------ measures

def rank_metrics(ranking: list[str], gold: list[str], any_gold: bool = False) -> dict:
    G = set(gold)
    first = next((i + 1 for i, t in enumerate(ranking) if t in G), None)
    out = {"MRR": 1.0 / first if first else 0.0}
    for k in (1, 5, 10, 20):
        top = set(ranking[:k])
        hit = len(top & G)
        out[f"R@{k}"] = float(hit > 0) if any_gold else hit / len(G)
        out[f"Cpl@{k}"] = float(hit > 0) if any_gold else float(G <= top)
        out[f"P@{k}"] = hit / k
    dcg = sum(1 / math.log2(i + 2) for i, t in enumerate(ranking[:10]) if t in G)
    ideal = sum(1 / math.log2(i + 2) for i in range(1 if any_gold else min(len(G), 10)))
    out["nDCG@10"] = min(dcg / ideal, 1.0) if ideal else 0.0
    return out


def selection_metrics(r) -> dict:
    S, G = set(r.selected), set(r.gold)
    inter = len(S & G)
    if r.any_gold:
        good = float(inter > 0)
        return {"recall": good, "precision": inter / len(S) if S else 0.0, "exact": good,
                "answerable": good, "unnecessary": float(bool(S - G)), "empty": float(not S),
                "f1": good}
    recall = inter / len(G)
    precision = inter / len(S) if S else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "exact": float(S == G),
            "answerable": float(G <= S), "unnecessary": float(bool(S - G)), "empty": float(not S)}


def coding_flags(r) -> dict | None:
    """Mean over the codings of the selected set: executed, exact, lenient, correct."""
    codings = (r.extra or {}).get("codings") or []
    if not r.selected:
        return {"executed": 0.0, "exact": 0.0, "lenient": 0.0, "correct": 0.0}
    if not codings:
        return None
    rows = []
    for result in codings:
        ev = result.get("code_evaluation") or {}
        executed = ev.get("final_execution_success")
        executed = ev.get("execution_success") if executed is None else executed
        exact = bool(ev.get("exact_result_match") or ev.get("representation_equivalent_match"))
        rows.append({"executed": float(bool(executed)), "exact": float(exact),
                     "lenient": float(exact or bool(ev.get("lenient_result_match"))),
                     "correct": float(solved_flag(result))})
    return {k: mean(row[k] for row in rows) for k in rows[0]}


# ------------------------------------------------------------------ tables

def bold_best(rows: list[list], columns: range) -> list[list[str]]:
    best = {c: max((round(r[c], 2) for r in rows if isinstance(r[c], float)), default=None) for c in columns}
    out = []
    for r in rows:
        cells = []
        for c, v in enumerate(r):
            if c in columns and isinstance(v, float):
                text = f2(v)
                cells.append(f"\\textbf{{{text}}}" if best[c] is not None and round(v, 2) == best[c] else text)
            else:
                cells.append(str(v))
        out.append(cells)
    return out


def main() -> None:
    records = load()
    meta = json.loads((OUT / "question_meta.json").read_text())
    rerank_rows = [json.loads(l) for l in (OUT / "rerank.jsonl").open()]
    tables, numbers = {}, {}
    portal = [r for r in records if r.lake in PORTALS]
    # one retrieval per (lake, strategy, question) on the portals: the selectors share it
    shared = {(r.lake, r.strategy, r.qid): r for r in portal if r.selector == "llm"}

    # ---- RQ1 retrieval per portal and strategy
    keys = ["R@1", "R@5", "R@10", "R@20", "MRR", "nDCG@10", "Cpl@20"]
    per = defaultdict(list)
    for (lake, strategy, qid), r in shared.items():
        per[(lake, strategy)].append(rank_metrics(r.ranking, r.gold))
    lines, rq1 = [], {}
    # TARGET and Chicago come from the whole-pipeline runs: every run retrieved on
    # its own (agent keywords), so both selectors' runs are pooled.
    for r in records:
        if r.lake in BENCH:
            per[(r.lake, r.strategy)].append(rank_metrics(r.ranking, r.gold, r.any_gold))
    for lake in PORTALS + ("all",) + BENCH:
        rows = []
        for strategy in STRATEGIES:
            if lake == "all":
                ms = [m for l in PORTALS for m in per[(l, strategy)]]
            else:
                ms = per[(lake, strategy)]
            vals = {k: mean(m[k] for m in ms) for k in keys}
            rq1[f"{lake}|{strategy}"] = vals
            label = NAME.get(lake, "Mean") + ("$^{\\ddagger}$" if lake in BENCH else "")
            rows.append([label if strategy == "keyword" else "", STRAT[strategy]] + [vals[k] for k in keys])
        for cells in bold_best(rows, range(2, 2 + len(keys))):
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline\\hline" if lake == "all" else "\\hline")
    tables["RQ1_ROWS"] = "\n".join(lines)
    numbers["rq1"] = rq1

    # ---- RQ1 scope / relation breakdown (portals pooled, Cpl@20 and R@10)
    groups = [("single", lambda q: q["scope"] == "single_table"), ("multi", lambda q: q["scope"] == "multi_table"),
              ("join", lambda q: q["relation"] == "join"), ("union", lambda q: q["relation"] == "union"),
              ("join-correlation", lambda q: q["relation"] == "join-correlation")]
    lines, scope_numbers = [], {}
    for strategy in STRATEGIES:
        cells = [STRAT[strategy]]
        for name, test in groups:
            ms = [rank_metrics(r.ranking, r.gold) for (l, s, q), r in shared.items()
                  if s == strategy and test(meta[l][q])]
            cpl, r10 = mean(m["Cpl@20"] for m in ms), mean(m["R@10"] for m in ms)
            scope_numbers[f"{strategy}|{name}"] = {"Cpl@20": cpl, "R@10": r10, "n": len(ms)}
            cells += [f2(r10), f2(cpl)]
        lines.append(" & ".join(cells) + " \\\\")
    counts = [sum(1 for (l, s, q) in shared if s == "keyword" and test(meta[l][q])) for _, test in groups]
    tables["SCOPE_ROWS"] = "\n".join(lines)
    tables["SCOPE_N"] = " & ".join(f"\\multicolumn{{2}}{{c}}{{$n={n}$}}" for n in counts)
    numbers["scope"] = scope_numbers

    # ---- RQ1 re-ranking: none vs Cohere on the same 20 candidates
    arms = defaultdict(list)
    for row in rerank_rows:
        arms[(row["strategy"], row["arm"], row["lake"])].append(rank_metrics(row["ranking"], row["gold"]))
    rkeys = ["R@1", "R@5", "R@10", "MRR", "nDCG@10"]
    lines, rr = [], {}
    for strategy in STRATEGIES:
        pair = {}
        for arm in ("none", "cohere"):
            ms = [m for lake in PORTALS for m in arms[(strategy, arm, lake)]]
            pair[arm] = {k: mean(m[k] for m in ms) for k in rkeys}
            pair[arm]["lakes_better"] = None
        better = sum(1 for lake in PORTALS
                     if mean(m["MRR"] for m in arms[(strategy, "cohere", lake)]) >
                     mean(m["MRR"] for m in arms[(strategy, "none", lake)]) + 1e-9)
        rr[strategy] = {"none": pair["none"], "cohere": pair["cohere"], "portals_mrr_up": better}
        production = "cohere" if strategy in ("keyword", "semantic", "hybrid") else "none"
        for arm, label in (("none", "none"), ("cohere", "Cohere")):
            cells = [STRAT[strategy] if arm == "none" else "", label + ("$^{\\dagger}$" if arm == production else "")]
            for k in rkeys:
                v, other = pair[arm][k], pair["cohere" if arm == "none" else "none"][k]
                text = f2(v)
                cells.append(f"\\textbf{{{text}}}" if round(v, 2) > round(other, 2) else text)
            cells.append(f"{better}/5" if arm == "cohere" else "")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")
    tables["RERANK_ROWS"] = "\n".join(lines)
    numbers["rerank"] = rr

    # ---- RQ2 selection (portals pooled), per strategy and selector
    skeys = ["recall", "precision", "f1", "exact", "answerable", "unnecessary", "empty"]
    lines, sel = [], {}
    for strategy in STRATEGIES:
        for selector in ("llm", "jev"):
            rs = [r for r in portal if r.strategy == strategy and r.selector == selector]
            ms = [selection_metrics(r) for r in rs]
            vals = {k: mean(m[k] for m in ms) for k in skeys}
            ok = [selection_metrics(r)["exact"] for r in rs if r.in_context()]
            vals["exact_given_retrieval"] = mean(ok)
            sel[f"{strategy}|{selector}"] = vals
            cells = [STRAT[strategy] if selector == "llm" else "", "LLM" if selector == "llm" else "Jev"]
            cells += [f2(vals[k]) for k in ("recall", "precision", "f1", "exact", "exact_given_retrieval", "answerable", "unnecessary", "empty")]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")
    tables["SEL_ROWS"] = "\n".join(lines)
    numbers["selection"] = sel
    for selector in ("llm", "jev"):
        ms = [selection_metrics(r) for r in portal if r.selector == selector]
        numbers[f"selection_all|{selector}"] = {k: mean(m[k] for m in ms) for k in skeys}

    # ---- RQ3 cross-lake summary under the default strategy
    lines, cross = [], {}
    for lake in PORTALS + BENCH:
        rs = [r for r in records if r.lake == lake and r.strategy == DEFAULT]
        cpl = {s: mean(float(r.in_context()) for r in records if r.lake == lake and r.strategy == s) for s in STRATEGIES}
        best = max(STRATEGIES, key=lambda s: cpl[s] if cpl[s] is not None else -1)
        row = {"cpl": cpl[DEFAULT], "best": best, "best_cpl": cpl[best]}
        for selector in ("llm", "jev"):
            sr = [r for r in rs if r.selector == selector]
            row[f"exact_{selector}"] = mean(selection_metrics(r)["exact"] for r in sr)
            row[f"qa_{selector}"] = mean(r.solved for r in sr)
        row["keyword_cpl"], row["semantic_cpl"] = cpl["keyword"], cpl["semantic"]
        cross[lake] = row
        mark = "$^{\\ddagger}$" if lake in BENCH else ""
        lines.append(" & ".join([NAME[lake] + mark, PLATFORM[lake], LANG[lake], f2(row["keyword_cpl"]), f2(row["semantic_cpl"]),
                                 f2(row["cpl"]), f"{SHORT[best]} ({f2(row['best_cpl'])})",
                                 f2(row["exact_llm"]), f2(row["exact_jev"]), f2(row["qa_llm"]), f2(row["qa_jev"])]) + " \\\\")
        if lake == PORTALS[-1]:
            lines.append("\\hline")
    tables["CROSS_ROWS"] = "\n".join(lines)
    numbers["cross"] = cross

    # ---- RQ4 waterfall (default strategy): retrieved -> answerable -> executed -> correct
    def waterfall(rs) -> dict:
        stages = {"retrieved": [], "answerable": [], "executed": [], "correct": [], "offpath": [], "exact_sel": []}
        for r in rs:
            flags = coding_flags(r)
            retrieved = r.in_context()
            answerable = retrieved and selection_metrics(r)["answerable"] == 1.0
            stages["retrieved"].append(float(retrieved))
            stages["exact_sel"].append(float(retrieved and selection_metrics(r)["exact"] == 1.0))
            stages["answerable"].append(float(answerable))
            if flags is None:
                continue
            stages["executed"].append(flags["executed"] if answerable else 0.0)
            stages["correct"].append(flags["correct"] if answerable else 0.0)
            stages["offpath"].append(0.0 if answerable else (r.solved or 0.0))
        return {k: mean(v) for k, v in stages.items()} | {"n": len(rs)}

    lines, wf = [], {}
    rows_spec = [(lake, "llm") for lake in PORTALS] + [("portals", "llm")] + [(l, s) for l in BENCH for s in ("llm", "jev")]
    for lake, selector in rows_spec:
        rs = [r for r in records if r.strategy == DEFAULT and r.selector == selector
              and (r.lake in PORTALS if lake == "portals" else r.lake == lake)]
        w = waterfall(rs)
        wf[f"{lake}|{selector}"] = w
        label = ("All portals" if lake == "portals" else NAME[lake]) + (" (Jev)" if selector == "jev" else " (LLM)" if lake in BENCH else "")
        lines.append(" & ".join([label, pct(w["retrieved"]), pct(w["answerable"]), pct(w["executed"]),
                                 pct(w["correct"]), pct(w["exact_sel"]), pct(w["offpath"])]) + " \\\\")
        if lake == PORTALS[-1] or lake == "portals":
            lines.append("\\hline")
    tables["WATERFALL_ROWS"] = "\n".join(lines)
    numbers["waterfall"] = wf
    # pooled over every strategy too, for the text
    numbers["waterfall_all_strategies"] = {
        f"{group}|{selector}": waterfall([r for r in records if r.selector == selector and
                                          (r.lake in PORTALS if group == "portals" else r.lake == group)])
        for group in ("portals",) + BENCH for selector in ("llm", "jev") if not (group == "portals" and selector == "jev")}

    # ---- the waterfall figure (portals, default strategy, LLM selector)
    w = wf["portals|llm"]
    bars = [("Questions", 1.0), ("Retrieved", w["retrieved"]), ("Answerable", w["answerable"]),
            ("Executed", w["executed"]), ("Correct", w["correct"])]
    fig = ["\\begin{tikzpicture}[x=1cm,y=3.0cm]"]
    for i, (label, value) in enumerate(bars):
        x = 2.0 * i
        drop = "" if not i else f" ($-{100 * (bars[i - 1][1] - value):.0f}$)"
        fig.append(f"\\fill[black!{70 - 12 * i}] ({x:.2f},0) rectangle ({x + 1.3:.2f},{value:.3f});")
        fig.append(f"\\node[above,font=\\footnotesize] at ({x + 0.65:.2f},{value:.3f}) {{{100 * value:.0f}\\%{drop}}};")
        fig.append(f"\\node[below,font=\\footnotesize] at ({x + 0.65:.2f},0) {{{label}}};")
    fig.append("\\end{tikzpicture}")
    tables["WATERFALL_FIGURE"] = "\n".join(fig)

    # ---- RQ4 difficulty, scope and relation (portals, default strategy)
    buckets = [("Easy", lambda q: q["difficulty"] == "easy"), ("Medium", lambda q: q["difficulty"] == "medium"),
               ("Hard", lambda q: q["difficulty"] == "hard"), ("Single-table", lambda q: q["scope"] == "single_table"),
               ("Multi-table", lambda q: q["scope"] == "multi_table"), ("\\quad Join", lambda q: q["relation"] == "join"),
               ("\\quad Union", lambda q: q["relation"] == "union"),
               ("\\quad Join-correlation", lambda q: q["relation"] == "join-correlation")]
    lines, diff = [], {}
    for label, test in buckets:
        rs = {s: [r for r in portal if r.strategy == DEFAULT and r.selector == s and test(meta[r.lake][r.qid])] for s in ("llm", "jev")}
        ms = [rank_metrics(r.ranking, r.gold) for r in rs["llm"]]
        row = {"n": len(rs["llm"]), "R@10": mean(m["R@10"] for m in ms), "Cpl@20": mean(m["Cpl@20"] for m in ms)}
        for s in ("llm", "jev"):
            sm = [selection_metrics(r) for r in rs[s]]
            row[f"f1_{s}"] = mean(m["f1"] for m in sm)
            row[f"exact_{s}"] = mean(m["exact"] for m in sm)
            row[f"qa_{s}"] = mean(r.solved for r in rs[s])
        diff[label.replace("\\quad ", "")] = row
        lines.append(" & ".join([label, str(row["n"]), f2(row["R@10"]), f2(row["Cpl@20"]), f2(row["f1_llm"]), f2(row["f1_jev"]),
                                 f2(row["exact_llm"]), f2(row["exact_jev"]), f2(row["qa_llm"]), f2(row["qa_jev"])]) + " \\\\")
        if label in ("Hard",):
            lines.append("\\hline")
    tables["DIFF_ROWS"] = "\n".join(lines)
    numbers["difficulty"] = diff
    # TARGET by BIRD difficulty, for the text
    numbers["target_difficulty"] = {
        f"{d}|{s}": {"n": len(rs), "exact": mean(selection_metrics(r)["exact"] for r in rs), "qa": mean(r.solved for r in rs)}
        for d in ("easy", "medium", "hard") for s in ("llm", "jev")
        for rs in [[r for r in records if r.lake == "target_bird" and r.strategy == DEFAULT and r.selector == s
                    and meta["target_bird"][r.qid]["difficulty"] == d]]}

    # ---- RQ4 downstream metrics, pooled over strategies
    lines, down = [], {}
    for group, selector in [("portals", "llm"), ("portals", "jev"), ("target_bird", "llm"), ("target_bird", "jev"),
                            ("chicago", "llm"), ("chicago", "jev")]:
        rs = [r for r in records if r.selector == selector and (r.lake in PORTALS if group == "portals" else r.lake == group)]
        coded = [(r, coding_flags(r)) for r in rs if r.selected]
        coded = [(r, f) for r, f in coded if f is not None]
        right = [(r, f) for r, f in coded if selection_metrics(r)["answerable"] == 1.0]
        row = {k: mean(f[k] for _, f in coded) for k in ("executed", "exact", "lenient", "correct")}
        row["correct_given_answerable"] = mean(f["correct"] for _, f in right)
        row["final"] = mean(r.solved for r in rs)
        row["coded"] = len(coded)
        down[f"{group}|{selector}"] = row
        label = ("Portals" if group == "portals" else NAME[group] + "$^{\\ddagger}$") + (" (LLM)" if selector == "llm" else " (Jev)")
        lines.append(" & ".join([label, str(len(coded)), f2(row["executed"]), f2(row["exact"]), f2(row["lenient"]),
                                 f2(row["correct"]), f2(row["correct_given_answerable"]), f2(row["final"])]) + " \\\\")
    tables["DOWN_ROWS"] = "\n".join(lines)
    numbers["downstream"] = down

    (OUT / "section_tables.json").write_text(json.dumps(tables, indent=1))
    (OUT / "section_numbers.json").write_text(json.dumps(numbers, indent=1, default=str))
    for name, text in tables.items():
        print(f"===== {name}\n{text}\n")


if __name__ == "__main__":
    main()
