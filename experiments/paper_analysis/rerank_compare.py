"""Second-stage re-ranking on and off, on the portals' saved candidate lists.

For every portal question and strategy, the selector comparison saved the 20
candidates it retrieved. Re-ranking only reorders those 20, so both orders are
available without retrieving again:

  keyword, semantic, hybrid   production re-ranks them: ``cohere`` is the saved
                              order, ``none`` sorts the candidates by their
                              first-stage score (ties by table id)
  content, pneuma, seeker     production does not re-rank them: ``none`` is the
                              saved order, ``cohere`` is Cohere rerank-v4.0-fast
                              (OCI) on the production rerank text, built from
                              each candidate's catalogue entry

Writes results/rerank.jsonl (one row per question, strategy and arm).

  PYTHONPATH=.:src .venv/bin/python experiments/paper_analysis/rerank_compare.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failures as F  # noqa: E402
from records import PORTALS, SELCMP, resource_id  # noqa: E402

from lakegen.integrations.llm.rerank import OCICohereReranker, candidate_text  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "rerank.jsonl"
PRODUCTION_RERANKED = {"keyword", "semantic", "hybrid"}
MODEL = "cohere.rerank-v4.0-fast"


def document(lake: str, rid: str) -> dict:
    entry = F.doc(lake, rid)[1]
    return {"title": entry.get("title") or rid, "publisher": entry.get("publisher") or "",
            "description": entry.get("description") or entry.get("resource_description") or "",
            "columns": [c for c in entry.get("columns") or [] if isinstance(c, dict)]}


def main() -> None:
    done = set()
    if OUT.exists():
        done = {(r["key"], r["arm"]) for r in map(json.loads, OUT.open())}
    reranker = OCICohereReranker(MODEL)
    jobs = []
    with (SELCMP / "contexts.jsonl").open() as handle:
        for line in handle:
            ctx = json.loads(line)
            if ctx["lake"] in PORTALS:
                jobs.append(ctx)

    def run(ctx: dict) -> list[dict]:
        lake, kind = ctx["lake"], ctx["kind"]
        candidates = sorted(ctx["context"].get("candidates") or [], key=lambda c: c.get("retrieval_rank") or 0)[:20]
        saved = list(dict.fromkeys(resource_id(c["dataset"]) for c in candidates))
        score = {resource_id(c["dataset"]): float((c.get("scores") or {}).get("score") or 0.0) for c in candidates}
        rows = []
        if kind in PRODUCTION_RERANKED:
            orders = {"cohere": saved, "none": sorted(saved, key=lambda t: (-score[t], t))}
        else:
            orders = {"none": saved}
            if (ctx["key"], "cohere") not in done and saved:
                texts = [candidate_text(document(lake, t)) for t in saved]
                scores = reranker.scores(ctx["question"], texts)
                orders["cohere"] = [saved[i] for i in sorted(range(len(saved)), key=lambda i: (-scores[i], i))]
            elif not saved:
                orders["cohere"] = []
        for arm, ranking in orders.items():
            if (ctx["key"], arm) in done:
                continue
            rows.append({"key": ctx["key"], "lake": lake, "strategy": kind, "qid": str(ctx["id"]),
                         "arm": arm, "gold": [resource_id(g) for g in ctx["gold"]], "ranking": ranking})
        return rows

    with OUT.open("a") as out, ThreadPoolExecutor(max_workers=6) as pool:
        for count, rows in enumerate(pool.map(run, jobs), 1):
            for row in rows:
                out.write(json.dumps(row) + "\n")
            if count % 300 == 0:
                print(f"{count}/{len(jobs)}", flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    main()
