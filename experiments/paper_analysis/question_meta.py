"""Per-question metadata the 100q files dropped: difficulty, scope, operations, relation.

The portal benchmarks were sampled by build_benchmark.py from OrQA's generated
queries (benchmark/100q_<lake>.json -> sample_metadata.source); the sampler kept
the question and its gold tables but not the generator's difficulty and plan.
This maps every question id back to its OrQA record and keeps

  difficulty   easy / medium / hard (OrQA's tier)
  scope        single_table / multi_table
  ops          the operations of OrQA's structured plan (filter, aggregate, join, ...)
  relation     for multi-table questions, OrQA's link type between the tables:
               join, union or join-correlation (a join whose question relates
               columns of the two tables)

TARGET keeps BIRD's own difficulty (simple / moderate / challenging -> easy /
medium / hard); Chicago has none. Writes results/question_meta.json.

  PYTHONPATH=.:src .venv/bin/python experiments/paper_analysis/question_meta.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import PORTALS, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "build"))
import build_benchmark as bb  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SHARED = Path("/home/gabriele.martinelli/LakeGen/lakegen_demo")
BIRD_DIFFICULTY = {"simple": "easy", "moderate": "medium", "challenging": "hard"}


def _source(meta: dict) -> Path:
    path = Path(meta["source"])
    if path.is_absolute():
        return path
    for root in (ROOT, SHARED):
        if (root / path).exists():
            return root / path
    raise FileNotFoundError(path)


def _relation(ops: list[str], plan: dict) -> str:
    """The generator's own link type between the tables (table_links[].type)."""
    types = {str(link.get("type") or "").lower() for link in plan.get("table_links") or [] if isinstance(link, dict)}
    for relation in ("join-correlation", "union", "join"):
        if relation in types:
            return relation
    text = " ".join(ops)
    return "union" if "union" in text else "join" if "join" in text else "other"


def main() -> None:
    out: dict[str, dict[str, dict]] = {}
    for lake in PORTALS:
        benchmark = json.loads((ROOT / f"benchmark/100q_{lake}.json").resolve().read_text())
        ids = {str(case["id"]) for case in benchmark["cases"]}
        items = bb._records(json.loads(_source(benchmark["sample_metadata"]).read_text()))
        found = {}
        for item in items:
            record = item["record"]
            cid = str(record.get("client_id") or f"{item['engine']}-{item['query_kind']}-{item['group']}-{item['record_key']}")
            if cid not in ids:
                continue
            plan = record.get("structured_plan") or {}
            ops = [str(step.get("op") or "").lower() for step in plan.get("steps") or [] if isinstance(step, dict)]
            scope = "multi_table" if len({t["name"] if isinstance(t, dict) else t for t in record.get("tables") or []}) > 1 else "single_table"
            found[cid] = {
                "difficulty": bb._difficulty(record.get("difficulty")),
                "scope": scope,
                "ops": ops,
                "relation": _relation(ops, plan) if scope == "multi_table" else "single",
            }
        missing = ids - set(found)
        if missing:
            print(f"{lake}: {len(missing)} questions not found in the OrQA source")
        out[lake] = found
    target = json.loads((ROOT / "benchmark/100q_target_bird.json").resolve().read_text())
    out["target_bird"] = {}
    for case in target["cases"]:
        raw = str((case.get("target") or {}).get("difficulty") or "").lower()
        out["target_bird"][str(case["id"])] = {
            "difficulty": BIRD_DIFFICULTY.get(raw, raw or "unknown"),
            "scope": "multi_table" if len(case["relevant_table_ids"]) > 1 else "single_table",
            "ops": [], "relation": "join" if len(case["relevant_table_ids"]) > 1 else "single",
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "question_meta.json").write_text(json.dumps(out, indent=1))
    for lake, questions in out.items():
        print(lake, dict(Counter(q["difficulty"] for q in questions.values())),
              dict(Counter(q["relation"] for q in questions.values())))
    ops = Counter(op for lake in PORTALS for q in out[lake].values() for op in set(q["ops"]))
    print("operations (questions using each, portals):", ops.most_common(20))


if __name__ == "__main__":
    main()
