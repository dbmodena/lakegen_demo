"""One record per (lake, strategy, selector, question) for the paper's results.

Two protocols, as the runs exist:

  portals (valencia, paris, bologna, uk, nyc)
      experiments/selector_compare: retrieval ran once per question with the
      benchmark's keywords; both selectors chose from the same prepared context
      (up to 20 candidates); coding outcomes come from coding those table sets
      (results/coded/<lake>.jsonl, keyed by question and table set; a set coded
      several times counts its mean outcome; a set never coded is pending).
  benchmark lakes (target_bird, chicago)
      the full-pipeline grid of experiments/benchmark_lakes (divided
      architecture, one job per strategy and selector; the `bench-grid` spec of
      experiments/specs, read through the run store): the agent wrote the
      keywords, the ranking is the final retrieval attempt, the selection is the
      final one, the outcome is the run's own code evaluation.

Gold semantics: every gold table is needed, except on Chicago, whose gold is a
set of alternatives (Pneuma's answer tables), any one of which answers.

The stored results are read through experiment_runner.analysis: API jobs with
`iter_runs` (which also undoes the provider switches), the selector comparison
and the benchmark files with `experiment_runner.analysis.sources`.
"""

from __future__ import annotations

import glob
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from experiment_runner.analysis.jobs import is_job_file, read_job
from experiment_runner.analysis.runs import Run, iter_runs, runs_of_job
from experiment_runner.analysis.sources import selector_compare
from experiment_runner.analysis.sources.benchmarks import benchmark_cases, gold_is_any_of
from lakegen.config.paths import BASE_DIR
from lakegen.utils.table_ids import resource_id
from experiment_runner.batches.results import solved

ROOT = BASE_DIR
PORTALS = ("valencia", "paris", "bologna", "uk", "nyc")
BENCH = ("target_bird", "chicago")
LAKES = PORTALS + BENCH
STRATEGIES = ("keyword", "semantic", "hybrid", "duckdb", "pneuma", "seeker")
SELECTORS = ("llm", "jev")
BENCH_MODE = {"keyword": "keyword", "semantic": "semantic", "hybrid": "hybrid",
              "duckdb_agentic": "duckdb", "pneuma": "pneuma", "pneuma_seeker": "seeker"}
BENCH_EXPERIMENT = "bench-grid"
SELCMP = selector_compare.RESULTS_DIR
# The portals' full-pipeline runs: ours and the shared copy, on OCI gpt-oss-120b.
PIPELINE_MODEL = "openai.gpt-oss-120b"
PIPELINE_CHECKOUTS = (ROOT, Path("/home/gabriele.martinelli/LakeGen/lakegen_demo"))

__all__ = ["BENCH", "LAKES", "PORTALS", "ROOT", "SELCMP", "SELECTORS", "STRATEGIES",
           "Record", "load", "resource_id", "solved_flag"]


def solved_flag(result: dict) -> bool:
    """The final outcome: exact or equivalent result, or accepted by the semantic judge."""
    return solved(result)


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def _pipeline_runs():
    """The portals' pipeline results, job files in directory order as the old report read them."""
    for checkout in PIPELINE_CHECKOUTS:
        for job_path in glob.glob(str(checkout / ".lakegen_jobs*/*.json")):
            if not is_job_file(Path(job_path)):
                continue
            try:
                job = read_job(Path(job_path))
            except (OSError, ValueError):
                continue
            yield from runs_of_job(job, include_results=True)


def portal_outcomes() -> dict[str, list[dict]]:
    """(lake|question id|sorted tables) -> every coding result of that table set.

    The same sources and filters as selector_compare/llm_coding_report.py: full
    pipeline runs on OCI gpt-oss-120b (ours and the shared copy), and the coding
    harness run on each selection, minus provider failures.
    """
    by_text = {
        (lake, _norm(case["question"])): question_id
        for lake in PORTALS
        for question_id, case in benchmark_cases(lake).items()
    }
    found: dict[str, list[dict]] = defaultdict(list)
    for run in _pipeline_runs():
        if run.lake not in PORTALS or run.model != PIPELINE_MODEL:
            continue
        result = run.result or {}
        evaluation = result.get("code_evaluation") or {}
        ran = (evaluation.get("generation_attempt_count") or 0) > 0 or (result.get("tokens") or {}).get("p3", 0) > 0
        question_id = by_text.get((run.lake, _norm(run.question)))
        tables = sorted(set(run.tables))
        if ran and tables and question_id:
            found[f"{run.lake}|{question_id}|{','.join(tables)}"].append(result)
    for record in selector_compare.coded():
        result = record.get("result") or {}
        if (((result.get("pipeline_stages") or {}).get("table_selection")) == "selected"
                and not selector_compare.provider_failure(record)):
            found[record["key"]].append(result)
    return found


@dataclass
class Record:
    lake: str
    strategy: str
    selector: str
    qid: str
    question: str
    gold: list[str]
    any_gold: bool                      # Chicago: any one gold table answers
    keywords: list[str]
    ranking: list[str]                  # retrieved tables, best first (up to 20)
    selected: list[str]
    solved: float | None                # None = no coding outcome (pending)
    result: dict | None = None          # the coded/pipeline result, for failure analysis
    expansion_used: bool = False
    protocol: str = "same-context"
    extra: dict = field(default_factory=dict)

    @property
    def multi(self) -> bool:
        return len(self.gold) > 1 and not self.any_gold

    def in_context(self, k: int = 20) -> bool:
        top = set(self.ranking[:k])
        return bool(top & set(self.gold)) if self.any_gold else set(self.gold) <= top

    def hit_at(self, k: int) -> float:
        return float(bool(set(self.ranking[:k]) & set(self.gold)))

    def recall_at(self, k: int) -> float:
        if self.any_gold:
            return self.hit_at(k)
        return len(set(self.ranking[:k]) & set(self.gold)) / len(set(self.gold))

    @property
    def selection_correct(self) -> bool:
        chosen = set(self.selected)
        if self.any_gold:
            return bool(chosen & set(self.gold))
        return chosen == set(self.gold)


_PORTAL_OUTCOMES: dict[str, list[dict]] = {}


def _portal_records(lake: str) -> list[Record]:
    cases = benchmark_cases(lake)
    contexts = selector_compare.contexts(lake)
    selections = selector_compare.selections(lake)
    if not _PORTAL_OUTCOMES:
        _PORTAL_OUTCOMES.update(portal_outcomes())
    outcomes = _PORTAL_OUTCOMES
    records = []
    for key, ctx in contexts.items():
        candidates = sorted(ctx["context"].get("candidates") or [], key=lambda c: c.get("retrieval_rank") or 0)
        ranking = list(dict.fromkeys(resource_id(c["dataset"]) for c in candidates))[:20]
        case = cases.get(str(ctx["id"]), {})
        for selector in SELECTORS:
            sel = selections.get(f"{key}|{selector}")
            if sel is None:
                continue
            selected = [resource_id(t) for t in sel.get("selected") or []]
            coded = outcomes.get(f"{lake}|{ctx['id']}|{','.join(sorted(set(selected)))}") if selected else None
            if not selected:
                solved_value, result = 0.0, None
            elif coded:
                passes = [solved_flag(r) for r in coded]
                solved_value, result = sum(passes) / len(passes), coded[-1]
            else:
                solved_value, result = None, None
            records.append(Record(
                lake=lake, strategy=ctx["kind"], selector=selector, qid=str(ctx["id"]),
                question=ctx["question"], gold=[resource_id(g) for g in ctx["gold"]], any_gold=False,
                keywords=list(case.get("keywords") or []), ranking=ranking, selected=selected,
                solved=solved_value, result=result, extra={"context_key": key, "codings": coded or []},
            ))
    return records


def _bench_record(run: Run, case: dict, strategy: str, selector: str) -> Record:
    result = run.result or {}
    attempts = (result.get("discovery") or {}).get("selection_attempts") or []
    keywords = (attempts[-1].get("keywords") if attempts else None) or result.get("keywords") or []
    selected = [resource_id(t) for t in (result.get("tables") or (result.get("discovery") or {}).get("selected_datasets") or [])]
    return Record(
        lake=run.lake, strategy=strategy, selector=selector, qid=run.question_id,
        question=case["question"], gold=[resource_id(g) for g in case["relevant_table_ids"]],
        any_gold=gold_is_any_of(case), keywords=list(keywords),
        ranking=list(run.ranking[:20]), selected=selected,
        solved=float(run.solved), result=result, expansion_used=run.expansion_used,
        protocol="pipeline", extra={"model": run.effective_model or "", "codings": [result]},
    )


def _bench_records(lake: str) -> list[Record]:
    """The lake's runs of the bench grid: per selector, per job, the last result of each question."""
    cases = benchmark_cases(lake)
    by_job: dict[str, dict[str, Run]] = {}
    for run in iter_runs(BENCH_EXPERIMENT, include_results=True):
        if run.lake == lake:
            by_job.setdefault(run.job_id, {})[run.question_id] = run  # the last result of a question counts
    records = []
    for selector in SELECTORS:
        for runs in by_job.values():
            first = next(iter(runs.values()))
            _, job_selector, _, mode = first.experiment_id.split("-", 3)
            if job_selector != selector:
                continue
            records.extend(
                _bench_record(run, cases[run.question_id], BENCH_MODE[mode], selector) for run in runs.values()
            )
    return records


def load(lakes=LAKES) -> list[Record]:
    records = []
    for lake in lakes:
        records += _bench_records(lake) if lake in BENCH else _portal_records(lake)
    return records
