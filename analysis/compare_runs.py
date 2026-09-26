#!/usr/bin/env python3
"""Compare the LakeGen batches of a factorial experiment grid.

Each configuration file names one run (its stem is the ``experiment_id``); all
completed jobs with that id (one per stage) are merged. Every analysis is per
core, because the two portals have different questions.

A benchmark with ``sample_metadata.stages`` (e.g. [100, 400]) gets one report
per cumulative stage, each in its own folder: ``100q/`` covers only the first
100 questions, ``500q/`` all of them.

Outcomes are end-to-end: a question blocked before the coder counts as a
failure, unlike the batch metrics, which drop it from the denominator.
Confidence intervals come from a bootstrap over questions, so repeated use of
the same question across configurations stays paired.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

from generate_report import CONTEXT_ORDER, load_job, pct, per_question_rows


RETRIEVAL_LABELS = {
    ("keyword", "weighted"): "BM25",
    ("keyword", "rrf"): "BM25",
    ("semantic", "weighted"): "Denso",
    ("semantic", "rrf"): "Denso",
    ("hybrid", "weighted"): "Ibrido",
    ("hybrid", "rrf"): "RRF",
}
RETRIEVAL_ORDER = ("BM25", "Denso", "Ibrido", "RRF")
MODEL_LABELS = {
    "openai.gpt-oss-120b": "GPT-OSS 120B",
    "meta.llama-3.3-70b-instruct": "Llama 3.3 70B",
}
ACCESS_LABELS = {"agentic": "Agentica", "orchestrated_context": "Orchestrata"}
FACTORS = (
    ("retrieval", "Retrieval"),
    ("model", "Modello"),
    ("access", "Interazione"),
)
# Question-level outcomes, all binary. Selection does not depend on the coder
# context, so it is read from the "full" rows only.
OUTCOMES = (
    ("exact_selection", "Exact selection"),
    ("exact_result_match", "Exact match"),
    ("lenient_result_match", "Lenient match"),
    ("supported_correct", "Supported"),
)
RUN_METRICS = ("Recall@10", "Hit@5", "MRR")


def find_runs(jobs_dir: Path, experiment_ids: Iterable[str]) -> dict[str, list[Path]]:
    """Every completed job of each wanted experiment id, oldest first.

    A configuration run in stages (scripts/run_thesis_suite.py) has one job per
    stage; the caller merges them, keeping the newest answer to a question.
    """
    import json

    wanted = set(experiment_ids)
    found: dict[str, list[tuple[str, Path]]] = {}
    for path in jobs_dir.glob("*.json"):
        if path.name.endswith(".questions.json"):
            continue
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if job.get("status") != "completed" or not job.get("batch_metrics"):
            continue
        experiment_id = (
            job.get("settings", {}).get("resolved_config", {}).get("experiment_id")
        )
        if experiment_id not in wanted:
            continue
        found.setdefault(experiment_id, []).append((str(job.get("finished_at") or ""), path))
    return {
        experiment_id: [path for _, path in sorted(entries)]
        for experiment_id, entries in found.items()
    }


def label_row(row: dict[str, Any]) -> dict[str, Any]:
    row["retrieval"] = RETRIEVAL_LABELS.get(
        (row["retrieval_mode"], row["fusion_method"]), str(row["retrieval_mode"])
    )
    row["model_label"] = MODEL_LABELS.get(row["model"], row["model"])
    row["access"] = ACCESS_LABELS.get(row["tool_access"], row["tool_access"])
    row["model"] = row["model_label"]
    return row


def value(row: dict[str, Any], outcome: str) -> float | None:
    raw = row.get(outcome)
    return None if raw in ("", None) else float(raw)


def per_question_means(
    rows: list[dict[str, Any]], outcome: str
) -> dict[str, float]:
    """Mean outcome of each question over the given rows (end-to-end)."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if not int(row["applicable"]) and outcome != "exact_selection":
            continue
        observed = value(row, outcome)
        if observed is not None:
            grouped.setdefault(str(row["question_id"]), []).append(observed)
    return {question: statistics.mean(values) for question, values in grouped.items()}


def bootstrap(
    by_question: dict[str, float], *, reps: int, seed: int
) -> tuple[float, float, float] | None:
    """Mean over questions with a 95% percentile interval."""
    values = list(by_question.values())
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)
    )
    return (
        statistics.mean(values),
        means[int(0.025 * reps)],
        means[min(reps - 1, int(0.975 * reps))],
    )


def paired_difference(
    a: dict[str, float], b: dict[str, float], *, reps: int, seed: int
) -> tuple[float, float, float, int] | None:
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    result = bootstrap({q: a[q] - b[q] for q in shared}, reps=reps, seed=seed)
    return (*result, len(shared)) if result else None


def ci_text(result: tuple[float, ...] | None) -> str:
    if result is None:
        return "—"
    mean, low, high = result[:3]
    return f"{mean * 100:.1f}% [{low * 100:.1f}, {high * 100:.1f}]"


def diff_text(result: tuple[float, ...] | None) -> str:
    if result is None:
        return "—"
    mean, low, high = result[:3]
    marker = " *" if low > 0 or high < 0 else ""
    return f"{mean * 100:+.1f} [{low * 100:+.1f}, {high * 100:+.1f}]{marker}"


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]


def outcome_rows(rows: list[dict[str, Any]], outcome: str) -> list[dict[str, Any]]:
    """Selection is context-independent: use one context to avoid triple counting."""
    if outcome == "exact_selection":
        return [row for row in rows if row["coder_context"] == "full"]
    return rows


def analyse_core(
    core: str,
    rows: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    *,
    reps: int,
    seed: int,
    effects: list[dict[str, Any]],
) -> list[str]:
    lines = [f"## Core {core.upper()}", ""]
    full_rows = [row for row in rows if row["coder_context"] == "full"]
    questions = {row["question_id"] for row in full_rows}
    lines.append(
        f"{len({row['experiment_id'] for row in rows})} run, {len(questions)} domande. "
        "Valori end-to-end in percentuale con intervallo di confidenza al 95% "
        f"(bootstrap sulle domande, {reps} ripetizioni)."
    )
    lines.append("")

    # 1. Main table: every configuration, coder context fixed to full.
    lines.extend(["### 1. Tabella principale (contesto coder: full)", ""])
    header = ["Retrieval", "Modello", "Interazione", *RUN_METRICS,
              *(name for _, name in OUTCOMES), "Tempo medio", "Token medi"]
    table = []
    for experiment_id, run in sorted(
        runs.items(),
        key=lambda item: (
            RETRIEVAL_ORDER.index(item[1]["retrieval"])
            if item[1]["retrieval"] in RETRIEVAL_ORDER else 99,
            item[1]["model"], item[1]["access"],
        ),
    ):
        if run["core"] != core:
            continue
        run_rows = [row for row in full_rows if row["experiment_id"] == experiment_id]
        cells = [run["retrieval"], run["model"], run["access"]]
        cells += [pct(run["retrieval_metrics"][key]) if key in run["retrieval_metrics"]
                  else "—" for key in RUN_METRICS]
        for outcome, _ in OUTCOMES:
            means = per_question_means(run_rows, outcome)
            cells.append(pct(statistics.mean(means.values())) if means else "—")
        elapsed = [float(row["elapsed_seconds"] or 0) for row in run_rows]
        tokens = [float(row["tokens_total"] or 0) for row in run_rows]
        cells.append(f"{statistics.mean(elapsed):.0f}s" if elapsed else "—")
        cells.append(f"{statistics.mean(tokens):,.0f}" if tokens else "—")
        table.append(cells)
    lines.extend(md_table(header, table))
    lines.append(
        "I token comprendono le tre varianti del coder, eseguite in ogni run. "
        "Le metriche di retrieval sono le medie del batch, senza intervallo."
    )
    lines.append("")

    # 2. Main effects: one factor at a time, averaged over the others.
    lines.extend(["### 2. Effetti principali (contesto coder: full)", ""])
    for factor, factor_name in FACTORS:
        levels = [level for level in (
            RETRIEVAL_ORDER if factor == "retrieval"
            else sorted({row[factor] for row in full_rows})
        ) if any(row[factor] == level for row in full_rows)]
        lines.extend([f"#### {factor_name}", ""])
        level_table = []
        level_means: dict[tuple[str, str], dict[str, float]] = {}
        for level in levels:
            cells = [level]
            for outcome, name in OUTCOMES:
                subset = [row for row in full_rows if row[factor] == level]
                means = per_question_means(subset, outcome)
                level_means[(level, outcome)] = means
                result = bootstrap(means, reps=reps, seed=seed)
                cells.append(ci_text(result))
                if result:
                    effects.append({
                        "core": core, "factor": factor, "comparison": level,
                        "outcome": outcome, "mean": result[0],
                        "ci_low": result[1], "ci_high": result[2],
                        "questions": len(means),
                    })
            level_table.append(cells)
        lines.extend(md_table([factor_name, *(n for _, n in OUTCOMES)], level_table))
        diff_table = []
        for a, b in combinations(levels, 2):
            cells = [f"{a} − {b}"]
            for outcome, _ in OUTCOMES:
                result = paired_difference(
                    level_means[(a, outcome)], level_means[(b, outcome)],
                    reps=reps, seed=seed,
                )
                cells.append(diff_text(result))
                if result:
                    effects.append({
                        "core": core, "factor": factor, "comparison": f"{a} - {b}",
                        "outcome": outcome, "mean": result[0],
                        "ci_low": result[1], "ci_high": result[2],
                        "questions": result[3],
                    })
            diff_table.append(cells)
        if diff_table:
            lines.append("Differenze appaiate in punti percentuali "
                         "(* = intervallo che non contiene lo zero):")
            lines.append("")
            lines.extend(md_table(["Confronto", *(n for _, n in OUTCOMES)], diff_table))

    # 3. Coder context: same selection, only the coder's context changes.
    lines.extend([
        "### 3. Contesto del coder", "",
        "Le tre varianti usano la stessa selezione di tabelle, quindi il confronto "
        "isola l'effetto del contesto. Condizionato = solo domande in cui il coder "
        "è partito (come nelle metriche del batch).", "",
    ])
    code_outcomes = [item for item in OUTCOMES if item[0] != "exact_selection"]
    context_table = []
    context_means: dict[tuple[str, str, str], dict[str, float]] = {}
    models = sorted({row["model"] for row in rows})
    for model in [*models, "Tutti"]:
        for context in CONTEXT_ORDER:
            subset = [row for row in rows if row["coder_context"] == context
                      and (model == "Tutti" or row["model"] == model)]
            if not subset:
                continue
            cells = [model, context]
            for outcome, _ in code_outcomes:
                means = per_question_means(subset, outcome)
                context_means[(model, context, outcome)] = means
                cells.append(ci_text(bootstrap(means, reps=reps, seed=seed)))
            ran = [row for row in subset if int(row["counted_in_batch_metrics"])]
            conditional = per_question_means(ran, "exact_result_match")
            cells.append(pct(statistics.mean(conditional.values())) if conditional else "—")
            context_table.append(cells)
    lines.extend(md_table(
        ["Modello", "Contesto", *(n for _, n in code_outcomes), "Exact condizionato"],
        context_table,
    ))
    diff_table = []
    for model in [*models, "Tutti"]:
        for other in ("schema_only", "minimal"):
            cells = [model, f"full − {other}"]
            for outcome, _ in code_outcomes:
                a = context_means.get((model, "full", outcome), {})
                b = context_means.get((model, other, outcome), {})
                result = paired_difference(a, b, reps=reps, seed=seed)
                cells.append(diff_text(result))
                if result and model == "Tutti":
                    effects.append({
                        "core": core, "factor": "coder_context",
                        "comparison": f"full - {other}", "outcome": outcome,
                        "mean": result[0], "ci_low": result[1],
                        "ci_high": result[2], "questions": result[3],
                    })
            diff_table.append(cells)
    lines.extend(md_table(["Modello", "Confronto", *(n for _, n in code_outcomes)], diff_table))

    # 4. The one interaction worth checking a priori: tool calling depends on the model.
    lines.extend(["### 4. Interazione modello × modalità (exact match, full)", ""])
    accesses = sorted({row["access"] for row in full_rows})
    grid = []
    for model in models:
        cells = [model]
        for access in accesses:
            subset = [row for row in full_rows
                      if row["model"] == model and row["access"] == access]
            cells.append(ci_text(bootstrap(
                per_question_means(subset, "exact_result_match"), reps=reps, seed=seed,
            )))
        grid.append(cells)
    lines.extend(md_table(["Modello", *accesses], grid))
    return lines




def benchmark_scopes(config_path: Path) -> list[tuple[str, set[str]]]:
    """Question-id sets to report separately: each cumulative stage, then all.

    With stages [100, 400] this gives ("100q", first 100) and ("500q", all).
    Without stages (or without the benchmark file) there is a single "all" scope.
    """
    import json

    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark_path = Path(str((config.get("benchmark") or {}).get("path") or ""))
    if not benchmark_path.is_file():
        return []
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    ids = [str(case["id"]) for case in benchmark.get("cases", [])]
    stages = benchmark.get("sample_metadata", {}).get("stages") or [len(ids)]
    scopes, end = [], 0
    for size in stages:
        end += size
        scopes.append((f"{end}q", set(ids[:end])))
    return scopes


def load_run(paths: list[Path]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], set[str]]]]:
    """Merge the stage jobs of one configuration; newer answers win."""
    by_question: dict[str, list[dict[str, Any]]] = {}
    jobs: list[tuple[dict[str, Any], set[str]]] = []
    for job_path in paths:
        job, questions, results = load_job(job_path)
        job_rows = per_question_rows(job, questions, results)
        if not job_rows:
            continue
        job_questions = {str(row["question_id"]) for row in job_rows}
        jobs.append((job, job_questions))
        for question_id in job_questions:
            by_question[question_id] = [
                row for row in job_rows if str(row["question_id"]) == question_id
            ]
    rows = [label_row(row) for group in by_question.values() for row in group]
    # A metric the run never recorded (e.g. lenient in older jobs) is missing,
    # not a failure, even for the questions blocked before the coder. Where the
    # run does record it, an absent field means no result to compare (a failed
    # execution), so it counts as a failure.
    for outcome, _ in OUTCOMES:
        recorded = any(row.get(outcome) != "" for row in rows if int(row["coder_ran"]))
        for row in rows:
            if not recorded:
                row[outcome] = ""
            elif row.get(outcome) == "":
                row[outcome] = 0
    return rows, jobs


def write_report(
    output: Path,
    title: str,
    loaded: dict[str, tuple[list[dict[str, Any]], list[tuple[dict[str, Any], set[str]]]]],
    scope: dict[str, set[str] | None],
    experiment_ids: list[str],
    missing: list[str],
    *,
    reps: int,
    seed: int,
) -> None:
    rows: list[dict[str, Any]] = []
    runs: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []
    for experiment_id, (run_rows, jobs) in sorted(loaded.items()):
        wanted = scope.get(experiment_id)
        kept = [row for row in run_rows if wanted is None or str(row["question_id"]) in wanted]
        if not kept:
            continue
        answered = {str(row["question_id"]) for row in kept}
        if wanted is not None and len(answered) < len(wanted):
            incomplete.append(f"{experiment_id}: {len(answered)} / {len(wanted)} domande")
        rows.extend(kept)
        # Batch retrieval means only from jobs that lie inside this scope,
        # weighted by their question count.
        scoped_jobs = [
            job for job, questions in jobs if wanted is None or questions <= wanted
        ]
        weights = [job["batch_metrics"]["table_selection"].get("case_count", 0) for job in scoped_jobs]
        retrieval_metrics = {
            key: sum(
                weight * job["batch_metrics"]["table_selection"]["mean_metrics"][key]
                for weight, job in zip(weights, scoped_jobs)
            ) / sum(weights)
            for key in RUN_METRICS
            if sum(weights) and all(
                key in job["batch_metrics"]["table_selection"].get("mean_metrics", {})
                for job in scoped_jobs
            )
        }
        first = kept[0]
        runs[experiment_id] = {
            "job_id": ", ".join(job["job_id"] for job in scoped_jobs),
            "core": first["core"],
            "retrieval": first["retrieval"],
            "model": first["model"],
            "access": first["access"],
            "questions": len(answered),
            "retrieval_metrics": retrieval_metrics,
        }

    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / "per_question_all.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    effects: list[dict[str, Any]] = []
    lines = [
        f"# Confronto delle configurazioni — {title}", "",
        f"Run trovate: {len(runs)} / {len(experiment_ids)}.", "",
    ]
    if incomplete:
        lines.extend([
            "**Attenzione: run incomplete in questo insieme di domande** "
            "(i confronti usano solo le domande presenti):", "",
            *(f"- {item}" for item in incomplete), "",
        ])
    for core in sorted({run["core"] for run in runs.values()}):
        lines.extend(analyse_core(
            core, [row for row in rows if row["core"] == core], runs,
            reps=reps, seed=seed, effects=effects,
        ))
    if missing:
        lines.extend(["## Run mancanti", "", *(f"- {item}" for item in sorted(missing)), ""])
    lines.extend([
        "## Run usate", "",
        *md_table(
            ["Experiment", "Domande", "Job"],
            [[eid, str(run["questions"]), f"`{run['job_id']}`"] for eid, run in sorted(runs.items())],
        ),
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if effects:
        with (output / "effects.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
            writer.writeheader()
            writer.writerows(effects)
    print(f"{title}: {len(runs)} run, riepilogo in {output / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config/thesis"),
                        help="Cartella dei config; il nome di ogni file è l'experiment_id")
    parser.add_argument("--jobs-dir", type=Path, default=Path(".lakegen_jobs"))
    parser.add_argument("--output", type=Path, default=Path("reports/thesis"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_paths = {path.stem: path for path in sorted(args.config_dir.rglob("*.yaml"))}
    experiment_ids = sorted(config_paths)
    if not experiment_ids:
        parser.error(f"nessun config in {args.config_dir}")
    job_paths = find_runs(args.jobs_dir, experiment_ids)
    missing = [eid for eid in experiment_ids if eid not in job_paths]

    loaded = {}
    for experiment_id, paths in sorted(job_paths.items()):
        run_rows, jobs = load_run(paths)
        if run_rows:
            loaded[experiment_id] = (run_rows, jobs)
        else:
            missing.append(f"{experiment_id} (senza domande/risultati)")

    # One report per scope name (e.g. 100q and 500q), each in its own folder.
    scopes_by_name: dict[str, dict[str, set[str] | None]] = {}
    for experiment_id, path in config_paths.items():
        scopes = benchmark_scopes(path) or [("all", None)]
        for name, ids in scopes:
            scopes_by_name.setdefault(name, {})[experiment_id] = ids
    for name, scope in sorted(scopes_by_name.items(), key=lambda item: len(item[0])):
        write_report(
            args.output / name, name, loaded, scope, experiment_ids, missing,
            reps=args.bootstrap, seed=args.seed,
        )


if __name__ == "__main__":
    main()
