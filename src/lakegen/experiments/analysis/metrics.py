"""A batch's table-selection report: the metrics the API logs when a job completes.

Every question of a metric-ready batch names its gold tables, so each result's
final ranking and selection are scored against them:

* retrieval: Hit, Recall and nDCG at 1, 5, 10, 15 and 20, MRR, and
  FullCoverage@k (every gold table within the first k);
* selection: hit, recall, precision and exact match of the selected set;
* augmented: the same for questions that accept alternative tables, where a
  gold table counts as found when any table of its group is;
* expansion: how often a gold table selected beyond the first ten came from
  the candidate expansion;
* by cardinality: the retrieval, selection, code and timing figures for the
  questions with one, two, three or more gold tables;
* performance: elapsed time and tokens per query;
* code: ``summarize_code_evaluations`` over the results, once per coder
  context level when the automatic coder sweep ran.

The report holds one experiment per coder context level; the API logs it as
CSV rows and keeps ``report_summary`` in the job.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lakegen.config.experiment import CoderContextLevel
from lakegen.domain.tables import resource_id
from lakegen.experiments.analysis.results import expansion_used, final_ranking, selected_tables
from lakegen.retrieval.evaluation import evaluate_ranking, mean_metrics
from lakegen.workflow.evaluation.code import summarize_code_evaluations

K_VALUES = (1, 5, 10, 15, 20)
CODER_CONTEXTS = ("full", "schema_only", "minimal")
CARDINALITY_BUCKETS = ("1", "2", "3", "4+")


@dataclass(frozen=True)
class CaseMetrics:
    """One question's scores, as the report's ``cases`` entry records them."""

    row: dict[str, Any]
    retrieval: dict[str, float]
    selection: dict[str, float]
    augmented: dict[str, Any] | None
    expansion_contributed: bool | None
    """Whether a gold table ranked beyond 10 was selected; None when no expansion ran."""

    @property
    def successful(self) -> bool:
        return not self.row["error"]


def _gold(source: Mapping[str, Any]) -> list[str] | None:
    gold = source.get("log_fields", {}).get("SOURCE_RELEVANT_TABLE_IDS")
    if not isinstance(gold, list) or not gold:
        return None
    return list(dict.fromkeys(resource_id(value) for value in gold))


def _accepted_groups(source: Mapping[str, Any], relevant: Sequence[str]) -> list[set[str]]:
    """Each gold table with the alternatives the benchmark accepts for it."""
    raw = source.get("log_fields", {}).get("SOURCE_ACCEPTED_TABLE_ALTERNATIVES", {})
    lookup = {
        resource_id(key): {resource_id(value) for value in values if str(value).strip()}
        for key, values in raw.items()
        if isinstance(values, list)
    } if isinstance(raw, dict) else {}
    return [{gold, *lookup.get(gold, set())} for gold in relevant]


def _augmented(
    ranking: Sequence[str], selected: set[str], relevant: set[str], groups: list[set[str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """The scores when any table of a gold table's group counts as that table."""
    retrieval: dict[str, float] = {}
    for k in K_VALUES:
        candidates = set(ranking[:k])
        covered = sum(bool(group & candidates) for group in groups)
        retrieval[f"AlternativeHit@{k}"] = float(covered > 0)
        retrieval[f"AlternativeRecall@{k}"] = covered / len(groups)
        retrieval[f"AlternativeFullCoverage@{k}"] = float(covered == len(groups))
    first_rank = next(
        (rank for rank, value in enumerate(ranking, 1) if any(value in group for group in groups)),
        None,
    )
    retrieval["AlternativeMRR"] = 1.0 / first_rank if first_rank else 0.0
    covered_groups = sum(bool(group & selected) for group in groups)
    strict_hit = bool(relevant & selected)
    selection = {
        "AlternativeSelectionHit": float(covered_groups > 0),
        "AlternativeSelectionRecall": covered_groups / len(groups),
        "AlternativeSelectionPrecision": covered_groups / len(selected) if selected else 0.0,
        "AlternativeExactSelection": float(
            covered_groups == len(groups) and len(selected) == len(groups)
        ),
        "AlternativeOnlySelection": float(covered_groups > 0 and not strict_hit),
        "StrictMissRescued": float(not strict_hit and covered_groups > 0),
    }
    return retrieval, selection


def case_metrics(source: Mapping[str, Any], entry: Mapping[str, Any]) -> CaseMetrics | None:
    """Score one result against its question's gold tables; None without gold tables."""
    relevant = _gold(source)
    if relevant is None:
        return None
    result = entry.get("result", {})
    ranking = final_ranking(result)
    retrieval = evaluate_ranking(ranking, relevant, k_values=K_VALUES)
    selected = selected_tables(result)
    relevant_set, selected_set = set(relevant), set(selected)
    for k in K_VALUES:
        retrieval[f"FullCoverage@{k}"] = float(relevant_set.issubset(set(ranking[:k])))
    matches = len(relevant_set & selected_set)
    selection = {
        "SelectionHit": float(matches > 0),
        "SelectionRecall": matches / len(relevant_set),
        "SelectionPrecision": matches / len(selected_set) if selected_set else 0.0,
        "ExactSelection": float(selected_set == relevant_set),
    }
    case_id = str(source.get("source_id") or source.get("source_path"))
    augmented = None
    groups = _accepted_groups(source, relevant)
    if any(len(group) > 1 for group in groups):
        augmented_retrieval, augmented_selection = _augmented(
            ranking, selected_set, relevant_set, groups,
        )
        augmented = {
            "case_id": case_id,
            "question": source["question"],
            "gold_table_ids": relevant,
            "accepted_table_groups": [sorted(group) for group in groups],
            "selected_table_ids": selected,
            "retrieval_metrics": augmented_retrieval,
            "selection_metrics": augmented_selection,
        }
    expansion_contributed = None
    if expansion_used(result):
        ranks = {value: rank for rank, value in enumerate(ranking, 1)}
        expansion_contributed = any(
            value in relevant_set and ranks.get(value, 0) > 10 for value in selected_set
        )
    row = {
        "case_id": case_id,
        "question": source["question"],
        "relevant_table_ids": relevant,
        "ranking": ranking,
        "metrics": retrieval,
        "gold_table_count": len(relevant_set),
        "selected_table_count": len(selected_set),
        "selection_metrics": selection,
        "selection_outcome": (
            "exact" if selected_set == relevant_set
            else "miss" if not matches
            else "over_selection" if relevant_set.issubset(selected_set)
            else "partial"
        ),
        "expansion_used": expansion_contributed is not None,
        "elapsed_seconds": float(result.get("elapsed_seconds") or 0.0),
        "error": str(result.get("error") or ""),
    }
    return CaseMetrics(row, retrieval, selection, augmented, expansion_contributed)


def _source_key(entry: Mapping[str, Any]) -> tuple[Any, Any] | None:
    """What names a question in a job: its source path and id, when the entry carries them."""
    if entry.get("source_id") is None and not entry.get("source_path"):
        return None
    return (entry.get("source_path"), entry.get("source_id"))


def _paired(
    questions: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]] | None:
    """Each question with its result: by source when both sides name theirs, else by position.

    A re-queued question has several results; the last one is its result.
    Pairing by position is what older jobs allow, and it is wrong once a
    result was re-queued, which is why the source is preferred.
    """
    keys = [_source_key(question) for question in questions]
    result_keys = [_source_key(entry) for entry in results]
    if all(keys) and all(result_keys) and len(set(keys)) == len(keys):
        latest = dict(zip(result_keys, results, strict=True))
        pairs = [(question, latest.get(key)) for question, key in zip(questions, keys, strict=True)]
        return None if any(entry is None for _, entry in pairs) else pairs  # type: ignore[return-value]
    if len(questions) != len(results):
        return None
    return list(zip(questions, results, strict=True))


def batch_cases(
    questions: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> list[CaseMetrics] | None:
    """A case per question, paired with its result; None unless every question has gold tables and a result.

    Older query documents carry no gold tables; a batch with any such question
    is skipped whole rather than scored in part.
    """
    if not questions:
        return None
    pairs = _paired(questions, results)
    if pairs is None:
        return None
    cases = []
    for source, entry in pairs:
        case = case_metrics(source, entry)
        if case is None:
            return None
        cases.append(case)
    return cases


def _percentile_90(values: Sequence[float]) -> float:
    return sorted(values)[max(0, math.ceil(0.9 * len(values)) - 1)]


def _performance(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elapsed = [float(entry.get("result", {}).get("elapsed_seconds") or 0.0) for entry in results]
    token_totals = {
        key: sum(int(entry.get("result", {}).get("tokens", {}).get(key, 0) or 0) for entry in results)
        for key in ("p1_p2", "p3", "p4")
    }
    coder_tokens = {
        context: sum(
            int(
                entry.get("result", {}).get("coder_context_experiment", {})
                .get("variants", {}).get(context, {}).get("tokens", 0) or 0
            )
            for entry in results
        )
        for context in CODER_CONTEXTS
    }
    coder_tokens["unattributed"] = max(0, token_totals["p3"] - sum(coder_tokens.values()))
    return {
        "query_count": len(results),
        "total_query_elapsed_seconds": sum(elapsed),
        "mean_query_elapsed_seconds": statistics.mean(elapsed),
        "median_query_elapsed_seconds": statistics.median(elapsed),
        "p90_query_elapsed_seconds": _percentile_90(elapsed),
        "token_totals": token_totals,
        "mean_tokens_per_query": {key: total / len(results) for key, total in token_totals.items()},
        "coder_token_totals_by_context": coder_tokens,
        "mean_coder_tokens_per_query_by_context": {
            key: total / len(results) for key, total in coder_tokens.items()
        },
    }


def _evaluation(result: Mapping[str, Any], variant: str | None) -> Mapping[str, Any]:
    if variant is None:
        return result.get("code_evaluation", {})
    return (
        result.get("coder_context_experiment", {})
        .get("variants", {}).get(variant, {})
        .get("code_evaluation", {})
    )


def _rate(evaluations: Sequence[Mapping[str, Any]], flag: str) -> float:
    """The share of evaluations with ``flag`` set; zero when there are none."""
    if not evaluations:
        return 0.0
    return sum(bool(evaluation.get(flag)) for evaluation in evaluations) / len(evaluations)


def _by_cardinality(
    cases: Sequence[CaseMetrics], results: Sequence[Mapping[str, Any]], variant: str | None,
) -> dict[str, Any]:
    """The figures of the questions with 1, 2, 3 or 4+ gold tables."""
    groups: dict[str, list[tuple[CaseMetrics, Mapping[str, Any]]]] = {}
    for case, entry in zip(cases, results, strict=True):
        count = int(case.row["gold_table_count"])
        groups.setdefault(str(count) if count < 4 else "4+", []).append((case, entry))
    analysis: dict[str, Any] = {}
    for bucket in CARDINALITY_BUCKETS:
        group = groups.get(bucket, [])
        if not group:
            continue
        rows = [case.row for case, _ in group]
        evaluations = [
            evaluation
            for _, entry in group
            if (evaluation := _evaluation(entry.get("result", {}), variant)).get("applicable")
        ]
        elapsed = sorted(row["elapsed_seconds"] for row in rows)
        analysis[bucket] = {
            "case_count": len(rows),
            "mean_gold_table_count": sum(row["gold_table_count"] for row in rows) / len(rows),
            "retrieval": mean_metrics([row["metrics"] for row in rows]),
            "selection": mean_metrics([row["selection_metrics"] for row in rows]),
            "selection_outcomes": {
                outcome: sum(row["selection_outcome"] == outcome for row in rows)
                for outcome in ("exact", "partial", "over_selection", "miss")
            },
            "code_applicable_count": len(evaluations),
            "exact_result_match_rate": _rate(evaluations, "exact_result_match"),
            "supported_result_rate": _rate(evaluations, "supported_correct"),
            "execution_success_rate": _rate(evaluations, "execution_success"),
            "mean_elapsed_seconds": statistics.mean(elapsed),
            "median_elapsed_seconds": statistics.median(elapsed),
            "p90_elapsed_seconds": _percentile_90(elapsed),
            "expansion_usage_rate": sum(row["expansion_used"] for row in rows) / len(rows),
        }
    return analysis


def _experiment(
    cases: Sequence[CaseMetrics],
    results: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
    level: CoderContextLevel,
    *,
    sweep: bool,
    performance: dict[str, Any],
) -> dict[str, Any]:
    retrieval = resolved["retrieval"]
    augmented = [case.augmented for case in cases if case.augmented is not None]
    expanded = [case for case in cases if case.expansion_contributed is not None]
    contributed = sum(case.expansion_contributed for case in expanded)
    successful = [case.retrieval for case in cases if case.successful]
    return {
        "config": {
            **retrieval,
            "mode": retrieval["mode"],
            "fusion_method": retrieval["fusion_method"],
            "coder_context_level": level.value,
            "semantic_code_judge_model": resolved.get("semantic_code_judge_model", ""),
        },
        "mean_metrics": mean_metrics([case.retrieval for case in cases]),
        "mean_metrics_successful_queries": mean_metrics(successful),
        "mean_selection_metrics": mean_metrics([case.selection for case in cases]),
        "augmented_retrieval": {
            "eligible_case_count": len(augmented),
            "mean_retrieval_metrics": mean_metrics([case["retrieval_metrics"] for case in augmented]),
            "mean_selection_metrics": mean_metrics([case["selection_metrics"] for case in augmented]),
            "cases": augmented,
        },
        "expansion_metrics": {
            "query_count": len(expanded),
            "gold_selected_beyond_initial_count": contributed,
            "gold_selected_beyond_initial_rate": contributed / len(expanded) if expanded else 0.0,
        },
        "successful_case_count": len(successful),
        "failed_case_count": len(cases) - len(successful),
        "code_metrics": summarize_code_evaluations(
            results, coder_context_level=(level.value if sweep else None)
        ),
        "table_count_analysis": _by_cardinality(cases, results, level.value if sweep else None),
        "performance_metrics": performance,
        "cases": [case.row for case in cases],
    }


def batch_report(
    cases: Sequence[CaseMetrics],
    results: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    """The report of a batch run with ``resolved`` (its stored flat configuration).

    One experiment, labelled by the experiment id; when the automatic coder
    sweep ran, one per coder context level, labelled ``<id>-coder-<level>``.
    """
    base_label = str(resolved.get("experiment_id") or "batch")
    sweep = bool(resolved.get("automatic_test_coder"))
    levels = (
        list(CoderContextLevel) if sweep
        else [CoderContextLevel(resolved.get("coder_context_level", "full"))]
    )
    performance = _performance(results)
    experiments = {
        (f"{base_label}-coder-{level.value}" if sweep else base_label): _experiment(
            cases, results, resolved, level, sweep=sweep, performance=performance,
        )
        for level in levels
    }
    return {
        "benchmark_type": "batch-table-selection",
        "created_at": created_at,
        "case_count": len(cases),
        "experiments": experiments,
    }


def report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The aggregates of the report's first experiment: what a job keeps as its batch metrics."""
    experiment = next(iter(report["experiments"].values()))
    return {
        "case_count": report["case_count"],
        **{
            key: experiment[key]
            for key in (
                "successful_case_count", "failed_case_count", "mean_metrics",
                "mean_metrics_successful_queries", "mean_selection_metrics",
                "augmented_retrieval", "expansion_metrics", "table_count_analysis",
                "performance_metrics",
            )
        },
    }
