#!/usr/bin/env python3
"""Generate a self-contained HTML/CSV/Markdown report for a LakeGen batch."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
from pathlib import Path
from typing import Any


RETRIEVAL_ORDER = (
    "Hit@1", "Hit@5", "Hit@10", "Hit@15", "Hit@20",
    "Recall@1", "Recall@5", "Recall@10", "Recall@15", "Recall@20",
    "MRR", "nDCG@1", "nDCG@5", "nDCG@10", "nDCG@15", "nDCG@20",
)
CODE_RATES = (
    "generation_success_rate",
    "execution_success_rate",
    "structured_output_rate",
    "result_type_match_rate",
    "format_compliance_rate",
    "exact_result_match_rate",
    "supported_result_rate",
    "pass_at_1",
    "success_within_3",
    "semantic_pass_at_1",
    "semantic_success_within_3",
    "mean_requirement_pass_rate",
)
CONTEXT_ORDER = ("full", "schema_only", "minimal")
LABEL_OVERRIDES = {
    "SelectionHit": "Selection Hit",
    "SelectionRecall": "Selection Recall",
    "SelectionPrecision": "Selection Precision",
    "ExactSelection": "Exact Selection",
}
METRIC_DESCRIPTIONS = (
    ("Retrieval Hit@k", "Quota di domande per cui il retriever colloca almeno una tabella gold nelle prime k posizioni."),
    ("Retrieval Recall@k", "Quota media delle tabelle gold presenti nelle prime k posizioni del retriever."),
    ("MRR", "Premia i casi in cui la prima tabella rilevante appare molto in alto nel ranking."),
    ("nDCG@k", "Misura la qualità dell'ordinamento delle tabelle rilevanti entro le prime k posizioni."),
    ("Selection Hit", "Quota di domande per cui l'agente seleziona almeno una tabella gold."),
    ("Selection Recall", "Quota media delle tabelle gold effettivamente selezionate dall'agente."),
    ("Selection Precision", "Quota delle tabelle selezionate dall'agente che sono realmente gold."),
    ("Exact Selection", "Quota di domande in cui le tabelle selezionate coincidono esattamente con tutte e sole le gold."),
    ("Execution success", "Quota di casi applicabili in cui il codice generato viene eseguito con successo."),
    ("Exact result match", "Quota di risultati che coincidono esattamente con il risultato di riferimento."),
    ("Supported result", "Quota di risultati esatti o equivalenti e supportati dalle evidenze disponibili."),
    ("Pass@1", "Quota di casi risolti correttamente al primo tentativo di generazione."),
    ("Requirement pass rate", "Percentuale media dei requisiti della domanda soddisfatti dal risultato."),
)


def label(value: str) -> str:
    return LABEL_OVERRIDES.get(
        value, value.replace("_", " ").title().replace("Ndcg", "nDCG")
    )


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def duration(value: Any) -> str:
    seconds = max(0, round(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:d}m {seconds:02d}s"


def bar(value: float, color: str = "#2563eb") -> str:
    width = max(0.0, min(100.0, value * 100))
    return (
        '<div class="bar-track"><div class="bar" '
        f'style="width:{width:.2f}%;background:{color}"></div></div>'
    )


def metric_table(
    title: str, metrics: dict[str, Any], keys: tuple[str, ...], *, rates: bool = True
) -> str:
    rows = []
    for key in keys:
        if key not in metrics:
            continue
        value = float(metrics[key])
        shown = pct(value) if rates else f"{value:.3f}"
        rows.append(
            f"<tr><td>{html.escape(label(key))}</td><td>{bar(value)}</td>"
            f'<td class="num">{shown}</td></tr>'
        )
    return f"<section><h2>{html.escape(title)}</h2><table>{''.join(rows)}</table></section>"


def retrieval_comparison(table_metrics: dict[str, Any]) -> str:
    all_rows = table_metrics.get("mean_metrics", {})
    ok_rows = table_metrics.get("mean_metrics_successful_queries", {})
    rows = []
    for key in RETRIEVAL_ORDER:
        if key not in all_rows:
            continue
        all_value = float(all_rows[key])
        ok_value = float(ok_rows.get(key, 0))
        rows.append(
            f"<tr><td>{html.escape(key)}</td>"
            f"<td>{bar(all_value, '#2563eb')}<small>Tutte: {pct(all_value)}</small></td>"
            f"<td>{bar(ok_value, '#16a34a')}<small>Riuscite: {pct(ok_value)}</small></td></tr>"
        )
    return (
        "<section><h2>1. Ranking prodotto dal retriever</h2>"
        "<p>Misura soltanto se le tabelle gold erano presenti nei candidati recuperati; "
        "non indica che l'agente le abbia selezionate.</p>"
        f"<table><thead><tr><th>Metrica</th><th>Tutte</th><th>Query riuscite</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def selection_summary(table_metrics: dict[str, Any]) -> str:
    metrics = table_metrics.get("mean_selection_metrics", {})
    keys = ("SelectionHit", "SelectionRecall", "SelectionPrecision", "ExactSelection")
    return metric_table("2. Tabelle finali selezionate dall'agente", metrics, keys)


def expansion_summary(table_metrics: dict[str, Any]) -> str:
    metrics = table_metrics.get("expansion_metrics", {})
    queries = int(metrics.get("query_count", 0))
    contributions = int(metrics.get("gold_selected_beyond_initial_count", 0))
    rate = float(metrics.get("gold_selected_beyond_initial_rate", 0))
    return (
        "<section><h2>3. Contributo osservabile dell'espansione</h2>"
        "<p>Conta i casi in cui l'agente ha usato l'espansione e ha poi selezionato "
        "una tabella gold che nel ranking originale era oltre le prime 10.</p>"
        f"<div class=\"cards\"><div class=\"card\"><span>Espansioni usate</span><strong>{queries}</strong></div>"
        f"<div class=\"card\"><span>Espansioni con contributo</span><strong>{contributions}</strong></div>"
        f"<div class=\"card\"><span>Tasso di contributo</span><strong>{pct(rate)}</strong></div></div></section>"
    )


def table_count_analysis(table_metrics: dict[str, Any]) -> str:
    groups = table_metrics.get("table_count_analysis", {})
    rows = []
    outcome_rows = []
    for bucket in ("1", "2", "3", "4+"):
        values = groups.get(bucket)
        if not values:
            continue
        retrieval = values.get("retrieval", {})
        selection = values.get("selection", {})
        applicable = int(values.get("code_applicable_count", 0))
        exact_result = pct(values.get("exact_result_match_rate", 0)) if applicable else "—"
        rows.append(
            f"<tr><td>{html.escape(bucket)}</td><td>{int(values['case_count'])}</td>"
            f"<td>{applicable}</td>"
            f"<td>{pct(retrieval.get('Recall@20', 0))}</td>"
            f"<td>{pct(retrieval.get('FullCoverage@20', 0))}</td>"
            f"<td>{pct(selection.get('SelectionRecall', 0))}</td>"
            f"<td>{pct(selection.get('ExactSelection', 0))}</td>"
            f"<td>{exact_result}</td>"
            f"<td>{float(values.get('median_elapsed_seconds', 0)):.1f}s</td>"
            f"<td>{pct(values.get('expansion_usage_rate', 0))}</td></tr>"
        )
        outcomes = values.get("selection_outcomes", {})
        outcome_rows.append(
            f"<tr><td>{html.escape(bucket)}</td><td>{int(values['case_count'])}</td>"
            f"<td>{int(outcomes.get('exact', 0))}</td>"
            f"<td>{int(outcomes.get('partial', 0))}</td>"
            f"<td>{int(outcomes.get('over_selection', 0))}</td>"
            f"<td>{int(outcomes.get('miss', 0))}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<section><h2>4. Comportamento all'aumentare delle tabelle richieste</h2>"
        "<p>Le fasce sono definite dal numero di tabelle gold, non dal numero scelto dall'agente. "
        "Full Coverage@20 richiede che tutte le gold siano presenti nelle prime 20.</p>"
        "<table><thead><tr><th>Gold richieste</th><th>Query</th><th>Casi risultato applicabili</th><th>Retrieval Recall@20</th>"
        "<th>Full Coverage@20</th><th>Selection Recall</th><th>Exact Selection</th>"
        "<th>Exact Result</th><th>Tempo mediano</th><th>Uso espansione</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<h3>Esiti della selezione</h3><p>Exact: tutte e sole le gold; Partial: almeno una gold ma non tutte; "
        "Over-selection: tutte le gold più fonti estranee; Miss: nessuna gold.</p>"
        "<table><thead><tr><th>Gold richieste</th><th>Query</th><th>Exact</th><th>Partial</th>"
        f"<th>Over-selection</th><th>Miss</th></tr></thead><tbody>{''.join(outcome_rows)}</tbody></table></section>"
    )


def performance_summary(job: dict[str, Any], table_metrics: dict[str, Any]) -> str:
    values = table_metrics.get("performance_metrics", {})
    try:
        wall_seconds = (
            datetime.fromisoformat(str(job["finished_at"]))
            - datetime.fromisoformat(str(job["started_at"]))
        ).total_seconds()
    except (KeyError, TypeError, ValueError):
        wall_seconds = values.get("total_query_elapsed_seconds", 0)
    means = values.get("mean_tokens_per_query", {})
    totals = values.get("token_totals", {})
    token_labels = (
        ("p1_p2", "Discovery (P1+P2)"),
        ("p3", "Coder complessivo (P3)"),
        ("p4", "Sintesi risposta (P4)"),
    )
    automatic_coder = bool(
        job.get("settings", {}).get("resolved_config", {}).get("automatic_test_coder")
    )
    coder_note = (
        " Poiché automatic_test_coder è attivo, P3 comprende le varianti full, "
        "schema_only e minimal; il residuo non attribuito copre lavoro P3 senza variante salvata."
        if automatic_coder else ""
    )
    token_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class=\"num\">{float(means.get(key, 0)):,.0f}</td>"
        f"<td class=\"num\">{int(totals.get(key, 0)):,}</td></tr>"
        for key, name in token_labels
    )
    if automatic_coder:
        context_means = values.get("mean_coder_tokens_per_query_by_context", {})
        context_totals = values.get("coder_token_totals_by_context", {})
        for key, name in (("full", "↳ Coder full"), ("schema_only", "↳ Coder schema only"), ("minimal", "↳ Coder minimal"), ("unattributed", "↳ Coder condiviso/non attribuito")):
            token_rows += (
                f"<tr><td>{html.escape(name)}</td><td class=\"num\">{float(context_means.get(key, 0)):,.0f}</td>"
                f"<td class=\"num\">{int(context_totals.get(key, 0)):,}</td></tr>"
            )
    return (
        "<section><h2>5. Costi e tempi</h2>"
        "<div class=\"cards\">"
        f"<div class=\"card\"><span>Tempo medio per query</span><strong>{duration(values.get('mean_query_elapsed_seconds', 0))}</strong></div>"
        f"<div class=\"card\"><span>Tempo mediano per query</span><strong>{duration(values.get('median_query_elapsed_seconds', 0))}</strong></div>"
        f"<div class=\"card\"><span>P90 per query</span><strong>{duration(values.get('p90_query_elapsed_seconds', 0))}</strong></div>"
        f"<div class=\"card\"><span>Durata totale batch</span><strong>{duration(wall_seconds)}</strong></div></div>"
        "<p>La durata totale è wall-clock da <code>started_at</code> a <code>finished_at</code>. "
        f"P1 e P2 sono tracciate congiuntamente; P4 è la sintesi finale.{coder_note}</p>"
        "<table><thead><tr><th>Fase</th><th>Token medi/query</th><th>Token totali</th></tr></thead>"
        f"<tbody>{token_rows}</tbody></table></section>"
    )


def metric_descriptions() -> str:
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(description)}</td></tr>"
        for name, description in METRIC_DESCRIPTIONS
    )
    return (
        "<section><h2>Come leggere le metriche principali</h2>"
        "<p>Tutte le metriche sono migliori quando il valore è più alto. "
        "Le metriche del codice sono calcolate sui casi dichiarati applicabili.</p>"
        "<table class=\"metric-guide\"><thead><tr><th>Metrica</th>"
        f"<th>Descrizione</th></tr></thead><tbody>{rows}</tbody></table></section>"
    )


def code_comparison(code: dict[str, Any]) -> str:
    contexts = [name for name in CONTEXT_ORDER if name in code]
    header = "".join(f"<th>{html.escape(label(name))}</th>" for name in contexts)
    rows = []
    for key in CODE_RATES:
        if not any(key in code[name] for name in contexts):
            continue
        cells = []
        for name in contexts:
            value = float(code[name].get(key, 0))
            cells.append(f"<td>{bar(value)}<small>{pct(value)}</small></td>")
        rows.append(f"<tr><td>{html.escape(label(key))}</td>{''.join(cells)}</tr>")
    return (
        "<section><h2>Esecuzione del codice per livello di contesto</h2>"
        "<p>Le percentuali misurano affidabilità tecnica e correttezza del risultato.</p>"
        f"<table><thead><tr><th>Metrica</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def count_comparison(title: str, code: dict[str, Any], field: str) -> str:
    contexts = [name for name in CONTEXT_ORDER if name in code]
    categories = sorted({key for name in contexts for key in code[name].get(field, {})})
    maximum = max(
        (int(code[name].get(field, {}).get(category, 0)) for name in contexts for category in categories),
        default=1,
    )
    rows = []
    for category in categories:
        cells = []
        for name in contexts:
            count = int(code[name].get(field, {}).get(category, 0))
            cells.append(f"<td>{bar(count / maximum, '#dc2626')}<small>{count}</small></td>")
        rows.append(f"<tr><td>{html.escape(label(category))}</td>{''.join(cells)}</tr>")
    header = "".join(f"<th>{html.escape(label(name))}</th>" for name in contexts)
    return (
        f"<section><h2>{html.escape(title)}</h2><table><thead><tr><th>Categoria</th>"
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
    )


def write_csv(path: Path, job: dict[str, Any]) -> None:
    metrics = job["batch_metrics"]
    rows: list[tuple[str, str, str, Any]] = []
    retrieval = metrics["table_selection"]
    for group in (
        "mean_metrics", "mean_metrics_successful_queries", "mean_selection_metrics",
    ):
        for name, value in retrieval.get(group, {}).items():
            area = "selection" if group == "mean_selection_metrics" else "retrieval"
            rows.append((area, group, name, value))
    for name, value in retrieval.get("expansion_metrics", {}).items():
        rows.append(("expansion", "observed_contribution", name, value))
    for bucket, values in retrieval.get("table_count_analysis", {}).items():
        for name, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append(("table_count", bucket, name, value))
        for subgroup in ("retrieval", "selection", "selection_outcomes"):
            for name, value in values.get(subgroup, {}).items():
                rows.append(("table_count", f"{bucket}:{subgroup}", name, value))
    performance = retrieval.get("performance_metrics", {})
    for name, value in performance.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append(("performance", "timing", name, value))
    for group in (
        "mean_tokens_per_query", "token_totals",
        "mean_coder_tokens_per_query_by_context", "coder_token_totals_by_context",
    ):
        for name, value in performance.get(group, {}).items():
            rows.append(("performance", group, name, value))
    for context, values in metrics.get("code", {}).items():
        for name, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append(("code", context, name, value))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("area", "group", "metric", "value"))
        writer.writerows(rows)


def write_markdown(path: Path, job: dict[str, Any]) -> None:
    table = job["batch_metrics"]["table_selection"]
    retrieval = table["mean_metrics"]
    lines = [
        f"# LakeGen batch {job['job_id']}", "",
        f"- Stato: **{job['status']}**",
        f"- Domande: **{job['processed']} / {job['question_count']}**",
        f"- Fallite: **{job['failed']}**", "",
        "## Come leggere le metriche principali", "",
        "Tutte le metriche sono migliori quando il valore è più alto. Le metriche del codice "
        "sono calcolate sui casi dichiarati applicabili.", "",
        "| Metrica | Descrizione |", "|---|---|",
    ]
    lines.extend(f"| {name} | {description} |" for name, description in METRIC_DESCRIPTIONS)
    lines.extend(["", "## Retrieval", "", "| Metrica | Valore |", "|---|---:|"])
    lines.extend(f"| {key} | {float(retrieval[key]):.3f} |" for key in RETRIEVAL_ORDER if key in retrieval)
    selection = table.get("mean_selection_metrics", {})
    lines.extend(["", "## Selezione finale dell'agente", "", "| Metrica | Valore |", "|---|---:|"])
    lines.extend(f"| {key} | {pct(selection[key])} |" for key in ("SelectionHit", "SelectionRecall", "SelectionPrecision", "ExactSelection") if key in selection)
    expansion = table.get("expansion_metrics", {})
    lines.extend([
        "", "## Contributo osservabile dell'espansione", "",
        f"- Espansioni usate: **{int(expansion.get('query_count', 0))}**",
        f"- Espansioni che hanno portato a selezionare una gold oltre la posizione 10: **{int(expansion.get('gold_selected_beyond_initial_count', 0))}**",
        f"- Tasso di contributo: **{pct(expansion.get('gold_selected_beyond_initial_rate', 0))}**",
    ])
    groups = table.get("table_count_analysis", {})
    lines.extend(["", "## Prestazioni per numero di tabelle gold", "", "| Gold | Query | Applicabili | Recall@20 | Full coverage@20 | Selection recall | Exact selection | Exact result | Tempo mediano |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for bucket in ("1", "2", "3", "4+"):
        values = groups.get(bucket)
        if values:
            applicable = int(values.get("code_applicable_count", 0))
            exact_result = pct(values.get("exact_result_match_rate", 0)) if applicable else "—"
            lines.append(
                f"| {bucket} | {values['case_count']} | {applicable} | {pct(values['retrieval'].get('Recall@20', 0))} | "
                f"{pct(values['retrieval'].get('FullCoverage@20', 0))} | {pct(values['selection'].get('SelectionRecall', 0))} | "
                f"{pct(values['selection'].get('ExactSelection', 0))} | {exact_result} | "
                f"{float(values.get('median_elapsed_seconds', 0)):.1f}s |"
            )
    performance = table.get("performance_metrics", {})
    try:
        wall_seconds = (datetime.fromisoformat(str(job["finished_at"])) - datetime.fromisoformat(str(job["started_at"]))).total_seconds()
    except (KeyError, TypeError, ValueError):
        wall_seconds = performance.get("total_query_elapsed_seconds", 0)
    lines.extend([
        "", "## Costi e tempi", "",
        f"- Tempo medio per query: **{duration(performance.get('mean_query_elapsed_seconds', 0))}**",
        f"- Tempo mediano per query: **{duration(performance.get('median_query_elapsed_seconds', 0))}**",
        f"- P90 per query: **{duration(performance.get('p90_query_elapsed_seconds', 0))}**",
        f"- Durata totale batch: **{duration(wall_seconds)}**", "",
        "| Fase | Token medi/query | Token totali |", "|---|---:|---:|",
    ])
    for key, name in (("p1_p2", "Discovery (P1+P2)"), ("p3", "Coder complessivo (P3)"), ("p4", "Sintesi (P4)")):
        lines.append(f"| {name} | {float(performance.get('mean_tokens_per_query', {}).get(key, 0)):,.0f} | {int(performance.get('token_totals', {}).get(key, 0)):,} |")
    if job.get("settings", {}).get("resolved_config", {}).get("automatic_test_coder"):
        for key, name in (("full", "↳ Coder full"), ("schema_only", "↳ Coder schema only"), ("minimal", "↳ Coder minimal"), ("unattributed", "↳ Coder condiviso/non attribuito")):
            lines.append(
                f"| {name} | {float(performance.get('mean_coder_tokens_per_query_by_context', {}).get(key, 0)):,.0f} | "
                f"{int(performance.get('coder_token_totals_by_context', {}).get(key, 0)):,} |"
            )
        lines.extend(["", "Nota: con `automatic_test_coder` attivo, P3 comprende le varianti full, schema_only e minimal; l'eventuale residuo condiviso/non attribuito copre lavoro P3 non associato a una variante salvata."])
    lines.extend(["", "## Esecuzione codice", "", "| Contesto | Execution | Exact match | Pass@1 |", "|---|---:|---:|---:|"])
    for context in CONTEXT_ORDER:
        values = job["batch_metrics"].get("code", {}).get(context)
        if values:
            lines.append(
                f"| {context} | {pct(values['execution_success_rate'])} | "
                f"{pct(values['exact_result_match_rate'])} | {pct(values['pass_at_1'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(job: dict[str, Any], output_dir: Path) -> None:
    if job.get("status") != "completed" or not job.get("batch_metrics"):
        raise ValueError("Il job deve essere completato e contenere batch_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = job["batch_metrics"]
    table = metrics["table_selection"]
    code = metrics.get("code", {})
    cards = (
        ("Domande", job.get("question_count", 0)),
        ("Riuscite", table.get("successful_case_count", 0)),
        ("Fallite", table.get("failed_case_count", 0)),
        ("Retrieval Hit@5", pct(table["mean_metrics"].get("Hit@5", 0))),
        ("MRR", pct(table["mean_metrics"].get("MRR", 0))),
    )
    cards_html = "".join(
        f'<div class="card"><span>{html.escape(str(name))}</span><strong>{html.escape(str(value))}</strong></div>'
        for name, value in cards
    )
    body = "".join((
        metric_descriptions(),
        retrieval_comparison(table),
        selection_summary(table),
        expansion_summary(table),
        table_count_analysis(table),
        performance_summary(job, table),
        code_comparison(code),
        count_comparison("Categorie di errore del codice", code, "error_categories"),
        count_comparison("Esiti della valutazione", code, "evaluation_dispositions"),
    ))
    document = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LakeGen batch report</title><style>
:root{{--bg:#f5f7fb;--panel:#fff;--text:#172033;--muted:#64748b;--line:#e2e8f0}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px}}h1{{margin-bottom:4px}}h2{{margin-top:0}}p,small{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:24px 0}}
.card,section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 8px #0f172a0a}}
.card{{padding:16px}}.card span{{display:block;color:var(--muted)}}.card strong{{font-size:25px}}
section{{padding:22px;margin:18px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:680px}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--muted)}}td:first-child{{font-weight:600}}
.bar-track{{height:10px;background:#e8edf5;border-radius:10px;overflow:hidden;min-width:120px}}.bar{{height:100%;border-radius:10px}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}small{{display:block;margin-top:3px}}
@media(max-width:600px){{main{{padding:16px}}}}
</style></head><body><main>
<h1>LakeGen batch report</h1><p>Job <code>{html.escape(str(job['job_id']))}</code> · {html.escape(str(job.get('finished_at', '')))}</p>
<div class="cards">{cards_html}</div>{body}
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    write_csv(output_dir / "metrics.csv", job)
    write_markdown(output_dir / "summary.md", job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="ID del batch oppure percorso del relativo JSON")
    parser.add_argument("--jobs-dir", type=Path, default=Path(".lakegen_jobs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    candidate = Path(args.job_id)
    job_path = candidate if candidate.is_file() else args.jobs_dir / f"{args.job_id}.json"
    if not job_path.is_file():
        parser.error(f"job non trovato: {job_path}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = args.output or Path("reports") / str(job["job_id"])
    try:
        generate(job, output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Report HTML: {output / 'report.html'}")
    print(f"Metriche CSV: {output / 'metrics.csv'}")
    print(f"Riepilogo: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
