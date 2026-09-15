#!/usr/bin/env python3
"""Generate a self-contained HTML/CSV/Markdown report for a LakeGen batch."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


RETRIEVAL_ORDER = (
    "Hit@1", "Hit@5", "Hit@10", "Recall@1", "Recall@5", "Recall@10",
    "MRR", "nDCG@1", "nDCG@5", "nDCG@10",
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


def label(value: str) -> str:
    return value.replace("_", " ").title().replace("Ndcg", "nDCG")


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


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
        "<section><h2>Metriche di retrieval</h2>"
        "<p>Confronto tra tutte le domande e le sole query completate senza errore.</p>"
        f"<table><thead><tr><th>Metrica</th><th>Tutte</th><th>Query riuscite</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
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
    for group in ("mean_metrics", "mean_metrics_successful_queries"):
        for name, value in retrieval.get(group, {}).items():
            rows.append(("retrieval", group, name, value))
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
        f"- Fallite: **{job['failed']}**", "", "## Retrieval", "",
        "| Metrica | Valore |", "|---|---:|",
    ]
    lines.extend(f"| {key} | {float(retrieval[key]):.3f} |" for key in RETRIEVAL_ORDER if key in retrieval)
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
        ("Hit@5", pct(table["mean_metrics"].get("Hit@5", 0))),
        ("MRR", pct(table["mean_metrics"].get("MRR", 0))),
    )
    cards_html = "".join(
        f'<div class="card"><span>{html.escape(str(name))}</span><strong>{html.escape(str(value))}</strong></div>'
        for name, value in cards
    )
    body = "".join((
        retrieval_comparison(table),
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
