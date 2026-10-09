"""Rule-based failure analysis over the records of records.py.

Each run fails at the first stage that loses the answer:

  retrieval   a gold table is missing from the top 20 (Chicago: no gold
              alternative in the top 20). Counted once per lake, strategy and
              question on the portals, whose two selectors share one context,
              and once per run on the pipeline lakes.
  selection   the gold tables are all in the top 20 but the selection is wrong
              (Chicago: no gold alternative selected). Once per run.
  downstream  the selection is right but the outcome is not solved. Once per
              distinct coding (question + table set) on the portals, once per
              run on the pipeline lakes.

Every failure gets the first category whose rule holds, in the order of the
lists below. The facts behind each rule (keyword matches in the catalogue entry,
terms found in the cells, sibling tables, window ranks, judged requirements) are
written next to the label in results/failures.jsonl.

  PYTHONPATH=.:src .venv/bin/python experiments/paper_analysis/failures.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv  # noqa: E402
from records import LAKES, ROOT, load, resource_id, solved_flag  # noqa: E402

load_dotenv(ROOT / ".env")

import duckdb  # noqa: E402

from lakegen.config.paths import resolve_portal_tables_dir  # noqa: E402
from lakegen.retrieval.factory import get_solr  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
NON_ENGLISH = {"valencia", "paris", "bologna"}
CATALOGUE_HAS_COLUMNS = {"paris", "bologna", "uk", "target_bird", "chicago"}
AND_STRATEGIES = {"keyword", "hybrid"}
LLM_WINDOW, LLM_WINDOW_EXPANDED, JEV_WINDOW = 10, 15, 8
CELL_ROWS, CELL_COLUMNS = 200_000, 60

RETRIEVAL = ("missing_metadata", "wrong_year", "wrong_geography", "multilingual",
             "value_in_cells", "and_restrictive", "terminology", "outranked")
SELECTION = ("not_inspected", "join_missed", "extra_table", "column_ambiguity", "wrong_rejection")
DOWNSTREAM = ("coder_rejected", "execution_error", "cleaning", "wrong_filter",
              "wrong_aggregation", "wrong_query", "unverified")

GEO_WORDS = {
    "district", "districts", "borough", "boroughs", "county", "counties", "ward", "wards",
    "region", "regions", "city", "town", "municipality", "neighbourhood", "neighborhood",
    "state", "province", "zone", "area", "areas", "quartiere", "quartieri", "quartier",
    "quartiers", "arrondissement", "arrondissements", "barrio", "barrios", "distrito",
    "distritos", "districte", "comune", "municipio", "zona", "zone", "provincia", "regione",
    "manhattan", "brooklyn", "queens", "bronx", "staten", "london", "england", "scotland",
    "wales", "valencia", "paris", "bologna", "chicago",
}
STOP = {
    "the", "of", "and", "for", "in", "on", "by", "to", "a", "an", "de", "del", "la", "le",
    "les", "el", "los", "las", "di", "della", "delle", "dei", "e", "y", "et", "du", "des",
    "da", "per", "en", "con", "al", "a", "il", "lo", "i", "gli", "un", "una", "une",
}
TEMPORAL = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
    "octubre", "noviembre", "diciembre", "gener", "febrer", "marc", "maig", "juny", "juliol", "agost",
    "setembre", "octubre", "novembre", "desembre", "janvier", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "aout", "septembre", "octobre", "decembre", "gennaio", "febbraio", "aprile", "maggio",
    "giugno", "luglio", "settembre", "ottobre", "dicembre", "q1", "q2", "q3", "q4", "quarter",
    "trimestre", "semestre", "annual", "anual", "annuel", "annuale", "monthly", "mensual", "mensuel",
}
CLEANING = re.compile(
    r"\b(cast(?:ing)?|convert(?:ed|ing)? (?:to|into) (?:a )?(?:numeric|number|integer|float|date)"
    r"|to_numeric|non-numeric|stored as (?:text|strings?)|string comparison|compared as strings?"
    r"|missing values|null values|\bnan\b|duplicate rows|duplicates|whitespace|trailing spaces"
    r"|thousands? separator|decimal (?:comma|separator)|encoding|case[- ]sensitiv\w*|normali[sz]\w*)",
    re.I)


def fold(text) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", fold(text)) if t not in STOP]


def years(text) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", str(text or "")))


# ---------------------------------------------------------------- catalogue

@lru_cache(maxsize=None)
def solr(lake):
    return get_solr(lake)


def _filter_for(rid: str) -> list[str]:
    quoted = rid.replace('"', '\\"')
    return [f'resource_id:"{quoted}"', f'dataset_id:"{quoted}"', f'resource_id:*___{quoted}']


@lru_cache(maxsize=None)
def doc(lake: str, rid: str) -> tuple[str, dict]:
    """(filter that finds the entry, the entry) for a table id, or ("", {})."""
    for fq in _filter_for(rid):
        response = solr(lake).select(["*:*"], def_type="lucene", rows=1, fq=fq, fl="*")
        docs = response.get("response", {}).get("docs", [])
        if docs:
            return fq, docs[0]
    return "", {}


def column_names(entry: dict) -> list[str]:
    """Catalogue column names and labels; the Solr client folds columns.* into a list."""
    names = []
    for column in entry.get("columns") or []:
        if isinstance(column, dict):
            names += [column.get("name"), column.get("label")]
    for key in ("columns.name", "columns.label"):
        value = entry.get(key)
        names += value if isinstance(value, list) else [value]
    return [str(n) for n in names if n]


def meta_text(entry: dict) -> str:
    parts = [entry.get(k) for k in ("title", "description", "resource_description", "resource_name", "publisher")]
    tags = entry.get("tags")
    parts += tags if isinstance(tags, list) else [tags]
    parts += column_names(entry)
    return fold(" ".join(str(p) for p in parts if p))


def title_of(lake: str, rid: str) -> str:
    entry = doc(lake, rid)[1]
    title = entry.get("title") or ""
    name = entry.get("resource_name") or ""
    return f"{title} {name}".strip() if name and fold(name) not in fold(title) else str(title)


@lru_cache(maxsize=None)
def keyword_matches(lake: str, rid: str, keyword: str) -> bool:
    """Would retrieval's AND query for this one keyword match the table's entry?"""
    fq = doc(lake, rid)[0]
    if not fq or not str(keyword).strip():
        return False
    response = solr(lake).select([str(keyword)], q_op="AND", rows=0, fq=fq)
    return int(response.get("response", {}).get("numFound", 0)) > 0


@lru_cache(maxsize=None)
def catalogue_df(lake: str, keyword: str) -> int:
    response = solr(lake).select([str(keyword)], q_op="AND", rows=0)
    return int(response.get("response", {}).get("numFound", 0))


def missing_metadata(lake: str, entry: dict) -> bool:
    title = fold(entry.get("title"))
    description = " ".join(str(entry.get(k) or "") for k in ("description", "resource_description")).strip()
    generic_title = title in {"csv", "data", "dataset", "file", "xls", "xlsx", "json", "table"} or len(title) <= 3
    no_columns = lake in CATALOGUE_HAS_COLUMNS and not column_names(entry)
    return generic_title or len(description) < 20 or no_columns


# ---------------------------------------------------------------- tables

@lru_cache(maxsize=None)
def table_files(lake: str) -> dict[str, Path]:
    directory = Path(resolve_portal_tables_dir(lake))
    return {resource_id(p.name): p for p in directory.glob("*.parquet")}


@lru_cache(maxsize=None)
def columns_of(lake: str, rid: str) -> tuple[tuple[str, str], ...]:
    path = table_files(lake).get(rid)
    if path is None:
        return ()
    try:
        rows = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    except Exception:
        return ()
    return tuple((str(r[0]), str(r[1])) for r in rows)


@lru_cache(maxsize=None)
def terms_in_cells(lake: str, rid: str, terms: tuple[str, ...]) -> frozenset[str]:
    """The terms found in the table's text cells (first CELL_ROWS rows, CELL_COLUMNS text columns)."""
    path = table_files(lake).get(rid)
    text_columns = [name for name, kind in columns_of(lake, rid) if kind.upper().startswith("VARCHAR")][:CELL_COLUMNS]
    if path is None or not text_columns or not terms:
        return frozenset()
    quoted = ['"' + c.replace('"', '""') + '"' for c in text_columns]
    folded = [f"strip_accents(lower({q}))" for q in quoted]
    selects = []
    for i, _ in enumerate(terms):
        selects.append("bool_or(" + " OR ".join(f"contains({f}, ${i + 1})" for f in folded) + f") AS t{i}")
    sql = (f"SELECT {', '.join(selects)} FROM (SELECT {', '.join(quoted)} "
           f"FROM read_parquet('{path}') LIMIT {CELL_ROWS})")
    try:
        row = duckdb.execute(sql, list(terms)).fetchone()
    except Exception:
        return frozenset()
    return frozenset(t for t, hit in zip(terms, row) if hit)


# ---------------------------------------------------------------- question terms

def question_entities(question: str) -> list[str]:
    """Values the question names: numbers, codes and capitalised words after the first."""
    found = re.findall(r"\b\d+(?:[.,]\d+)?\b", question)
    found = [n for n in found if len(re.sub(r"\D", "", n)) >= 2]
    words = re.findall(r"[\wÀ-ÿ'’\-]+", question)
    for i, word in enumerate(words):
        if i == 0 or len(word) < 3 or word.lower() in STOP:
            continue
        if word[0].isupper() or (any(c.isdigit() for c in word) and any(c.isalpha() for c in word)):
            found.append(word)
    return list(dict.fromkeys(found))[:10]


# ---------------------------------------------------------------- rules

def classify_retrieval(r) -> tuple[str, dict]:
    missing = [g for g in r.gold if g not in r.ranking[:20]]
    g = missing[0]
    fq, entry = doc(r.lake, g)
    facts = {"gold": g, "found_in_catalogue": bool(fq), "title": title_of(r.lake, g)}
    keywords = [k for k in r.keywords if str(k).strip()]
    matched = [k for k in keywords if keyword_matches(r.lake, g, k)]
    facts.update(keywords=keywords, matched_keywords=matched)
    meta = meta_text(entry)
    candidates = [fold(k) for k in keywords] + [fold(e) for e in question_entities(r.question)]
    terms = tuple(dict.fromkeys(t for t in candidates if len(t) >= 3 or t.isdigit()))
    in_cells = terms_in_cells(r.lake, g, terms)
    cells_only = sorted(t for t in in_cells if t not in meta and not any(fold(k) == t for k in matched))
    facts["cells_only_terms"] = cells_only

    if fq and missing_metadata(r.lake, entry):
        return "missing_metadata", facts
    gold_title = title_of(r.lake, g)
    variant = lambda title: {t for t in tokens(title) if not t.isdigit() and t not in TEMPORAL}
    period = lambda title: years(title) | {t for t in tokens(title) if t in TEMPORAL}
    gold_tokens = variant(gold_title)
    # a question's place: a capitalised word without digits that does not open the question
    named = {fold(w) for w in re.findall(r"(?<!^)(?<![.?!¿¡] )\b[A-ZÀ-Ý][^\W\d_]+", r.question)}
    places = {t for w in named for t in tokens(w)} - TEMPORAL
    for t in r.ranking[:20]:
        title = title_of(r.lake, t)
        if t not in r.gold and gold_tokens and variant(title) == gold_tokens and period(title) != period(gold_title):
            facts.update(sibling=t, sibling_title=title)
            return "wrong_year", facts
    for t in r.ranking[:20]:
        title = title_of(r.lake, t)
        other = variant(title)
        if t in r.gold or not other or not gold_tokens:
            continue
        diff = (other ^ gold_tokens) - STOP
        jaccard = len(other & gold_tokens) / len(other | gold_tokens)
        if 1 <= len(diff) <= 4 and jaccard >= 0.5 and (
                any(d in GEO_WORDS for d in diff) or any(d in places for d in gold_tokens - other)):
            facts.update(sibling=t, sibling_title=title, differs_by=sorted(diff))
            return "wrong_geography", facts
    if r.lake in NON_ENGLISH:
        foreign = [k for k in keywords if k not in matched and catalogue_df(r.lake, k) == 0 and catalogue_df("nyc", k) > 0]
        if foreign:
            facts["foreign_keywords"] = foreign
            return "multilingual", facts
    if cells_only:
        return "value_in_cells", facts
    if r.strategy in AND_STRATEGIES and keywords and 0 < len(matched) < len(keywords):
        return "and_restrictive", facts
    if keywords and not matched:
        return "terminology", facts
    return "outranked", facts


def _column_jaccard(lake, a, b) -> float:
    ca = {fold(n) for n, _ in columns_of(lake, a)}
    cb = {fold(n) for n, _ in columns_of(lake, b)}
    return len(ca & cb) / len(ca | cb) if ca and cb else 0.0


def classify_selection(r) -> tuple[str, dict]:
    gold, chosen = set(r.gold), set(r.selected)
    unselected = [g for g in r.gold if g not in chosen]
    rank = {t: i + 1 for i, t in enumerate(r.ranking)}
    facts = {"gold": r.gold, "selected": r.selected,
             "gold_ranks": {g: rank.get(g) for g in r.gold}}
    if r.protocol == "pipeline":
        window = JEV_WINDOW if r.selector == "jev" else (LLM_WINDOW_EXPANDED if r.expansion_used else LLM_WINDOW)
        facts["window"] = window
        # Chicago: the run failed only if no alternative was reachable within the window
        beyond = [g for g in unselected if (rank.get(g) or 99) > window]
        reachable = [g for g in unselected if (rank.get(g) or 99) <= window]
        if beyond and (not r.any_gold or not reachable):
            return "not_inspected", facts
    if not r.any_gold and r.multi and chosen & gold and not gold <= chosen:
        return "join_missed", facts
    if not r.any_gold and gold < chosen:
        facts["extra"] = sorted(chosen - gold)
        return "extra_table", facts
    for t in chosen - gold:
        for g in unselected:
            similarity = _column_jaccard(r.lake, t, g)
            if similarity >= 0.5:
                facts.update(confused=t, with_gold=g, column_jaccard=round(similarity, 2))
                return "column_ambiguity", facts
    facts["empty"] = not chosen
    return "wrong_rejection", facts


def classify_downstream(r) -> tuple[str, dict]:
    result = r.result or {}
    evaluation = result.get("code_evaluation") or {}
    facts = {"error_category": evaluation.get("error_category"),
             "judge": evaluation.get("semantic_correctness")}
    stages = result.get("pipeline_stages") or {}
    if (result.get("execution_outcome") or {}).get("status") == "tables_rejected" or stages.get("code_execution") == "tables_rejected":
        return "coder_rejected", facts
    executed = evaluation.get("final_execution_success")
    if executed is None:
        executed = evaluation.get("execution_success")
    if not executed or evaluation.get("error_category") in ("execution_error", "generation_error"):
        return "execution_error", facts
    judgment = evaluation.get("semantic_judgment") or {}
    types = {q.get("id"): str(q.get("type") or "") for q in judgment.get("requirements") or []}
    descriptions = {q.get("id"): str(q.get("description") or "") for q in judgment.get("requirements") or []}
    failed = [a for a in judgment.get("requirement_assessments") or [] if a.get("status") == "failed"]
    facts["failed_types"] = [types.get(a.get("requirement_id")) for a in failed]
    if failed:
        text = " ".join(str(a.get("evidence") or "") + " " + descriptions.get(a.get("requirement_id"), "") for a in failed)
        if CLEANING.search(text):
            facts["cleaning_evidence"] = CLEANING.search(text).group(0)
            return "cleaning", facts
        kinds = " ".join(facts["failed_types"])
        if re.search(r"filter|scope|row_operation", kinds):
            return "wrong_filter", facts
        if re.search(r"aggregation|computation|grouping", kinds):
            return "wrong_aggregation", facts
        return "wrong_query", facts
    if judgment and evaluation.get("semantic_correctness") == "indeterminate":
        return "unverified", facts
    return "wrong_query", facts


def main() -> None:
    records = load()
    rows = []
    seen_retrieval, seen_coding = set(), set()
    for r in records:
        if not r.in_context():
            key = (r.lake, r.strategy, r.qid) if r.protocol != "pipeline" else (r.lake, r.strategy, r.selector, r.qid)
            if key in seen_retrieval:
                continue
            seen_retrieval.add(key)
            label, facts = classify_retrieval(r)
            stage = "retrieval"
        elif not r.selection_correct:
            label, facts = classify_selection(r)
            stage = "selection"
        elif r.result is not None and not solved_flag(r.result):
            key = (r.lake, r.qid, frozenset(r.selected)) if r.protocol != "pipeline" else (r.lake, r.strategy, r.selector, r.qid)
            if key in seen_coding:
                continue
            seen_coding.add(key)
            label, facts = classify_downstream(r)
            stage = "downstream"
        else:
            continue
        rows.append({"lake": r.lake, "strategy": r.strategy, "selector": r.selector, "qid": r.qid,
                     "protocol": r.protocol, "stage": stage, "label": label, "facts": facts})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "failures.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    counts = defaultdict(Counter)
    for row in rows:
        counts[(row["stage"], row["label"])][row["lake"]] += 1
    lines = ["| stage | category | " + " | ".join(LAKES) + " | total |", "|---|---|" + "---|" * (len(LAKES) + 1)]
    for stage, labels in (("retrieval", RETRIEVAL), ("selection", SELECTION), ("downstream", DOWNSTREAM)):
        for label in labels:
            c = counts[(stage, label)]
            lines.append(f"| {stage} | {label} | " + " | ".join(str(c[l]) for l in LAKES) + f" | {sum(c.values())} |")
    (OUT / "failures.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
