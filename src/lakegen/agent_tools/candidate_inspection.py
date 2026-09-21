"""Parallel, single-shot per-candidate table inspection, returning a compact
verdict instead of a raw schema/row dump.

Ported from the scratchpad research that validated this design
(`miniagent_100_test.py`): on 100 UK OrQa questions, keyword mode, replacing
the main agent's serial `inspect_columns` accumulation with independent,
parallel inspector calls -- each judging ONE candidate against a
deterministic value-distribution report (not a raw row sample) plus
title/description/publisher/tags/column-overlap -- beat the real unmodified
production pipeline.

The compact verdict (RELEVANT/PARTIAL/IRRELEVANT + one-sentence reason) is
what should enter the main agent's own context; the value-distribution
report and column list are working data for the inspector call only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from llama_index.core.llms import ChatMessage

from lakegen.core.table_io import read_table

MAX_COLUMNS_REQUESTED = 6
DISTRIBUTION_ROW_CAP = 5000


def _table_path(csv_dir: Path, file_name: str) -> Path:
    return Path(csv_dir) / file_name


def list_columns(csv_dir: Path, file_name: str) -> list[str]:
    path = _table_path(csv_dir, file_name)
    if not path.exists():
        return []
    df = read_table(path, nrows=1)
    return list(df.columns)


def column_distribution_report(csv_dir: Path, file_name: str, columns: list[str]) -> str:
    path = _table_path(csv_dir, file_name)
    if not path.exists():
        return f"Error: file missing: {file_name}"
    try:
        df = read_table(path, nrows=DISTRIBUTION_ROW_CAP, columns=columns)
    except Exception:  # noqa: BLE001
        try:
            df = read_table(path, nrows=DISTRIBUTION_ROW_CAP)
        except Exception as exc2:  # noqa: BLE001
            return f"Error reading table: {exc2}"
        df = df[[c for c in columns if c in df.columns]]
    total = len(df)
    lines = [f"Value distribution report ({total} rows profiled):"]
    for col in columns:
        if col not in df.columns:
            lines.append(f"- {col}: NOT FOUND in this table")
            continue
        series = df[col]
        n_null = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        header = f"- {col} ({dtype}): {n_null}/{total} null, {n_unique} distinct"
        if pd.api.types.is_numeric_dtype(series) and n_unique > 20:
            desc = series.describe()
            lines.append(
                f"{header}\n"
                f"    min={desc.get('min'):.4g} max={desc.get('max'):.4g} "
                f"mean={desc.get('mean'):.4g} median={series.median():.4g}"
            )
        else:
            counts = series.dropna().astype(str).value_counts().head(10)
            value_str = ", ".join(f"{v!r}: {c}" for v, c in counts.items())
            more = n_unique - len(counts)
            suffix = f" (+{more} more distinct values)" if more > 0 else ""
            lines.append(f"{header}\n    top values: {value_str}{suffix}")
    return "\n".join(lines)


def pick_columns(llm, question: str, title: str, description: str, columns: list[str]) -> list[str]:
    prompt = (
        "You are a table inspector deciding what to check, before you have "
        "seen any actual data. You will get a value-distribution report only "
        "for the columns you pick -- not a row sample -- so pick the columns "
        "most likely to confirm or contradict the question.\n\n"
        "First identify every specific, checkable constraint the question "
        "names -- a date or period, a place or org name, a category, a "
        "threshold, an id. For EACH such constraint, you must include "
        "whichever available column looks most likely to encode it, even if "
        "you are not certain it exists or applies -- a wrong guess costs "
        "nothing, but skipping the check means you cannot verify that "
        "constraint at all. Only after covering every named constraint should "
        "you spend remaining slots on general topical columns.\n\n"
        f"Question: {question}\n\n"
        f"Table title: {title}\nTable description: {description}\n"
        f"Available columns: {columns}\n\n"
        f"Respond with ONLY a JSON list of up to {MAX_COLUMNS_REQUESTED} exact "
        "column names from the list above, constraint-covering ones first, "
        'e.g. ["col_a", "col_b"].'
    )
    try:
        resp = llm.chat([ChatMessage(role="user", content=prompt)])
        text = str(resp.message.content or "")
        match = re.search(r"\[.*\]", text, re.DOTALL)
        picked = json.loads(match.group(0)) if match else []
    except Exception:  # noqa: BLE001
        picked = []
    valid = [c for c in picked if c in columns][:MAX_COLUMNS_REQUESTED]
    return valid or columns[:MAX_COLUMNS_REQUESTED]


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was",
    "were", "how", "many", "what", "total", "all", "and", "or", "by", "at",
    "according", "recorded", "during", "that", "this", "with", "from",
    "have", "been", "each", "into", "over", "does", "did", "there",
}


def question_content_words(question: str) -> list[str]:
    """Deterministic: the question's own distinctive nouns, stripped of
    stopwords/short filler. No LLM involved."""
    words = re.findall(r"[A-Za-z]+", question)
    seen: list[str] = []
    for w in words:
        if len(w) > 3 and w.lower() not in _STOPWORDS and w not in seen:
            seen.append(w)
    return seen


def _tokenize_column_name(name: str) -> list[str]:
    """Split a column name into word-like tokens: camelCase, snake_case,
    and punctuation all become boundaries (e.g. "InfraType" -> ["Infra",
    "Type"])."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", str(name))
    spaced = re.sub(r"[_\-./]", " ", spaced)
    return [t for t in re.findall(r"[A-Za-z]+", spaced) if len(t) > 2]


def column_lexical_overlap(question: str, columns: list[str]) -> list[str]:
    """Deterministic: which question content-words share a stem with any
    column-name TOKEN (prefix match, not substring -- "infrastructure" is
    longer than "infracode", so plain substring containment never fires;
    tokenizing "InfraType" -> "Infra" and comparing shared 5-char prefixes
    catches it)."""
    col_tokens = {
        tok.lower() for c in columns for tok in _tokenize_column_name(c)
    }
    matches = []
    for w in question_content_words(question):
        wl = w.lower()
        for tok in col_tokens:
            prefix_len = min(len(wl), len(tok), 5)
            if prefix_len >= 4 and wl[:prefix_len] == tok[:prefix_len]:
                matches.append(w)
                break
    return matches


def miniagent_inspect(
    llm, question: str, csv_dir: Path, rank: int, file_name: str, meta: dict
) -> dict:
    """Judge ONE candidate against the question. Returns a compact verdict
    dict: {rank, file, title, publisher, verdict, raw, columns_requested}.
    `raw` is the full 3-line VERDICT/COLUMNS/REASON response -- short enough
    to enter the main agent's context directly, unlike a raw schema dump."""
    title = meta.get("title", "")
    description = str(meta.get("description", ""))[:400]
    publisher = meta.get("publisher", "")
    tags = meta.get("tags", [])
    try:
        columns = list_columns(csv_dir, file_name)
        overlap = column_lexical_overlap(question, columns)
        overlap_line = (
            f"Column-name / question wording overlap: {overlap}"
            if overlap else
            "Column-name / question wording overlap: NONE -- no column name "
            "shares any wording with the question's own distinctive nouns. "
            "Treat this as real negative evidence, not a neutral gap: it can "
            "mean this table is a same-publisher, same-topic-sounding, but "
            "DIFFERENT resource -- not the right one."
        )
        requested = pick_columns(llm, question, title, description, columns)
        report = column_distribution_report(csv_dir, file_name, requested)
        prompt = (
            "You are a focused table inspector. You already chose and received "
            "a value-distribution report for the columns you judged most "
            "relevant (not a raw row sample). Judge this ONE candidate table "
            f"against the question using this report AND the table's title/"
            "description/publisher/tags/column-name overlap.\n\n"
            "A constraint the question names (a date, period, place, category) "
            "can be confirmed two ways: a matching value in the distribution "
            "report, OR the title/description stating it explicitly and "
            "exactly (e.g. a title naming one specific month is just as valid "
            "confirmation as a column value, even if no column in this table "
            "encodes that period at all -- do not penalize a table for lacking "
            "a column that its title already answers). Only mark a table down "
            "for a constraint when neither source confirms it, or when one "
            "source actively contradicts it. A column value that merely "
            "overlaps a broader range is weaker evidence than a title that "
            "names the exact period -- do not let a vague column-based match "
            "outrank an exact title-based one.\n\n"
            "If you cannot find anything that distinguishes this table from "
            "what the question needs, but its publisher and tags match the "
            "question's organisation/topic exactly, say so explicitly in your "
            "REASON (e.g. \"same publisher as expected, no data in this table "
            "contradicts the question\") rather than guessing IRRELEVANT for "
            "lack of a distinguishing detail that may not exist to find. BUT "
            "weigh the column-name overlap line below seriously -- a same-"
            "publisher table whose columns share NONE of the question's "
            "wording is a real signal it covers a different aspect of that "
            "publisher's data (a different table, not just a different "
            "period/edition of the same one), even without a value to point to.\n\n"
            f"Question: {question}\n\n"
            f"Table title: {title}\nTable description: {description}\n"
            f"Publisher: {publisher}\nTags: {tags}\n"
            f"{overlap_line}\n"
            f"Columns available (not all profiled): {columns}\n\n{report}\n\n"
            "Respond with exactly 3 lines:\n"
            "VERDICT: RELEVANT|PARTIAL|IRRELEVANT\n"
            "COLUMNS: comma-separated column names that matter (or none)\n"
            "REASON: one sentence citing the SPECIFIC value, count, title, "
            "publisher/tag, or column-overlap detail that confirms or "
            "contradicts the question"
        )
        resp = llm.chat([ChatMessage(role="user", content=prompt)])
        text = str(resp.message.content or "")
    except Exception as exc:  # noqa: BLE001
        text = f"VERDICT: IRRELEVANT\nCOLUMNS: none\nREASON: inspector error: {exc}"
        requested = []
    verdict_match = re.search(r"VERDICT:\s*(RELEVANT|PARTIAL|IRRELEVANT)", text, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
    return {
        "rank": rank, "file": file_name, "title": title, "publisher": publisher,
        "verdict": verdict, "raw": text.strip(), "columns_requested": requested,
    }


MAX_OBSERVATION_CHARS = 1500


def _observation_prompt(
    question: str, title: str, description: str, publisher: str, tags: list,
    columns: list[str], report: str,
) -> str:
    return (
        "You are a table inspector. Another agent will decide which tables "
        "answer the question; your job is only to report what THIS one table "
        "shows, so that agent does not have to open it. You are given a "
        "value-distribution report for the columns judged most relevant.\n\n"
        "Report facts, not a decision. Do not say whether the table is "
        "relevant, irrelevant, suitable, sufficient, complete or partial, do "
        "not say it matches or fails the question, and do not compare it with "
        "other tables.\n\n"
        f"Question: {question}\n\n"
        f"Table title: {title}\nTable description: {description}\n"
        f"Publisher: {publisher}\nTags: {tags}\n"
        f"Columns available (not all profiled): {columns}\n\n{report}\n\n"
        "For EACH specific, checkable constraint the question names (a date "
        "or period, a place or organisation, a category, a threshold, an id, "
        "a measure), write one line stating what the table itself shows about "
        "it: quote the exact values, ranges or counts from the report, or from "
        "the title or description when they state it (say which). When neither "
        "the report nor the metadata shows it, write 'not observable' and why "
        "(for example the column is missing, or the report only profiled the "
        "first rows). The report may cover only the first rows of a large "
        "table; say so when that limits a statement.\n\n"
        "Respond with only those lines, each starting with '- '."
    )


def inspect_candidate(
    llm, question: str, csv_dir: Path, rank: int, file_name: str, meta: dict,
    profile_header: str = "",
) -> dict:
    """Inspect ONE candidate for the main agent and report facts, no verdict.

    `profile_header` is the deterministic head of the report (row count and
    whole-file date coverage), supplied by the caller. The LLM only chooses
    which columns to profile, as in `miniagent_inspect`, and then states what
    the profile shows about each constraint the question names.
    Returns {rank, file, title, publisher, report, columns_requested, ok}.
    `report` is compact enough to enter the main agent's context directly.
    `ok` is False when the model call failed: the report then still carries
    the deterministic head and says the observations are unavailable.
    """
    title = meta.get("title", "")
    description = str(meta.get("description", ""))[:400]
    publisher = meta.get("publisher", "")
    tags = meta.get("tags", [])
    requested: list[str] = []
    ok = True
    try:
        columns = list_columns(csv_dir, file_name)
        if not columns:
            return {
                "rank": rank, "file": file_name, "title": title,
                "publisher": publisher, "columns_requested": [], "ok": False,
                "report": f"Error: table could not be read: {file_name}",
            }
        requested = pick_columns(llm, question, title, description, columns)
        report = column_distribution_report(csv_dir, file_name, requested)
        resp = llm.chat([ChatMessage(role="user", content=_observation_prompt(
            question, title, description, publisher, tags, columns, report,
        ))])
        text = str(resp.message.content or "").strip()
        lines = [line.rstrip() for line in text.splitlines() if line.lstrip().startswith("-")]
        observations = "\n".join(lines) if lines else text
    except Exception as exc:  # noqa: BLE001
        ok = False
        observations = f"- unavailable: inspector error: {exc}"
    observations = observations[:MAX_OBSERVATION_CHARS]
    parts = [profile_header.strip()] if profile_header.strip() else []
    if requested:
        parts.append("Columns examined: " + ", ".join(requested))
    parts.append("Observations:\n" + observations)
    return {
        "rank": rank, "file": file_name, "title": title, "publisher": publisher,
        "report": "\n".join(parts), "columns_requested": requested, "ok": ok,
    }
