"""Delegated final table selection: a separate, clean-context LLM call fed
only compact per-candidate verdicts -- never raw data, never the main
agent's own accumulated search-loop context -- makes the actual pick.

Ported from the scratchpad research that validated this design
(`miniagent_100_test.py:391-429`, `final_selection_delegation_test.py`).
Standalone, layered on the real production pipeline (30 questions, keyword
mode): precise 7/30 -> 14/30, and `FOUND_IMPRECISE` (gold selected but
bundled with an extra, wrong table) dropped from 9/30 to 0/30.

Known, repeatable bias (not fixed by more prompting alone, but mitigated
below): a tendency to prefer an already-aggregated/pre-summed candidate over
a more granular one that is actually the benchmark's correct table, when
both plausibly answer the question (clearest case: a 3/3 -> 0/0 flip across
independent replicates on a question about a per-entity breakdown where the
pipeline consistently picked a pre-totaled table instead). The added
paragraph below is the mitigation; re-validate against that failure pattern
before enabling this in an experiment by default.
"""
from __future__ import annotations

import json
import re

from llama_index.core.llms import ChatMessage


def final_selection(llm, question: str, verdicts: list[dict]) -> dict:
    """`verdicts` is a list of dicts shaped like `candidate_inspection.
    miniagent_inspect`'s return value (rank, file, title, publisher, raw).
    Returns {"selected": [file_name, ...], "reasoning": str}; `selected` is
    filtered to only file names actually present in `verdicts`, so a
    hallucinated name can never reach the caller."""
    if not verdicts:
        return {"selected": [], "reasoning": "No candidates were retrieved."}
    digest = "\n".join(
        f"[{v['rank']}] {v['file']} -- title: {v.get('title') or '(none)'}, "
        f"publisher: {v.get('publisher') or '(none)'}\n{v['raw']}\n"
        for v in sorted(verdicts, key=lambda x: x["rank"])
    )
    prompt = (
        "You are the table-selection agent. Independent inspectors already "
        "examined every candidate below and reported a verdict for each. Do "
        "not invent evidence beyond what they reported, and never name a "
        "file that is not listed below. Pick the table(s) needed to answer "
        "the question, or none if nothing suffices.\n\n"
        "If two or more RELEVANT candidates share the same publisher and are "
        "clearly duplicate or re-published versions of the same underlying "
        "dataset (nothing in their reports actually distinguishes them for "
        "this question), select only ONE of them -- whichever the reports "
        "make the strongest case for, or the first if truly tied. Do not "
        "select every duplicate just because each one individually looks "
        "relevant; that adds no information and is not what \"the table(s) "
        "needed\" means.\n\n"
        "A table that is already pre-aggregated or pre-summed is not "
        "automatically more sufficient than a more granular, per-entity "
        "table -- judge sufficiency strictly from what the question actually "
        "asks and what the verdicts actually establish, never from which "
        "candidate looks like it would save further aggregation work. When "
        "two candidates both plausibly answer the question at different "
        "levels of aggregation, prefer whichever one the reported evidence "
        "most concretely and specifically ties to the question's exact "
        "wording, not whichever requires less further computation.\n\n"
        f"Question: {question}\n\n"
        f"Candidates with inspector verdicts:\n{digest}\n\n"
        "Respond with ONLY a JSON object: "
        '{"selected": ["exact_file_name.parquet", ...], "reasoning": "..."}'
    )
    try:
        resp = llm.chat([ChatMessage(role="user", content=prompt)])
        text = str(resp.message.content or "")
    except Exception as exc:  # noqa: BLE001
        return {"selected": [], "reasoning": f"delegated selection error: {exc}"}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"selected": [], "reasoning": f"UNPARSEABLE: {text[:300]}"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"selected": [], "reasoning": f"JSON_ERROR: {text[:300]}"}
    valid = {v["file"] for v in verdicts}
    selected = [str(t) for t in parsed.get("selected", []) if str(t) in valid]
    return {"selected": selected, "reasoning": str(parsed.get("reasoning", ""))}
