"""The selector's tool calls in the webapp: each one a step under the selection step, in order with the text."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from lakegen.datastructures.types import ToolCall
from lakegen.entrypoints.webapp import sections, streaming
from lakegen.pipeline.discovery.selectors.memgpt import TRACE_HEADER


class _Parent:
    id = "parent"

    def __init__(self, log):
        self.log = log

    async def stream_token(self, token):
        self.log.append(("text", token))


def _fake_step(log):
    class Step:
        def __init__(self, name, type, parent_id, show_input):
            self.name, self.type, self.parent_id, self.show_input = (
                name,
                type,
                parent_id,
                show_input,
            )
            self.output, self.is_error = "", False

        async def send(self):
            log.append(
                (
                    "step",
                    self.name,
                    self.type,
                    self.parent_id,
                    self.output,
                    self.is_error,
                    self.show_input,
                )
            )

    return Step


def test_a_tool_call_becomes_a_step_under_the_selection_step(monkeypatch):
    log = []
    monkeypatch.setattr(streaming, "cl", SimpleNamespace(Step=_fake_step(log)))

    async def run():
        bridge = streaming.StepStreamBridge(_Parent(log))
        async with bridge:
            bridge.emit("starting\n")
            bridge.emit(ToolCall("inspect", "inspect · #1 Parks", "**#1 Parks** — 10 rows"))
            bridge.emit(
                ToolCall(
                    "confirm", "⚠ confirm · refused", "Confirm refused: R2 still open.", failed=True
                )
            )
            bridge.emit("done\n")

    asyncio.run(run())
    assert log == [
        ("text", "starting\n"),
        ("step", "inspect · #1 Parks", "tool", "parent", "**#1 Parks** — 10 rows", False, False),
        (
            "step",
            "⚠ confirm · refused",
            "tool",
            "parent",
            "Confirm refused: R2 still open.",
            True,
            False,
        ),
        ("text", "done\n"),
    ]


def _session(trace):
    return SimpleNamespace(
        tables=["a.parquet"],
        candidates=["a.parquet", "b.parquet"],
        full_trace=trace,
        solr_metadata_map={
            "a.parquet": {"title": "Parks 2024"},
            "b.parquet": {"title": "Parks 2023"},
        },
        architect_reasoning="a covers 2024.",
        alternatives=[],
        tokens={"p2": 7},
    )


def test_a_memgpt_selection_is_summed_up_by_title_and_other_traces_keep_the_activity_log():
    memgpt = sections.build_phase2_summary(_session(f"{TRACE_HEADER}\n[turn 1] ..."))
    assert "- **Parks 2024** · `a.parquet`" in memgpt and "a covers 2024." in memgpt
    assert (
        "[turn 1]" not in memgpt and "Parks 2023" not in memgpt
    )  # the candidates and the calls are steps
    other = sections.build_phase2_summary(_session("--- Unified Agent ---\nranked"))
    assert "ranked" in other and "- `a.parquet`" in other


def test_the_alternatives_the_selector_offers_follow_the_selected_tables():
    session = _session(f"{TRACE_HEADER}\n[turn 1] ...")
    assert "Other selections" not in sections.build_phase2_summary(session)
    session.alternatives = [
        {"label": "A1", "tables": ["b.parquet"], "note": "the 2023 parks: one year earlier"}
    ]
    summary = sections.build_phase2_summary(session)
    assert "- **Parks 2023** · `b.parquet` — the 2023 parks: one year earlier" in summary
    assert (
        summary.index("Parks 2024")
        < summary.index("Other selections that also answer the question")
        < summary.index("a covers 2024.")
    )
