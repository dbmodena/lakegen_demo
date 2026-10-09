"""Keep the main agent's context lean: withdraw spent tools, never print a
missing column description as "None", and never re-send an inspection."""

import itertools

import pytest
from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool
from pydantic import Field

from lakegen.config.experiment import DiscoveryConfig
from lakegen.discovery import tools as tools_p12_v1
from lakegen.discovery.candidates import (
    _candidate_columns,
    format_candidate_context,
    solr_metadata_from_doc,
)
from lakegen.discovery.tools import inspection as inspection_tools
from lakegen.discovery.tools import search as search_tools
from lakegen.llm.agents.runner import run_agent_workflow
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode

# agent_tools_2 was the experimental v2 module; its improvements were promoted
# into agent_tools/tools_p12.py and the duplicate package was removed, so only
# the promoted module is left to exercise here.
BOTH = pytest.mark.parametrize("module", [tools_p12_v1])


def _hit(resource_id, rank=1):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id, "columns": []},
        score=1.0 / rank,
        rank=rank,
        lexical_rank=rank,
    )


def _manager(module, monkeypatch, tmp_path, *, found=True, mode=RetrievalMode.KEYWORD, **discovery):
    class FakeService:
        def retrieve(self, **_kwargs):
            return [_hit("table")] if found else []

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    state = module.P12State()
    manager = module.Phase12ToolsManager(
        state, object(), ["table.parquet"], tmp_path,
        retrieval_config=RetrievalConfig(mode=mode),
        discovery_config=DiscoveryConfig(**discovery),
    )
    return state, manager


def _candidates(state, total, visible):
    state.all_candidates = [f"t{i}.parquet" for i in range(total)]
    state.visible_candidate_count = visible


# --- the runner withdraws tools on the real LlamaIndex loop -----------------


class _RecordingLLM(MockFunctionCallingLLM):
    offered: list = Field(default_factory=list)

    async def astream_chat_with_tools(self, tools, *args, **kwargs):
        self.offered.append([tool.metadata.name for tool in tools])
        return await super().astream_chat_with_tools(tools, *args, **kwargs)


def test_runner_stops_offering_a_tool_once_it_is_spent():
    script = iter([("search", {}), ("search", {}), ("confirm", {}), None])
    ids = itertools.count()

    def respond(_messages):
        step = next(script)
        if step is None:
            return ChatMessage(role="assistant", content="done")
        name, kwargs = step
        return ChatMessage(role="assistant", blocks=[
            ToolCallBlock(tool_name=name, tool_kwargs=kwargs, tool_call_id=f"call-{next(ids)}")
        ])

    spent = {"search": False}

    def search() -> str:
        """Search."""
        if spent["search"]:
            return "SPENT: search refused"
        spent["search"] = True
        return "results"

    def confirm() -> str:
        """Confirm."""
        return "confirmed"

    llm = _RecordingLLM(response_generator=respond)
    streamed = []
    run_agent_workflow(
        llm=llm, system_prompt="s", user_prompt="u", agent_name="t",
        emit_stream=streamed.append, max_iterations=8, max_repeats=5,
        tools=[FunctionTool.from_defaults(fn=search), FunctionTool.from_defaults(fn=confirm)],
        tool_available=lambda name: not (name == "search" and spent["search"]),
    )

    assert llm.offered == [["search", "confirm"]] + [["confirm"]] * 3
    # A stray call to the withdrawn tool still gets that tool's own refusal.
    assert any("SPENT: search refused" in chunk for chunk in streamed)
    assert not any("not found" in chunk for chunk in streamed)


def test_runner_without_a_predicate_offers_every_tool():
    llm = _RecordingLLM(response_generator=lambda _m: ChatMessage(role="assistant", content="done"))

    def search() -> str:
        """Search."""
        return "results"

    run_agent_workflow(
        llm=llm, system_prompt="s", user_prompt="u", agent_name="t",
        emit_stream=lambda _t: None, tools=[FunctionTool.from_defaults(fn=search)],
    )

    assert llm.offered == [["search"]]


# --- the manager says when a tool can only refuse ---------------------------


@BOTH
def test_every_tool_is_available_before_anything_has_run(module, monkeypatch, tmp_path):
    _, manager = _manager(module, monkeypatch, tmp_path)

    assert all(
        manager.is_tool_available(name)
        for name in ("search_tables", "inspect_columns", "expand_candidates")
    )


@BOTH
def test_search_is_withdrawn_once_a_search_found_candidates(module, monkeypatch, tmp_path):
    _, manager = _manager(module, monkeypatch, tmp_path)

    manager.search_keyword_concepts(["alpha"])

    assert not manager.is_tool_available("search_tables")
    # ...exactly when the tool itself would refuse.
    assert manager.search_keyword_concepts(["beta"]).startswith("Search limit reached")


@BOTH
def test_search_stays_available_for_zero_result_retries_then_is_withdrawn(
    module, monkeypatch, tmp_path
):
    _, manager = _manager(module, monkeypatch, tmp_path, found=False)

    for word in ("alpha", "beta"):
        manager.search_keyword_concepts([word])
        assert manager.is_tool_available("search_tables")
    manager.search_keyword_concepts(["gamma"])  # 1 attempt + 2 zero-result retries

    assert not manager.is_tool_available("search_tables")


@BOTH
def test_a_search_rejected_for_a_banned_subset_spends_nothing(module, monkeypatch, tmp_path):
    _, manager = _manager(module, monkeypatch, tmp_path, found=False)

    manager.search_keyword_concepts(["alpha"])
    assert manager.search_keyword_concepts(["alpha", "beta"]).startswith("Search rejected")

    assert manager.is_tool_available("search_tables")


@BOTH
def test_search_is_withdrawn_after_an_inspection(module, monkeypatch, tmp_path):
    monkeypatch.setattr(inspection_tools, "_inspect_columns", lambda *_a: "Schema")
    state, manager = _manager(module, monkeypatch, tmp_path)
    _candidates(state, total=1, visible=1)
    state.all_candidates = ["table.parquet"]

    manager.inspect_columns(candidate_number=1)

    assert not manager.is_tool_available("search_tables")
    assert manager.search_keyword_concepts(["alpha"]).startswith("Search refinement blocked")


@BOTH
def test_search_is_withdrawn_once_retrieval_has_failed_for_the_request(
    module, monkeypatch, tmp_path
):
    state, manager = _manager(module, monkeypatch, tmp_path, mode=RetrievalMode.SEMANTIC)
    state.semantic_failure = "embedding service down"
    assert not manager.is_tool_available("search_tables")

    # The failure only ends retrieval that embeds; a keyword search is unaffected.
    state, manager = _manager(module, monkeypatch, tmp_path)
    state.semantic_failure = "embedding service down"
    assert manager.is_tool_available("search_tables")


@BOTH
def test_expansion_is_withdrawn_at_its_limit_or_when_none_is_allowed(
    module, monkeypatch, tmp_path
):
    state, manager = _manager(module, monkeypatch, tmp_path)
    assert manager.is_tool_available("expand_candidates")
    state.expansion_count = 1
    assert not manager.is_tool_available("expand_candidates")

    _, disabled = _manager(module, monkeypatch, tmp_path, max_expansions=0)
    assert not disabled.is_tool_available("expand_candidates")


@BOTH
def test_inspection_stays_available_at_the_shortlist_while_an_expansion_can_raise_it(
    module, monkeypatch, tmp_path
):
    state, manager = _manager(module, monkeypatch, tmp_path)
    _candidates(state, total=10, visible=5)
    state.inspection_counts = {f"t{i}.parquet": 1 for i in range(3)}

    assert manager.is_tool_available("inspect_columns")


@BOTH
@pytest.mark.parametrize(
    ("discovery", "total", "visible"),
    [
        ({"max_expansions": 0}, 10, 5),   # expansion is switched off
        ({}, 5, 5),                       # nothing is left to reveal
    ],
)
def test_inspection_is_withdrawn_at_the_shortlist_when_no_expansion_can_raise_it(
    module, discovery, total, visible, monkeypatch, tmp_path
):
    state, manager = _manager(module, monkeypatch, tmp_path, **discovery)
    _candidates(state, total=total, visible=visible)
    state.inspection_counts = {f"t{i}.parquet": 1 for i in range(3)}

    assert not manager.is_tool_available("inspect_columns")


@BOTH
def test_inspection_is_withdrawn_at_the_post_expansion_limit(module, monkeypatch, tmp_path):
    state, manager = _manager(module, monkeypatch, tmp_path)
    _candidates(state, total=10, visible=10)
    state.expansion_count = 1

    state.inspection_counts = {f"t{i}.parquet": 1 for i in range(5)}
    assert manager.is_tool_available("inspect_columns")
    state.inspection_counts = {f"t{i}.parquet": 1 for i in range(6)}
    assert not manager.is_tool_available("inspect_columns")


@BOTH
def test_inspection_stays_available_for_a_carried_schema_the_agent_has_not_seen(
    module, monkeypatch, tmp_path
):
    state, manager = _manager(module, monkeypatch, tmp_path, max_expansions=0)
    _candidates(state, total=5, visible=5)
    state.inspection_counts = {f"t{i}.parquet": 1 for i in range(3)}
    state.inspection_cache = {"t4.parquet": "Schema carried over from the last round"}

    assert manager.is_tool_available("inspect_columns")
    manager.inspect_columns(candidate_number=5)  # now it has been served
    assert not manager.is_tool_available("inspect_columns")


@BOTH
def test_selection_tools_are_never_withdrawn(module, monkeypatch, tmp_path):
    state, manager = _manager(module, monkeypatch, tmp_path, max_expansions=0)
    _candidates(state, total=3, visible=3)
    state.expansion_count = 1
    state.inspection_counts = {f"t{i}.parquet": 1 for i in range(9)}
    manager.search_keyword_concepts(["alpha"])

    assert not any(
        manager.is_tool_available(name)
        for name in ("search_tables", "inspect_columns", "expand_candidates")
    )
    assert all(
        manager.is_tool_available(name)
        for name in ("confirm_unified_selection", "reject_unified_selection", "check_join_union")
    )


# --- candidate previews never spell a missing value "None" ------------------


def test_missing_column_metadata_is_left_empty_not_spelled_none():
    document = {
        "resource_id": "r", "title": "T",
        "columns": [
            {"name": "A", "type": "int64", "description": None},
            {"name": "B", "type": None, "description": "Real text"},
            {"name": None, "type": "x", "description": None},
        ],
    }

    metadata = solr_metadata_from_doc(document)
    preview = format_candidate_context(["f.parquet"], {"f.parquet": metadata})

    assert [c["name"] for c in metadata["columns"]] == ["A", "B"]
    assert "None" not in preview
    assert "    - A [int64]\n" in preview
    assert "    - B — Real text" in preview


def test_legacy_flat_column_metadata_is_left_empty_not_spelled_none():
    meta = {"columns.name": ["A", None], "columns.description": [None], "columns.type": [None]}

    assert _candidate_columns(meta) == [{"name": "A", "type": "", "description": ""}]


# --- a repeated inspection points back instead of repeating the schema ------


@BOTH
def test_reinspection_points_back_instead_of_repeating_the_schema(
    module, monkeypatch, tmp_path
):
    schema = "Schema for table.parquet:\n" + "\n".join(f"- column_{i} (int64)" for i in range(60))
    monkeypatch.setattr(inspection_tools, "_inspect_columns", lambda *_a: schema)
    state, manager = _manager(module, monkeypatch, tmp_path)
    state.all_candidates = ["other.parquet", "table.parquet"]
    state.visible_candidate_count = 2

    first = manager.inspect_columns(candidate_number=2)
    second = manager.inspect_columns(candidate_number=2)
    third = manager.inspect_columns(candidate_number=2)

    assert first == schema
    assert second.startswith("Cached inspection (attempt 2/2)")
    assert "Candidate 2" in second and "column_0" not in second
    assert len(second) < len(first) / 5
    assert third.startswith("Inspection skipped")


@BOTH
def test_a_carried_schema_is_served_in_full_the_first_time(module, monkeypatch, tmp_path):
    state, manager = _manager(module, monkeypatch, tmp_path)
    state.all_candidates = ["table.parquet"]
    state.visible_candidate_count = 1
    state.inspection_cache["table.parquet"] = "Schema carried over from the last round"

    first = manager.inspect_columns(candidate_number=1)
    second = manager.inspect_columns(candidate_number=1)

    assert first == "Cached inspection (attempt 1/2):\nSchema carried over from the last round"
    assert "carried over" not in second and "Candidate 1" in second


# --- the real manager's real tool list, on the real agent loop --------------


def _drive(manager, calls):
    """Run the agent loop with a scripted model; return what each turn was offered."""
    script = iter([*calls, None])
    ids = itertools.count()

    def respond(_messages):
        step = next(script)
        if step is None:
            return ChatMessage(role="assistant", content="done")
        name, kwargs = step
        return ChatMessage(role="assistant", blocks=[
            ToolCallBlock(tool_name=name, tool_kwargs=kwargs, tool_call_id=f"call-{next(ids)}")
        ])

    llm = _RecordingLLM(response_generator=respond)
    streamed = []
    run_agent_workflow(
        llm=llm, system_prompt="s", user_prompt="u", agent_name="t",
        emit_stream=streamed.append, tools=manager.get_tools(),
        tool_available=manager.is_tool_available,
        max_iterations=12, max_repeats=5, max_tool_calls=12,
    )
    return llm.offered, "\n".join(streamed)


def _search(*words):
    return ("search_tables", {"concepts": list(words)})


_REJECT = ("reject_unified_selection", {"reasoning": "No table matches.", "suggestion": "Try other words."})


@BOTH
def test_search_leaves_the_tool_list_the_moment_its_limit_is_reached(module, monkeypatch, tmp_path):
    _, manager = _manager(module, monkeypatch, tmp_path, found=False)

    # 1 attempt + 2 zero-result retries = 3 searches; the model then tries a 4th.
    offered, streamed = _drive(manager, [_search("alpha"), _search("beta"), _search("gamma"),
                                         _search("delta"), _REJECT])

    assert ["search_tables" in tools for tools in offered] == [True, True, True, False, False]
    assert set(offered[3]) == set(offered[0]) - {"search_tables"}
    assert "reject_unified_selection" in offered[3]
    # The stray 4th call was answered by the tool's own refusal, not "not found".
    assert "Search limit reached" in streamed and "not found" not in streamed


@BOTH
@pytest.mark.parametrize("mode", [RetrievalMode.KEYWORD, RetrievalMode.SEMANTIC])
def test_search_leaves_the_tool_list_right_after_the_one_search_that_found_candidates(
    module, mode, monkeypatch, tmp_path
):
    _, manager = _manager(module, monkeypatch, tmp_path, found=True, mode=mode)

    offered, _ = _drive(manager, [_search("alpha"), _REJECT])

    assert ["search_tables" in tools for tools in offered] == [True, False]


@BOTH
def test_a_search_rejected_as_banned_does_not_remove_the_tool(module, monkeypatch, tmp_path):
    _, manager = _manager(module, monkeypatch, tmp_path, found=False)

    offered, streamed = _drive(manager, [_search("alpha"), _search("alpha", "beta"), _REJECT])

    assert "Search rejected before retrieval" in streamed
    assert ["search_tables" in tools for tools in offered] == [True, True, True]
