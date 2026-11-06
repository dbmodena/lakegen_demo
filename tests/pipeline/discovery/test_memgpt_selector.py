"""The MemGPT table selector: its loop and gates with a scripted model, its selection, its place in phase 2."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.llm import ToolSelection

from lakegen.config.experiment import DiscoveryConfig, ExperimentConfig, TableSelector
from lakegen.datastructures.errors import WorkflowCancelled
from lakegen.datastructures.outcomes import SelectorKind
from lakegen.llm.prompts import get_prompt_manager
from lakegen.pipeline.discovery import divided as phase2
from lakegen.pipeline.discovery.selectors import (
    SelectionRequest,
    SelectorUnavailable,
    build,
    choose,
)
from lakegen.pipeline.discovery.selectors.memgpt import (
    TRACE_HEADER,
    MemGPTSelector,
    alternatives,
    contrast,
    drop,
)
from lakegen.pipeline.discovery.selectors.memgpt import agent as agent_module
from lakegen.pipeline.discovery.selectors.memgpt.agent import MemGPTAgent, recover_call
from lakegen.pipeline.discovery.selectors.memgpt.items import (
    Item,
    fold,
    is_period_label,
    named_items,
    period_evidence,
    question_periods,
    tightest_span,
    tokens_near,
)
from lakegen.pipeline.discovery.selectors.memgpt.memory import Memory, Requirement
from lakegen.pipeline.wiring.adapters import divided_agentic_outcome
from lakegen.retrieval import RetrievalConfig

QUESTION = "What is the average number of tickets sold at San Siro in June and August 2026?"
REQS = ["measure: tickets sold San Siro June 2026", "measure: tickets sold San Siro August 2026"]
JUNE, AUGUST = "san_siro_tickets_june_2026.parquet", "san_siro_tickets_august_2026.parquet"
MEMGPT = DiscoveryConfig(selector="memgpt")


class ScriptedLLM:
    """Returns the scripted tool calls one per turn and records what each turn offered and showed."""

    def __init__(self, script, tokens_per_call=0):
        self.script = list(script)
        self.offered: list[list[str]] = []
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.tokens_per_call = tokens_per_call
        self.token_usage_total = 0

    def reset_token_usage(self):
        self.token_usage_total = 0

    def chat_with_tools(self, tools, chat_history=None, **kwargs):
        assert "tool_required" not in kwargs
        self.offered.append([t.metadata.name for t in tools])
        self.systems.append(chat_history[0].content)
        self.prompts.append(chat_history[-1].content)
        self.token_usage_total += self.tokens_per_call
        name, args = (
            self.script.pop(0) if self.script else ("confirm", {"reasoning": "out of script"})
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="", additional_kwargs={"call": (name, args)}
            )
        )

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True):
        name, args = response.message.additional_kwargs["call"]
        return (
            [] if name is None else [ToolSelection(tool_id="x", tool_name=name, tool_kwargs=args)]
        )


class ChoosyLLM(ScriptedLLM):
    """A client that declares tool choice: records every request's choice; a scripted ("prose", text) item answers with text only."""

    tool_choice_supported = True

    def __init__(self, script):
        super().__init__(script)
        self.choices: list = []

    def chat_with_tools(self, tools, chat_history=None, **kwargs):
        assert (
            "tool_required" not in kwargs
        )  # the client's flag reaches the SDK as auto; the choice itself is passed
        self.choices.append(kwargs.get("tool_choice") or "auto")
        self.offered.append([t.metadata.name for t in tools])
        self.systems.append(chat_history[0].content)
        self.prompts.append(chat_history[-1].content)
        name, args = (
            self.script.pop(0) if self.script else ("confirm", {"reasoning": "out of script"})
        )
        if name == "prose":
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=args,
                    additional_kwargs={"call": (None, None)},
                )
            )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="", additional_kwargs={"call": (name, args)}
            )
        )


@pytest.fixture()
def lake(tmp_path):
    """Two San Siro partitions (June at #1, August at #5) among 16 unrelated tables."""
    frames = {
        JUNE: pd.DataFrame(
            {"match_date": [f"2026-06-{d:02d}" for d in range(1, 21)], "tickets_sold": range(20)}
        ),
        AUGUST: pd.DataFrame(
            {"match_date": [f"2026-08-{d:02d}" for d in range(1, 21)], "tickets_sold": range(20)}
        ),
    }
    for i in range(16):
        frames[f"other_{i:02d}.parquet"] = pd.DataFrame({"a": range(5), "b": [f"x{i}"] * 5})
    for name, frame in frames.items():
        frame.to_parquet(tmp_path / name, index=False)
    order = [
        JUNE,
        *[f"other_{i:02d}.parquet" for i in range(3)],
        AUGUST,
        *[f"other_{i:02d}.parquet" for i in range(3, 16)],
    ]
    titles = {JUNE: "San Siro tickets sold June 2026", AUGUST: "San Siro tickets sold August 2026"}
    metadata = {
        name: {
            "title": titles.get(name, f"Unrelated dataset {name[6:8]}"),
            "description": "",
            "columns.name": list(frames[name].columns),
        }
        for name in order
    }
    return SimpleNamespace(dir=tmp_path, candidates=order, metadata=metadata)


def _agent(lake, llm, **kwargs) -> MemGPTAgent:
    return MemGPTAgent(
        question=QUESTION,
        candidates=lake.candidates,
        metadata=lake.metadata,
        table_dir=lake.dir,
        llm=llm,
        prompts=get_prompt_manager(),
        portal="Test portal",
        **kwargs,
    )


def _request(lake, llm, **kwargs) -> SelectionRequest:
    return SelectionRequest(
        QUESTION,
        list(lake.candidates),
        lake.metadata,
        lake.dir,
        MEMGPT,
        llm=llm,
        portal_name="Test portal",
        **kwargs,
    )


# ---------------------------------------------------------------- the loop and the gates


def test_the_queue_gate_can_be_switched_off(lake):
    class Open(MemGPTAgent):
        queue_gate = False

    for cls, answer in ((MemGPTAgent, "Confirm refused: the catalogue of #5"), (Open, "CONFIRMED")):
        agent = cls(
            question="How many tickets were sold at San Siro?",
            candidates=lake.candidates,
            metadata=lake.metadata,
            table_dir=lake.dir,
            llm=None,
            prompts=get_prompt_manager(),
            portal="Test portal",
        )
        agent.define_requirements(["measure: tickets sold San Siro"])
        agent.inspect([1])
        agent.select(
            1, ["R1"]
        )  # nothing open; #5 (August) matches as well as #1 and is queued as an alternative
        assert agent.memory.queue()[0][0] == 5 and agent.confirm("done").startswith(answer)


def test_first_turn_offers_only_define_requirements(lake):
    llm = ScriptedLLM(
        [
            ("define_requirements", {"requirements": REQS}),
            ("reject", {"reasoning": "x", "suggestion": "y"}),
        ]
    )
    agent = _agent(lake, llm)
    agent.run()
    assert llm.offered[0] == ["define_requirements"]
    assert [r.id for r in agent.memory.requirements] == ["R1", "R2"]
    assert "REQUIREMENTS (read-only" in agent.system_message()
    # the periods of the question became the requirements' items, so no requirement was added
    assert agent.items_added == [] and all(r.item is not None for r in agent.memory.requirements)


def test_confirm_is_refused_until_august_is_covered(lake):
    script = [
        ("define_requirements", {"requirements": REQS}),
        ("inspect", {"candidates": [1]}),
        ("select", {"table": 1, "covers": ["R1"]}),
        ("confirm", {"reasoning": "June found"}),  # refused: R2 open, #5 queued
        ("inspect", {"candidates": [1]}),  # repeat: answered from STATE, no action
        ("inspect", {"candidates": [5]}),
        ("select", {"table": 5, "covers": ["R2"]}),
        ("confirm", {"reasoning": "June and August", "tables": ["ignored"]}),
    ]
    llm = ScriptedLLM(script)
    agent = _agent(lake, llm)
    agent.run()
    assert agent.outcome == "confirmed" and sorted(agent.memory.selection) == [1, 5]
    assert agent.refusals["confirm"] == 1 and agent.actions == 2
    assert "R2 still open" in llm.prompts[4] and "#5 (R2" in llm.prompts[4]
    plan, _ = agent.plan()
    assert plan["combination_strategy"] == "concat_partitions"
    assert plan["requirement_coverage"]["tickets sold San Siro June 2026"]["table"] == JUNE


def test_reject_is_refused_while_a_queued_table_matches_an_open_requirement(lake):
    llm = ScriptedLLM(
        [
            ("define_requirements", {"requirements": REQS}),
            ("inspect", {"candidates": [2, 3]}),
            ("reject", {"reasoning": "nothing", "suggestion": "more"}),
        ]
    )
    agent = _agent(lake, llm)
    agent.run()
    assert agent.refusals["reject"] >= 1 and agent.outcome != "rejected"


def test_third_identical_call_enters_decision_mode_and_fallback_keeps_the_selection(lake):
    script = [
        ("define_requirements", {"requirements": REQS}),
        ("inspect", {"candidates": [1]}),
        ("select", {"table": 1, "covers": ["R1"]}),
        ("recall", {"table": 1}),
        ("recall", {"table": 1}),
        ("recall", {"table": 1}),
        (None, {}),
        (None, {}),
    ]
    llm = ScriptedLLM(script)
    agent = _agent(lake, llm)
    agent.run()
    assert agent.decision_mode and llm.offered[6] == ["confirm", "reject"]
    assert (
        agent.outcome == "confirmed"
    )  # the scripted model confirms once only terminal tools are offered
    assert agent.fallback() == [JUNE] and agent.no_call_turns == 2


def test_history_is_a_fifo_with_a_summary(lake):
    script = [
        ("define_requirements", {"requirements": REQS}),
        ("inspect", {"candidates": [1]}),
        ("inspect", {"candidates": [2, 3]}),
        ("inspect", {"candidates": [4]}),
        ("inspect", {"candidates": [5]}),
        ("select", {"table": 1, "covers": ["R1"]}),
        ("select", {"table": 5, "covers": ["R2"]}),
        ("confirm", {"reasoning": "ok"}),
    ]
    llm = ScriptedLLM(script)
    agent = _agent(lake, llm)
    agent.run()
    last = llm.prompts[-1]
    assert "SUMMARY OF EARLIER TURNS" in last and "turn 2: inspected [1]" in last
    assert last.count("[turn ") == 3  # only the three most recent exchanges verbatim


def test_a_call_written_as_text_is_recovered():
    offered = ["select", "unselect", "waive", "recall", "confirm", "reject"]
    assert recover_call('{"reasoning": "All requirements are covered"}', offered) == (
        "confirm",
        {"reasoning": "All requirements are covered"},
    )
    assert (
        recover_call('text {"reasoning": "none", "suggestion": "other data"}', offered)[0]
        == "reject"
    )
    assert recover_call('{"table": 3, "covers": ["R1"]}', offered)[0] == "select"
    assert recover_call('{"candidates": [1, 2]}', offered) is None  # inspect not offered
    assert recover_call("no json here", offered) is None
    # the tool name wrapping the arguments, with or without the functions. prefix
    assert recover_call('{"confirm": {"reasoning": "Dataset #1 holds every column"}}', offered) == (
        "confirm",
        {"reasoning": "Dataset #1 holds every column"},
    )
    assert recover_call('{"functions.select": {"table": 2, "covers": ["R1"]}}', offered) == (
        "select",
        {"table": 2, "covers": ["R1"]},
    )
    assert recover_call('{"inspect": {"candidates": [3]}}', offered) is None
    # a Python-style call, nested quotes and brackets included
    text = 'I will select it now: select(table=1, covers=["R1", "R3"], note="R1 (subject), R3 by \'location\'")'
    assert recover_call(text, offered) == (
        "select",
        {"table": 1, "covers": ["R1", "R3"], "note": "R1 (subject), R3 by 'location'"},
    )
    assert recover_call('confirm(reasoning="done")', offered) == ("confirm", {"reasoning": "done"})
    assert recover_call("inspect (see the tables above) then select", offered) is None
    assert (
        recover_call("select(1, ['R1'])", offered) is None
    )  # positional arguments are not guessed


def test_alternatives_flag_inspected_tables_showing_the_same_requirements(lake):
    llm = ScriptedLLM(
        [
            ("define_requirements", {"requirements": REQS}),
            ("inspect", {"candidates": [1, 5]}),
            ("select", {"table": 5, "covers": ["R2"]}),
            ("reject", {"reasoning": "x", "suggestion": "y"}),
        ]
    )
    agent = _agent(lake, llm)
    agent.run()
    m = agent.memory
    assert m.cards[1].shows and m.cards[5].shows
    assert any(s == 5 for _, s, _ in m.alternatives()) == ("R2" in m.cards[1].shows)


def test_a_turn_without_a_tool_call_is_asked_again_with_one_required(lake):
    llm = ChoosyLLM(
        [
            ("define_requirements", {"requirements": REQS}),
            ("prose", "I'm ready to inspect the first table, it looks like the right one."),
            ("inspect", {"candidates": [1]}),
            ("prose", "**Answer:** 2,206 tickets were sold."),
            ("confirm", {"reasoning": "done"}),
        ]
    )
    agent = _agent(lake, llm)
    agent.run()
    assert llm.choices[:4] == ["auto", "auto", "required", "auto"]
    assert llm.prompts[1] == llm.prompts[2]  # the required retry repeats the same turn
    assert agent.constrained_retries == 2 and agent.constrained_calls == 2
    assert agent.no_call_turns == 0 and agent.recovered_calls == 0
    assert agent.calls["inspect"] == 1
    assert "asked again with a tool call required" in agent.trace.getvalue()


def test_tool_choice_is_not_sent_to_a_client_without_it_nor_when_switched_off(lake):
    llm = ScriptedLLM(
        [
            ("define_requirements", {"requirements": REQS}),
            (None, None),
            ("confirm", {"reasoning": "x"}),
        ]
    )
    agent = _agent(lake, llm)
    agent.run()
    assert agent.no_call_turns == 1 and agent.constrained_retries == 0
    assert "You answered without a tool call" in llm.prompts[2]

    class Off(MemGPTAgent):
        tool_choice = False

    llm = ChoosyLLM(
        [
            ("define_requirements", {"requirements": REQS}),
            ("prose", "text"),
            ("confirm", {"reasoning": "x"}),
        ]
    )
    agent = Off(
        question="Q",
        candidates=list(lake.candidates),
        metadata=lake.metadata,
        table_dir=lake.dir,
        llm=llm,
        prompts=agent.prompts,
        portal="Test portal",
    )
    agent.run()
    assert (
        set(llm.choices) == {"auto"} and agent.no_call_turns == 1 and agent.constrained_retries == 0
    )


def test_a_provider_error_costs_a_turn_and_three_in_a_row_end_the_run(lake):
    class Failing(ScriptedLLM):
        def chat_with_tools(self, tools, chat_history=None, **kwargs):
            super().chat_with_tools(tools, chat_history, **kwargs)
            raise RuntimeError("HTTP 503")

    llm = Failing([])
    agent = _agent(lake, llm)
    agent.run()
    assert agent.outcome.startswith("error: HTTP 503") and agent.turn == 3
    assert "failed on the provider's side" in llm.prompts[1] and agent.no_call_turns == 0


# ---------------------------------------------------------------- items: periods and named items


def test_the_read_only_block_names_the_portal_the_keywords_and_the_item_rules(lake):
    text = _agent(lake, ScriptedLLM([]), keywords=["tickets", "San Siro"]).system_message()
    assert (
        "Active Dataset Portal: 'Test portal'." in text
        and "Retrieval concepts: tickets San Siro" in text
    )
    assert "named_items" in text and "partitions to add" in text


def test_periods_become_their_own_requirements_and_need_evidence(lake):
    script = [
        (
            "define_requirements",
            {"requirements": ["measure: tickets sold at San Siro"], "named_items": []},
        ),
        ("inspect", {"candidates": [1]}),
        ("select", {"table": 1, "covers": ["R1", "R2", "R3"]}),  # claims August too: refused for R3
        ("confirm", {"reasoning": "done"}),  # refused: R3 open, #5 queued
        ("inspect", {"candidates": [5]}),
        ("select", {"table": 5, "covers": ["R3"]}),
        ("confirm", {"reasoning": "June and August"}),
    ]
    llm = ScriptedLLM(script)
    agent = _agent(lake, llm)
    agent.run()
    m = agent.memory
    assert [r.label for r in m.requirements][1:] == [
        "R2 [period] june 2026",
        "R3 [period] august 2026",
    ]
    assert agent.items_added == ["R2 [period] june 2026", "R3 [period] august 2026"]
    assert m.selection == {1: ["R1", "R2"], 5: ["R3"]} and agent.cover_refusals == 1
    assert "Not accepted: #1 does not show R3 (august 2026) -- shown by #5" in llm.prompts[3]
    assert agent.outcome == "confirmed" and agent.refusals["confirm"] == 1


def test_item_requirements_cannot_be_waived_and_partitions_are_listed(lake):
    llm = ScriptedLLM(
        [
            ("define_requirements", {"requirements": ["measure: tickets sold at San Siro"]}),
            ("inspect", {"candidates": [1, 5]}),
            ("waive", {"requirement": "R3", "reason": "August is not really needed here"}),
            ("select", {"table": 1, "covers": ["R1", "R2"]}),
            ("reject", {"reasoning": "x", "suggestion": "y"}),
        ]
    )
    agent = _agent(lake, llm)
    agent.run()
    assert "cannot be waived" in llm.prompts[3]
    assert (
        "Partitions to add (inspected, not selected, showing an OPEN requirement): #5 shows R3"
        in llm.prompts[4]
    )
    assert agent.refusals["reject"] >= 1


def test_named_items_are_checked_against_the_question_and_matched_near_each_other():
    q = "¿Cuál de los sensores, el de Cádiz 16, el de cuba 3 y el de Cádiz 3, registró más ruido?"
    items = named_items(q, ["Cádiz 16", "cuba 3", "Cádiz 3", "Valencia"])
    assert [i.label for i in items] == [
        "Cádiz 16",
        "cuba 3",
        "Cádiz 3",
    ]  # in question order; Valencia is not in it
    assert named_items(q, ["Cádiz 16"]) == []  # a single item is not an enumeration
    description = fold(
        "Sensor de ruido en la calle Cádiz, 16 (T248671). Ubicación: 39.4631424, -0.3746647"
    )
    assert tokens_near(description, items[0].tokens) and not tokens_near(
        description, items[2].tokens
    )
    assert [i.label for i in question_periods("entre 2018‑2020, et en 2013 ?")] == [
        "2018 to 2020",
        "2013",
    ]
    assert question_periods("Buildings in the 2020s in the 2025 report") == []


def test_named_items_must_form_a_list_in_the_question():
    olympics = (
        "Quelle est la longueur moyenne des parcours prioritaire des Quartiers d'Accessibilité Augmentée (QAA) "
        "pour chaque secteur administratif de Paris pendant les Jeux Olympiques 2024 ?"
    )
    assert (
        named_items(
            olympics,
            [
                "Quartiers d'Accessibilité Augmentée (QAA)",
                "secteur administratif de Paris",
                "Jeux Olympiques 2024",
            ],
        )
        == []
    )
    sensors = "el etiquetado t248655, el sensor de la calle doctor Serrano, el sensor de la calle cuba, y el sensor de la calle Cádiz 16"
    assert [
        i.label for i in named_items(sensors, ["t248655", "doctor Serrano", "cuba", "Cádiz 16"])
    ] == ["t248655", "doctor Serrano", "cuba", "Cádiz 16"]
    assert (
        named_items(
            "for september 2019, june 2019 and january 2020", ["September 2019", "January 2020"]
        )
        == []
    )  # periods are the system's


def test_named_items_survive_month_streets_shared_head_words_and_any_list_shape():
    q = (
        "El 4 de septiembre de 2026, ¿qué sensor entre los de Cádiz 16, Cádiz 3, carles Cervera, salvador Abril y cuba "
        "registró el nivel de ruido total más alto?"
    )
    listed = ["Cádiz 16", "Cádiz 3", "Carles Cervera", "Salvador Abril", "Cuba", "4 September 2026"]
    five = ["Cádiz 16", "Cádiz 3", "Carles Cervera", "Salvador Abril", "Cuba"]
    assert [
        i.label for i in named_items(q, listed)
    ] == five  # "Cádiz 3" at its own place, "Abril" a street, the date a period
    assert [
        i.label for i in named_items(q, {"sensors": five, "date": "4 September 2026"})
    ] == five  # a dict of lists
    assert [i.label for i in named_items(q, "Cádiz 16; Cádiz 3, Carles Cervera")] == five[
        :3
    ]  # one string
    assert [i.label for i in named_items(q, [five[:2], ["Carles Cervera"]])] == five[
        :3
    ]  # nested lists
    assert (
        not is_period_label("Salvador Abril")
        and is_period_label("abril 2024")
        and is_period_label("15 de abril")
    )
    folded = fold(q)
    assert folded[slice(*tightest_span(folded, ("cadiz", "3")))] == "cadiz 3"
    assert tightest_span(folded, ("sueca",)) is None


def test_a_preposition_alone_does_not_make_a_list():
    assert (
        named_items(
            "percorsi pedonali costruiti ogni anno a Bologna e in quale anno", ["Bologna", "anno"]
        )
        == []
    )
    assert (
        named_items(
            "en el presupuesto municipal de la Ajuntament de València de 2024",
            ["presupuesto municipal", "Ajuntament de València"],
        )
        == []
    )
    assert (
        len(
            named_items(
                "le liste Terzo Polo Centro, Bologna FUTURA e Sinistra per Bologna",
                ["Terzo Polo Centro", "Bologna FUTURA", "Sinistra per Bologna"],
            )
        )
        == 3
    )


def test_month_with_a_two_digit_year_is_evidence():
    june = Item("june 2019", "period", (2019,), 6)
    assert (
        period_evidence(june, "Animal Welfare and Hygiene Compliance Sections June 19", "")
        == "title/description"
    )
    assert (
        period_evidence(june, "Animal Welfare and Hygiene Compliance Sections September 19", "")
        is None
    )
    assert (
        period_evidence(june, "Enforcement Non-compliance - April 2019 to December 2019", "")
        is None
    )


def test_a_year_in_a_description_is_not_a_period_of_the_table():
    meta = {
        "t.parquet": {
            "title": "Elections 2022",
            "description": "Paris Vote depuis 2017",
            "columns.name": [],
        }
    }
    m = Memory("q", ["t.parquet"], meta, 1)
    r2017 = Requirement("R1", "period", "2017", item=Item("2017", "period", (2017,)))
    r2022 = Requirement("R2", "period", "2022", item=Item("2022", "period", (2022,)))
    assert m.item_evidence(1, r2017) is None and m.item_evidence(1, r2022) == "title/description"


# ---------------------------------------------------------------- the selector

CONFIRMING = [
    ("define_requirements", {"requirements": REQS}),
    ("inspect", {"candidates": [1, 5]}),
    ("select", {"table": 1, "covers": ["R1"]}),
    ("select", {"table": 5, "covers": [2]}),
    ("confirm", {"reasoning": "June and August"}),
]
# Three identical recalls put the agent in decision mode, where reject is accepted.
REJECTING = [
    ("define_requirements", {"requirements": REQS}),
    ("inspect", {"candidates": [1]}),
    ("recall", {"table": 1}),
    ("recall", {"table": 1}),
    ("recall", {"table": 1}),
    ("reject", {"reasoning": "no table holds the prices", "suggestion": "ticket prices"}),
]


def test_memgpt_is_a_registered_selector():
    selector = choose(MEMGPT)
    assert isinstance(selector, MemGPTSelector) and selector.kind is TableSelector.MEMGPT
    assert build("memgpt", MEMGPT).title == "MemGPT table selector"


def test_a_confirmed_selection_hands_on_a_plan_in_the_judges_keys(lake):
    llm = ScriptedLLM(CONFIRMING, tokens_per_call=100)
    streamed = []
    selection = MemGPTSelector(MEMGPT).select(
        _request(lake, llm, keywords=("tickets",), stream_callback=streamed.append)
    )
    assert selection.selected == [JUNE, AUGUST] and selection.reasoning == "June and August"
    assert selection.source == "memgpt_selector" and selection.trace.startswith(TRACE_HEADER)
    assert (
        selection.llm_calls == 8 and selection.tokens == 800
    )  # five turns, no look-alike to contrast, two drop calls, one alternatives call (two drops to read)
    plan = selection.plan
    assert {
        "requirement_coverage",
        "table_roles",
        "combination_strategy",
        "uncovered_requirements",
        "alternatives_rejected",
        "requirement_ledger",
    } <= set(plan)
    assert plan["alternatives"] == {"pool": 2, "offered": []}  # out of script: nothing kept
    assert (
        plan["combination_strategy"] == "concat_partitions" and plan["uncovered_requirements"] == []
    )
    assert plan["table_roles"][JUNE] == "covers [measure] tickets sold San Siro June 2026"
    assert isinstance(plan["requirement_ledger"], list)
    assert "Retrieval concepts: tickets" in llm.systems[0] and "'Test portal'" in llm.systems[0]
    assert (
        streamed[0].startswith("\n**Phase 2 – MemGPT table selector started**")
        and len(streamed) == 8
    )  # the start line, five turns, the drop check, the alternatives check


def test_a_rejection_asks_for_new_keywords_and_carries_the_tables_that_show_requirements(lake):
    selection = MemGPTSelector(MEMGPT).select(_request(lake, ScriptedLLM(REJECTING)))
    assert selection.selected == [] and selection.plan is None
    assert (
        selection.reasoning
        == "REJECT_KEYWORDS: no table holds the prices\nSuggestion: ticket prices"
    )
    assert selection.keep_tables == (JUNE,) and selection.skip_tables == ()


def test_a_run_cut_short_keeps_its_selection(lake):
    script = [
        ("define_requirements", {"requirements": REQS}),
        ("inspect", {"candidates": [1]}),
        ("select", {"table": 1, "covers": ["R1"]}),
    ] + [(None, {})] * agent_module.MAX_TURNS
    selection = MemGPTSelector(MEMGPT).select(_request(lake, ScriptedLLM(script)))
    assert selection.selected == [JUNE] and selection.source == "memgpt_fallback"
    assert "stopped before confirming (max_turns)" in selection.reasoning
    assert (
        selection.plan["requirement_coverage"]["tickets sold San Siro June 2026"]["table"] == JUNE
    )


def test_a_selector_that_cannot_select_says_so(lake, monkeypatch):
    with pytest.raises(SelectorUnavailable, match="needs the run's LLM"):
        MemGPTSelector(MEMGPT).select(_request(lake, None))
    with pytest.raises(SelectorUnavailable, match="without a selection"):
        MemGPTSelector(MEMGPT).select(
            _request(lake, ScriptedLLM([(None, {})] * agent_module.MAX_TURNS))
        )

    def broken(*_a, **_k):
        raise KeyError("memgpt_selector")

    monkeypatch.setattr(MemGPTAgent, "system_message", broken)
    with pytest.raises(SelectorUnavailable, match="failed: KeyError"):
        MemGPTSelector(MEMGPT).select(_request(lake, ScriptedLLM(CONFIRMING)))


def test_a_cancelled_run_stops_at_once(lake):
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) > 2:
            raise WorkflowCancelled("stopped")

    llm = ScriptedLLM(CONFIRMING)
    with pytest.raises(WorkflowCancelled):
        MemGPTSelector(MEMGPT).select(_request(lake, llm, cancel_check=cancel))
    assert len(llm.prompts) == 2


# ---------------------------------------------------------------- phase 2 and the config


def _phase2(
    monkeypatch,
    lake,
    llm,
    state=None,
    *,
    candidates=None,
    discovery=MEMGPT,
    hint="",
    retrieval_config=None,
    stream=None,
):
    pool = lake.candidates if candidates is None else candidates
    monkeypatch.setattr(
        phase2,
        "retrieve_candidates",
        lambda *_a, **_k: (list(pool), {c: lake.metadata[c] for c in pool}),
    )
    return phase2.select_tables(
        query=QUESTION,
        llm=llm,
        pm=get_prompt_manager(),
        all_files=list(lake.candidates),
        keywords=["tickets", "San Siro"],
        solr_client=object(),
        csv_dir=lake.dir,
        hint=hint,
        portal_name="Test portal",
        stream_callback=stream,
        retrieval_config=retrieval_config,
        selection_state=state,
        discovery=discovery,
    )


def _state():
    return SimpleNamespace(
        rejection_keep_tables=None, rejection_skip_tables=None, selection_plan=None
    )


def _outcome(result, state):
    return divided_agentic_outcome((["tickets"], "raw", 1, "phase 1"), result, state)


def test_phase2_uses_the_memgpt_selector_and_records_its_plan(monkeypatch, lake):
    state = _state()
    llm = ScriptedLLM(CONFIRMING)
    result = _phase2(monkeypatch, lake, llm, state)
    selected, candidates, _meta, reasoning, trace, _tokens = result
    assert (
        selected == [JUNE, AUGUST]
        and candidates == lake.candidates
        and reasoning == "June and August"
    )
    assert state.selection_plan_source == "memgpt_selector" and state.confirmed_tables == selected
    assert "requirement_coverage" in state.selection_plan and trace.startswith(TRACE_HEADER)
    outcome = _outcome(result, state)
    assert outcome.selector is SelectorKind.MEMGPT and not outcome.rejected


def test_a_memgpt_rejection_in_phase2_keeps_its_tables_for_the_next_attempt(monkeypatch, lake):
    state = _state()
    outcome = _outcome(_phase2(monkeypatch, lake, ScriptedLLM(REJECTING), state), state)
    assert outcome.rejected and outcome.keep_tables == [JUNE] and outcome.skip_tables == []


def test_no_candidates_reject_the_keywords_before_the_selector_runs(monkeypatch, lake):
    llm = ScriptedLLM([])
    selected, candidates, metadata, reasoning, trace, tokens = _phase2(
        monkeypatch, lake, llm, candidates=[]
    )
    assert (selected, candidates, metadata, tokens) == ([], [], {}, 0) and llm.prompts == []
    assert (
        reasoning
        == "REJECT_KEYWORDS: No tables found with keyword retrieval for [tickets, San Siro]. Try broader or different terms."
    )
    assert trace.startswith(
        "[phase2] keyword retrieval with keywords=['tickets', 'San Siro'] returned 0 results."
    )


def test_phase2_asks_for_new_keywords_when_memgpt_cannot_select(monkeypatch, lake):
    state = _state()
    result = _phase2(monkeypatch, lake, None, state)
    selected, candidates, _meta, reasoning, trace, tokens = result
    assert selected == [] and candidates == lake.candidates and tokens == 0
    assert reasoning.startswith(
        "REJECT_KEYWORDS: No selection could be made from the 18 retrieved candidates"
    )
    assert "needs the run's LLM" in reasoning and trace.startswith("[memgpt selector unavailable]")
    assert _outcome(result, state).rejected and state.rejection_keep_tables == []


def test_the_removed_jev_is_refused(monkeypatch, lake):
    with pytest.raises(ValueError, match="removed on 2026-10-29"):
        _phase2(
            monkeypatch, lake, ScriptedLLM([]), _state(), discovery=DiscoveryConfig(selector="jev")
        )


def test_the_removed_judge_is_refused(monkeypatch, lake):
    with pytest.raises(ValueError, match="replaced on 2026-10-21"):
        _phase2(
            monkeypatch, lake, ScriptedLLM([]), _state(), discovery=DiscoveryConfig(selector="llm")
        )


def test_feedback_and_question_only_retrieval_reach_the_read_only_block(monkeypatch, lake):
    llm = ScriptedLLM(CONFIRMING)
    _phase2(
        monkeypatch,
        lake,
        llm,
        _state(),
        hint="the coder found no 2026 rows in other_00",
        retrieval_config=RetrievalConfig(mode="semantic"),
    )
    assert (
        "Retrieval ranked the candidates by the question alone; these concepts were not searched: tickets San Siro"
        in llm.systems[0]
    )
    assert (
        "Feedback from the previous attempt: the coder found no 2026 rows in other_00"
        in llm.systems[0]
    )
    plain = ScriptedLLM(CONFIRMING)
    _phase2(monkeypatch, lake, plain, _state())
    assert (
        "\nRetrieval concepts: tickets San Siro\n" in plain.systems[0]
        and "Feedback from" not in plain.systems[0]
    )


def test_the_selectors_live_lines_are_printed_once_and_forwarded_once(monkeypatch, lake, capsys):
    streamed = []
    _phase2(monkeypatch, lake, ScriptedLLM(CONFIRMING), _state(), stream=streamed.append)
    printed = capsys.readouterr().out
    turns = [line for line in streamed if line.startswith("- turn ")]
    assert len(turns) == 7 and all(
        printed.count(line) == 1 for line in turns
    )  # five turns, the drop check and the alternatives check


def test_memgpt_is_the_default_selector_of_the_default_divided_pipeline():
    config = ExperimentConfig()
    assert (
        config.discovery_architecture == "divided"
        and config.discovery.selector is TableSelector.MEMGPT
    )
    ExperimentConfig(discovery_architecture="unified")  # the unified agent selects in its own turn
    # the removed selectors still load, for the configurations of the runs made with them
    assert ExperimentConfig(discovery={"selector": "llm"}).discovery.selector is TableSelector.LLM
    assert ExperimentConfig(discovery={"selector": "jev"}).discovery.selector is TableSelector.JEV


# ---------------------------------------------------------------- the contrast check

PREF, VOTES = "elezioni_2009_preferenze_lista_alfa.parquet", "elezioni_2009_voti_alle_liste.parquet"
LOOK_QUESTION = "How many preference votes did the list Alfa get in section 1 in 2009?"
LOOK_REQS = ["measure: preference votes of the list Alfa", "filter: section 1", "period: 2009"]
CHOSE_VOTES = [
    ("define_requirements", {"requirements": LOOK_REQS}),
    ("inspect", {"candidates": [1, 6]}),
    ("select", {"table": 1, "covers": ["R1", "R2", "R3"]}),
    ("confirm", {"reasoning": "votes of the lists"}),
]


@pytest.fixture()
def lookalikes(tmp_path):
    """The preference votes of a list and the votes of the lists: two tables that look alike."""
    frames = {
        PREF: pd.DataFrame(
            {
                "sezione": [1, 1, 2],
                "candidato": ["Rossi", "Bianchi", "Rossi"],
                "preferenze": [10, 5, 7],
            }
        ),
        VOTES: pd.DataFrame(
            {"sezione": [1, 1, 2], "lista": ["Alfa", "Beta", "Alfa"], "voti": [100, 80, 90]}
        ),
    }
    for i in range(8):
        frames[f"other_{i}.parquet"] = pd.DataFrame({"a": range(3), "b": [f"x{i}"] * 3})
    for name, frame in frames.items():
        frame.to_parquet(tmp_path / name, index=False)
    order = [
        VOTES,
        *[f"other_{i}.parquet" for i in range(4)],
        PREF,
        *[f"other_{i}.parquet" for i in range(4, 8)],
    ]
    titles = {
        PREF: "Elezioni amministrative 2009 - Preferenze lista Alfa",
        VOTES: "Elezioni amministrative 2009 - Voti alle liste",
    }
    metadata = {
        n: {
            "title": titles.get(n, f"Unrelated {n[6]}"),
            "description": "",
            "columns.name": list(frames[n].columns),
        }
        for n in order
    }
    return SimpleNamespace(dir=tmp_path, candidates=order, metadata=metadata)


def _look_request(lake, llm, **kwargs) -> SelectionRequest:
    return SelectionRequest(
        LOOK_QUESTION,
        list(lake.candidates),
        lake.metadata,
        lake.dir,
        MEMGPT,
        llm=llm,
        portal_name="Test portal",
        **kwargs,
    )


def test_the_contrast_check_swaps_to_the_look_alike_the_model_prefers(lookalikes):
    alts = contrast.alternatives(lookalikes.metadata, [VOTES], lookalikes.candidates)
    label = next(a.label for a in alts if a.tables == [PREF])
    llm = ScriptedLLM(
        [*CHOSE_VOTES, ("choose", {"alternative": label, "reasoning": "the preferenze column"})]
    )
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert selection.selected == [PREF] and selection.llm_calls == 6  # one alternatives call
    contrast_system, contrast_prompt = (
        llm.systems[-2],
        llm.prompts[-2],
    )  # the last call is the alternatives check
    assert (
        "Keep A0 unless another alternative clearly holds" in contrast_system
        and "T6 replaced" not in contrast_prompt
    )
    assert "T1 replaced by T6" in contrast_prompt and "preferenze" in contrast_prompt
    assert selection.plan["contrast"] == {
        "chosen": label,
        "alternatives": len(alts),
        "swaps": [[VOTES, PREF]],
    }
    assert all(c["table"] == PREF for c in selection.plan["requirement_coverage"].values())
    assert "[contrast] swapped #1 for #6 (1 call): the preferenze column" in selection.trace


def test_the_contrast_check_keeps_the_selection_without_an_answer(lookalikes):
    llm = ScriptedLLM([*CHOSE_VOTES, (None, {}), (None, {})])
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert selection.selected == [VOTES] and selection.llm_calls == 7
    assert (
        "[contrast] kept the selection over" in selection.trace
        and selection.plan["contrast"]["chosen"] == "A0"
    )


# ---------------------------------------------------------------- the drop check

CHOSE_BOTH = [
    ("define_requirements", {"requirements": LOOK_REQS}),
    ("inspect", {"candidates": [1, 6]}),
    ("select", {"table": 6, "covers": ["R1", "R2", "R3"]}),
    ("select", {"table": 1, "covers": ["R2"]}),
    ("confirm", {"reasoning": "both"}),
]
KEPT_A0 = ("choose", {"alternative": "A0", "reasoning": "fine"})


def test_the_drop_check_leaves_out_the_table_both_calls_name(lookalikes):
    # three options: the pair, then without each table in rank-free order; S2 under the first order and S0 under
    # the reversed one are the same option, the drop of the table lettered second
    llm = ScriptedLLM(
        [
            *CHOSE_BOTH,
            ("choose", {"alternative": "S2", "reasoning": "B holds nothing"}),
            ("choose", {"alternative": "S0", "reasoning": "B is idle"}),
        ]
    )  # both look-alikes selected: no contrast call
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert selection.llm_calls == 8 and len(selection.selected) == 1
    dropped = selection.plan["drop"]["dropped"]
    assert selection.plan["drop"] == {"options": 3, "dropped": dropped} and selection.selected == [
        t for t in (PREF, VOTES) if t != dropped
    ]
    assert selection.plan["alternatives_rejected"][dropped].startswith(
        "left out by the drop check: B holds nothing"
    )
    first, second = llm.prompts[-3], llm.prompts[-2]  # the last call is the alternatives check
    assert "whole set, and the set with one table left out" in llm.systems[-2] and first != second
    assert (
        "S0: Table A, Table B\nS1: Table B -- Table A left out\nS2: Table A -- Table B left out"
        in first
    )
    assert (
        "S0: Table A -- Table B left out\nS1: Table B -- Table A left out\nS2: Table A, Table B"
        in second
    )
    assert 'Table A "' in first and "T1 " not in first and "T6 " not in first
    assert (
        "[drop] left out #" in selection.trace
        and "(2 calls): B holds nothing | B is idle" in selection.trace
    )


def test_the_drop_check_keeps_the_tables_unless_both_calls_agree(lookalikes):
    llm = ScriptedLLM(
        [
            *CHOSE_BOTH,
            ("choose", {"alternative": "S1", "reasoning": "x"}),
            ("choose", {"alternative": "S2", "reasoning": "y"}),
        ]
    )
    selection = MemGPTSelector(MEMGPT).select(
        _look_request(lookalikes, llm)
    )  # S1 then S2: a drop, then the whole set under the reversed order
    assert (
        sorted(selection.selected) == sorted([PREF, VOTES])
        and selection.plan["drop"]["dropped"] is None
    )
    assert "[drop] kept the 2 tables (the two calls disagreed) (2 calls)" in selection.trace
    llm = ScriptedLLM(
        [
            *CHOSE_BOTH,
            ("choose", {"alternative": "S0", "reasoning": "all needed"}),
            (None, {}),
            (None, {}),
        ]
    )
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert sorted(selection.selected) == sorted([PREF, VOTES]) and selection.llm_calls == 9
    assert (
        "[drop] kept the 2 tables (the two calls disagreed) (3 calls): all needed"
        in selection.trace
    )


def test_one_selected_table_gets_no_drop_check(lookalikes):
    llm = ScriptedLLM([*CHOSE_VOTES, KEPT_A0])
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert (
        selection.selected == [VOTES]
        and selection.llm_calls == 6
        and "drop" not in selection.plan
        and "[drop]" not in selection.trace
    )
    m = Memory(LOOK_QUESTION, list(lookalikes.candidates), lookalikes.metadata, 10)
    assert drop.options(m, [VOTES, PREF]) == [[PREF, VOTES], [VOTES], [PREF]] and drop.letters(
        m, [VOTES, PREF]
    ) == {PREF: "Table A", VOTES: "Table B"}


def test_a_rejection_gets_no_contrast_check(lookalikes):
    script = [
        ("define_requirements", {"requirements": LOOK_REQS}),
        ("inspect", {"candidates": [1]}),
        ("recall", {"table": 1}),
        ("recall", {"table": 1}),
        ("recall", {"table": 1}),
        ("reject", {"reasoning": "none fits", "suggestion": "ballots"}),
    ]
    llm = ScriptedLLM(script)
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert (
        selection.selected == [] and len(llm.prompts) == 6 and "[contrast]" not in selection.trace
    )


# ---------------------------------------------------------------- the alternatives check

KEPT_PREF = (
    "keep",
    {
        "alternatives": ["A1"],
        "notes": [
            "the preference votes of the list Alfa by candidate rather than the votes of the lists"
        ],
        "reasoning": "both hold section 1 in 2009",
    },
)


def test_the_alternatives_check_offers_the_look_alike_that_also_answers_and_keeps_the_selection(
    lookalikes,
):
    calls = []
    llm = ScriptedLLM([*CHOSE_VOTES, KEPT_A0, KEPT_PREF])
    selection = MemGPTSelector(MEMGPT).select(
        _look_request(lookalikes, llm, tool_callback=calls.append)
    )
    assert selection.selected == [VOTES] and selection.llm_calls == 6  # the selection never moves
    assert selection.plan["alternatives"] == {
        "pool": 1,
        "offered": [
            {
                "label": "A1",
                "tables": [PREF],
                "change": [VOTES, PREF],
                "note": "the preference votes of the list Alfa by candidate rather than the votes of the lists",
            }
        ],
    }
    assert (
        llm.offered[-1] == ["keep"]
        and "Keep every alternative whose tables hold" in llm.systems[-1]
    )
    assert (
        "A0 (the current selection): T1\nA1: T6 -- T1 replaced by T6" in llm.prompts[-1]
        and "preferenze" in llm.prompts[-1]
    )
    assert "[alternatives] offered 1 of 1 (1 call): both hold section 1 in 2009" in selection.trace
    step = calls[-1]
    assert step.name == "alternatives" and step.title == "alternatives · 1 offered of 1"
    assert (
        "- **A1**: #6 Elezioni amministrative 2009 - Preferenze lista Alfa — the preference votes"
        in step.output
    )


def test_the_alternatives_check_offers_nothing_without_a_kept_alternative_or_an_answer(lookalikes):
    llm = ScriptedLLM(
        [
            *CHOSE_VOTES,
            KEPT_A0,
            ("keep", {"alternatives": [], "reasoning": "votes are not preferences"}),
        ]
    )
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert selection.plan["alternatives"] == {"pool": 1, "offered": []}
    assert (
        "[alternatives] none of 1 held every requirement (1 call): votes are not preferences"
        in selection.trace
    )
    llm = ScriptedLLM([*CHOSE_VOTES, KEPT_A0, (None, {}), (None, {})])
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert selection.plan["alternatives"] == {"pool": 1, "offered": []} and selection.llm_calls == 7
    assert "[alternatives] none of 1 held every requirement (2 calls)" in selection.trace


def test_the_alternatives_check_reads_the_drops_of_a_pair_and_a_note_written_inline(lookalikes):
    llm = ScriptedLLM(
        [
            *CHOSE_BOTH,
            ("choose", {"alternative": "S0", "reasoning": "both"}),
            ("choose", {"alternative": "S2", "reasoning": "both"}),
            (
                "keep",
                {
                    "alternatives": "A2 -- A2: only the votes of the lists, no candidates",
                    "reasoning": "x",
                },
            ),
        ]
    )  # both look-alikes selected: no contrast call; the drop calls disagree and keep the pair
    selection = MemGPTSelector(MEMGPT).select(_look_request(lookalikes, llm))
    assert sorted(selection.selected) == sorted([PREF, VOTES]) and selection.llm_calls == 8
    m = Memory(LOOK_QUESTION, list(lookalikes.candidates), lookalikes.metadata, 10)
    pool = alternatives.pool(lookalikes.metadata, [VOTES, PREF], lookalikes.candidates)
    assert [(a.label, a.tables, a.change) for a in pool] == [
        ("A1", [PREF], (VOTES, None)),
        ("A2", [VOTES], (PREF, None)),
    ]
    assert alternatives.describe(m, pool[1]) == "A2: T1 -- T6 left out"
    assert "T6 left out" in llm.prompts[-1]
    offered = selection.plan["alternatives"]["offered"]
    assert len(offered) == 1 and offered[0]["label"] == "A1" and len(offered[0]["tables"]) == 1
    assert offered[0]["note"] == "only the votes of the lists, no candidates"
    assert offered[0]["change"][1] is None


# ---------------------------------------------------------------- how a person sees the tool calls

REFUSED_ONCE = [
    ("define_requirements", {"requirements": REQS}),
    ("inspect", {"candidates": [1]}),
    ("select", {"table": 1, "covers": ["R1"], "note": "the June partition"}),
    ("confirm", {"reasoning": "June found"}),  # refused: R2 open
    ("inspect", {"candidates": [5]}),
    ("select", {"table": 5, "covers": ["R2"]}),
    ("confirm", {"reasoning": "June and August"}),
]


def test_tool_calls_name_tables_by_title_and_mark_refusals(lake):
    calls, streamed = [], []
    agent = _agent(
        lake, ScriptedLLM(REFUSED_ONCE), tool_callback=calls.append, emit=streamed.append
    )
    agent.run()
    assert [c.title for c in calls] == [
        "define_requirements · R1–R2",
        "inspect · #1 San Siro tickets sold June 2026",
        "select · #1 San Siro tickets sold June 2026 → R1",
        "⚠ confirm · refused",
        "inspect · #5 San Siro tickets sold August 2026",
        "select · #5 San Siro tickets sold August 2026 → R2",
        "confirm · #1, #5",
    ]
    assert [c.failed for c in calls] == [False, False, False, True, False, False, False]
    assert "R2 still open" in calls[3].output
    inspect = calls[1].output
    assert (
        inspect.startswith("**#1 San Siro tickets sold June 2026** — 20 rows")
        and "Shows **R1**" in inspect
    )
    assert "```text\nSchema for san_siro_tickets_june_2026.parquet:" in inspect
    assert "> the June partition" in calls[2].output
    assert "**Measured against the selection**\n\n- #1 ↔ #5: " in calls[5].output
    assert "**Confirmed.**" in calls[-1].output and "covers R1" in calls[-1].output
    assert streamed == []  # with a tool channel, the calls are not repeated in the stream


def test_without_a_tool_channel_each_call_is_one_line_of_the_stream(lake):
    streamed = []
    _agent(lake, ScriptedLLM(REFUSED_ONCE), emit=streamed.append).run()
    assert streamed[0].startswith(
        "- turn 1 · define_requirements · R1–R2: R1 measure: tickets sold San Siro June 2026"
    )
    assert streamed[3].startswith("- turn 4 · ⚠ confirm · refused: Confirm refused: R2 still open")
    assert len(streamed) == len(REFUSED_ONCE)


def test_the_selector_shows_its_candidates_as_the_first_tool_call(lake):
    calls, streamed = [], []
    MemGPTSelector(MEMGPT).select(
        _request(
            lake,
            ScriptedLLM(REFUSED_ONCE),
            tool_callback=calls.append,
            stream_callback=streamed.append,
        )
    )
    first = calls[0]
    assert first.name == "candidates" and first.title == "retrieval · 18 candidates, 10 shown first"
    assert (
        first.output.startswith("1. San Siro tickets sold June 2026")
        and "*Hidden at first" in first.output
    )
    assert not any("started" in s for s in streamed)


# ---------------------------------------------------------------- the catalogue the model reads

LONG_TITLE = (
    "Niveles sonoros registrados por el sensor de ruido de la calle Carles Cervera, Chaflán Reina Doña María, "
    "barrio de Ruzafa, con los valores diarios"
)


def _titled(lake):
    meta = {name: dict(entry) for name, entry in lake.metadata.items()}
    meta[JUNE].update(
        title=LONG_TITLE,
        resource_name="Nivells sonors del sensor del carrer Carles Cervera, Chaflán Reina Doña María",
        publisher="Ajuntament de València",
        responsible_entity="JCDecaux",
    )
    meta[AUGUST].update(
        title="CSV",
        resource_name="CSV",
        dataset_title="San Siro tickets sold by month",
        publisher="Stadium office",
    )
    return SimpleNamespace(dir=lake.dir, candidates=lake.candidates, metadata=meta)


def test_the_model_reads_whole_titles_second_titles_and_publishers(lake):
    titled = _titled(lake)
    llm = ScriptedLLM(CONFIRMING)
    selection = MemGPTSelector(MEMGPT).select(
        SelectionRequest(
            QUESTION,
            list(titled.candidates),
            titled.metadata,
            titled.dir,
            MEMGPT,
            llm=llm,
            portal_name="Test portal",
        )
    )
    assert selection.selected == [JUNE, AUGUST]
    first, last = llm.prompts[0], llm.prompts[len(CONFIRMING) - 1]
    # the candidate list: the whole title, the file's other name and the publisher on their own lines
    assert (
        f"#1 [not inspected] {LONG_TITLE}\n    also titled: Nivells sonors del sensor del carrer Carles Cervera, Chaflán Reina Doña María\n"
        in first
    )
    assert "    publisher: Ajuntament de València; responsible entity: JCDecaux\n" in first
    assert (
        "#5 [not inspected] CSV\n    dataset: San Siro tickets sold by month\n    publisher: Stadium office\n"
        in first
    )
    assert "also titled: CSV" not in first
    # STATE: the cards and the selection name the tables whole, with the dataset's title
    assert (
        f'#1 "{LONG_TITLE}" --' in last
        and '#5 "CSV (dataset: San Siro tickets sold by month)" --' in last
    )
    assert (
        f"Selection: #1 {LONG_TITLE} covers R1; #5 CSV (dataset: San Siro tickets sold by month) covers R2"
        in last
    )
    # the read-only block explains the lines
    assert (
        "CATALOGUE: a candidate is listed with its title and, when the portal has them, the title of the dataset"
        in llm.systems[0]
    )


def test_the_read_only_block_carries_the_portals_catalogue_note(lake):
    def block(lake_name: str) -> str:
        agent = MemGPTAgent(
            question=QUESTION,
            candidates=lake.candidates,
            metadata=lake.metadata,
            table_dir=lake.dir,
            llm=ScriptedLLM([]),
            prompts=get_prompt_manager(),
            portal="UK Open Data portal",
            lake=lake_name,
        )
        return agent.system_message()

    assert "Each candidate is one file (resource) of a dataset on data.gov.uk" in block("uk")
    assert "CATALOGUE: a candidate is listed with its title" in block(
        ""
    ) and "data.gov.uk" not in block("")
    assert "data.gov.uk" not in block("nowhere")


def test_phase2_hands_the_lake_to_the_search_and_the_selector(monkeypatch, lake):
    seen = {}

    def search(*args, **kwargs):
        seen["lake"] = kwargs.get("lake")
        return list(lake.candidates), {c: lake.metadata[c] for c in lake.candidates}

    monkeypatch.setattr(phase2, "retrieve_candidates", search)
    llm = ScriptedLLM(CONFIRMING)
    phase2.select_tables(
        query=QUESTION,
        llm=llm,
        pm=get_prompt_manager(),
        all_files=list(lake.candidates),
        keywords=["tickets"],
        solr_client=object(),
        csv_dir=lake.dir,
        portal_name="UK Open Data portal",
        lake="uk",
        discovery=MEMGPT,
    )
    assert seen["lake"] == "uk" and "data.gov.uk" in llm.systems[0]


def test_the_contrast_check_names_the_publisher_and_the_other_title(lookalikes):
    meta = {name: dict(entry) for name, entry in lookalikes.metadata.items()}
    meta[PREF].update(publisher="Comune di Bologna - Ufficio elettorale")
    meta[VOTES].update(
        resource_name="Voti alle liste - sezioni",
        publisher="Comune di Bologna - Ufficio elettorale",
    )
    alts = contrast.alternatives(meta, [VOTES], lookalikes.candidates)
    label = next(a.label for a in alts if a.tables == [PREF])
    llm = ScriptedLLM(
        [*CHOSE_VOTES, ("choose", {"alternative": label, "reasoning": "the preferenze column"})]
    )
    MemGPTSelector(MEMGPT).select(
        SelectionRequest(
            LOOK_QUESTION, list(lookalikes.candidates), meta, lookalikes.dir, MEMGPT, llm=llm
        )
    )
    assert "  publisher: Comune di Bologna - Ufficio elettorale" in llm.prompts[-1]
    assert "  also titled: Voti alle liste - sezioni" in llm.prompts[-1]
