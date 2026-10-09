import io
import json
import re
from collections.abc import Callable
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import LLM

from lakegen.config.experiment import DiscoveryConfig
from lakegen.config.settings import get_settings
from lakegen.discovery.ledger import (
    build_minimal_selection_fallback,
    requirement_ledger_blockers,
)
from lakegen.discovery.tools import P12State, Phase12ToolsManager
from lakegen.domain.errors import WorkflowCancelled
from lakegen.domain.types import SolrMetadata, StreamCallback
from lakegen.lakes.table_io import read_table
from lakegen.llm.agents.activity import (
    Phase2AgentStall,
)
from lakegen.llm.agents.instrumentation import ThinkingCapture
from lakegen.llm.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.prompts.prompt_manager import PromptManager
from lakegen.retrieval import RetrievalConfig
from lakegen.retrieval.models import RetrievalRun
from lakegen.retrieval.solr_client import LocalSolrClient


def _reasoning_with_selection_plan(
    reasoning: str,
    plan: dict[str, object] | None,
    advisories: list[str] | None,
) -> str:
    """Add the architect's agentic plan to the existing coder context."""
    if not plan:
        return reasoning
    lines = [reasoning, "", "AGENTIC SELECTION PLAN"]
    lines.append(f"Combination strategy: {plan.get('combination_strategy', 'unspecified')}")
    roles = plan.get("table_roles", {})
    if isinstance(roles, dict) and roles:
        lines.append("Table roles:")
        lines.extend(f"- {table}: {role}" for table, role in roles.items())
    coverage = plan.get("requirement_coverage", {})
    if isinstance(coverage, dict) and coverage:
        lines.append("Requirement coverage:")
        for requirement, evidence in coverage.items():
            if isinstance(evidence, dict):
                table = evidence.get("table", "unspecified")
                columns = evidence.get("columns", [])
                columns_text = ", ".join(map(str, columns)) if isinstance(columns, list) else str(columns)
                lines.append(f"- {requirement}: {table} [{columns_text}]")
    uncovered = plan.get("uncovered_requirements", [])
    if isinstance(uncovered, list) and uncovered:
        lines.append("Uncovered requirements: " + "; ".join(map(str, uncovered)))
    semantic_plan = plan.get("semantic_plan")
    if isinstance(semantic_plan, dict):
        lines.append("Semantic analysis plan (authoritative, non-oracle JSON):")
        lines.append(json.dumps(semantic_plan, ensure_ascii=False, sort_keys=True))
    alternatives = plan.get("alternatives_rejected", {})
    if isinstance(alternatives, dict) and alternatives:
        lines.append("Inspected alternatives rejected:")
        for table, evidence in list(alternatives.items())[:2]:
            if isinstance(evidence, dict):
                matched = ", ".join(map(str, evidence.get("matched_requirements", [])))
                missing = evidence.get("missing_requirement", "not specified")
                lines.append(f"- {table}: matches [{matched}]; lacks {missing}")
            else:
                lines.append(f"- {table}: lacks {evidence}")
    if advisories:
        lines.append("Non-blocking architect advisories:")
        lines.extend(f"- {advisory}" for advisory in advisories)
    return "\n".join(lines).strip()


def _recover_minimal_selection_plan(
    selected: list[str], reasoning: str
) -> tuple[dict[str, object], list[str]]:
    """Recover non-blocking coder guidance when discovery exits without a plan."""
    return build_minimal_selection_fallback(selected, reasoning)


def _recover_blocked_selection(
    state: P12State, all_files: list[str]
) -> tuple[list[str], str, dict[str, object], str] | None:
    """Recover the last ledger-blocked confirm_unified_selection call.

    For an attempt that ended without a terminal tool call (prose, loop
    guard): the blocked call's tables and bindings are still the architect's
    own inspected choice. A live run that fell back to the top three
    inspected candidates with an empty plan instead handed the coder no
    binding for the snapshot date the architect had identified.
    """
    blocked = state.blocked_selection
    tables = [table for table in blocked.get("tables", []) if table in all_files]
    if not tables:
        return None
    plan = {
        **dict(blocked.get("selection_plan") or {}),
        "recovered_from_existing_discovery_context": True,
    }
    unbound = requirement_ledger_blockers(
        list(plan.get("requirement_ledger") or []), tables
    )
    advisory = (
        "The selection plan was recovered from the architect's last "
        "confirm_unified_selection call, which was blocked and never "
        "validated. No selected-table column was bound for: "
        + (", ".join(unbound) or "(none recorded)")
        + ". Verify those against the data before relying on the plan."
    )
    return tables, str(blocked.get("reasoning") or ""), plan, advisory


def _inspected_runtime_evidence(state: P12State, max_chars: int = 24000) -> str:
    """Serialize benchmark-blind cached inspections for a fresh recovery agent."""
    evidence: dict[str, str] = {}
    remaining = max_chars
    for table in state.inspected_candidates():
        inspection = str(state.inspection_cache.get(table.casefold(), "")).strip()
        if not inspection or remaining <= 0:
            continue
        excerpt = inspection[:remaining]
        evidence[table] = excerpt
        remaining -= len(excerpt)
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def _extract_plausible_json(text: str) -> dict[str, object] | None:
    """Recover one complete JSON object from assistant text without accepting it."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _conservative_draft_from_requirements(
    requirements: dict[str, object], selected: list[str], schema: set[str]
) -> dict[str, object] | None:
    """Build only a single-table draft whose choices are explicit and unambiguous."""
    if len(selected) != 1:
        return None
    canonical = {column.casefold(): column for column in schema}
    grouping: list[list[object]] = []
    for raw in requirements.get("grouping", []) if isinstance(requirements.get("grouping"), list) else []:
        column = canonical.get(str(raw).strip().casefold())
        if not column:
            return None
        grouping.append([column, column])
    measures: list[list[object]] = []
    for raw in (
        requirements.get("measures", []) if isinstance(requirements.get("measures"), list) else []
    ):
        text = str(raw).strip()
        lowered = text.casefold()
        if lowered in {"count rows", "row count", "count_rows"}:
            measures.append(["row_count", "count_rows", []])
            continue
        operation = next((
            normalized for term, normalized in (
                ("average", "mean"), ("mean", "mean"), ("sum", "sum"),
                ("distinct", "count_distinct"), ("minimum", "min"),
                ("maximum", "max"),
            ) if term in lowered
        ), "")
        matched = [column for folded, column in canonical.items() if folded in lowered]
        if not operation or len(matched) != 1:
            return None
        measures.append([f"{operation}_{matched[0]}", operation, matched])
    if not measures:
        return None
    filters: list[list[object]] = []
    for raw in requirements.get("filters", []) if isinstance(requirements.get("filters"), list) else []:
        match = re.fullmatch(r"\s*([^=<>]+?)\s*(=|contains)\s*(.+?)\s*", str(raw), re.I)
        if not match:
            return None
        column = canonical.get(match.group(1).strip().casefold())
        if not column:
            return None
        filters.append([
            column, "equals" if match.group(2) == "=" else "contains",
            match.group(3).strip(),
        ])
    ordering: list[list[object]] = []
    ordering_text = str(requirements.get("ordering") or "").casefold()
    if ordering_text:
        direction = "descending" if "desc" in ordering_text else (
            "ascending" if "asc" in ordering_text else ""
        )
        if not direction:
            return None
        output = next((str(item[0]) for item in measures if str(item[0]).casefold() in ordering_text), str(measures[0][0]))
        ordering.append([output, direction])
    return {
        "filters": filters, "temporal_filters": [], "dimensions": grouping,
        "measures": measures, "joins": [], "ordering": ordering,
        "limit": requirements.get("limit"),
    }


def _semantic_planner_prompt(query: str, state: P12State) -> str:
    """Build the complete planner context from runtime-only allowlisted inputs."""
    return (
        "Create a compact SemanticPlanDraft for the confirmed selection. "
        "You cannot search or change tables. Call submit_semantic_plan_draft. "
        "Use only these forms: filters [[column, operator, value]], "
        "dimensions [[output, column]], measures [[output, operation, [columns]]], "
        "ordering [[output, direction]], limit, and structured joins. "
        "Allowed operations: count_rows, count_distinct, sum, mean, min, max, "
        "ratio, difference, custom. Correct only fields reported by validation; "
        "at most two tool attempts. Never use benchmark, gold, reference, expected "
        "answer, evaluator, judge, or prior results.\n\n"
        f"QUESTION:\n{query}\n\n"
        "CONFIRMED TABLES:\n"
        + json.dumps(state.confirmed_tables, ensure_ascii=False)
        + "\n\nDECLARED REQUIREMENTS:\n"
        + json.dumps(state.selection_requirements, ensure_ascii=False)
        + "\n\nINSPECTED RUNTIME EVIDENCE:\n"
        + _inspected_runtime_evidence(state)
    )

STALL_RECOVERY_INSTRUCTIONS = (
    "The discovery exploration stopped before producing its required typed "
    "semantic plan. Do not search again. Using only the user question and "
    "the schemas/values already obtained through prior tool calls, inspect "
    "a selected table only if essential evidence is still missing, then call "
    "confirm_unified_selection. The semantic_plan must explicitly contain "
    "filters, temporal_filters, dimensions, measures, joins, ordering, limit, "
    "output_columns, null_policy, and table_roles. If runtime evidence proves "
    "the selection cannot answer the question, call reject_unified_selection: "
    "map every inspected candidate you judge highly irrelevant -- proven to "
    "satisfy nothing -- to the concrete evidence proving that in ban_tables, "
    "so it is excluded from later retrieval; every other inspected candidate "
    "is kept by default and carries over to the next attempt, and state what "
    "is still missing in reasoning and suggestion. Do not just describe this "
    "in prose -- "
    "call the tool. Never use benchmark, gold, expected answer, reference "
    "output, or prior evaluator output.\n\n"
)
ERROR_RECOVERY_INSTRUCTIONS = (
    "Discovery ended without its required semantic plan. Do not search "
    "again. Using only the user question and already inspected runtime "
    "schemas/values, call confirm_unified_selection now. Supply every "
    "semantic_plan field: filters, temporal_filters, dimensions, measures, "
    "joins, ordering, limit, output_columns, null_policy, and table_roles. "
    "Never use benchmark, reference, gold, expected-answer, evaluator, or "
    "prior-judge data. If the inspected tables are insufficient, call "
    "reject_unified_selection: map every inspected candidate you judge "
    "highly irrelevant -- proven to satisfy nothing -- to the concrete "
    "evidence proving that in ban_tables, so it is excluded from later "
    "retrieval; every other inspected candidate is kept by default and "
    "carries over to the next attempt, and state what is still missing in "
    "reasoning and suggestion. Do not just describe this in prose -- call "
    "the tool.\n\n"
)
EVIDENCE_HEADER = "INSPECTED RUNTIME EVIDENCE (authoritative, benchmark-blind):\n"
DEDICATED_PLANNER_SYSTEM_PROMPT = (
    "You are a benchmark-blind semantic planner. Compile explicit "
    "question semantics against inspected runtime schemas."
)
CARDINALITY_CHECK = (
    "\n\nEXPERIMENTAL SELECTION CHECK: Before confirming, state the expected "
    "table count internally. Select one table unless the question has a "
    "separate source, period, or join requirement. For every table after "
    "the first, require one unique required measure, filter, period, or join "
    "key that no already selected table covers; otherwise exclude it."
)
TRACE_HEADER = "--- Unified Phase Activity Log ---\n"


def _rejection_payload(reason: str) -> str:
    return "FINAL_PAYLOAD: " + json.dumps({"tables": "", "reasoning": reason})


def _fallback_payload(tables: list[str], reasoning: str) -> str:
    return f"FINAL_PAYLOAD: {json.dumps({'tables': ', '.join(tables), 'reasoning': reasoning})}"


class _UnifiedRun:
    """One run of the unified agent: the exploration, its recoveries, the dedicated planner, the trace."""

    def __init__(
        self,
        query: str,
        llm: LLM,
        pm: PromptManager,
        all_files: list[str],
        solr_client: LocalSolrClient,
        csv_dir: Path,
        *,
        hint: str,
        retrieval_memory: str,
        portal_name: str,
        stream_callback: StreamCallback | None,
        cancel_check: Callable[[], None] | None,
        retrieval_config: RetrievalConfig | None,
        state: P12State,
        retrieval_observer: Callable[[RetrievalRun], None] | None,
        require_semantic_plan: bool,
        discovery_config: DiscoveryConfig | None,
    ) -> None:
        self.query, self.llm, self.csv_dir, self.state = query, llm, csv_dir, state
        self.stream_callback, self.cancel_check = stream_callback, cancel_check
        self.require_semantic_plan = require_semantic_plan
        self.stream_trace = io.StringIO()
        self.response = ""
        self.tools_manager = Phase12ToolsManager(
            state, solr_client, all_files, csv_dir, question=query,
            retrieval_config=retrieval_config, retrieval_observer=retrieval_observer,
            discovery_config=discovery_config,
            # Terminal, frontend, and the saved activity log -- but never the prompt.
            notice_callback=self.emit, llm=llm,
        )
        self.tools = self.tools_manager.get_tools()
        self.context_editor = self.tools_manager.context.edit if self.tools_manager.context.enabled else None
        retrieval = retrieval_config or RetrievalConfig()
        discovery = discovery_config or DiscoveryConfig()
        self.system_prompt = pm.render(
            "unified_architect", "system_prompt",
            portal_name=portal_name, hint=hint, retrieval_memory=retrieval_memory,
            value_search=retrieval.mode.value_keywords, verbatim_entities=retrieval.mode.verbatim_entities,
            initial_shortlist_size=discovery.initial_shortlist_size,
            max_inspected_candidates=discovery.max_inspected_candidates,
            state_board=self.tools_manager.discovery.state_board,
            compact_history=self.tools_manager.discovery.compact_history,
        )
        if get_settings().selection_cardinality_experiment:
            self.system_prompt += CARDINALITY_CHECK
        self.max_tool_calls = max(12, discovery.initial_shortlist_size + 5)
        self.token_counter = next(
            (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")), None,
        )
        if self.token_counter:
            self.token_counter.reset_counts()
        reset_llm_token_usage(llm)
        self.agent_prompt = pm.render("unified_architect", "user_prompt", question=query)
        self.thinking_capture = ThinkingCapture()
        self.dispatcher = get_dispatcher()
        self.dispatcher.add_event_handler(self.thinking_capture)

    def emit(self, delta: str) -> None:
        if not delta:
            return
        self.stream_trace.write(delta)
        print(delta, end="", flush=True)
        if self.stream_callback is not None:
            self.stream_callback(delta)

    def _agent(self, user_prompt: str, *, agent_name: str, max_iterations: int, max_repeats: int,
               max_tool_calls: int, timeout_seconds: int) -> str:
        """One agent over the discovery tools; a stall or an error reaches the caller."""
        from lakegen.llm.agents.runner import run_agent_workflow

        return run_agent_workflow(
            llm=self.llm, system_prompt=self.system_prompt, user_prompt=user_prompt,
            agent_name=agent_name, emit_stream=self.emit, cancel_check=self.cancel_check,
            tools=self.tools, tool_available=self.tools_manager.is_tool_available,
            context_editor=self.context_editor, max_iterations=max_iterations,
            max_repeats=max_repeats, max_tool_calls=max_tool_calls, timeout_seconds=timeout_seconds,
        )

    def explore(self) -> None:
        self.emit(
            "\n**Unified Architect & Search agent started**\n"
            "- Streaming model output and tool inspections below.\n"
        )
        self.response = self._agent(
            self.agent_prompt, agent_name="unified_explorer", max_iterations=16, max_repeats=3,
            max_tool_calls=self.max_tool_calls, timeout_seconds=300,
        )

    def _can_recover(self) -> bool:
        """A required plan can still be written from what was inspected."""
        return self.require_semantic_plan and bool(self.state.inspected_candidates())

    def _planner_recovery(self, instructions: str, heading: str) -> None:
        """A bounded planner over the inspected evidence; its own stall rejects the keywords."""
        self.state.recovery_started = True
        self.emit(f"\n\n**{heading}**\n- Using only the question and already inspected runtime evidence.\n")
        prompt = instructions + EVIDENCE_HEADER + _inspected_runtime_evidence(self.state)
        try:
            self.response = self._agent(
                prompt, agent_name="semantic_planner_recovery", max_iterations=6, max_repeats=2,
                max_tool_calls=4, timeout_seconds=180,
            )
        except Phase2AgentStall as planner_stall:
            self.state.recovery_stop_reason = str(planner_stall)
            self.response = _rejection_payload(
                "REJECT_KEYWORDS: semantic planner could not establish a "
                f"complete runtime-grounded plan after bounded recovery: {planner_stall}"
            )

    def after_stall(self, stall: Phase2AgentStall) -> None:
        """The loop guard stopped the exploration: recover a plan, or settle for the inspected tables."""
        self.state.initial_stall_reason = str(stall)
        if self._can_recover():
            self._planner_recovery(STALL_RECOVERY_INSTRUCTIONS, "Semantic planner recovery started")
            return
        self.emit(
            "\n\n**Loop guard triggered**\n"
            f"- Reason: `{stall}`\n"
            "- Action: using only inspected candidates.\n"
        )
        self.response = _fallback_payload(
            self.state.inspected_candidates()[:2],
            f"Phase loop guard triggered: {stall}. Fallback restricted to inspected candidates.",
        )

    def after_error(self, error: Exception) -> None:
        """The exploration failed: recover a plan, or settle for the inspected tables."""
        message = str(error)
        self.state.initial_stall_reason = f"{type(error).__name__}: {message}"
        if self._can_recover():
            self._planner_recovery(
                ERROR_RECOVERY_INSTRUCTIONS, "Semantic planner recovery started after discovery error",
            )
            return
        if "Max iterations" in message:
            reason = "Agent exceeded maximum iterations without a final decision. Fallback to top candidates."
        else:
            reason = f"Agent error: {message[:120]}. Fallback to top 2."
        self.emit(f"\n[agent error] {message[:160]}\n")
        self.response = _fallback_payload(self.state.inspected_candidates()[:2], reason)

    def _plan_needed(self) -> bool:
        plan = self.state.selection_plan
        return bool(
            self.require_semantic_plan and self.state.confirmed_tables
            and not plan.get("coder_brief") and not plan.get("semantic_plan")
        )

    def _dedicated_planner(self) -> None:
        """A planner with only the drafting tool, over the confirmed selection and its evidence."""
        from lakegen.llm.agents.runner import run_agent_workflow

        state, manager = self.state, self.tools_manager
        try:
            planner_response = run_agent_workflow(
                llm=self.llm, system_prompt=DEDICATED_PLANNER_SYSTEM_PROMPT,
                user_prompt=_semantic_planner_prompt(self.query, state),
                agent_name="dedicated_semantic_planner", emit_stream=self.emit,
                cancel_check=self.cancel_check, tools=manager.get_semantic_planner_tools(),
                max_iterations=5, max_repeats=3, max_tool_calls=2, timeout_seconds=180,
            )
            if state.selection_plan.get("semantic_plan"):
                self.response = planner_response
            elif state.semantic_planner_attempts < 2:
                candidate = _extract_plausible_json(planner_response)
                if isinstance(candidate, dict):
                    draft = candidate.get("draft", candidate.get("semantic_plan", candidate))
                    if isinstance(draft, dict):
                        self.response = manager.submit_semantic_plan_draft(draft)
        except (Phase2AgentStall, ValueError) as planner_error:
            state.recovery_stop_reason = str(planner_error)

    def _requirements_draft(self) -> None:
        """The last resort: a single-table draft from the requirements the architect stated."""
        state = self.state
        try:
            table = state.confirmed_tables[0] if len(state.confirmed_tables) == 1 else ""
            schema = {str(column) for column in read_table(self.csv_dir / table, nrows=0).columns} if table else set()
            draft = _conservative_draft_from_requirements(state.selection_requirements, state.confirmed_tables, schema)
            if draft and state.semantic_planner_attempts < 2:
                self.response = self.tools_manager.submit_semantic_plan_draft(draft)
                state.selection_plan_source = "semantic_requirements_fallback"
        except ValueError as fallback_error:
            state.recovery_stop_reason = str(fallback_error)

    def plan_if_missing(self) -> None:
        """A confirmed selection without a plan gets one from the dedicated planner, then from the requirements."""
        if not self._plan_needed():
            return
        self.state.recovery_started = True
        self.emit(
            "\n\n**Dedicated semantic planner started**\n"
            "- Search tools are unavailable; using only confirmed selection and runtime evidence.\n"
        )
        self._dedicated_planner()
        if not self.state.selection_plan.get("semantic_plan"):
            self._requirements_draft()

    def close(self) -> str:
        """The activity log of the run; the stream and the thinking capture are released."""
        full_trace = TRACE_HEADER + self.stream_trace.getvalue()
        self.stream_trace.close()
        self.dispatcher.event_handlers.remove(self.thinking_capture)
        return full_trace

    def tokens(self) -> int:
        tokens = 0
        if self.token_counter:
            tokens = self.token_counter.prompt_llm_token_count + self.token_counter.completion_llm_token_count
            self.token_counter.reset_counts()
        return max(tokens, get_llm_token_usage(self.llm))


def _parse_final_payload(
    response: str, all_files: list[str],
) -> tuple[list[str], str, dict[str, object] | None, list[str]]:
    """The tables, reasoning, plan and advisories of a FINAL_PAYLOAD; prose is its own reasoning."""
    selected: list[str] = []
    reasoning = ""
    plan: dict[str, object] | None = None
    advisories: list[str] = []
    try:
        match = re.search(r"FINAL_PAYLOAD:\s*(\{.*\})", response, re.DOTALL)
        if match:
            payload = json.loads(match.group(1))
            reasoning = payload.get("reasoning", "")
            plan = payload.get("selection_plan")
            advisories = payload.get("advisories") or []
            for table in [item.strip() for item in payload.get("tables", "").split(",")]:
                if table in all_files and table not in selected:
                    selected.append(table)
        else:
            reasoning = response
    except json.JSONDecodeError:
        pass
    return selected, reasoning, plan, advisories


def _settle_selection(state: P12State, all_files: list[str], response: str) -> tuple[list[str], str]:
    """The run's selection and reasoning, with a plan recovered where the agent left none.

    confirm_unified_selection persists the validated contract before
    returning its terminal payload, so that state is preferred when the
    response was truncated or its wrapper malformed; a confirmation the
    ledger blocked is recovered next; a minimal plan is the last resort.
    """
    selected, reasoning, plan, advisories = _parse_final_payload(response, all_files)
    if not selected:
        selected = state.inspected_candidates()[:3]
    if not plan and state.selection_plan:
        plan = dict(state.selection_plan)
        if state.confirmed_tables:
            selected = list(state.confirmed_tables)
        advisories.extend(state.selection_advisories)
        if state.selection_plan_source == "none":
            state.selection_plan_source = "confirmed_state_recovery"
    recovery = None if plan or reasoning.startswith("REJECT_KEYWORDS") else _recover_blocked_selection(state, all_files)
    if recovery is not None:
        selected, blocked_reasoning, plan, advisory = recovery
        reasoning = blocked_reasoning or reasoning
        advisories.append(advisory)
        state.selection_plan_source = "blocked_confirmation_recovery"
    if not plan:
        plan, recovered_advisories = _recover_minimal_selection_plan(selected, reasoning)
        advisories.extend(recovered_advisories)
        state.selection_plan_source = "minimal_fallback"
    if plan:
        state.selection_plan = plan
        state.selection_advisories = advisories
        if state.selection_plan_source == "none":
            state.selection_plan_source = "final_payload"
    return selected, _reasoning_with_selection_plan(reasoning, plan, advisories)


def phase12_agent(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    solr_client: LocalSolrClient,
    csv_dir: Path,
    hint: str = "",
    retrieval_memory: str = "",
    portal_name: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check: Callable[[], None] | None = None,
    retrieval_config: RetrievalConfig | None = None,
    state: P12State | None = None,
    retrieval_observer: Callable[[RetrievalRun], None] | None = None,
    require_semantic_plan: bool = True,
    discovery_config: DiscoveryConfig | None = None,
) -> tuple[list[str], list[str], SolrMetadata, str, str, int]:
    """The unified agent on one question: the selected tables, keywords, metadata, reasoning, trace and tokens."""
    state = state or P12State()
    run = _UnifiedRun(
        query, llm, pm, all_files, solr_client, csv_dir, hint=hint, retrieval_memory=retrieval_memory,
        portal_name=portal_name, stream_callback=stream_callback, cancel_check=cancel_check,
        retrieval_config=retrieval_config, state=state, retrieval_observer=retrieval_observer,
        require_semantic_plan=require_semantic_plan, discovery_config=discovery_config,
    )
    try:
        run.explore()
    except Phase2AgentStall as stall:
        run.after_stall(stall)
    except WorkflowCancelled:
        raise
    except Exception as error:
        run.after_error(error)
    finally:
        run.plan_if_missing()
        full_trace = run.close()
    tokens = run.tokens()
    selected, reasoning = _settle_selection(state, all_files, run.response)
    return selected, state.used_keywords, state.solr_meta, reasoning, full_trace, tokens
