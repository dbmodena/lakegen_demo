"""The step-by-step terminal driver: the engine's steps, shown and answered in the terminal.

The driver walks ``Pipeline.steps`` and renders each signal the way the CLI
always has (the phase headers, the keywords, the tables, the code's
outcome, the answer and the summary), answers each request at the same
prompts, and records every answer as a human intervention for the run's
record. The engine's logic underneath is the batch service's.
"""

from __future__ import annotations

from lakegen.domain.results import QueryResult
from lakegen.interfaces.cli import terminal
from lakegen.observability.tracing import HumanGate, HumanInterventionRecorder
from lakegen.pipeline.context import RunContext
from lakegen.pipeline.engine import MAX_CODE_ATTEMPTS, MAX_TABLE_ATTEMPTS, Pipeline
from lakegen.pipeline.events import (
    Abort,
    Answer,
    Approve,
    CoderRejected,
    Force,
    Rediscover,
    Regenerate,
    Reject,
    ReviewKeywords,
    ReviewTables,
    RunStarted,
    StageFinished,
    StageStarted,
)

_c = terminal.colour


class TerminalDriver:
    def __init__(self, pipeline: Pipeline, ctx: RunContext) -> None:
        self.pipeline = pipeline
        self.ctx = ctx
        self.unified = ctx.experiment.use_unified_agent
        self.interventions = HumanInterventionRecorder()
        self.rediscovering = False
        """The round in progress follows a coder rejection the person sent back."""
        self.approve_forced = False
        """The round forced a selection after its last attempt; it is coded as it is."""
        self.answered = False
        self.discovery_tokens = (0, 0)

    def run(self, question: str) -> QueryResult:
        steps = self.pipeline.steps(question, interventions=self.interventions)
        answer: Answer | None = None
        try:
            while True:
                event = steps.send(answer)
                answer = self.handle(event)
        except StopIteration as stop:
            result = stop.value
        self.discovery_tokens = tuple(self.pipeline.last_discovery_tokens)
        if result.status != "cancelled" and self.answered:
            self.summary(result)
        return result

    # --- signals -------------------------------------------------------------

    def handle(self, event) -> Answer | None:
        if isinstance(event, RunStarted):
            print(_c(f"Portal: {self.ctx.portal_name}", "cyan"))
            print(_c(f"Model:  {self.ctx.model_name}", "cyan"))
            print(_c(f"Tables: {event.tables} files", "cyan"))
        elif isinstance(event, StageStarted):
            self.stage_started(event)
        elif isinstance(event, StageFinished):
            self.stage_finished(event)
        elif isinstance(event, ReviewKeywords):
            return self.review_keywords(event)
        elif isinstance(event, ReviewTables):
            return self.review_tables(event)
        elif isinstance(event, CoderRejected):
            return self.coder_rejected(event)
        return None

    def stage_started(self, event: StageStarted) -> None:
        if event.stage == "discovery":
            if self.unified:
                terminal.header("Re-running Phase 1 & 2 (Unified)" if self.rediscovering else "Phase 1 & 2 – Unified Architect & Search")
            else:
                terminal.header("Re-running Phase 1 – Keyword Selection" if self.rediscovering else "Phase 1 – Keyword Selection (AND logic)")
        elif event.stage == "code":
            terminal.header(f"Phase 3 – Code Generation (attempt {event.attempt}/{MAX_CODE_ATTEMPTS})")
        elif event.stage == "answer":
            self.answered = True
            terminal.header("Phase 4 – Answer Synthesis")

    def stage_finished(self, event: StageFinished) -> None:
        if event.stage == "discovery":
            self.discovery_finished(event)
        elif event.stage == "code":
            self.code_finished(event)

    def discovery_finished(self, event: StageFinished) -> None:
        exhausted = event.attempt >= MAX_TABLE_ATTEMPTS
        if event.detail == "keywords_rejected":
            if not self.unified:
                reason = event.message.replace("REJECT_KEYWORDS:", "").strip()
                print(_c(f"\n⚠ Keywords rejected by table judge: {reason}", "yellow"))
            if exhausted:
                print(_c(
                    "No datasets found after the configured retries." if self.unified
                    else f"Max keyword retries ({MAX_TABLE_ATTEMPTS}) reached.", "red",
                ))
        elif event.detail == "forced_selection":
            print(_c(
                "No datasets found after the configured retries. Using best available." if self.unified
                else f"Max keyword retries ({MAX_TABLE_ATTEMPTS}) reached. Using best available.", "red",
            ))
            self.approve_forced = True

    def code_finished(self, event: StageFinished) -> None:
        if event.detail == "succeeded":
            print(_c("\n✓ Code executed successfully!", "green"))
            print(f"{_c('Output:', 'bold')}\n{event.message}")
        elif event.detail in {"failed", "empty", "blocked_by_selection"}:
            print(_c(f"\n✗ Execution error: {event.message}", "red"))
            if event.more:
                print(f"Retrying ({event.attempt}/{MAX_CODE_ATTEMPTS})...")
            elif event.detail != "blocked_by_selection":
                print(_c(f"Failed after {event.attempt} attempts.", "red"))

    # --- requests ------------------------------------------------------------

    def review_keywords(self, event: ReviewKeywords) -> Answer:
        if self.rediscovering:
            # After a coder rejection the CLI never reviewed the new keywords.
            terminal.header("Re-running Phase 2 – Table Search & Selection")
            return Approve()
        print(f"\n\n{_c('AND Keywords:', 'bold')} {terminal.keyword_list(event.keywords)}")
        print(f"{_c('Tokens:', 'dim')}    {event.tokens}")
        if terminal.ask_yes_no(
            f"\n{_c('Approve these keywords?', 'yellow')}",
            recorder=self.interventions, phase="discovery", gate=HumanGate.KEYWORD_APPROVAL,
        ):
            terminal.header(f"Phase 2 – Table Search & Selection ({self.ctx.retrieval.mode})")
            return Approve()
        hint = terminal.ask_input(
            "Hint for keyword selection (or press Enter to retry without hint)",
            recorder=self.interventions, phase="discovery", gate=HumanGate.KEYWORD_HINT,
        )
        terminal.header("Phase 1 – Keyword Selection (AND logic)")
        return Regenerate(hint)

    def review_tables(self, event: ReviewTables) -> Answer:
        if event.after_rejection:
            self.rediscovering = False
            print(f"\n{_c('New tables:', 'bold')} {', '.join(event.selected)}")
            if terminal.ask_yes_no(
                "Approve?", recorder=self.interventions, phase="discovery", gate=HumanGate.DATASET_APPROVAL,
            ):
                return Approve()
            print(_c("Workflow cancelled.", "red"))
            return Abort()
        if self.approve_forced:
            self.approve_forced = False
            return Approve()
        print(f"\n\n{_c('Keywords:', 'bold')} {terminal.keyword_list(event.keywords)}")
        print(f"{_c('Tables:', 'bold')}   {', '.join(event.selected) if event.selected else '(none)'}")
        print(f"{_c('Reasoning:', 'bold')} {event.reasoning}")
        print(f"{_c('Tokens:', 'dim')}    {event.tokens}")
        if terminal.ask_yes_no(
            f"\n{_c('Approve this selection?', 'yellow')}",
            recorder=self.interventions, phase="discovery", gate=HumanGate.DATASET_APPROVAL,
        ):
            return Approve()
        hint = terminal.ask_input(
            "Hint for the agent (or press Enter to retry without hint)" if self.unified
            else "Hint for the agent (or press Enter to retry)",
            recorder=self.interventions, phase="discovery", gate=HumanGate.DATASET_HINT,
        )
        return Reject(hint)

    def coder_rejected(self, event: CoderRejected) -> Answer:
        print(_c(f"\n⚠ Tables rejected: {event.reason}", "yellow"))
        if terminal.ask_yes_no(
            "Force execution anyway?", default=False,
            recorder=self.interventions, phase="code", gate=HumanGate.FORCE_EXECUTION_CONFIRMATION,
        ):
            return Force()
        self.rediscovering = True
        return Rediscover()

    # --- the end -------------------------------------------------------------

    def summary(self, result: QueryResult) -> None:
        print(f"\n{_c('Final Answer:', 'bold')}")
        print(result.answer)
        terminal.header("Summary")
        p1, p2 = self.discovery_tokens
        tokens = {"p1": p1, "p2": p2, "p3": result.tokens["p3"], "p4": result.tokens["p4"]}
        print(f"  Keywords:  {terminal.keyword_list(result.keywords)}")
        print(f"  Tables:    {', '.join(result.tables)}")
        print(f"  Tokens:    P1={tokens['p1']}  P2={tokens['p2']}  P3={tokens['p3']}  P4={tokens['p4']}")
        print(f"  Total:     {sum(tokens.values())}")
        print(_c("\nExperiment log saved.", "dim"))
