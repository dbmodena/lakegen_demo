from types import SimpleNamespace

import pytest

from lakegen.experiment_config import ExperimentConfig
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.tracing import HumanGate
from lakegen.ui import workflow
from lakegen.ui.sections import format_retrieval_keywords
from lakegen.ui.state import LakeGenSession, WorkflowTimedOut
from src import app as chainlit_app


def _session() -> LakeGenSession:
    config = ExperimentConfig(interaction_mode="human_gated")
    runtime = SimpleNamespace(
        model_name=config.model,
        experiment=config,
        retrieval=RetrievalConfig(),
    )
    session = LakeGenSession(runtime=runtime, query="Question?")
    session.manifest = {"run_id": session.run_id, "resolved_config": {"core": "nyc"}}
    return session


def test_chat_flags_keywords_that_question_only_retrieval_never_searches():
    session = _session()
    assert format_retrieval_keywords(session, ["Home", "Office"]) == "`Home`, `Office`"

    session.runtime.retrieval = RetrievalConfig(mode=RetrievalMode.PNEUMA)
    flagged = format_retrieval_keywords(session, ["Home", "Office"])

    assert flagged.startswith("_none; retrieval uses only the question_")
    assert "not searched: `Home`, `Office`" in flagged


@pytest.mark.parametrize(
    ("output", "expected", "provided"),
    [
        ("real hint", "real hint", True),
        ("", "", False),
        ("   ", "", False),
        ("skip", "", False),
        ("none", "", False),
        ("no", "", False),
        ("SKIP", "", False),
        ("NoNe", "", False),
        ("NO", "", False),
    ],
)
@pytest.mark.asyncio
async def test_chainlit_hint_normalization(output, expected, provided, monkeypatch):
    session = _session()

    class Ask:
        def __init__(self, **_kwargs):
            pass

        async def send(self):
            return {"output": output}

        async def remove(self):
            pass

    monkeypatch.setattr(workflow, "get_session", lambda: session)
    monkeypatch.setattr(workflow.cl, "AskUserMessage", Ask)

    value = await workflow._ask_hint(
        "Hint?", phase="discovery", gate=HumanGate.DATASET_HINT
    )

    assert value == expected
    assert session.intervention_recorder.to_list()[0]["provided"] is provided


@pytest.mark.asyncio
async def test_chainlit_interaction_timeout_is_explicit(monkeypatch):
    session = _session()

    class Ask:
        def __init__(self, **_kwargs):
            pass

        async def send(self):
            return None

        async def remove(self):
            pass

    monkeypatch.setattr(workflow, "get_session", lambda: session)
    monkeypatch.setattr(workflow.cl, "AskUserMessage", Ask)

    with pytest.raises(WorkflowTimedOut):
        await workflow._ask_hint(
            "Hint?", phase="discovery", gate=HumanGate.DATASET_HINT
        )
    assert session.intervention_recorder.to_list() == []


class _Step:
    def __init__(self, **_kwargs):
        self.output = ""
        self.default_open = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def update(self):
        pass


class _Bridge:
    def __init__(self, _step):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def emit(self, _delta):
        pass


class _Message:
    def __init__(self, **_kwargs):
        pass

    async def send(self):
        pass


def _engine_run(monkeypatch, session, tmp_path, *, phase12, ask_choice):
    """A unified run on the engine with scripted phases, its log rows captured."""
    config = ExperimentConfig(interaction_mode="human_gated")
    runtime = SimpleNamespace(
        model_name=config.model, solr_core=config.core, csv_dir=tmp_path, portal_name="UK",
        retrieval=config.retrieval.to_runtime(), experiment=config, discovery=config.discovery,
        use_unified_agent=True,
    )
    logged = []
    monkeypatch.setattr(workflow, "get_session", lambda: session)
    monkeypatch.setattr(workflow, "get_runtime_settings", lambda: runtime)
    monkeypatch.setattr(workflow, "get_llm", lambda _name: (object(), None))
    monkeypatch.setattr(workflow, "get_solr", lambda _core: object())
    monkeypatch.setattr(workflow, "get_prompt_manager", object)
    monkeypatch.setattr(workflow, "get_all_table_files", lambda _path: ["table.csv"])
    monkeypatch.setattr(workflow, "persist_manifest", lambda *_args: None)
    monkeypatch.setattr(workflow, "log_retrieval_decision", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "phase12_agent", phase12)
    monkeypatch.setattr(workflow, "_ask_choice", ask_choice)
    monkeypatch.setattr(workflow, "save_experiment_log", lambda **kwargs: logged.append(kwargs))
    monkeypatch.setattr(workflow, "StepStreamBridge", _Bridge)
    monkeypatch.setattr(workflow.cl, "Step", _Step)
    monkeypatch.setattr(workflow.cl, "Message", _Message)
    return logged


@pytest.mark.asyncio
async def test_workflow_exception_is_logged_without_masking_original(monkeypatch, tmp_path):
    """A phase that raises ends the run as failed, with the error, in exactly one record."""
    session = _session()

    def fail(**_kwargs):
        raise ValueError("discovery failed")

    async def never(*_args, **_kwargs):
        raise AssertionError("no gate is reached")

    logged = _engine_run(monkeypatch, session, tmp_path, phase12=fail, ask_choice=never)
    await workflow.run_lakegen_workflow("Question?")

    assert len(logged) == 1
    assert logged[0]["status"] == "failed"
    assert "ValueError: discovery failed" in logged[0]["error"]


@pytest.mark.asyncio
async def test_workflow_timeout_is_logged_once(monkeypatch, tmp_path):
    """A gate that times out ends the run as timed out, recorded once, before the chat goes on."""
    session = _session()

    def select(**_kwargs):
        return ["table.csv"], ["water"], {}, "Selected.", "", 3

    async def time_out(*_args, **_kwargs):
        raise WorkflowTimedOut("dataset_hint expired")

    logged = _engine_run(monkeypatch, session, tmp_path, phase12=select, ask_choice=time_out)
    await workflow.run_lakegen_workflow("Question?")

    assert len(logged) == 1
    assert logged[0]["status"] == "timed_out"
    assert "WorkflowTimedOut" in logged[0]["error"]


@pytest.mark.asyncio
async def test_busy_workflow_is_rejected_without_session_manifest_or_log(monkeypatch):
    messages = []

    class Message:
        def __init__(self, *, content):
            self.content = content

        async def send(self):
            messages.append(self.content)

    monkeypatch.setattr(workflow.cl, "Message", Message)
    monkeypatch.setattr(
        workflow, "get_session", lambda: pytest.fail("active session was accessed")
    )
    monkeypatch.setattr(
        workflow, "persist_manifest", lambda *_a, **_k: pytest.fail("manifest created")
    )
    monkeypatch.setattr(
        workflow, "save_experiment_log", lambda **_k: pytest.fail("log created")
    )

    await workflow.WORKFLOW_LOCK.acquire()
    try:
        await workflow.run_lakegen_workflow("Second question?")
    finally:
        workflow.WORKFLOW_LOCK.release()

    assert messages == [workflow.t("workflow.locked")]


@pytest.mark.asyncio
async def test_on_message_rejects_before_replacing_active_session(monkeypatch):
    active_session = _session()
    messages = []

    class Message:
        def __init__(self, *, content):
            self.content = content

        async def send(self):
            messages.append(self.content)

    monkeypatch.setattr(chainlit_app.cl, "Message", Message)
    monkeypatch.setattr(chainlit_app, "get_session", lambda: active_session)
    monkeypatch.setattr(
        chainlit_app.cl.user_session,
        "set",
        lambda *_a, **_k: pytest.fail("active session was replaced"),
    )
    monkeypatch.setattr(
        chainlit_app,
        "run_lakegen_workflow",
        lambda *_a, **_k: pytest.fail("second workflow was started"),
    )

    await workflow.WORKFLOW_LOCK.acquire()
    try:
        await chainlit_app.on_message(SimpleNamespace(content="Second question?"))
    finally:
        workflow.WORKFLOW_LOCK.release()

    assert messages == [workflow.t("workflow.locked")]
    assert active_session.finalized is False
    assert active_session.tokens == {"p1": 0, "p2": 0, "p3": 0, "p4": 0}


@pytest.mark.asyncio
async def test_new_request_is_allowed_after_previous_workflow_finishes(monkeypatch):
    session = _session()
    runs = []

    async def complete(question):
        runs.append(question)
        return "completed"

    monkeypatch.setattr(workflow, "get_session", lambda: session)
    monkeypatch.setattr(workflow, "_run_locked_workflow", complete)

    await workflow.run_lakegen_workflow("First question?")
    await workflow.run_lakegen_workflow("Next question?")

    assert runs == ["First question?", "Next question?"]


def test_next_round_hint_merges_architect_and_user_feedback():
    from lakegen.pipeline.engine import next_round_hint

    assert next_round_hint("Architect says X.", "try Y") == (
        "Architect says X. User feedback: try Y"
    )
    assert next_round_hint("Architect says X.", "") == "Architect says X."
    assert next_round_hint("", "try Y") == "try Y"
    assert next_round_hint("", "") == ""


@pytest.mark.asyncio
async def test_unified_rejection_feedback_and_memory_reach_the_next_round(monkeypatch, tmp_path):
    """The chat UI used to start the round after a rejection with only the
    user's typed hint: no architect feedback, no retrieval memory. On the
    engine the architect's feedback, the kept table and the round's retrieval
    evidence all reach the next round, which no one has to approve."""
    session = _session()
    rejection = (
        "REJECT_KEYWORDS: all candidates are fishing licences\n"
        "Suggestion: water abstraction licence NI"
    )
    calls = []

    def fake_phase12_agent(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            state = kwargs["state"]
            state.rejection_keep_tables = ["kept.parquet"]
            state.retrieval_memory_events.append({
                "outcome": "insufficient_coverage", "terms": ["water", "licence"],
                "tables": ["kept.parquet"], "reason": "all candidates are fishing licences",
                "suggestion": "water abstraction licence NI",
            })
            return [], ["water", "licence"], {}, rejection, "", 0
        return ["kept.parquet"], ["water"], {}, "Selected.", "", 0

    async def approve(*_args, **_kwargs):
        return "approve"

    logged = _engine_run(monkeypatch, session, tmp_path, phase12=fake_phase12_agent, ask_choice=approve)
    monkeypatch.setattr(workflow, "phase3_generate_and_execute", lambda *a, **k: (_ for _ in ()).throw(ValueError("stop here")))
    await workflow.run_lakegen_workflow("How much water is abstracted?")

    first, second = calls
    assert first["retrieval_memory"] == ""
    assert second["hint"].startswith("The previous attempt already found tables")
    assert f"Architect feedback: {rejection}." in second["hint"]
    assert "water abstraction licence NI" in second["retrieval_memory"]
    assert second["state"].carried_tables == ["kept.parquet"]
    assert len(logged) == 1
