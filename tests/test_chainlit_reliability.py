import sys
from types import ModuleType, SimpleNamespace

import pytest

from lakegen.experiment_config import ExperimentConfig
from lakegen.tracing import HumanGate
from lakegen.ui.state import LakeGenSession, WorkflowTimedOut

# The production UI configures the optional local embedding package at import
# time. Supply a no-I/O stand-in so these workflow tests remain fully offline.
from llama_index.core.base.embeddings.base import BaseEmbedding


class _OfflineEmbedding(BaseEmbedding):
    def _get_query_embedding(self, _query):
        return []

    async def _aget_query_embedding(self, _query):
        return []

    def _get_text_embedding(self, _text):
        return []


_embedding_package = ModuleType("llama_index.embeddings")
_embedding_module = ModuleType("llama_index.embeddings.huggingface")
_embedding_module.HuggingFaceEmbedding = _OfflineEmbedding
sys.modules.setdefault("llama_index.embeddings", _embedding_package)
sys.modules.setdefault("llama_index.embeddings.huggingface", _embedding_module)

from lakegen.ui import workflow
from src import app as chainlit_app


def _session() -> LakeGenSession:
    config = ExperimentConfig(interaction_mode="human_gated")
    runtime = SimpleNamespace(
        model_name=config.model,
        experiment=config,
    )
    session = LakeGenSession(runtime=runtime, query="Question?")
    session.manifest = {"run_id": session.run_id, "resolved_config": {"core": "nyc"}}
    return session


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


@pytest.mark.parametrize(
    ("status", "phase"),
    [
        ("completed", "result"),
        ("cancelled", "discovery"),
        ("failed", "initialization"),
        ("failed", "discovery"),
        ("timed_out", "discovery"),
    ],
)
def test_chainlit_finalizer_saves_once_with_partial_phases(
    status, phase, monkeypatch
):
    session = _session()
    session.phase = phase
    session.intervention_recorder.record_hint(
        phase="discovery",
        gate=HumanGate.DATASET_HINT,
        provided=True,
        elapsed_seconds=0.2,
    )
    logged = []
    monkeypatch.setattr(
        workflow, "save_experiment_log", lambda **kwargs: logged.append(kwargs)
    )

    workflow._finalize_run(session, status, "safe failure" if status == "failed" else "")
    workflow._finalize_run(session, status)

    assert len(logged) == 1
    assert logged[0]["status"] == status
    trace = logged[0]["extra_fields"]["RUN_TRACE_JSON"]
    assert trace["phase_reached"] == phase
    assert len(trace["human_interventions"]) == 1
    if phase in {"initialization", "discovery"}:
        assert trace["code"] is None
        assert trace["execution_outcome"]["raw_result"] is None
        assert trace["llm_calls"][1]["total_tokens"] == 0
        assert trace["llm_calls"][2]["total_tokens"] == 0


@pytest.mark.parametrize("phase", ["initialization", "discovery"])
@pytest.mark.asyncio
async def test_workflow_exception_is_logged_without_masking_original(
    phase, monkeypatch
):
    session = _session()
    logged = []

    async def fail(_question):
        session.phase = phase
        raise ValueError(f"{phase} failed")

    monkeypatch.setattr(workflow, "get_session", lambda: session)
    monkeypatch.setattr(workflow, "_run_locked_workflow", fail)
    monkeypatch.setattr(
        workflow, "save_experiment_log", lambda **kwargs: logged.append(kwargs)
    )

    with pytest.raises(ValueError, match=f"{phase} failed"):
        await workflow.run_lakegen_workflow("Question?")

    assert len(logged) == 1
    assert logged[0]["status"] == "failed"
    assert f"ValueError: {phase} failed" in logged[0]["error"]


@pytest.mark.asyncio
async def test_workflow_timeout_is_logged_once(monkeypatch):
    session = _session()
    logged = []

    async def time_out(_question):
        session.phase = "discovery"
        raise WorkflowTimedOut("dataset_hint expired")

    monkeypatch.setattr(workflow, "get_session", lambda: session)
    monkeypatch.setattr(workflow, "_run_locked_workflow", time_out)
    monkeypatch.setattr(
        workflow, "save_experiment_log", lambda **kwargs: logged.append(kwargs)
    )

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
        workflow, "create_manifest", lambda *_a, **_k: pytest.fail("manifest created")
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
    monkeypatch.setattr(workflow, "_finalize_run", lambda *_a, **_k: None)

    await workflow.run_lakegen_workflow("First question?")
    await workflow.run_lakegen_workflow("Next question?")

    assert runs == ["First question?", "Next question?"]
