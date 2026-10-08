"""ThreadSafeStepEmitter: cross-thread bridging to Chainlit's async cl.Step,
verified against a fake Step (recording lifecycle calls) driven from a real
background thread -- not just same-thread calls, since the whole point of
call_soon_threadsafe is safety across threads."""

import asyncio
import contextvars
import threading

import pytest

from lakegen.ui import live_steps

# Simulates chainlit.context.context_var: a ContextVar set once on the main
# event-loop thread (e.g. by Chainlit's own request handling) that must
# still be visible inside the task ThreadSafeStepEmitter creates later, even
# though that task is scheduled from a background thread that never set it
# itself. This is exactly what chainlit.context.get_context() depends on,
# and exactly what a missing `context=` on call_soon_threadsafe breaks.
_fake_chainlit_context_var: "contextvars.ContextVar[str]" = contextvars.ContextVar("fake_chainlit")


class ContextCheckingStep:
    """Like FakeStep, but raises the same shape of error chainlit.Step
    raises (a LookupError-derived exception) if the simulated Chainlit
    context var isn't visible at construction time -- reproducing the exact
    failure this test guards against, without importing real chainlit
    internals."""

    events: list[dict] = []

    def __init__(self, name, type="tool"):
        _fake_chainlit_context_var.get()  # raises LookupError if unset, like chainlit's own context lookup
        self.name = name
        self.type = type
        self.parent_id = None
        self.output = None
        self.is_error = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        ContextCheckingStep.events.append({"name": self.name, "output": self.output})
        return False


class FakeStep:
    """Records what a real cl.Step would have done, as an async context
    manager with the same shape (`.output`, `.is_error`, `.parent_id`)."""

    events: list[dict] = []

    def __init__(self, name, type="tool"):
        self.name = name
        self.type = type
        self.parent_id = None
        self.output = None
        self.is_error = False
        self.id = f"step-{len(FakeStep.events)}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        FakeStep.events.append({
            "name": self.name, "type": self.type, "parent_id": self.parent_id,
            "output": self.output, "is_error": self.is_error,
        })
        return False


@pytest.fixture(autouse=True)
def _fake_cl_step(monkeypatch):
    FakeStep.events = []
    monkeypatch.setattr(live_steps.cl, "Step", FakeStep)
    yield


@pytest.mark.asyncio
async def test_instant_records_name_and_output():
    emitter = live_steps.ThreadSafeStepEmitter()
    emitter.instant("s1", "Solr query", "keywords=['a'], q_op=AND, 5 hits")
    await emitter.aclose()

    assert len(FakeStep.events) == 1
    assert FakeStep.events[0]["name"] == "Solr query"
    assert FakeStep.events[0]["output"] == "keywords=['a'], q_op=AND, 5 hits"


@pytest.mark.asyncio
async def test_start_finish_pair_from_a_real_background_thread():
    emitter = live_steps.ThreadSafeStepEmitter()

    def worker():
        emitter.start("cand-1", "Inspecting: table.parquet")
        emitter.finish("cand-1", "VERDICT: RELEVANT -- matches the question.")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    await emitter.aclose()

    assert len(FakeStep.events) == 1
    assert FakeStep.events[0]["name"] == "Inspecting: table.parquet"
    assert "RELEVANT" in FakeStep.events[0]["output"]


@pytest.mark.asyncio
async def test_multiple_started_steps_from_different_threads_all_complete():
    emitter = live_steps.ThreadSafeStepEmitter()
    n = 5

    def worker(i):
        emitter.start(f"cand-{i}", f"Inspecting: table{i}.parquet")
        emitter.finish(f"cand-{i}", f"VERDICT: RELEVANT ({i})")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    await emitter.aclose()

    assert len(FakeStep.events) == n
    names = {e["name"] for e in FakeStep.events}
    assert names == {f"Inspecting: table{i}.parquet" for i in range(n)}


@pytest.mark.asyncio
async def test_parent_step_id_is_propagated():
    class FakeParent:
        id = "parent-123"

    emitter = live_steps.ThreadSafeStepEmitter(parent_step=FakeParent())
    emitter.instant("s1", "Solr query", "ok")
    await emitter.aclose()

    assert FakeStep.events[0]["parent_id"] == "parent-123"


@pytest.mark.asyncio
async def test_aclose_reports_unfinished_steps_instead_of_hanging():
    emitter = live_steps.ThreadSafeStepEmitter()
    emitter.start("cand-1", "Inspecting: table.parquet")
    # No finish() call -- aclose() must not hang forever.
    await asyncio.wait_for(emitter.aclose(), timeout=2)

    assert len(FakeStep.events) == 1
    assert FakeStep.events[0]["output"] == "(no result reported)"


@pytest.mark.asyncio
async def test_step_creation_sees_the_context_set_on_the_constructing_thread(monkeypatch):
    """Regression test for the real bug this session hit live: a Solr-query
    step opened via `instant()` from a background thread (the retrieval
    observer, called from inside _search running under cl.make_async) must
    still see whatever contextvars.Context was active when
    ThreadSafeStepEmitter was constructed -- even though the thread that
    later calls instant()/start()/finish() never set that var itself."""
    monkeypatch.setattr(live_steps.cl, "Step", ContextCheckingStep)
    ContextCheckingStep.events = []

    token = _fake_chainlit_context_var.set("session-abc")
    try:
        emitter = live_steps.ThreadSafeStepEmitter()  # constructed WITH the var set
    finally:
        _fake_chainlit_context_var.reset(token)
    # From here on, the var is unset again on THIS (main) context -- exactly
    # like a background thread that never had it in the first place.
    assert _fake_chainlit_context_var not in contextvars.copy_context()

    def worker():
        # A real background thread: contextvars are thread-local, so this
        # thread's own top-level context never had the var set at all.
        emitter.instant("s1", "Solr query", "keywords=['a'], 5 hits")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    await emitter.aclose()  # would raise LookupError here if context weren't pinned

    assert len(ContextCheckingStep.events) == 1
    assert ContextCheckingStep.events[0]["name"] == "Solr query"
