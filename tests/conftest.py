"""Test-suite configuration: the quarantine and a hermetic runtime state.

``needs_testing_grounds`` holds tests written against code that exists only on
the ``testing-grounds`` branch (modules such as ``lakegen.code_judge`` or the
parallel-inspection and PRP-rerank features). They are kept so that work can
be ported later, but they are not collected here.

Hermetic state: every run writes under a temporary root. ``LAKEGEN_STATE_DIR``
is set in ``pytest_configure``, before any ``lakegen`` import, because
``core.config.LOG_DIR`` and ``api.JOB_DIR`` are computed at import time; the
autouse fixture then points the call-time paths (``lakegen.core.paths``) at
each test's own directory and resets the module-level caches that hold a
path. The environment is also cleared of every LakeGen and provider variable,
and ``.env`` is not loaded, so no test can reach a real service with real keys.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

collect_ignore = ["needs_testing_grounds"]

_SCRUBBED_PREFIXES = (
    "LAKEGEN_", "GROQ_", "GROP_", "TOGETHER", "OPENAI_", "OCI_", "JEV_", "TYPESAFE_",
)
_state_root: str | None = None


def _no_dotenv(*args, **kwargs) -> bool:
    """Stands in for ``dotenv.load_dotenv``, which ``lakegen.core.bootstrap`` calls at import."""
    return False


def pytest_configure(config: pytest.Config) -> None:
    global _state_root
    for name in list(os.environ):
        if name.startswith(_SCRUBBED_PREFIXES):
            del os.environ[name]
    _state_root = tempfile.mkdtemp(prefix="lakegen-tests-")
    os.environ["LAKEGEN_STATE_DIR"] = _state_root
    import dotenv

    dotenv.load_dotenv = _no_dotenv


def pytest_unconfigure(config: pytest.Config) -> None:
    if _state_root:
        shutil.rmtree(_state_root, ignore_errors=True)


@pytest.fixture(autouse=True)
def _hermetic_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKEGEN_STATE_DIR", str(tmp_path / "state"))
    resources = sys.modules.get("lakegen.core.resources")
    if resources is not None:
        resources.get_solr.cache_clear()
        resources.get_retrieval_run_logger.cache_clear()
    groq = sys.modules.get("lakegen.core.groq_offload")
    if groq is not None:
        monkeypatch.setattr(groq, "_offload", None)
    yield
