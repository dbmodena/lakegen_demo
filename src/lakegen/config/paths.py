"""Where LakeGen writes its runtime state, resolved at call time.

Everything a run writes lives under one root, ``core.config.resolve_state_dir``:

    coding/<run id>/              the scripts the coder generates and runs
    logs/                         experiment logs, retrieval rankings, the Groq ledger
    logs/manifests/               one manifest per run
    .lakegen_jobs*/               API batch jobs (``LAKEGEN_JOB_DIR`` names the folder)
    .lakegen_keyword_memory.json  the durable keyword memory of standalone runs
    runs/<experiment>/memory.json the keyword memory of one experiment

``core.config.LOG_DIR`` and ``api.JOB_DIR`` are the same locations computed
once at import; they stay because modules and tests refer to them by name.
The functions here re-read the environment on every call, so one run can be
pointed elsewhere, which is how the test suite isolates each test.
"""

from __future__ import annotations

from pathlib import Path

from lakegen.config.locations import LOGS_DIR_NAME, resolve_state_dir
from lakegen.config.settings import read

DEFAULT_JOBS_DIR_NAME = ".lakegen_jobs"
KEYWORD_MEMORY_FILE = ".lakegen_keyword_memory.json"


def state_dir() -> Path:
    return resolve_state_dir()


def log_dir() -> Path:
    return resolve_state_dir() / LOGS_DIR_NAME


def manifests_dir() -> Path:
    return log_dir() / "manifests"


def coding_dir() -> Path:
    return resolve_state_dir() / "coding"


def jobs_dir() -> Path:
    """The API's job folder. ``LAKEGEN_JOB_DIR`` gives an API instance its own."""
    return resolve_state_dir() / (read("LAKEGEN_JOB_DIR") or DEFAULT_JOBS_DIR_NAME)


def runs_dir() -> Path:
    """The run store: what each experiment submitted and where its results are."""
    return resolve_state_dir() / "runs"


def services_dir() -> Path:
    """Logs and pids of the services ``lakegen services up`` started."""
    return resolve_state_dir() / "services"


def keyword_memory_path(experiment_id: str | None = None) -> Path:
    """The keyword memory of an experiment, or the lake-wide file of standalone runs.

    Each experiment remembers on its own (``runs/<experiment>/memory.json``),
    so concurrent experiments neither share bans nor contend for one file;
    a run outside an experiment (the default id) keeps today's file.
    """
    if experiment_id and experiment_id != "default":
        return runs_dir() / experiment_id / "memory.json"
    return resolve_state_dir() / KEYWORD_MEMORY_FILE
