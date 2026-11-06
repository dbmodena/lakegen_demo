"""Where LakeGen writes its runtime state, resolved at call time.

Everything a run writes lives under one root, ``resolve_state_dir``:

    coding/<run id>/              the scripts the coder generates and runs
    logs/                         experiment logs, retrieval rankings, the Groq ledger
    logs/manifests/               one manifest per run
    .lakegen_jobs*/               API batch jobs (``LAKEGEN_JOB_DIR`` names the folder)

``LOG_DIR`` and ``api.JOB_DIR`` are the same locations computed
once at import; they stay because modules and tests refer to them by name.
The functions here re-read the environment on every call, so one run can be
pointed elsewhere, which is how the test suite isolates each test.
"""

from __future__ import annotations

from pathlib import Path

from lakegen.config.settings import read

_src_dir = Path(__file__).resolve()
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent

if _src_dir.name == "src":
    BASE_DIR = _src_dir.parent
else:
    BASE_DIR = Path.cwd()


def resolve_state_dir() -> Path:
    """The root under which LakeGen writes its runtime state.

    Coding runs, logs, manifests, job folders and the keyword memory all live
    under one root: ``LAKEGEN_STATE_DIR`` when it is set, else the repository
    root, where they have always been. The test suite points the variable at
    a temporary directory so that no test touches the real state.

    Returns:
        The state root.
    """
    value = read("LAKEGEN_STATE_DIR")
    return Path(value).expanduser() if value else BASE_DIR


STATE_DIR = resolve_state_dir()
LOGS_DIR_NAME = "logs"
LOG_DIR = STATE_DIR / LOGS_DIR_NAME


def resolve_portal_tables_dir(portal: str) -> Path:
    """The lake's tables directory; the rules live in config/lakes.yaml."""
    from lakegen.config.registry import get_lake  # the registry imports BASE_DIR from here

    return get_lake(portal).tables_dir


def resolve_portal_metadata_path(portal: str) -> Path:
    """The lake's metadata catalogue (the cleaned one for the UK)."""
    from lakegen.config.registry import get_lake

    return get_lake(portal).metadata_path


DEFAULT_JOBS_DIR_NAME = ".lakegen_jobs"


def log_dir() -> Path:
    """The logs folder under the state root."""
    return resolve_state_dir() / LOGS_DIR_NAME


def manifests_dir() -> Path:
    """The run manifests' folder, inside the logs folder."""
    return log_dir() / "manifests"


def coding_dir() -> Path:
    """The folder of the coder's generated scripts, one subfolder per run."""
    return resolve_state_dir() / "coding"


def jobs_dir() -> Path:
    """The API's job folder. ``LAKEGEN_JOB_DIR`` gives an API instance its own."""
    return resolve_state_dir() / (read("LAKEGEN_JOB_DIR") or DEFAULT_JOBS_DIR_NAME)


def runs_dir() -> Path:
    """The run store: what each experiment submitted and where its results are."""
    return resolve_state_dir() / "runs"


def services_dir() -> Path:
    """Logs and pids of the services ``python -m experiment_runner services up`` started."""
    return resolve_state_dir() / "services"


def pneuma_index_dir() -> Path:
    """The folder of the Pneuma indexes: ``LAKEGEN_PNEUMA_INDEX_DIR``, else ``pneuma-out``."""
    value = read("LAKEGEN_PNEUMA_INDEX_DIR")
    return Path(value).expanduser() if value else BASE_DIR / "pneuma-out"


def prompts_dir() -> Path:
    """The folder of the agents' prompt templates: ``LAKEGEN_PROMPTS_DIR``, else ``prompts`` at the root."""
    value = read("LAKEGEN_PROMPTS_DIR")
    return Path(value).expanduser() if value else BASE_DIR / "prompts"
