"""Where LakeGen writes its runtime state, resolved at call time.

Everything a run writes lives under one root, ``resolve_state_dir``:

    coding/<run id>/              the scripts the coder generates and runs
    logs/                         experiment logs, retrieval rankings, the Groq ledger
    logs/manifests/               one manifest per run
    .lakegen_jobs*/               API batch jobs (``LAKEGEN_JOB_DIR`` names the folder)
    .lakegen_keyword_memory.json  the durable keyword memory of standalone runs
    runs/<experiment>/memory.json the keyword memory of one experiment

``LOG_DIR`` and ``api.JOB_DIR`` are the same locations computed
once at import; they stay because modules and tests refer to them by name.
The functions here re-read the environment on every call, so one run can be
pointed elsewhere, which is how the test suite isolates each test.
"""

from __future__ import annotations

import json
from pathlib import Path

from lakegen.config.settings import read

_src_dir = Path(__file__).resolve()
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent

if _src_dir.name == "src":
    BASE_DIR = _src_dir.parent
else:
    BASE_DIR = Path.cwd()

CONFIG_FILE = BASE_DIR / "config_paths.json"
if CONFIG_FILE.exists():
    with open(CONFIG_FILE) as f:
        config_data = json.load(f)
else:
    config_data = {"paths": {}}

_configured = config_data.get("paths", {})

DATA_DIR = BASE_DIR / _configured.get("data_dir", "Data")
TABLES_DIR = BASE_DIR / _configured.get(
    "tables_dir",
    _configured.get("csv_dir", "data/nyc/datasets/parquets"),
)
# Backward-compatible alias for older imports.
CSV_DIR = TABLES_DIR
JSON_DIR = BASE_DIR / _configured.get("json_metadata_dir", "Data/bologna_update/metadata")
DB_PATH = BASE_DIR / _configured.get("blend_db_path", "Data/blend_index.db")
INDEXES_DIR = BASE_DIR / _configured.get("indexes_dir", "Data/indexes")


def resolve_state_dir() -> Path:
    """The root under which LakeGen writes its runtime state.

    Coding runs, logs, manifests, job folders and the keyword memory all live
    under one root: ``LAKEGEN_STATE_DIR`` when it is set, else the repository
    root, where they have always been. The test suite points the variable at
    a temporary directory so that no test touches the real state.
    """
    value = read("LAKEGEN_STATE_DIR")
    return Path(value).expanduser() if value else BASE_DIR


STATE_DIR = resolve_state_dir()
LOGS_DIR_NAME = _configured.get("logs_dir", "logs")
LOG_DIR = STATE_DIR / LOGS_DIR_NAME


def resolve_portal_tables_dir(portal: str) -> Path:
    """The lake's tables directory; the rules live in config/lakes.yaml."""
    from lakegen.config.registry import get_lake  # the registry imports BASE_DIR from here

    return get_lake(portal).tables_dir


def resolve_portal_metadata_path(portal: str) -> Path:
    """The lake's metadata catalogue (the cleaned one for the UK)."""
    from lakegen.config.registry import get_lake

    return get_lake(portal).metadata_path


def resolve_portal_dataset_statistics_path(portal: str) -> Path:
    """The lake's per-table statistics CSV."""
    from lakegen.config.registry import get_lake

    return get_lake(portal).statistics_path


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
