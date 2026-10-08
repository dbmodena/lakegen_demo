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

paths = config_data.get("paths", {})

DATA_DIR = BASE_DIR / paths.get("data_dir", "Data")
TABLES_DIR = BASE_DIR / paths.get(
    "tables_dir",
    paths.get("csv_dir", "data/nyc/datasets/parquets"),
)
# Backward-compatible alias for older imports.
CSV_DIR = TABLES_DIR
JSON_DIR = BASE_DIR / paths.get("json_metadata_dir", "Data/bologna_update/metadata")
DB_PATH = BASE_DIR / paths.get("blend_db_path", "Data/blend_index.db")
INDEXES_DIR = BASE_DIR / paths.get("indexes_dir", "Data/indexes")


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
LOGS_DIR_NAME = paths.get("logs_dir", "logs")
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
