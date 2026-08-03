import json
from pathlib import Path

_src_dir = Path(__file__).resolve()
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent

if _src_dir.name == "src":
    BASE_DIR = _src_dir.parent
else:
    BASE_DIR = Path.cwd()

CONFIG_FILE = BASE_DIR / "config_paths.json"
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
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
LOG_DIR = BASE_DIR / paths.get("logs_dir", "logs")


def resolve_portal_tables_dir(portal: str) -> Path:
    datasets_dir = BASE_DIR / "data" / portal / "datasets"
    candidates = (
        datasets_dir / "parquets",
        datasets_dir / "parquet",
        datasets_dir / "csv",
    )
    supported_suffixes = {".parquet", ".pq", ".csv"}
    for candidate in candidates:
        if candidate.is_dir() and any(
            entry.is_file() and entry.suffix.casefold() in supported_suffixes
            for entry in candidate.iterdir()
        ):
            return candidate
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
