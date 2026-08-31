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
    portal_dir = BASE_DIR / "data" / portal
    dataset_roots = (
        portal_dir / "clean_datasets",
        portal_dir / "datasets",
    ) if portal.casefold() == "uk" else (portal_dir / "datasets",)
    candidates = tuple(
        root / name
        for root in dataset_roots
        for name in ("parquets", "parquet", "csv")
    )
    supported_suffixes = {".parquet", ".pq", ".csv"}
    for candidate in candidates:
        if candidate.is_dir() and any(
            entry.is_file() and entry.suffix.casefold() in supported_suffixes
            for entry in candidate.iterdir()
        ):
            return candidate
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def resolve_portal_metadata_path(portal: str) -> Path:
    """Return the filtered metadata catalog, preferring cleaned UK metadata."""
    metadata_dir = BASE_DIR / "data" / portal / "metadata"
    names = (
        "metadata_retrieved_cleaned.json",
        "metadata_retrieved_only.json",
        "metadata.json",
    ) if portal.casefold() == "uk" else (
        "metadata_retrieved_only.json",
        "metadata.json",
    )
    return next(
        (metadata_dir / name for name in names if (metadata_dir / name).is_file()),
        metadata_dir / names[0],
    )


def resolve_portal_dataset_statistics_path(portal: str) -> Path:
    """Return the cleaned per-table statistics CSV for a portal."""
    return BASE_DIR / "data" / portal / "metadata" / "datasets_metadata.csv"
