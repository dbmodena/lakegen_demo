"""Shared setup for the pair-evidence evaluations: LakeGen tables and COMA candidate pairs."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lakegen.agent_tools.schema_matching import verify_pair_schema  # noqa: E402
from lakegen.core.config import resolve_portal_tables_dir  # noqa: E402
from lakegen.core.table_io import read_table  # noqa: E402

PORTAL = os.environ.get("PORTAL", "uk")
TABLE_DIR = resolve_portal_tables_dir(PORTAL)
# Union evidence only needs column names and value samples, so big tables are row-sampled at load
# (NYC has tables of up to 19M rows). Join evidence uses whole tables up to this cap as well.
MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))

_cache: dict[str, pd.DataFrame] = {}


def table(table_id: str) -> pd.DataFrame:
    if table_id not in _cache:
        frame = read_table(TABLE_DIR / (table_id + ".parquet"))
        if len(frame) > MAX_ROWS:
            frame = frame.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
        _cache[table_id] = frame
    return _cache[table_id]


def coma_matches(q: pd.DataFrame, r: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Name-similar column pairs, best first: what the tool starts from."""
    return verify_pair_schema(q, r)["matches"]  # type: ignore[return-value]
