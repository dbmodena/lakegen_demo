"""Cheap, LLM-free per-column data-quality evidence for the coder prompt and
error repair. Every value reported here is observed directly in the sampled
data, never inferred or guessed, so the coder can clean columns using literal
tokens instead of inventing sentinel values."""

from __future__ import annotations

import pandas as pd

_SENTINEL_TOKENS = {
    "", "n/a", "na", "n.a.", "none", "null", "nil", "-", "--", "unknown",
    "unspecified", "not available", "not applicable", "#n/a", "nan",
    "999", "9999", "-1", "-999",
}

_NUMERIC_NOISE_PATTERN = r"[,$€£%]|(?<=\d)\s(?=\d)"


def profile_column_quality(series: pd.Series, *, sample_size: int = 2000) -> dict | None:
    """Return literal cleaning evidence for one column, or None when it looks clean."""
    sample = series.head(sample_size)
    total = len(sample)
    if total == 0:
        return None

    as_str = sample.astype(str).str.strip()
    blank_mask = sample.isna() | (as_str == "")
    blank_count = int(blank_mask.sum())

    non_blank = as_str[~blank_mask]
    sentinel_counts: dict[str, int] = {}
    if not non_blank.empty:
        for value, count in non_blank.str.casefold().value_counts().items():
            if value in _SENTINEL_TOKENS:
                sentinel_counts[value] = int(count)

    numeric_format_examples: list[str] = []
    if not pd.api.types.is_numeric_dtype(sample) and not non_blank.empty:
        try:
            candidates = non_blank[non_blank.str.contains(r"\d", regex=True, na=False)]
            noisy = candidates[
                candidates.str.contains(_NUMERIC_NOISE_PATTERN, regex=True, na=False)
            ]
            if not noisy.empty:
                cleaned = pd.to_numeric(
                    noisy.str.replace(_NUMERIC_NOISE_PATTERN, "", regex=True),
                    errors="coerce",
                )
                numeric_format_examples = (
                    noisy[cleaned.notna()].drop_duplicates().head(5).tolist()
                )
        except (TypeError, ValueError):
            numeric_format_examples = []

    if not blank_count and not sentinel_counts and not numeric_format_examples:
        return None

    evidence: dict[str, object] = {"sampled_rows": total}
    if blank_count:
        evidence["blank_or_null_count"] = blank_count
        evidence["blank_or_null_pct"] = round(100 * blank_count / total, 1)
    if sentinel_counts:
        evidence["sentinel_tokens_observed"] = sentinel_counts
    if numeric_format_examples:
        evidence["numeric_format_examples"] = numeric_format_examples
    return evidence


def profile_table_quality(df: pd.DataFrame, *, max_columns: int = 25) -> dict[str, dict]:
    """Per-column literal cleaning evidence, skipping columns that look clean."""
    findings: dict[str, dict] = {}
    for column in list(df.columns)[:max_columns]:
        try:
            result = profile_column_quality(df[column])
        except Exception:
            continue
        if result:
            findings[str(column)] = result
    return findings
