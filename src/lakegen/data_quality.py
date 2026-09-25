"""Cheap, LLM-free per-column data-quality evidence for the coder prompt and
error repair. Every value reported here is observed directly in the sampled
data, never inferred or guessed, so the coder can clean columns using literal
tokens instead of inventing sentinel values."""

from __future__ import annotations

import pandas as pd

_SENTINEL_TOKENS = {
    "", "n/a", "na", "n.a.", "none", "null", "nil", "-", "--", "unknown",
    "unspecified", "not available", "not applicable", "#n/a", "nan",
}
# Numeric codes are only *possibly* missing-value markers: 999 m3/day or -1 can
# be genuine measurements, so they are reported with the column's range and
# left to the coder's judgement instead of being prescribed as cleaning.
_AMBIGUOUS_NUMERIC_CODES = {"999", "9999", "-1", "-999"}

_NUMERIC_NOISE_PATTERN = r"[,$€£%]|(?<=\d)\s(?=\d)"


def _implausible_codes(non_blank: pd.Series, codes: dict[str, int]) -> dict[str, int]:
    """Keep only numeric codes that sit outside the column's ordinary range.

    A 999 among volumes ranging up to millions is an ordinary value; a -1 in a
    non-negative column, or a 9999 far above every other value, is not.
    """

    if not codes:
        return {}
    numbers = pd.to_numeric(non_blank, errors="coerce").dropna()
    others = numbers[~numbers.astype(str).str.removesuffix(".0").isin(codes)]
    if len(others) < 5:
        return {}
    low, high = others.quantile(0.01), others.quantile(0.99)
    return {
        code: count for code, count in codes.items()
        if float(code) < low or float(code) > high
    }


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
    numeric_code_counts: dict[str, int] = {}
    if not non_blank.empty:
        for value, count in non_blank.str.casefold().value_counts().items():
            normalized = value[:-2] if value.endswith(".0") else value
            if value in _SENTINEL_TOKENS:
                sentinel_counts[value] = int(count)
            elif normalized in _AMBIGUOUS_NUMERIC_CODES:
                numeric_code_counts[normalized] = int(count)

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

    numeric_code_counts = _implausible_codes(non_blank, numeric_code_counts)
    if (
        not blank_count and not sentinel_counts and not numeric_code_counts
        and not numeric_format_examples
    ):
        return None

    evidence: dict[str, object] = {"sampled_rows": total}
    if blank_count:
        evidence["blank_or_null_count"] = blank_count
        evidence["blank_or_null_pct"] = round(100 * blank_count / total, 1)
    if sentinel_counts:
        evidence["sentinel_tokens_observed"] = sentinel_counts
    if numeric_format_examples:
        evidence["numeric_format_examples"] = numeric_format_examples
    if numeric_code_counts:
        evidence["possible_missing_codes"] = numeric_code_counts
        numbers = pd.to_numeric(non_blank, errors="coerce").dropna()
        if not numbers.empty:
            evidence["numeric_median"] = float(numbers.median())
            evidence["numeric_max"] = float(numbers.max())
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


_DATE_LIKE = r"(?:19|20)\d{2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}"


def snapshot_stamp_years(path, *, sample_rows: int = 500) -> set[str]:
    """Years of date-like columns holding one non-null value in every row.

    Such a column stamps the whole file with its snapshot date, so a question
    "as of" that date is satisfied without any row filter.
    """

    from lakegen.core.table_io import read_table

    try:
        sample = read_table(path, nrows=sample_rows)
        constant = [
            column for column in sample.columns
            if sample[column].notna().all() and sample[column].nunique() == 1
            and pd.Series([str(sample[column].iloc[0])]).str.contains(_DATE_LIKE).iloc[0]
        ]
        if not constant:
            return set()
        full = read_table(path, columns=constant)
    except Exception:
        return set()
    years: set[str] = set()
    for column in full.columns:
        values = full[column]
        if values.notna().all() and values.nunique() == 1:
            years.update(pd.Series([str(values.iloc[0])]).str.findall(r"(?:19|20)\d{2}").iloc[0])
    return years
