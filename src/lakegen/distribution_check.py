"""Deterministic per-column data-distribution profiling, styled after OrQa's
ColumnStatistics -- surfaces a column's real numeric range/encoding shape
BEFORE code is written, catching the FY2016/17 bug class (a filter literal
that structurally can't match an encoded column) proactively rather than
only reactively (see `value_grounding.check_value_grounding`, which catches
the same shape of problem after code is generated).

No LLM; pandas only. Validated over four rounds of live 100-question
scratchpad testing before being ported here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# Consecutive fiscal/school-year span codes: "200001".."201617" (6 digits,
# YYYY + next-YY). This is the exact encoding that broke the motivating bug
# (df['Financial Year'] == 2016 against a column that only ever holds
# values like 201617).
_SPAN_6DIGIT_RE = re.compile(r"^(\d{4})(\d{2})$")


@dataclass
class ColumnDistribution:
    name: str
    dtype: str
    cardinality: int
    null_ratio: float
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_mean: float | None = None
    numeric_outliers: list = field(default_factory=list)
    pinned_extreme: str | None = None  # e.g. "312/900 rows pinned at max=90"
    top_values: list = field(default_factory=list)
    encoded_span: dict | None = None  # {"kind": "fiscal_year_span", "example": "201617", ...}


def _detect_encoded_span(series: pd.Series) -> dict | None:
    """Detect a column whose values are consecutive fiscal/school-year span
    codes (e.g. 200001, 200102, ..., 201617) rather than plain years. A bare
    4-digit-year filter against such a column matches nothing."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    as_str = (
        non_null.astype("int64", errors="ignore").astype(str)
        if pd.api.types.is_numeric_dtype(series) else non_null.astype(str)
    )
    matches = as_str.str.strip().str.fullmatch(_SPAN_6DIGIT_RE)
    hit_ratio = matches.fillna(False).mean()
    if hit_ratio < 0.9:
        return None
    parsed = as_str[matches.fillna(False)].str.extract(_SPAN_6DIGIT_RE)
    if parsed.empty:
        return None
    starts = parsed[0].astype(int)
    ends = parsed[1].astype(int)
    # Consecutive-span sanity: end = (start+1) mod 100, for most rows.
    consecutive_ratio = (ends == (starts + 1) % 100).mean()
    if consecutive_ratio < 0.8:
        return None
    example = str(as_str[matches.fillna(False)].iloc[0])
    return {
        "kind": "fiscal_year_span",
        "example": example,
        "start_year_min": int(starts.min()),
        "start_year_max": int(starts.max()),
    }


def profile_column_distribution(series: pd.Series, name: str) -> ColumnDistribution:
    dtype = str(series.dtype)
    n = len(series)
    null_ratio = float(series.isna().mean()) if n else 0.0
    cardinality = int(series.nunique(dropna=True))
    dist = ColumnDistribution(name=name, dtype=dtype, cardinality=cardinality, null_ratio=round(null_ratio, 4))

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if not non_null.empty:
            dist.numeric_min = float(non_null.min())
            dist.numeric_max = float(non_null.max())
            dist.numeric_mean = round(float(non_null.mean()), 4)
            q1, q3 = non_null.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = non_null[(non_null < lo) | (non_null > hi)]
                if not outliers.empty:
                    dist.numeric_outliers = sorted(outliers.unique().tolist())[:5]
            # Pinned-extreme: a value AT min or max repeating far more than
            # the spread predicts (top/bottom-coding).
            for label, extreme in (("min", dist.numeric_min), ("max", dist.numeric_max)):
                count = int((non_null == extreme).sum())
                if len(non_null) >= 20 and count / len(non_null) > 0.15:
                    dist.pinned_extreme = f"{count}/{len(non_null)} rows pinned at {label}={extreme}"
                    break
        span = _detect_encoded_span(series)
        if span:
            dist.encoded_span = span
    else:
        counts = series.dropna().astype(str).value_counts().head(8)
        dist.top_values = [(str(v), int(c)) for v, c in counts.items()]
        span = _detect_encoded_span(series)
        if span:
            dist.encoded_span = span

    return dist


def profile_table_distribution(df: pd.DataFrame, columns: list[str] | None = None) -> list[ColumnDistribution]:
    cols = columns or list(df.columns)
    return [profile_column_distribution(df[c], c) for c in cols if c in df.columns]


def format_distribution_report(distributions: list[ColumnDistribution]) -> str:
    """Structured text for the coder prompt, styled after OrQa's
    query_planner.md DATA QUALITY section -- explicit interpretive prose,
    not just raw stats."""
    lines = ["### DATA DISTRIBUTION (real values observed in this table, not a schema guess)"]
    for d in distributions:
        header = f"- {d.name} ({d.dtype}): cardinality={d.cardinality}, null_ratio={d.null_ratio}"
        lines.append(header)
        if d.encoded_span:
            lines.append(
                f"    ENCODED VALUE WARNING: values look like consecutive fiscal/school-year "
                f"span codes (e.g. {d.encoded_span['example']!r}, years "
                f"{d.encoded_span['start_year_min']}-{d.encoded_span['start_year_max']} "
                f"encoded as YYYY+next-YY). A bare 4-digit year filter (e.g. == 2016) will "
                f"match ZERO rows -- the correct literal for year Y is the 6-digit code "
                f"'{{Y}}{{(Y+1)%100:02d}}', e.g. 2016 -> '201617'."
            )
        if d.numeric_min is not None:
            lines.append(f"    range: min={d.numeric_min}, max={d.numeric_max}, mean={d.numeric_mean}")
        if d.numeric_outliers:
            lines.append(f"    outliers (Tukey IQR, far from the column's own spread): {d.numeric_outliers}")
        if d.pinned_extreme:
            lines.append(f"    PINNED EXTREME: {d.pinned_extreme} -- likely a sentinel/cap, not a real observation")
        if d.top_values:
            lines.append(f"    top values: {d.top_values}")
    return "\n".join(lines)
