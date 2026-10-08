"""Table selection with TypeSafe's Jev decision model.

One request per selection: the question plus every candidate, each described by
its catalogue metadata and a small profile of its data, with one yes/no
question per candidate and a choice of the primary table. Asking about each
table in a separate request made look-alike tables pass (experiments/jev_selector).
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lakegen.config.settings import jev_api_key
from lakegen.core.types import SolrMetadata
from lakegen.experiment_config import DiscoveryConfig

API_URL = "https://api.typesafe.ai/v1/systemone"
KEY_NAMES = ("TYPESAFE_API_KEY", "JEV_API_KEY")
# Jev allows 32k tokens for the state plus the longest question.
MAX_REQUEST_BYTES = 80_000
PROFILE_ROW_LIMIT = 200_000
PROFILE_COLUMNS = 40
TRACE_HEADER = "--- Jev Table Selector ---"

NEEDED_CRITERIA = {
    "true": (
        "The table holds data the answer is computed from: the entities, measures and the "
        "year, period or edition the question asks about. Also true when the question needs "
        "this table together with another one (to join, compare or combine them)."
    ),
    "false": (
        "The table is only on a related topic; or it is another version of a needed table "
        "that covers a different year, period, election or edition than the question asks "
        "about; or it holds a different kind of figure than asked (planned or voted amounts "
        "when the question asks about actual or executed ones, or the reverse); or its "
        "columns cannot provide what the question asks."
    ),
}

YEAR = r"(?:19[5-9]\d|20[0-4]\d)"
_RANGE = rf"({YEAR})\s*(?:-|–|—|‑|/|à|au|al|a|to)\s*({YEAR})"
_SINCE = rf"(?:à\s+partir\s+de|a\s+partir\s+de|depuis|desde|since|dal)\s+({YEAR})"
_YEAR_COLUMN_TOKENS = {
    "annee", "année", "an", "year", "exercice", "anno", "any", "ano", "año",
    "ejercicio", "exercici", "millesime", "millésime", "periode", "période", "anyo", "anio",
}


class JevSelectorError(RuntimeError):
    """Jev could not produce a selection (no key, API failure, malformed answer)."""


@dataclass(frozen=True)
class JevSelection:
    selected: list[str]
    reasoning: str
    trace: str
    scores: dict[str, float]
    input_tokens: int


def _years_in_text(text: str, current_year: int) -> set[int]:
    years: set[int] = set()
    for a, b in re.findall(_RANGE, text or ""):
        lo, hi = sorted((int(a), int(b)))
        if hi - lo <= 40:
            years.update(range(lo, hi + 1))
    for a in re.findall(_SINCE, text or "", flags=re.I):
        years.update(range(int(a), current_year + 1))
    years.update(int(y) for y in re.findall(rf"(?<!\d){YEAR}(?!\d)", text or ""))
    return years


def _compact_years(years: list[int]) -> str:
    if not years:
        return "unknown"
    runs: list[str] = []
    start = prev = years[0]
    for year in years[1:] + [None]:
        if year is not None and year == prev + 1:
            prev = year
            continue
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        if year is not None:
            start = prev = year
    return ", ".join(runs)


_profile_cache: dict[tuple[str, int], dict[str, Any]] = {}
_profile_lock = threading.Lock()


def profile_table(path: Path) -> dict[str, Any]:
    """Row count, year coverage and a few values per column, from the table itself."""
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return {}
    with _profile_lock:
        if key in _profile_cache:
            return _profile_cache[key]
    import duckdb

    quoted = str(path).replace("'", "''")
    source = f"read_parquet('{quoted}')" if path.suffix == ".parquet" else f"read_csv_auto('{quoted}')"
    profile: dict[str, Any] = {}
    try:
        con = duckdb.connect()
        names = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
        selected = ", ".join('"' + n.replace('"', '""') + '"' for n in names[:PROFILE_COLUMNS])
        con.execute(f"CREATE TABLE t AS SELECT {selected} FROM {source} LIMIT {PROFILE_ROW_LIMIT + 1}")
        rows = con.execute("SELECT count(*) FROM t").fetchone()[0]
        years: set[int] = set()
        samples: dict[str, str] = {}
        for name, dtype, *_ in con.execute("DESCRIBE t").fetchall():
            column = '"' + name.replace('"', '""') + '"'
            if set(re.split(r"[^a-zà-ÿ]+", name.lower())) & _YEAR_COLUMN_TOKENS:
                for (value,) in con.execute(
                    f"SELECT DISTINCT CAST({column} AS VARCHAR) FROM t WHERE {column} IS NOT NULL LIMIT 500"
                ).fetchall():
                    match = re.match(rf"^\s*({YEAR})", value or "")
                    if match:
                        years.add(int(match.group(1)))
            if any(kind in dtype.upper() for kind in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "DATE", "TIMESTAMP")):
                lo, hi = con.execute(f"SELECT min({column}), max({column}) FROM t").fetchone()
                if lo is not None:
                    samples[name] = f"range {lo} .. {hi}"
            else:
                values = con.execute(
                    f"SELECT CAST({column} AS VARCHAR), count(*) c FROM t WHERE {column} IS NOT NULL "
                    "GROUP BY 1 ORDER BY c DESC LIMIT 3"
                ).fetchall()
                if values:
                    samples[name] = " | ".join((value or "")[:50] for value, _ in values)
        con.close()
        profile = {
            "row_count": rows if rows <= PROFILE_ROW_LIMIT else f"more than {PROFILE_ROW_LIMIT}",
            "years": sorted(years),
            "samples": samples,
        }
    except Exception as exc:  # malformed files exist in real portals; selection must still run
        profile = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    with _profile_lock:
        _profile_cache[key] = profile
    return profile


def _table_entry(meta: dict[str, Any], profile: dict[str, Any], desc_chars: int, max_cols: int) -> dict[str, Any]:
    title = meta.get("title") or ""
    entry: dict[str, Any] = {"title": title}
    if meta.get("resource_name") and meta["resource_name"] != title:
        entry["resource_name"] = meta["resource_name"]
    description = " ".join((meta.get("description") or "").split())
    if len(description) > desc_chars:
        description = description[:desc_chars].rsplit(" ", 1)[0] + " ..."
    years = profile.get("years") or sorted(
        _years_in_text(f"{title} {meta.get('resource_name') or ''}", time.gmtime().tm_year)
    )
    entry |= {
        "description": description,
        "tags": [tag for tag in meta.get("tags") or [] if tag],
        "years_covered": _compact_years(years),
    }
    if "row_count" in profile:
        entry["row_count"] = profile["row_count"]
    columns = meta.get("columns") or [{"name": name} for name in meta.get("columns.name") or []]
    samples = profile.get("samples") or {}
    rendered = []
    for column in columns[:max_cols]:
        item = {"name": column.get("name")}
        if column.get("type"):
            item["type"] = column["type"]
        if column.get("description"):
            item["description"] = " ".join(str(column["description"]).split())[:200]
        sample = samples.get(column.get("name"))
        is_geo = "geo" in str(column.get("type") or "").lower() or (sample or "").lstrip().startswith(
            ("{", "POINT", "POLYGON", "MULTI", "LINESTRING")
        )
        if sample and not is_geo:
            item["values"] = sample
        rendered.append(item)
    if len(columns) > max_cols:
        rendered.append(f"... (+{len(columns) - max_cols} more columns)")
    entry["columns"] = rendered
    return entry


def _display_keys(candidates: list[str]) -> dict[str, str]:
    stems = [re.sub(r"\.(parquet|csv)$", "", name, flags=re.I) for name in candidates]
    if len(set(stems)) < len(stems):
        stems = list(candidates)
    return dict(zip(stems, candidates))


def build_request(
    query: str, keys: dict[str, str], metadata: SolrMetadata,
    profiles: dict[str, dict[str, Any]], model: str,
) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "primary_table": {
            "type": "choice",
            "instructions": "Which candidate table is the main source of data for answering the user question?",
            "criteria": {key: (metadata.get(name) or {}).get("title") or key for key, name in keys.items()},
        }
    }
    for key in keys:
        questions[f"needs::{key}"] = {
            "type": "noul",
            "instructions": f"Is the candidate table `{key}` needed to compute the answer to the user question?",
            "criteria": NEEDED_CRITERIA,
        }
    for desc_chars, max_cols in ((1500, 60), (800, 40), (400, 25), (200, 15)):
        tables = {
            key: _table_entry(metadata.get(name) or {}, profiles.get(name) or {}, desc_chars, max_cols)
            for key, name in keys.items()
        }
        payload = {"model": model, "state": {"user_question": query, "candidate_tables": tables},
                   "questions": questions}
        if len(json.dumps(payload, ensure_ascii=False, default=str).encode()) <= MAX_REQUEST_BYTES:
            return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    raise JevSelectorError("the candidate context does not fit in one Jev request")


def _api_key() -> str:
    key = jev_api_key()
    if key is None:
        raise JevSelectorError(f"no Jev API key: set one of {', '.join(KEY_NAMES)}")
    return key


def post_request(payload: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        # Cloudflare in front of the API rejects Python's default agent.
        "User-Agent": "lakegen-jev-selector",
    }
    last_error = ""
    for attempt in range(retries):
        request = urllib.request.Request(API_URL, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read()[:300]!r}"
            if exc.code not in (429, 500, 502, 503, 504):
                break
            retry_after = exc.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 5 * 2 ** attempt
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            delay = 5 * 2 ** attempt
        if attempt < retries - 1:
            time.sleep(min(delay, 30))
    raise JevSelectorError(f"Jev API request failed: {last_error}")


def choose_tables(
    candidates: list[str], scores: dict[str, float], primary: str | None,
    threshold: float, reject_floor: float,
) -> list[str]:
    """Tables Jev rates as needed, primary first; [] means reject the keywords."""
    ranked = sorted(candidates, key=lambda name: scores[name], reverse=True)
    chosen = [name for name in ranked if scores[name] >= threshold]
    if not chosen:
        if scores[ranked[0]] < reject_floor:
            return []
        chosen = [primary if primary in scores else ranked[0]]
    if primary in chosen:
        chosen.remove(primary)
        chosen.insert(0, primary)
    return chosen


def select_with_jev(
    query: str,
    candidates: list[str],
    metadata: SolrMetadata,
    table_dir: Path | None,
    discovery: DiscoveryConfig,
    post: Callable[[dict[str, Any]], dict[str, Any]] = post_request,
) -> JevSelection:
    pool = list(dict.fromkeys(candidates))[: discovery.jev_max_candidates]
    if not pool:
        raise JevSelectorError("no candidates to select from")
    profiles = {
        name: profile_table(Path(table_dir) / name) if table_dir is not None else {}
        for name in pool
    }
    keys = _display_keys(pool)
    started = time.monotonic()
    response = post(build_request(query, keys, metadata, profiles, discovery.jev_model))
    elapsed = time.monotonic() - started
    try:
        answers = response["answers"]
        scores = {name: float(answers[f"needs::{key}"]["noul"]) for key, name in keys.items()}
        primary_key = answers["primary_table"].get("choice")
    except (KeyError, TypeError, ValueError) as exc:
        raise JevSelectorError(f"unexpected Jev response: {str(response)[:300]}") from exc
    primary = keys.get(primary_key)
    selected = choose_tables(pool, scores, primary, discovery.jev_threshold, discovery.jev_reject_floor)

    def label(name: str) -> str:
        title = (metadata.get(name) or {}).get("title") or name
        return f"`{name}` ({title}, {scores[name]:.2f})"

    ranked = sorted(pool, key=lambda name: scores[name], reverse=True)
    if selected:
        others = [name for name in ranked if name not in selected][:3]
        reasoning = "Jev rated as needed: " + ", ".join(label(name) for name in selected) + "."
        if others:
            reasoning += " Not needed: " + ", ".join(label(name) for name in others) + "."
    else:
        reasoning = (
            "REJECT_KEYWORDS: Jev rated none of the retrieved tables as needed for the question "
            f"(best: {label(ranked[0])}). Search for tables that hold the data the question asks for."
        )
    usage = response.get("usage") or {}
    trace_lines = [
        TRACE_HEADER,
        f"model={response.get('model')} input_tokens={usage.get('input_tokens')} seconds={elapsed:.2f}",
        f"threshold={discovery.jev_threshold} reject_floor={discovery.jev_reject_floor} primary={primary}",
        *(f"{scores[name]:.3f}  {name}" + ("  [selected]" if name in selected else "") for name in ranked),
    ]
    return JevSelection(
        selected=selected,
        reasoning=reasoning,
        trace="\n".join(trace_lines),
        scores=scores,
        input_tokens=int(usage.get("input_tokens") or 0),
    )
