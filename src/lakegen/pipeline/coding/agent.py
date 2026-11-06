"""The coder: it writes the analysis over the selected tables and runs it.

``write_and_run_code`` is the pipeline's coder, a bounded tool-using
agent whose result is settled deterministically. ``phase3_generate_code``
streams one completion and returns the code without running it. Both describe
the tables with ``_build_coder_tables_info``; code runs in ``_execute_code``, a
subprocess with a guard against Cartesian merges prepended.
"""

import json
import logging
import re
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.llms import LLM, ChatMessage

from lakegen.config import paths
from lakegen.config.experiment import CoderContextLevel
from lakegen.datastructures.types import SolrMetadata
from lakegen.llm.prompts import PromptManager
from lakegen.llm.token_usage import (
    extract_total_tokens,
    get_llm_token_usage,
    reset_llm_token_usage,
)
from lakegen.pipeline.coding.context import CoderContext
from lakegen.pipeline.coding.tools.quality.column_resolution import resolve_generated_code_columns
from lakegen.pipeline.coding.tools.quality.data_quality import profile_table_quality
from lakegen.pipeline.discovery.keywords import split_thinking_blocks
from lakegen.pipeline.evaluation.code import (
    evaluation_output_instruction,
    extract_evaluation_payload,
)
from lakegen.utils.merge_limits import MERGE_BLOWUP_FACTOR, MERGE_BLOWUP_MIN_ROWS
from lakegen.utils.table_io import read_table, table_load_command

logger = logging.getLogger(__name__)


@dataclass
class CoderResult:
    """What one coder turn produced: the code, its result or failure, and the telemetry.

    Attributes:
        code_raw: The code as the coder submitted it, or its response when it submitted none.
        tokens: The tokens the turn used.
        raw_result: The printed output of the latest execution, or None when there is none.
        error: Why the turn failed, or None.
        clean_code: The code as resolved and executed.
        rejected_reason: Why the coder rejected the tables; empty when it did not.
        structured_result: The payload parsed out of the output.
        structured_result_error: Why no payload could be parsed, if so.
        execution_error: The latest error, classified and enriched, or None.
        coder_review: The finalization's review record, or None.
        coder_runs: The executions the turn spent.
        coder_lifecycle: Where the coder's run stood when the turn ended.
        stop_reason: Why the coding workflow stopped, when it stopped early.
        finalization_mode: How the run was finalized.
        operation_trace: What the executed code did, for telemetry.
        coder_context_audit: What the coder was shown and the plan's validation status.
        rejection_details: The rejection's missing requirements and evidence, or None.
        rejection_keep_tables: The selected tables that carry over to the next attempt.
        rejection_skip_tables: The selected tables banned as irrelevant.
        coder_attempt_trace: A snapshot of every execution attempt.
    """

    code_raw: str
    tokens: int
    raw_result: str | None = None
    error: str | None = None
    clean_code: str = ""
    rejected_reason: str = ""
    structured_result: object | None = None
    structured_result_error: str = ""
    execution_error: dict[str, object] | None = None
    coder_review: dict[str, str] | None = None
    coder_runs: int = 0
    coder_lifecycle: str = ""
    stop_reason: str = ""
    finalization_mode: str = ""
    operation_trace: dict[str, object] | None = None
    coder_context_audit: dict[str, object] | None = None
    rejection_details: dict[str, object] | None = None
    rejection_keep_tables: list[str] = field(default_factory=list)
    rejection_skip_tables: list[str] = field(default_factory=list)
    coder_attempt_trace: list[dict[str, object]] = field(default_factory=list)


_ERROR_PATTERNS = [
    "error:",
    "exception:",
    "traceback",
    "errno",
    "no such file",
    "filenotfounderror",
    "permissionerror",
    "modulenotfounderror",
    "importerror",
    "keyerror",
    "valueerror",
    "typeerror",
    "indexerror",
    "zerodivisionerror",
    # Generated scripts sometimes handle schema validation themselves, print a
    # failure, and return normally.  The process then exits with code 0 even
    # though no analysis result was produced, so these messages must trigger a
    # correction attempt too.
    "missing required column",
    "required column is missing",
    "required columns are missing",
]


# Prepended to every executed script. It never runs user code, only wraps
# pandas' merge entry points so a many-to-many join on non-unique keys (the
# "massive Cartesian product" failure mode already banned for geo joins in
# code_generator.yaml) raises a clear, coachable error instead of silently
# producing a huge, wrong result or hanging until the execution timeout.
_CARTESIAN_GUARD_PREAMBLE = """
import pandas as _lakegen_pd_guard


def _lakegen_check_merge_blowup(left_len, right_len, result_len, keys, label):
    baseline = max(left_len, right_len, 1)
    if result_len > __MIN_ROWS__ and result_len > baseline * __FACTOR__:
        raise RuntimeError(
            "CARTESIAN_PRODUCT_DETECTED: "
            f"{label} on {keys!r} produced {result_len} rows from left={left_len} "
            f"rows and right={right_len} rows. This indicates a many-to-many join "
            "on non-unique keys. Aggregate each side to one row per key before "
            "merging (e.g. groupby(key).agg(...)), or verify the join keys are "
            "actually unique identifiers."
        )


_lakegen_original_df_merge = _lakegen_pd_guard.DataFrame.merge


def _lakegen_guarded_df_merge(self, right, *args, **kwargs):
    result = _lakegen_original_df_merge(self, right, *args, **kwargs)
    keys = kwargs.get("on") or kwargs.get("left_on") or kwargs.get("right_on") or "unspecified"
    _lakegen_check_merge_blowup(
        len(self), len(right) if hasattr(right, "__len__") else 0, len(result), keys, "DataFrame.merge"
    )
    return result


_lakegen_pd_guard.DataFrame.merge = _lakegen_guarded_df_merge
_lakegen_original_pd_merge = _lakegen_pd_guard.merge


def _lakegen_guarded_pd_merge(left, right, *args, **kwargs):
    result = _lakegen_original_pd_merge(left, right, *args, **kwargs)
    keys = kwargs.get("on") or kwargs.get("left_on") or kwargs.get("right_on") or "unspecified"
    _lakegen_check_merge_blowup(
        len(left) if hasattr(left, "__len__") else 0,
        len(right) if hasattr(right, "__len__") else 0,
        len(result), keys, "pd.merge",
    )
    return result


_lakegen_pd_guard.merge = _lakegen_guarded_pd_merge
""".replace("__MIN_ROWS__", str(MERGE_BLOWUP_MIN_ROWS)).replace(
    "__FACTOR__", str(MERGE_BLOWUP_FACTOR)
)


def _extract_code(code_raw: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", code_raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_raw.replace("```python", "").replace("```", "").strip()


def _recover_fenced_agent_code(response: str, manager, state) -> bool:
    """Execute a complete program emitted without a run_analysis call."""
    if (
        state.run_count != 0
        or state.error is not None
        or state.rejected_reason
        or "```python" not in response.casefold()
    ):
        return False
    fallback_code = _extract_code(response)
    if "__LAKEGEN_EVAL_JSON__" not in fallback_code or not re.search(
        r"\b(?:import|from)\s+\w+", fallback_code
    ):
        return False
    if not state.analysis_contract:
        manager.infer_analysis_contract()
    fallback_run = json.loads(manager.run_analysis(fallback_code))
    if fallback_run.get("status") not in {"completed", "completed_with_warnings"}:
        return False
    state.stop_reason = "recovered_fenced_code_without_run_analysis_call"
    return True


def _recover_structured_rejection(response: str, manager, state) -> bool:
    """Route an uncalled, explicit rejection object through the normal evidence gate."""
    if state.finished or state.rejected_reason or not response.strip():
        return False
    decoder = json.JSONDecoder()
    required = {"reason", "missing_requirements", "inspected_evidence"}
    for start, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(response[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not required.issubset(payload):
            continue
        if not isinstance(payload["missing_requirements"], list):
            continue
        ban_tables = payload.get("ban_tables")
        if not isinstance(ban_tables, dict):
            ban_tables = None
        try:
            manager.reject_data(
                str(payload["reason"]),
                payload["missing_requirements"],
                str(payload["inspected_evidence"]),
                ban_tables,
            )
        except (TypeError, ValueError):
            # The same rejection validator used by the tool remains authoritative.
            return False
        state.error = None
        state.stop_reason = "recovered_structured_rejection_without_tool_call"
        return True
    return False


def _rejected_tables_reason(code_raw: str) -> str:
    if "REJECT_TABLES" not in code_raw:
        return ""
    for line in code_raw.splitlines():
        if "REJECT_TABLES" in line:
            # Extract just the reason part from the line
            return (
                line.replace("print(", "")
                .replace(")", "")
                .replace('"', "")
                .replace("'", "")
                .replace("REJECT_TABLES:", "")
                .replace("REJECT_TABLES", "")
                .strip()
            )
    return "Tables rejected by model."


def _keyerror_repair_hint(code: str, col_name: str, detail: str) -> str:
    """Point a KeyError at the intermediate frame, not only the source schema.

    A column present in the source file can still be missing at the failing
    line because an earlier rename/selection/aggregation removed it; telling
    the model to "use the schema name" then makes it repeat the same bug.

    Args:
        code: The executed code, without the merge-guard preamble.
        col_name: The missing column as the KeyError names it, possibly quoted.
        detail: The traceback, read for the failing line number.

    Returns:
        The hint: the generic advice, the failing line when the traceback
        locates it, and any earlier rename of the column.
    """
    column = col_name.strip().strip("'\"")
    parts = [
        "The column is missing from the dataframe AT THE FAILING LINE. It may "
        "exist in the source file but have been renamed, dropped, selected away, "
        "or turned into the index by an earlier step. Fix the reference using "
        "the columns the intermediate dataframe actually has at that point."
    ]
    offset = (_CARTESIAN_GUARD_PREAMBLE + "\n\n").count("\n")
    line_numbers = [int(number) for number in re.findall(r'script\.py", line (\d+)', detail)]
    code_lines = code.splitlines()
    if line_numbers and 0 < line_numbers[-1] - offset <= len(code_lines):
        failing = code_lines[line_numbers[-1] - offset - 1].strip()
        parts.append(f"Failing line {line_numbers[-1] - offset}: `{failing}`.")
    renamed = sorted(
        set(re.findall(rf"['\"]{re.escape(column)}['\"]\s*:\s*['\"]([^'\"]+)['\"]", code))
    )
    if column and renamed:
        parts.append(
            f"An earlier rename maps '{column}' to {renamed}; after that rename "
            f"use the new name, or reference '{column}' only before renaming."
        )
    return " ".join(parts)


def _execute_code(code_raw: str, run_dir: Path | None = None):
    code = _extract_code(code_raw)

    forbidden = ["import os", "import sys", "import shutil", "subprocess", "eval(", "exec("]
    forbidden_match = next((fragment for fragment in forbidden if fragment in code), None)
    if forbidden_match is not None:
        return (
            None,
            f"Security Error: forbidden code fragment {forbidden_match!r}. "
            "Remove it completely; use only data-analysis libraries required by the task.",
            code,
        )

    coding_dir = run_dir or paths.coding_dir() / uuid.uuid4().hex
    coding_dir.mkdir(parents=True, exist_ok=True)
    fp = coding_dir / "script.py"
    fp.write_text(_CARTESIAN_GUARD_PREAMBLE + "\n\n" + code, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(fp)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            stdout_lower = result.stdout.lower()
            if any(pat in stdout_lower for pat in _ERROR_PATTERNS):
                return None, result.stdout.strip(), code
            return result.stdout.strip(), None, code

        detail = result.stderr.strip() or result.stdout.strip()

        # Enhanced Error Parsing for Pandas
        if "KeyError:" in detail:
            match = re.search(r"KeyError:\s*(.*)", detail)
            col_name = match.group(1).strip() if match else "Unknown"
            error_msg = (
                f"FATAL ERROR: KeyError for column {col_name}. "
                + _keyerror_repair_hint(code, col_name, detail)
                + f"\n\nFull Traceback:\n{detail[-500:]}"
            )
        elif "FileNotFoundError:" in detail:
            error_msg = (
                "FATAL ERROR: FileNotFoundError. "
                "You hallucinated a file path or name. "
                "Use the EXACT file paths provided in the AVAILABLE TABLES section. "
                f"\n\nFull Traceback:\n{detail[-500:]}"
            )
        elif "CARTESIAN_PRODUCT_DETECTED" in detail:
            error_msg = (
                "FATAL ERROR: Cartesian product detected during a merge. "
                "The join key(s) are not unique on one or both sides, so the merge "
                "produced a huge, incorrect result. Aggregate each side to one row "
                "per key before merging, or verify the join keys are true unique "
                f"identifiers.\n\nFull Traceback:\n{detail[-500:]}"
            )
        else:
            error_msg = f"[Exit code {result.returncode}] {detail[-800:]}"

        return None, error_msg, code
    except subprocess.TimeoutExpired:
        return None, "Execution timed out (180s limit).", code
    except Exception as e:
        return None, f"[{type(e).__name__}] {e}", code


def _resolve_and_validate_columns(
    code_raw: str,
    tables: list[str],
    csv_dir: Path,
) -> tuple[str, str | None]:
    """Resolve generated aliases against the exact selected-table schemas."""
    code = _extract_code(code_raw)
    schemas: dict[str, list[str]] = {}
    try:
        for table in tables:
            path = Path(csv_dir) / table.strip()
            schemas[table] = [str(column) for column in read_table(path, nrows=0).columns]
        resolution = resolve_generated_code_columns(code, schemas)
    except (SyntaxError, ValueError) as exc:
        return code, f"Column preflight failed: {type(exc).__name__}: {exc}"

    if resolution.unresolved_required:
        available = sorted({column for columns in schemas.values() for column in columns})
        return resolution.code, (
            "Column preflight failed. Required generated column names do not match "
            f"the selected schemas: {list(resolution.unresolved_required)}. "
            f"Available exact columns: {available}"
        )
    return resolution.code, None


def _exact_column_labels(frame) -> list[str]:
    """Render the executable table schema without substituting Solr aliases."""
    return [f"{column}({frame[column].dtype})" for column in frame.columns]


def _temporal_value_profile(
    filepath: Path, temporal_columns: list[str], *, max_listed: int = 12
) -> list[str]:
    """Full-file distinct values of low-cardinality temporal columns.

    A column holding one value for every row is a snapshot/edition stamp:
    "as of <that date>" is then already satisfied by the whole file, and
    re-deriving it from another date column silently drops valid rows.

    Args:
        filepath: The table file, read in full for these columns only.
        temporal_columns: The candidate columns; the first 12 are read.
        max_listed: The most distinct values a column may have to be listed.

    Returns:
        The profile lines under a "Full-file value profile:" header, or no
        lines when nothing is worth saying or the file cannot be read.
    """
    if not temporal_columns:
        return []
    try:
        frame = read_table(filepath, columns=temporal_columns[:12])
    except Exception:
        logger.debug("No full-file value profile for %s", filepath, exc_info=True)
        return []
    lines: list[str] = []
    for column in frame.columns:
        values = frame[column].dropna()
        distinct = values.astype(str).unique()
        null_count = len(frame) - len(values)
        if len(distinct) == 1 and null_count:
            lines.append(
                f"     - {column}: only non-null value is {distinct[0]!r}, "
                f"{null_count}/{len(frame)} rows null (presence flag: the value "
                "marks the rows that have the attribute)"
            )
        elif len(distinct) == 1:
            date_like = bool(
                re.search(
                    r"(?:19|20)\d{2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}", distinct[0]
                )
            )
            lines.append(
                f"     - {column}: every row = {distinct[0]!r}"
                + (
                    " (snapshot/edition date stamp; a question 'as of' this "
                    "date is satisfied by all rows and needs no row filter)"
                    if date_like
                    else " (constant; not a discriminating filter)"
                )
            )
        elif 1 < len(distinct) <= max_listed:
            lines.append(
                f"     - {column}: {len(distinct)} distinct values {sorted(distinct.tolist())}"
            )
    if lines:
        lines.insert(0, "   Full-file value profile:")
    return lines


def _resource_lines(metadata: dict[str, Any]) -> list[str]:
    """The catalogue's title, publisher, resource name and description, whitespace-collapsed and cut."""
    title = " ".join(str(metadata.get("title") or "").split())[:160]
    publisher = " ".join(str(metadata.get("publisher") or "").split())[:160]
    resource_name = " ".join(str(metadata.get("resource_name") or "").split())[:160]
    description = " ".join(str(metadata.get("description") or "").split())[:320]
    lines = []
    if title:
        lines.append(f"   Resource title: {title}")
    if publisher:
        lines.append(f"   Publisher: {publisher}")
    if resource_name:
        lines.append(f"   Resource name: {resource_name}")
    if description:
        lines.append(f"   Resource description: {description}")
    return lines


def _column_lines(df) -> list[str]:
    """The file's exact columns with their dtypes: 25 in full, then 15 more by name."""
    max_detail_cols = 25
    # The local file is the execution schema and therefore the sole source
    # of truth for exact names. Solr may expose normalized API aliases.
    try:
        col_typed = _exact_column_labels(df)
    except Exception:
        logger.debug("Could not label the columns with their dtypes", exc_info=True)
        col_typed = ["Unknown columns"]

    total_cols = len(col_typed)
    shown = col_typed[:max_detail_cols]
    lines = [f"   Columns ({total_cols}): {', '.join(shown)}"]
    if total_cols > max_detail_cols:
        rest_names = [
            column.split("(")[0] if "(" in column else column
            for column in col_typed[max_detail_cols:]
        ]
        lines.append(f"   +{total_cols - max_detail_cols} more: {', '.join(rest_names[:15])}")
    return lines


def _column_semantics_lines(df, metadata: dict[str, Any]) -> list[str]:
    """The catalogue's descriptions of the file's columns, matched on normalized names, at most 12."""
    structured_columns = metadata.get("columns", [])
    if not isinstance(structured_columns, list):
        return []
    descriptions = {
        re.sub(r"[^a-z0-9]", "", str(column.get("name", "")).casefold()): " ".join(
            str(column.get("description") or "").split()
        )[:180]
        for column in structured_columns
        if isinstance(column, dict) and column.get("name") and column.get("description")
    }
    described = []
    for column in df.columns:
        description_hint = descriptions.get(re.sub(r"[^a-z0-9]", "", str(column).casefold()))
        if description_hint:
            described.append(f"{column}: {description_hint}")
        if len(described) >= 12:
            break
    if not described:
        return []
    return ["   Column semantics:", *(f"     - {item}" for item in described)]


def _quality_evidence_parts(evidence: dict[str, Any]) -> list[str]:
    """One column's data-quality findings, as the phrases of its line."""
    parts = []
    if "blank_or_null_pct" in evidence:
        parts.append(f"{evidence['blank_or_null_pct']}% blank/null")
    if evidence.get("sentinel_tokens_observed"):
        tokens = ", ".join(
            f"{token!r}x{count}" for token, count in evidence["sentinel_tokens_observed"].items()
        )
        parts.append(f"sentinel tokens observed: {tokens}")
    if evidence.get("possible_missing_codes"):
        codes = ", ".join(
            f"{code!r}x{count}" for code, count in evidence["possible_missing_codes"].items()
        )
        context = (
            f" (column median {evidence['numeric_median']:g}, max {evidence['numeric_max']:g})"
            if "numeric_median" in evidence
            else ""
        )
        parts.append(f"possible missing-value codes: {codes}{context}")
    if evidence.get("numeric_format_examples"):
        parts.append(f"numeric formatting noise, e.g. {evidence['numeric_format_examples']}")
    return parts


def _data_quality_lines(df) -> list[str]:
    """The sampled rows' data-quality findings, as literal evidence for cleaning."""
    quality_findings = profile_table_quality(df)
    if not quality_findings:
        return []
    lines = [
        "   Data quality (literal evidence from the sampled rows above — "
        "clean using exactly these observed tokens, never invent others; "
        "treat a possible missing-value code as missing only when metadata "
        "or its implausibility against the column range supports it, "
        "otherwise keep it as a real value):"
    ]
    for column, evidence in quality_findings.items():
        lines.append(
            f"     - {column}: {'; '.join(_quality_evidence_parts(evidence))} "
            f"(n={evidence['sampled_rows']} sampled)"
        )
    return lines


def _sample_lines(df) -> list[str]:
    """The first two rows of the first 15 columns, cells cut to 40 characters."""
    max_sample_cols = 15
    max_sample_rows = 2
    max_cell_width = 40
    sample_df = df.head(max_sample_rows).copy()
    if len(sample_df.columns) > max_sample_cols:
        sample_df = sample_df.iloc[:, :max_sample_cols].copy()
    for column in sample_df.columns:
        sample_df[column] = sample_df[column].astype(str).str.slice(0, max_cell_width)
    return [f"   Sample:\n{sample_df.to_string(index=False)}"]


def _build_coder_tables_info(
    tables: list[str],
    csv_dir: Path,
    context_level: CoderContextLevel,
    table_metadata: SolrMetadata | None = None,
) -> str:
    """Build the selected-table context exposed to the code generator.

    Every table gets its load command. Above ``MINIMAL`` it also gets the
    catalogue lines, the exact columns, column semantics and a note on
    ambiguous temporal columns; ``FULL`` adds a 500-row sample read, the
    full-file profile of temporal and constant columns, the data-quality
    findings and two sample rows.

    Args:
        tables: The selected table file names, relative to ``csv_dir``.
        csv_dir: The directory holding the table files.
        context_level: How much of each table the coder is shown.
        table_metadata: The catalogue metadata by file name or stem.

    Returns:
        The context as text, starting with "AVAILABLE TABLES:".
    """
    profile_sample_rows = 500

    info_lines = ["AVAILABLE TABLES:"]
    for idx, fn in enumerate(tables, 1):
        filepath = Path(csv_dir) / fn.strip()
        load_cmd = table_load_command(filepath)
        info_lines.append(f"{idx}. LOAD: {load_cmd}")

        if context_level == CoderContextLevel.MINIMAL:
            continue

        metadata = (table_metadata or {}).get(fn, (table_metadata or {}).get(Path(fn).stem, {}))
        info_lines.extend(_resource_lines(metadata))

        rows_to_read = profile_sample_rows if context_level == CoderContextLevel.FULL else 0
        df = read_table(filepath, nrows=rows_to_read)
        info_lines.extend(_column_lines(df))
        info_lines.extend(_column_semantics_lines(df, metadata))

        temporal_candidates = [
            str(column)
            for column in df.columns
            if re.search(
                r"(?:^|[_\W])(year|date|time|period|fy|sy)(?:$|[_\W])", str(column), re.IGNORECASE
            )
            or re.search(r"(?:year|date)$", str(column), re.IGNORECASE)
        ]
        if len(temporal_candidates) > 1:
            info_lines.append(
                "   Temporal ambiguity: "
                + ", ".join(temporal_candidates[:8])
                + ". Choose by resource/column meaning, not name alone."
            )

        if context_level == CoderContextLevel.SCHEMA_ONLY:
            continue

        sample_constant = [
            str(column) for column in df.columns if len(df) and df[column].nunique(dropna=True) == 1
        ]
        info_lines.extend(
            _temporal_value_profile(
                filepath, list(dict.fromkeys([*temporal_candidates, *sample_constant]))
            )
        )
        info_lines.extend(_data_quality_lines(df))
        info_lines.extend(_sample_lines(df))

    return "\n".join(info_lines)


def _phase3_user_prompt(
    pm: PromptManager,
    *,
    query: str,
    tables_info: str,
    retries: int,
    error_msg: str,
    previous_code: str,
    seed: int,
    force_execution: bool,
) -> str:
    """The single-completion coder's turn: the task or the correction, the seed, the output format, the cues.

    Args:
        pm: The prompt manager that renders the ``code_generator`` prompts.
        query: The user's question.
        tables_info: The selected-table context.
        retries: The correction round; 0 renders the initial prompt.
        error_msg: The previous attempt's error, for a correction.
        previous_code: The previous attempt's code, for a correction.
        seed: The effective seed the code must use.
        force_execution: Whether the user chose to continue despite a table-quality warning.

    Returns:
        The user prompt.
    """
    if retries == 0:
        user_prompt = pm.render(
            "code_generator",
            "initial_prompt",
            question=query,
            arch_reasoning="",
            tables_info=tables_info,
        )
    else:
        user_prompt = pm.render(
            "code_generator",
            "correction_prompt",
            question=query,
            error_message=error_msg,
            previous_code=previous_code,
            arch_reasoning="",
            tables_info=tables_info,
        )

    user_prompt += (
        "\n\n[REPRODUCIBILITY]\n"
        f"The effective seed for this run is {seed}. Use exactly this value for "
        "every stochastic operation that accepts a seed (including pandas "
        "sampling, train/test split, cross-validation splitters, and model "
        "random_state parameters). Do not add a seed to deterministic operations. "
        "Do not use another fixed seed.\n"
    )
    user_prompt += evaluation_output_instruction()

    if force_execution:
        user_prompt += (
            "\n\nFORCE EXECUTION OVERRIDE\n"
            "The user explicitly chose to continue with these tables despite "
            "the previous table-quality warning. Do not return REJECT_TABLES. "
            "Write the best executable Pandas script possible using only the "
            "available paths and columns. If the data is insufficient, the "
            "script must print a concise explanation of what is missing."
        )

    if retries > 0:
        # Correction turns bury the question under previous code, the error,
        # and table info. Restating it last (closest to where the model
        # actually writes the fix) keeps the full original ask in view instead
        # of narrowing to "make this error go away."
        user_prompt += (
            "\n\nORIGINAL QUESTION (verbatim, re-read before finalizing this "
            f'correction):\n"{query}"\n'
            "Fix only what the reported error requires. Do not silently drop, "
            "simplify, or narrow any other filter, grouping, measure, ordering, "
            "or limit that this question asks for and that the previous code "
            "already computed correctly."
        )
    return user_prompt


def _reasoning_loop_detected(structured_reasoning: str) -> bool:
    """Whether the reasoning, past 400 characters, repeats its last 150 at least four times."""
    cleaned_reasoning = structured_reasoning.strip()
    if len(cleaned_reasoning) > 400:
        tail = cleaned_reasoning[-150:]
        return cleaned_reasoning.count(tail) >= 4
    return False


def _chunk_tokens(tokens: int, chunk: Any) -> int:
    """The running token count, raised to what the chunk reports."""
    return max(
        tokens,
        extract_total_tokens(chunk.raw),
        extract_total_tokens(chunk.additional_kwargs),
    )


def _update_placeholders(
    raw_stream: str,
    structured_reasoning: str,
    stream_placeholder: Any,
    reasoning_placeholder: Any,
) -> None:
    """Show the visible stream and the reasoning (structured and tagged) in their placeholders."""
    visible_stream, tagged_reasoning = split_thinking_blocks(raw_stream)
    reasoning_parts = [
        part.strip() for part in (structured_reasoning, tagged_reasoning) if part.strip()
    ]
    reasoning_stream = "\n\n".join(reasoning_parts)

    if stream_placeholder is not None:
        stream_placeholder.markdown(visible_stream or raw_stream)
    if reasoning_placeholder is not None and reasoning_stream:
        reasoning_placeholder.markdown(reasoning_stream)


def _stream_code(
    llm: LLM,
    messages: list[ChatMessage],
    *,
    stream_reasoning: bool,
    cancel_check: Callable[[], None] | None,
    stream_placeholder: Any,
    reasoning_placeholder: Any,
) -> tuple[str, int, bool]:
    """Stream the completion to the console and the placeholders.

    The stream asks for structured reasoning when ``stream_reasoning`` is set
    and breaks when the reasoning repeats itself. If the stream fails before
    anything arrived, it is retried once without the reasoning request.

    Args:
        llm: The model.
        messages: The system and user messages.
        stream_reasoning: Whether to ask for structured reasoning (``think``).
        cancel_check: Called before each chunk; raises to cancel.
        stream_placeholder: Shows the visible stream, when given.
        reasoning_placeholder: Shows the reasoning, when given.

    Returns:
        The raw streamed text, the tokens the chunks reported, and whether a
        reasoning loop stopped the stream.

    Raises:
        Exception: The stream's error, when it failed after output arrived or
            without a reasoning request.
    """
    raw_stream = ""
    structured_reasoning = ""
    tokens = 0

    def update_placeholders() -> None:
        _update_placeholders(
            raw_stream, structured_reasoning, stream_placeholder, reasoning_placeholder
        )

    print("[phase3 code stream] ", end="", flush=True)

    stream_kwargs = {"think": True} if stream_reasoning else {}

    loop_detected = False

    try:
        chunk_stream = llm.stream_chat(messages, **stream_kwargs)
        for chunk in chunk_stream:
            if cancel_check is not None:
                cancel_check()
            thinking_delta = chunk.additional_kwargs.get("thinking_delta")
            if thinking_delta:
                structured_reasoning += thinking_delta
                print(thinking_delta, end="", flush=True)
                if _reasoning_loop_detected(structured_reasoning):
                    logger.warning(
                        "Block repetition loop detected in reasoning. Forcing stream break."
                    )
                    loop_detected = True
                    break

                update_placeholders()

            delta = chunk.delta or ""
            if delta:
                raw_stream += delta
                print(delta, end="", flush=True)
                update_placeholders()

            tokens = _chunk_tokens(tokens, chunk)

    except Exception as e:
        logger.warning("Error during the code stream: %s", e, exc_info=True)
        if raw_stream or structured_reasoning or not stream_reasoning:
            raise
        chunk_stream = llm.stream_chat(messages)
        for chunk in chunk_stream:
            if cancel_check is not None:
                cancel_check()
            delta = chunk.delta or ""
            if delta:
                raw_stream += delta
                print(delta, end="", flush=True)
                update_placeholders()

            tokens = _chunk_tokens(tokens, chunk)

    print("", flush=True)
    return raw_stream, tokens, loop_detected


def _final_code(raw_stream: str, loop_detected: bool) -> str:
    """The visible code of the stream, or a ``__GENERATION_ERROR__`` when there is none usable."""
    visible_content, _tagged_reasoning = split_thinking_blocks(raw_stream)

    final_code = visible_content.strip()
    if not final_code:
        if loop_detected:
            final_code = "__GENERATION_ERROR__: The model was forcefully stopped due to an infinite reasoning loop."
        else:
            final_code = "__GENERATION_ERROR__: No code was generated by the model."
    elif loop_detected:
        # Some partial code may have been emitted before the loop was detected.
        # If the code looks incomplete (no newlines, no import, no print), treat it as an error.
        if final_code.count("\n") < 2 or not any(
            kw in final_code for kw in ("import ", "print(", "pd.", "df")
        ):
            final_code = "__GENERATION_ERROR__: The model entered an infinite reasoning loop. Partial output was not valid Python."
    return final_code


def phase3_generate_code(
    query,
    tables,
    candidates,
    solr_meta: SolrMetadata,
    reasoning,
    llm: LLM,
    pm: PromptManager,
    csv_dir,
    retries=0,
    error_msg="",
    previous_code="",
    force_execution: bool = False,
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
    cancel_check: Callable[[], None] | None = None,
    seed: int = 0,
    seed_instruction_recorder: Callable[[], None] | None = None,
    coder_context_level: CoderContextLevel = CoderContextLevel.FULL,
    evaluation_result_type: str | None = None,
):
    """Generate an analysis script with one streamed completion, without running it.

    A correction round raises the model's temperature by 0.1 per retry (at
    most 0.8) to break out of reasoning local minimums; the temperature is
    restored once the stream ends.

    Args:
        query: The user's question.
        tables: The selected table file names.
        candidates: Unused; kept for the shared call signature.
        solr_meta: The catalogue metadata of the tables.
        reasoning: Unused; kept for the shared call signature.
        llm: The model that writes the code.
        pm: The prompt manager.
        csv_dir: The directory holding the table files.
        retries: The correction round; 0 is the first attempt.
        error_msg: The previous attempt's error, for a correction.
        previous_code: The previous attempt's code, for a correction.
        force_execution: Whether the user chose to continue despite a table-quality warning.
        stream_placeholder: Shows the visible stream, when given.
        reasoning_placeholder: Shows the reasoning, when given.
        stream_reasoning: Whether to ask the model for structured reasoning.
        cancel_check: Called before each chunk; raises to cancel.
        seed: The effective seed the code must use.
        seed_instruction_recorder: Called once the seed instruction is in the prompt.
        coder_context_level: How much of each table the coder is shown.
        evaluation_result_type: Unused; kept for the shared call signature.

    Returns:
        The generated code (or a ``__GENERATION_ERROR__`` message) and the
        tokens used.
    """
    coder_context = CoderContext.build(
        question=query,
        selected_tables=tables,
        table_metadata=solr_meta,
        selection_plan={},
        execution_error={"message": error_msg} if error_msg else {},
    )
    solr_meta = dict(coder_context.table_metadata)
    context_level = CoderContextLevel(coder_context_level)
    tables_info = _build_coder_tables_info(tables, Path(csv_dir), context_level, solr_meta)

    system_prompt = pm.render("code_generator", "system_prompt")
    user_prompt = _phase3_user_prompt(
        pm,
        query=query,
        tables_info=tables_info,
        retries=retries,
        error_msg=error_msg,
        previous_code=previous_code,
        seed=seed,
        force_execution=force_execution,
    )

    original_temperature = getattr(llm, "temperature", 0.1)
    if retries > 0:
        # Dynamic temperature adjustment to break out of reasoning local minimums
        if hasattr(llm, "temperature"):
            llm.temperature = min(original_temperature + (0.1 * retries), 0.8)

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    if seed_instruction_recorder is not None:
        seed_instruction_recorder()

    reset_llm_token_usage(llm)
    raw_stream, tokens, loop_detected = _stream_code(
        llm,
        messages,
        stream_reasoning=stream_reasoning,
        cancel_check=cancel_check,
        stream_placeholder=stream_placeholder,
        reasoning_placeholder=reasoning_placeholder,
    )
    final_code = _final_code(raw_stream, loop_detected)

    if hasattr(llm, "temperature"):
        llm.temperature = original_temperature

    tokens = max(tokens, get_llm_token_usage(llm))
    return final_code, tokens


RUNNABLE_PLAN_STATUSES = frozenset({"verified", "executable_with_obligations"})
"""The plan statuses the coder may start from when a semantic plan is required."""

CODER_AGENT_RULES = (
    "\n\nYou are a bounded coding agent. A semantic plan is authoritative only "
    "after deterministic runtime validation reports it locked. An invalid or "
    "insufficiently evidenced plan must be resolved using inspect_data or "
    "rejected with concrete evidence through reject_data. The validated coder "
    "brief remains the sole semantic contract. "
    "Treat a year that names a dataset edition or release (for example, "
    "'using the 2024 ... dataset') as resource provenance, not as a row-level "
    "filter: omit it from filters and do not require a year/date column. "
    "The contract must preserve every semantic "
    "qualifier from the question (for example Community, completed, distinct, "
    "and the exact measure being averaged). Objective contract violations block "
    "finalization and require corrected code. "
    "Do not add filters, limits, distinct-count semantics, deduplication, or "
    "temporal interpretations that the question does not request. Preserve every "
    "requested measure and identifying dimension in the final structured output. "
    "Do not invent bucket boundaries with qcut/cut, tertiles, or quantiles unless "
    "the question explicitly defines that bucketing rule. Do not replace a named "
    "measure with a nearby total, count, or identifier merely because it is easier "
    "to compute. For a qualified filter, inspect the exact column values when its "
    "encoding is unclear; a non-empty value is not evidence that a semantic "
    "qualifier is satisfied. "
    "For distinct counts of categories or types, use the column whose meaning "
    "represents the requested category; do not automatically prefer an identifier "
    "code when an explicit descriptive column is available. "
    "You have exactly three tools: inspect_data, run_analysis, and reject_data. "
    "Use inspect_data at most once and request all needed tables/columns together. "
    "Then write a complete Python program and call run_analysis. That tool performs "
    "preflight, execution, result inspection, adaptation, and finalization. If it "
    "returns revision_required, correct every reported problem together and call "
    "run_analysis once more. The second executed analysis is terminal, including "
    "when warnings remain. Do not call any separate inspection, contract, or finish "
    "tool. Call reject_data only when inspected evidence proves fundamental source "
    "facts are absent. Never print diagnostic schemas from analysis code. Never "
    "infer correctness from benchmark gold: it is not available. "
    "Do not import sys; it is unnecessary and forbidden by the execution sandbox. "
    "Do not merely describe code in chat."
)
NO_SEMANTIC_REVISION_RULE = (
    " For this experimental arm, semantic coverage warnings are terminal: "
    "preserve the inspected result and do not revise code because of those "
    "warnings. Technical execution and preflight errors remain actionable."
)
PLAN_COMPLETION_RULE = (
    "The computational part of the brief may be "
    "incomplete. Compare it with the complete question and add or refine any required "
    "final aggregation, derived calculation, ratio, correlation, ordering, limit, or "
    "output shaping. Do not stop at an intermediate grouped result. Call "
    "run_analysis with the code only; completion is inferred automatically. "
    "If the selected data cannot support an essential requirement, inspect once and "
    "reject_data with concrete evidence instead of inventing it.\n"
)
ADVISORY_BRIEF_RULE = (
    "Treat selected tables and evidenced join keys as fixed; verify the brief's "
    "column choices, row filters and null policy against the data evidence. "
)
IMMUTABLE_BRIEF_RULE = (
    "Treat selected tables, exact runtime columns, explicit question filters, and "
    "evidenced join keys as immutable. "
)
MISSING_PLAN_VIEW = {
    "valid": False,
    "locked": False,
    "status": "missing",
    "required_action": "inspect_data_then_run_analysis_or_reject_data",
}
STOP_WITHOUT_TERMINAL_STATE = {
    "needs_inspection": "Coder stopped after execution without inspecting the latest result.",
    "needs_revision": "Coder stopped while the inspected result still required revision.",
    "ready_to_finish": "Coder stopped before mandatory finalization.",
}


def _coder_system_prompt(
    pm: PromptManager, *, brief_advisory: bool, allow_semantic_revision: bool
) -> str:
    prompt = (
        pm.render("code_generator", "agentic_system_prompt", brief_advisory=brief_advisory)
        + CODER_AGENT_RULES
    )
    if not allow_semantic_revision:
        prompt += NO_SEMANTIC_REVISION_RULE
    return prompt


def _coder_user_prompt(
    pm: PromptManager,
    *,
    query: str,
    retries: int,
    error_msg: str,
    previous_code: str,
    tables_info: str,
    output_shape: Any,
    seed: int,
    force_execution: bool,
) -> str:
    """The coder's turn: the task or the correction, the cues, the seed, the output format."""
    if retries == 0:
        prompt = pm.render(
            "code_generator",
            "agentic_initial_prompt",
            question=query,
            arch_reasoning="",
            tables_info=tables_info,
        )
    else:
        prompt = pm.render(
            "code_generator",
            "agentic_correction_prompt",
            question=query,
            error_message=error_msg,
            previous_code=previous_code,
            arch_reasoning="",
            tables_info=tables_info,
        )
    prompt += "\n\nQUESTION-DERIVED ORDERING/LIMIT CUES (non-gold):\n" + json.dumps(
        output_shape, ensure_ascii=False, sort_keys=True
    )
    prompt += (
        "\n\n[REPRODUCIBILITY]\n"
        f"Use seed {seed} for every stochastic operation. "
        "Before finishing, inspect the actual output and check filters, measures, "
        "group coverage, ordering/limits, and output shape."
    )
    prompt += evaluation_output_instruction()
    if force_execution:
        prompt += (
            "\nThe selected tables must be used for the best possible executable "
            "answer; do not reject them."
        )
    return prompt


def _plan_sections(
    plan_view: dict[str, Any], *, brief_advisory: bool, retries: int, query: str
) -> str:
    """What the coder is told about the validated plan, how to complete it, and the question again on a correction."""
    sections = "\n\nSELECTION CONTRACT RUNTIME VALIDATION (non-gold):\n" + json.dumps(
        plan_view or MISSING_PLAN_VIEW, ensure_ascii=False, sort_keys=True, default=str
    )
    sections += (
        "\n\nPLAN COMPLETION RULE:\n"
        + (ADVISORY_BRIEF_RULE if brief_advisory else IMMUTABLE_BRIEF_RULE)
        + PLAN_COMPLETION_RULE
        + json.dumps(
            plan_view.get("plan_completion_policy", {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    if retries > 0:
        # Correction turns bury the question under previous code, the error,
        # and table/plan JSON. Restating it last (closest to where the model
        # actually writes the fix) keeps the full original ask in view instead
        # of narrowing to "make this error go away."
        sections += (
            "\n\nORIGINAL QUESTION (verbatim, re-read before finalizing this "
            f'correction):\n"{query}"\n'
            "Fix only what the reported error requires. Do not silently drop, "
            "simplify, or narrow any other filter, grouping, measure, ordering, "
            "or limit that this question asks for and that the previous code "
            "already computed correctly."
        )
    return sections


def _blocked_before_start(
    coder_context: CoderContext, plan_view: dict[str, Any], initial_plan_status: str
) -> CoderResult:
    """The selection-contract gate: the coder does not start on an unverified plan."""
    status = str(plan_view.get("status") or "invalid")
    audit = {
        **coder_context.audit(),
        **plan_view,
        "semantic_plan_status": status,
        "semantic_plan_initial_status": initial_plan_status,
        "semantic_plan_final_status": status,
        "semantic_plan_coder_start_status": "not_started",
        "validation_diagnostics": list(plan_view.get("diagnostics") or []),
        "evidence_count": len(plan_view.get("evidence") or []),
        "coder_started_after_verified_plan": False,
    }
    return CoderResult(
        code_raw="",
        tokens=0,
        error=f"Selection contract gate blocked coder startup: {status}.",
        execution_error={
            "stage": "selection_contract_gate",
            "category": "coder_brief_not_verified",
            "status": status,
            "retryable": status in {"missing", "invalid", "needs_runtime_evidence"},
            "diagnostics": audit["validation_diagnostics"],
        },
        coder_runs=0,
        coder_lifecycle="blocked_before_start",
        stop_reason="selection_contract_gate",
        coder_context_audit=audit,
    )


def _run_coder_agent(
    llm: LLM,
    system_prompt: str,
    user_prompt: str,
    *,
    state: Any,
    manager: Any,
    stream_placeholder: Any,
    cancel_check: Callable[[], None] | None,
) -> tuple[str, bool]:
    """The coder agent's turn: its response, and whether it ran out of iterations."""
    from lakegen.llm.agent_activity import AgentStall
    from lakegen.llm.agent_runner import run_agent_workflow

    def emit_stream(delta: str) -> None:
        if not delta:
            return
        print(delta, end="", flush=True)
        if stream_placeholder is not None:
            stream_placeholder.markdown(delta)

    try:
        response = run_agent_workflow(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name="coder",
            emit_stream=emit_stream,
            cancel_check=cancel_check,
            tools=manager.get_tools(),
            max_iterations=8,
            max_repeats=3,
            max_tool_calls=6,
            timeout_seconds=600,
        )
        return response, False
    except AgentStall as exc:
        state.stop_reason = str(exc)
        if not state.error and state.raw_result is None:
            state.error = f"Coder agent stalled: {exc}"
        return "", False
    except Exception as exc:
        workflow_error = f"Coder agent failed: {type(exc).__name__}: {exc}"
        # An orchestration limit must not invalidate a successful execution.
        # Keep it as stop telemetry and let deterministic inspection decide
        # whether the latest structured result can be finalized.
        if state.raw_result is not None and state.structured_result is not None:
            state.stop_reason = workflow_error
            state.error = None
        elif not state.error:
            state.error = workflow_error
        return "", "Max iterations" in str(exc)


def _rejection_reason(state: Any, manager: Any, response: str, *, force_execution: bool) -> str:
    """Why the coder rejected the tables, from its tool call or its free text; nothing under force."""
    if force_execution:
        return ""
    if state.rejected_reason:
        missing = state.rejection_details.get("missing_requirements", [])
        evidence = state.rejection_details.get("inspected_evidence", "")
        return f"{state.rejected_reason} Missing requirements: {missing}. Inspected evidence: {evidence}"
    rejected_reason = _rejected_tables_reason(response)
    if rejected_reason:
        # This is the last-resort path: the model rejected in free text
        # that didn't parse as the structured payload
        # _recover_structured_rejection understands, so there is no
        # ban_tables signal to trust. Ban nothing rather than banning the
        # whole selected set on no evidence -- the safe default matches
        # reject_tables' own "no justification, no ban" rule instead of
        # excluding a table that was never actually proven irrelevant.
        state.rejection_skip_tables = []
        state.rejection_keep_tables = list(manager.tables)
    return rejected_reason


def _settle_run(
    state: Any, manager: Any, rejected_reason: str, *, workflow_exhausted: bool
) -> None:
    """What the coder left undone, done deterministically: the inspection, the finalization, the degraded finish."""
    # A model can exhaust its normal reasoning turn immediately after inspection.
    # Inspection is deterministic bookkeeping over the latest runtime result.
    # Do it when the model executed successfully but omitted the protocol call.
    if (
        not state.finished
        and state.lifecycle.value == "needs_inspection"
        and state.raw_result is not None
        and not rejected_reason
    ):
        try:
            manager.inspect_result()
        except Exception as exc:
            state.stop_reason = f"automatic_inspection_failed: {type(exc).__name__}: {exc}"
    # Deterministic validation is the terminal gate; no second LLM is needed.
    if not state.finished and state.ready_for_finalization() and not rejected_reason:
        manager.finalize_validated_result(
            "Automatically finalized after a valid, structured, warning-free inspection."
        )
    # A latest inspected structured result remains useful even when semantic
    # advisories could not be resolved within the bounded coder loop. Preserve
    # it as a degraded completion; downstream evaluation still decides its
    # correctness and the advisories remain attached to the review telemetry.
    if (
        not state.finished
        and state.ready_for_degraded_finalization()
        and (state.run_count >= state.max_runs or workflow_exhausted)
        and not rejected_reason
    ):
        manager.recover_degraded_finish(
            "System recovery preserved the latest inspected structured result "
            "after the coder exhausted its correction/finalization budget or "
            "agent-iteration budget."
        )
    if not state.finished and state.error is None and not rejected_reason:
        state.error = STOP_WITHOUT_TERMINAL_STATE.get(
            state.lifecycle.value,
            "Coder stopped without reaching a terminal coding state.",
        )
    if (
        not state.finished
        and state.lifecycle.value == "needs_revision"
        and not state.execution_error
    ):
        state.execution_error = {
            "stage": "result_validation",
            "category": "result_needs_revision",
            "message": state.error or "The inspected result requires revision.",
            "retryable": state.run_count < state.max_runs,
            "coverage_warnings": list(state.coverage_warnings),
        }


def _phase3_result(
    state: Any,
    manager: Any,
    response: str,
    *,
    tokens: int,
    rejected_reason: str,
    coder_context: CoderContext,
    initial_plan_status: str,
    require_semantic_plan: bool,
) -> CoderResult:
    final_plan_view = manager.coder_plan_view()
    return CoderResult(
        code_raw=state.code_raw or response,
        tokens=tokens,
        raw_result=state.raw_result,
        error=state.error,
        clean_code=state.clean_code,
        rejected_reason=rejected_reason,
        structured_result=state.structured_result,
        structured_result_error=state.structured_result_error,
        execution_error=state.execution_error or None,
        coder_review=state.review or None,
        coder_runs=state.run_count,
        coder_lifecycle=state.lifecycle.value,
        stop_reason=state.stop_reason,
        finalization_mode=state.finalization_mode,
        operation_trace=state.operation_trace or None,
        coder_context_audit={
            **coder_context.audit(),
            **final_plan_view,
            "semantic_plan_validated_before_lock": bool(state.plan_validation.get("valid")),
            "semantic_plan_status": state.plan_validation.get("status"),
            "semantic_plan_initial_status": initial_plan_status,
            "semantic_plan_final_status": state.plan_validation.get("status"),
            "semantic_plan_coder_start_status": initial_plan_status,
            "validation_diagnostics": list(state.plan_validation.get("diagnostics", [])),
            "evidence_count": len(state.plan_validation.get("evidence", [])),
            "coder_started_after_verified_plan": (
                not require_semantic_plan or initial_plan_status in RUNNABLE_PLAN_STATUSES
            ),
            "verified_requirements": list(state.coverage_requirements),
            "inspect_result_executed": state.inspected_version > 0,
            "plan_completion": dict(
                (state.operation_trace or {}).get("inferred_plan_completion") or {}
            ),
        },
        rejection_details=state.rejection_details or None,
        rejection_keep_tables=list(state.rejection_keep_tables),
        rejection_skip_tables=list(state.rejection_skip_tables),
        coder_attempt_trace=list(state.execution_attempts),
    )


def write_and_run_code(
    query: str,
    tables: list[str],
    candidates: list[str],
    solr_meta: SolrMetadata,
    reasoning: str,
    llm: LLM,
    pm: PromptManager,
    csv_dir: Path,
    retries: int = 0,
    error_msg: str = "",
    previous_code: str = "",
    force_execution: bool = False,
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
    cancel_check: Callable[[], None] | None = None,
    run_dir: Path | None = None,
    seed: int = 0,
    seed_instruction_recorder: Callable[[], None] | None = None,
    coder_context_level: CoderContextLevel = CoderContextLevel.FULL,
    evaluation_result_type: str | None = None,
    max_run_calls: int = 3,
    selection_plan: dict[str, object] | None = None,
    source_field_names: list[str] | None = None,
    require_semantic_plan: bool = True,
    allow_semantic_revision: bool = True,
    brief_advisory: bool = False,
) -> CoderResult:
    """The coder on one selection: a bounded tool-using agent over the sandbox, its result settled deterministically.

    The coder sees only the allowlisted context (``CoderContext``). When a
    semantic plan is required and the plan is not runnable, it does not start.
    Afterwards, fenced code or a rejection object left in the response goes
    through the same tools, the latest result is inspected, and whatever the
    agent left undone (inspection, finalization) is done deterministically.

    Args:
        query: The user's question.
        tables: The selected table file names.
        candidates: Unused; kept for the shared call signature.
        solr_meta: The catalogue metadata of the tables.
        reasoning: Unused; kept for the shared call signature.
        llm: The model that drives the coder.
        pm: The prompt manager.
        csv_dir: The directory holding the table files.
        retries: The correction round; 0 renders the initial prompt.
        error_msg: The previous attempt's error, for a correction.
        previous_code: The previous attempt's code, for a correction.
        force_execution: Whether the selected tables must be used; disables rejection.
        stream_placeholder: Shows the agent's stream, when given.
        reasoning_placeholder: Unused; kept for the shared call signature.
        stream_reasoning: Unused; kept for the shared call signature.
        cancel_check: Called by the agent loop; raises to cancel.
        run_dir: Where the executed scripts are written; a fresh directory when None.
        seed: The effective seed the code must use.
        seed_instruction_recorder: Called once the seed instruction is in the prompt.
        coder_context_level: How much of each table the coder is shown.
        evaluation_result_type: Unused; the coder never sees the expected result type.
        max_run_calls: The executions allowed, clamped to one or two.
        selection_plan: The selection's plan and coder brief.
        source_field_names: The fields of the source payload; any but the
            question is excluded and its values scrubbed from the context.
        require_semantic_plan: Whether the coder starts only on a runnable plan.
        allow_semantic_revision: Whether semantic coverage warnings may trigger a revision.
        brief_advisory: Whether the brief's columns, filters and null policy are
            advisory rather than immutable.

    Returns:
        The turn's code, result or failure, and telemetry.
    """
    # Keep retrieval/discovery and the existing sandbox unchanged: only the
    # coder's generate/execute retry loop becomes a bounded tool-using agent.
    from lakegen.pipeline.coding.tools import CoderState, CoderTools

    coder_context = CoderContext.build(
        question=query,
        selected_tables=tables,
        table_metadata=solr_meta,
        selection_plan=selection_plan,
        source_payload={name: None for name in (source_field_names or [])},
        execution_error={"message": error_msg} if error_msg else {},
    )
    # From this point onward Phase 3 sees only the DTO's explicit allowlist.
    selection_plan = dict(coder_context.selection_plan)
    solr_meta = dict(coder_context.table_metadata)
    tables_info = _build_coder_tables_info(
        tables, Path(csv_dir), CoderContextLevel(coder_context_level), solr_meta
    )
    system_prompt = _coder_system_prompt(
        pm,
        brief_advisory=brief_advisory,
        allow_semantic_revision=allow_semantic_revision,
    )
    user_prompt = _coder_user_prompt(
        pm,
        query=query,
        retries=retries,
        error_msg=error_msg,
        previous_code=previous_code,
        tables_info=tables_info,
        output_shape=coder_context.output_shape,
        seed=seed,
        force_execution=force_execution,
    )
    if seed_instruction_recorder is not None:
        seed_instruction_recorder()

    # One initial analysis plus at most one evidence-directed revision.
    state = CoderState(max_runs=max(1, min(2, max_run_calls)))
    manager = CoderTools(
        state,
        tables=tables,
        csv_dir=Path(csv_dir),
        run_dir=run_dir,
        evaluation_result_type=None,
        question=query,
        table_metadata={
            table: solr_meta.get(table, solr_meta.get(Path(table).stem, {})) for table in tables
        },
        selection_plan=selection_plan,
        resolve_code=_resolve_and_validate_columns,
        execute_code=_execute_code,
        extract_payload=extract_evaluation_payload,
        require_semantic_plan=require_semantic_plan,
        allow_semantic_revision=allow_semantic_revision,
        brief_advisory=brief_advisory,
        # The coder brief is the sole semantic contract. Runtime code/result
        # inspection supplies independent evidence without a duplicate model-
        # authored manifest.
        require_analysis_manifest=False,
    )
    plan_view = manager.coder_plan_view()
    initial_plan_status = str(plan_view.get("status") or "missing")
    if require_semantic_plan and plan_view.get("status") not in RUNNABLE_PLAN_STATUSES:
        return _blocked_before_start(coder_context, plan_view, initial_plan_status)
    user_prompt += _plan_sections(
        plan_view, brief_advisory=brief_advisory, retries=retries, query=query
    )
    reset_llm_token_usage(llm)
    response, workflow_exhausted = _run_coder_agent(
        llm,
        system_prompt,
        user_prompt,
        state=state,
        manager=manager,
        stream_placeholder=stream_placeholder,
        cancel_check=cancel_check,
    )
    # Some model turns emit the complete program in a fenced response instead
    # of invoking run_code. Route that program through the exact same bounded
    # preflight, sandbox, result extraction and inspection path as a tool call.
    _recover_fenced_agent_code(response, manager, state)
    _recover_structured_rejection(response, manager, state)
    # inspect_result is deterministic and does not consume a run_code attempt.
    # If the model stops immediately after a valid structured execution, apply
    # the same inspection tool programmatically so closure/revision routing is
    # based on the actual latest result.
    if (
        state.error is None
        and state.raw_result is not None
        and state.structured_result is not None
        and state.inspected_version != state.result_version
    ):
        manager.inspect_result()
        state.stop_reason = state.stop_reason or "auto_inspected_latest_successful_result"
    rejected_reason = _rejection_reason(state, manager, response, force_execution=force_execution)
    _settle_run(state, manager, rejected_reason, workflow_exhausted=workflow_exhausted)
    return _phase3_result(
        state,
        manager,
        response,
        tokens=get_llm_token_usage(llm),
        rejected_reason=rejected_reason,
        coder_context=coder_context,
        initial_plan_status=initial_plan_status,
        require_semantic_plan=require_semantic_plan,
    )
