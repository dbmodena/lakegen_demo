"""Every environment variable LakeGen reads, in one registry.

Each variable has one name, one default, one parse rule and a note on when it
is read, so the environment can be documented (docs/configuration.md), scrubbed
(tests/conftest.py) and migrated from one place. ``get_settings()`` reads the
infrastructure and operational variables into a ``Settings`` snapshot;
``retrieval_env_values()`` reads the retrieval knobs that ``RetrievalConfig.from_env``
turns into a configuration. Nothing here is cached: every call re-reads the
environment, so a test can change a variable between calls.

The parse rules reproduce the rules the reading code used before the registry
existed, including which malformed values are tolerated (the API's pacing
variables fall back to their default with a warning) and which raise (the
retrieval knobs, so a typo cannot silently run a different experiment).

Groups:
    paths       where the runtime state lives
    solr        the Solr service and the per-lake core overrides
    pneuma      the per-lake Pneuma service overrides
    providers   OCI, Together, Ollama, Groq and Jev credentials and endpoints
    api         the HTTP API's workers, batch pacing and model split
    switches    behaviour switches that are not part of the experiment config
    retrieval   the defaults of ``RetrievalConfig.from_env``, one per field

This module must stay importable with the standard library alone.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Every registered name starts with one of these; the test suite scrubs them.
SCRUBBED_PREFIXES = (
    "LAKEGEN_", "SOLR_", "OCI_", "TOGETHER", "GROQ_", "GROP_", "JEV_", "TYPESAFE_",
)


# --- parse rules ------------------------------------------------------------
# Each takes the raw string and the variable's default and returns the value.

Parser = Callable[[str, Any, str], Any]


def text(raw: str, default: Any, name: str = "") -> str:
    return raw


def stripped(raw: str, default: Any, name: str = "") -> str:
    return raw.strip()


def optional_text(raw: str, default: Any, name: str = "") -> str | None:
    """The string, or the default when it is empty (``os.environ.get(x) or default``)."""
    return raw if raw else default


def integer(raw: str, default: Any, name: str = "") -> int:
    return int(raw)


def number(raw: str, default: Any, name: str = "") -> float:
    return float(raw)


def optional_int(raw: str, default: Any, name: str = "") -> int | None:
    """An integer, or None when blank: no cap."""
    value = raw.strip()
    return int(value) if value else None


def positive_int_or_default(raw: str, default: Any, name: str = "") -> int:
    """A positive integer; anything else keeps the default with a warning."""
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if value < 1:
        logger.warning("Ignoring non-positive %s=%r; using %s", name, raw, default)
        return default
    return value


def non_negative_float_or_default(raw: str, default: Any, name: str = "") -> float:
    """A non-negative number; blank or anything else keeps the default."""
    if not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if value < 0:
        logger.warning("Ignoring negative %s=%r; using %s", name, raw, default)
        return default
    return value


def non_negative_count_or_default(raw: str, default: Any, name: str = "") -> int:
    return int(non_negative_float_or_default(raw, default, name))


def share(raw: str, default: Any, name: str = "") -> float:
    """A share in [0, 1]: non-negative, capped at 1."""
    return min(non_negative_float_or_default(raw, default, name), 1.0)


def flag(raw: str, default: Any, name: str = "") -> bool:
    return raw.strip().casefold() in {"1", "true", "yes"}


def flag_or_on(raw: str, default: Any, name: str = "") -> bool:
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def not_disabled(raw: str, default: Any, name: str = "") -> bool:
    return raw.strip().casefold() not in {"0", "false", "no"}


def not_zero(raw: str, default: Any, name: str = "") -> bool:
    return raw.strip() != "0"


def is_one(raw: str, default: Any, name: str = "") -> bool:
    return raw == "1"


# --- the registry -------------------------------------------------------------

@dataclass(frozen=True)
class EnvVar:
    name: str
    default: Any
    parse: Parser
    group: str
    help: str
    read_at: str = "call"
    field: str | None = None  # the RetrievalConfig field a retrieval knob sets

    def read(self) -> Any:
        raw = os.environ.get(self.name)
        if raw is None:
            return self.default
        return self.parse(raw, self.default, self.name)


@dataclass(frozen=True)
class EnvFamily:
    """Variables whose names carry a parameter, such as a lake."""

    pattern: str
    group: str
    help: str
    read_at: str = "call"


RETRIEVAL_READ_AT = (
    "RetrievalConfig.from_env, and load_experiment_config, where a set variable overrides the "
    "static default and is recorded in the manifest's env_overrides"
)


def _retrieval(name: str, field: str, default: Any, parse: Parser, help: str) -> EnvVar:
    return EnvVar(name, default, parse, "retrieval", help, RETRIEVAL_READ_AT, field)


VARIABLES: tuple[EnvVar, ...] = (
    # paths
    EnvVar("LAKEGEN_STATE_DIR", None, stripped, "paths",
           "Root of the runtime state (coding runs, logs, manifests, jobs, keyword memory); "
           "the repository root when unset.", "import of core.config, and every call-time path"),
    EnvVar("LAKEGEN_JOB_DIR", None, optional_text, "paths",
           "Name of the API job folder under the state root (.lakegen_jobs when unset), "
           "so an API instance can own its jobs.", "import of api"),
    # solr
    EnvVar("SOLR_BASE_URL", "http://localhost:8983/solr", text, "solr",
           "Base URL of the Solr service.", "every Solr client; the retrieval benchmark's --solr-base-url default"),
    # providers
    EnvVar("OCI_CONFIG_FILE", "~/.oci/config", text, "providers", "Path of the OCI configuration file."),
    EnvVar("OCI_PROFILE", "DEFAULT", text, "providers", "Profile of the OCI configuration file."),
    EnvVar("OCI_COMPARTMENT_ID", None, optional_text, "providers",
           "Compartment for OCI Generative AI; the profile's oci_compartment_id when unset."),
    EnvVar("OCI_SERVICE_ENDPOINT", None, optional_text, "providers",
           "OCI Generative AI endpoint; derived from the profile's region when unset."),
    EnvVar("LAKEGEN_TOGETHER_BASE_URL", "https://api.together.xyz/v1", text, "providers",
           "Base URL of the Together AI API."),
    EnvVar("LAKEGEN_TOGETHER_TEMPERATURE", 0.1, number, "providers",
           "Sampling temperature of the Together models."),
    EnvVar("LAKEGEN_TOGETHER_PARALLEL_TOOL_CALLS", True, not_zero, "providers",
           "Let Together's gpt-oss make parallel tool calls (0 turns it off)."),
    EnvVar("LAKEGEN_TOGETHER_RECOVER_LEAKS", True, not_zero, "providers",
           "Recover tool calls that Together's gpt-oss writes as plain text (0 turns it off)."),
    EnvVar("LAKEGEN_OLLAMA_BASE_URL", "http://localhost:11434", text, "providers",
           "Base URL of the local Ollama server for the ollama.* models."),
    EnvVar("LAKEGEN_GROQ_OFFLOAD", False, flag, "providers",
           "Send small tool-free gpt-oss-120b calls to Groq's free tier (1, true or yes)."),
    EnvVar("LAKEGEN_GROQ_LEDGER", None, optional_text, "providers",
           "Path of the Groq call ledger; logs/groq_calls.jsonl under the state root when unset.",
           "first Groq call"),
    # api
    EnvVar("LAKEGEN_WORKFLOW_WORKERS", 1, positive_int_or_default, "api",
           "Questions the API runs at once.", "import of api"),
    EnvVar("LAKEGEN_BATCH_PAUSE_EVERY", 0, non_negative_count_or_default, "api",
           "After this many finished questions a batch sleeps LAKEGEN_BATCH_PAUSE_SECONDS; 0 disables it.",
           "import of api"),
    EnvVar("LAKEGEN_BATCH_PAUSE_SECONDS", 0.0, non_negative_float_or_default, "api",
           "Length of the periodic batch pause.", "import of api"),
    EnvVar("LAKEGEN_BATCH_TOKENS_PER_MINUTE", 0.0, non_negative_float_or_default, "api",
           "Token rate a batch is held to by sleeping after each question; 0 disables it.", "import of api"),
    EnvVar("LAKEGEN_BATCH_TOKENS_PER_MINUTE_FILE", "", stripped, "api",
           "A file whose number overrides the token rate, re-read before every rest.", "import of api"),
    EnvVar("LAKEGEN_SPLIT_MODEL", "", stripped, "api",
           "A second model that runs a share of every batch's questions.", "import of api"),
    EnvVar("LAKEGEN_SPLIT_SHARE", 0.0, share, "api",
           "The share of questions (0 to 1) that run on LAKEGEN_SPLIT_MODEL.", "import of api"),
    EnvVar("LAKEGEN_DASHBOARD_TOKEN", "", stripped, "api",
           "A token the run dashboard requires (open /?token=... once; it is kept in a cookie). Empty: open.",
           "start of the dashboard"),
    # switches
    EnvVar("LAKEGEN_SELECTION_CARDINALITY_EXPERIMENT", False, is_one, "switches",
           "Adds the experimental table-count check to the unified architect's prompt (1)."),
    EnvVar("LAKEGEN_ENABLE_TABPFN", False, flag_or_on, "switches",
           "Lets generated analyses use the optional TabPFN stack (1, true, yes or on)."),
    # retrieval: the defaults of RetrievalConfig.from_env
    _retrieval("LAKEGEN_RETRIEVAL_MODE", "mode", "keyword", text, "Retrieval mode."),
    _retrieval("LAKEGEN_RETRIEVAL_TOP_K", "top_k", 20, integer, "Ranked pool size."),
    _retrieval("LAKEGEN_HYBRID_ALPHA", "alpha", 0.5, number, "Lexical weight of weighted hybrid fusion."),
    _retrieval("LAKEGEN_CANDIDATE_MULTIPLIER", "candidate_multiplier", 5, integer,
               "Per-branch pool size as a multiple of top_k."),
    _retrieval("LAKEGEN_REPRESENTATION_VERSION", "representation_version", "metadata-v1", text,
               "Version of the indexed representation."),
    _retrieval("LAKEGEN_EMBEDDING_MODEL", "embedding_model", "cohere.embed-v4.0", text,
               "Embedding model of semantic retrieval."),
    _retrieval("LAKEGEN_EMBEDDING_BASE_URL", "embedding_base_url", "http://localhost:11434", text,
               "Ollama URL for Ollama embedding models."),
    _retrieval("LAKEGEN_VECTOR_FIELD", "vector_field", "table_embedding", text,
               "Solr dense-vector field."),
    _retrieval("LAKEGEN_BM25_QUERY_FIELDS", "lexical_query_fields", None, text,
               "BM25 fields and boosts in Solr qf format."),
    _retrieval("LAKEGEN_MISSING_SIGNAL_POLICY", "missing_signal_policy", "zero", text,
               "How hybrid fusion treats a signal missing from one branch."),
    _retrieval("LAKEGEN_FUSION_METHOD", "fusion_method", "weighted", text,
               "Hybrid fusion method: weighted or rrf."),
    _retrieval("LAKEGEN_RRF_K", "rrf_k", 60, integer, "RRF smoothing constant."),
    _retrieval("LAKEGEN_PNEUMA_INDEX_NAME", "pneuma_index_name", "lakegen-cohere-v4-1024", text,
               "Pneuma index name when the lake has no route."),
    _retrieval("LAKEGEN_PNEUMA_BASE_URL", "pneuma_base_url", "http://localhost:8767", text,
               "Pneuma service URL when the lake has no route."),
    _retrieval("LAKEGEN_PNEUMA_TIMEOUT_SECONDS", "pneuma_timeout_seconds", 120.0, number,
               "Pneuma request timeout."),
    _retrieval("LAKEGEN_PNEUMA_CONTENT_WEIGHT", "pneuma_content_weight", 0.5, number,
               "Pneuma-Seeker: weight of content search against Pneuma's ranking."),
    _retrieval("LAKEGEN_PNEUMA_CONTENT_FUSION", "pneuma_content_fusion", "weighted", text,
               "Pneuma-Seeker: weighted or fill."),
    _retrieval("LAKEGEN_PNEUMA_ENTITY_MODEL", "pneuma_entity_model", None, text,
               "Pneuma-Seeker: chat model that extracts entities when the caller gives none."),
    _retrieval("LAKEGEN_PNEUMA_ENUMERATE_TABLES", "pneuma_enumerate_tables", True, not_disabled,
               "Pneuma-Seeker: pull in tables whose names form a family (0, false or no turns it off)."),
    _retrieval("LAKEGEN_PNEUMA_TABLE_NAME_WEIGHT", "pneuma_table_name_weight", 3.0, number,
               "Pneuma-Seeker: weight of a title match."),
    _retrieval("LAKEGEN_PNEUMA_COLUMN_NAME_WEIGHT", "pneuma_column_name_weight", 2.0, number,
               "Pneuma-Seeker: weight of a column-name match."),
    _retrieval("LAKEGEN_PNEUMA_CELL_WEIGHT", "pneuma_cell_weight", 1.0, number,
               "Pneuma-Seeker: weight of a cell match."),
    _retrieval("LAKEGEN_PNEUMA_VALUE_SCAN_WEIGHT", "pneuma_value_scan_weight", 0.0, number,
               "Plain pneuma: weight of a value scan of the returned tables; 0 scans nothing."),
    _retrieval("LAKEGEN_DUCKDB_MAX_FILES", "duckdb_max_files", 250, integer, "duckdb_agentic: files scanned."),
    _retrieval("LAKEGEN_DUCKDB_MAX_COLUMNS_PER_FILE", "duckdb_max_columns_per_file", 40, integer,
               "duckdb_agentic: columns scanned per file."),
    _retrieval("LAKEGEN_DUCKDB_SAMPLE_ROWS", "duckdb_sample_rows", 3, integer,
               "duckdb_agentic: sample rows per file."),
    _retrieval("LAKEGEN_DUCKDB_MAX_SCAN_ROWS_PER_FILE", "duckdb_max_scan_rows_per_file", 100_000, integer,
               "duckdb_agentic: rows scanned per file."),
    _retrieval("LAKEGEN_DUCKDB_PROBE_FILES", "duckdb_probe_files", 25, integer,
               "duckdb_agentic: following files probed for value-only terms."),
    _retrieval("LAKEGEN_DUCKDB_PROBE_ROWS_PER_FILE", "duckdb_probe_rows_per_file", 1_000, integer,
               "duckdb_agentic: rows per probed file."),
    _retrieval("LAKEGEN_GREP_MAX_FILES", "grep_max_files", None, optional_int,
               "Pneuma-Seeker content scan: file cap; blank scans the whole lake."),
    _retrieval("LAKEGEN_GREP_MAX_COLUMNS_PER_FILE", "grep_max_columns_per_file", 40, integer,
               "Content scan: columns per file."),
    _retrieval("LAKEGEN_GREP_MAX_SCAN_ROWS_PER_FILE", "grep_max_scan_rows_per_file", 100_000, integer,
               "Content scan: rows per file."),
    _retrieval("LAKEGEN_GREP_SAMPLE_ROWS", "grep_sample_rows", 3, integer, "Content scan: sample rows."),
    _retrieval("LAKEGEN_GREP_PROBE_FILES", "grep_probe_files", 25, integer, "Content scan: probed files."),
    _retrieval("LAKEGEN_SCAN_WORKERS", "scan_workers", 16, integer,
               "Worker processes of the content scan."),
    _retrieval("LAKEGEN_GREP_VALUE_WEIGHT", "grep_value_weight", 1.0, number,
               "Content scan: weight of value evidence."),
    _retrieval("LAKEGEN_GREP_METADATA_WEIGHT", "grep_metadata_weight", 1.0, number,
               "Content scan: weight of metadata evidence."),
    _retrieval("LAKEGEN_GREP_PROBE_ROWS_PER_FILE", "grep_probe_rows_per_file", 1_000, integer,
               "Content scan: rows per probed file."),
    _retrieval("LAKEGEN_RERANK_MODEL", "rerank_model", "cohere.rerank-v4.0-fast", text,
               "Second-stage reranker; empty turns reranking off."),
    _retrieval("LAKEGEN_RERANK_DEPTH", "rerank_depth", 20, integer,
               "How many top hits the reranker reorders."),
)

FAMILIES: tuple[EnvFamily, ...] = (
    EnvFamily("LAKEGEN_SOLR_CORE_<LAKE>", "solr",
              "Serves a lake from a differently named Solr core; blank means the lake's own name."),
    EnvFamily("LAKEGEN_PNEUMA_<LAKE>_BASE_URL", "pneuma", "The lake's Pneuma service URL."),
    EnvFamily("LAKEGEN_PNEUMA_<LAKE>_INDEX_NAME", "pneuma", "The lake's Pneuma index name."),
    EnvFamily("TOGETHERAI_API_KEY, TOGETHER_API_KEY", "providers",
              "Together AI key; the first non-empty one wins."),
    EnvFamily("GROQ_API_KEY*, GROP_API_KEY*", "providers",
              "Groq keys for the offload, every variable matching GRO[QP]_API_KEY<digits>, in name order."),
    EnvFamily("TYPESAFE_API_KEY, JEV_API_KEY", "providers",
              "TypeSafe key for the Jev table selector; the first set one wins."),
)

BY_NAME = {variable.name: variable for variable in VARIABLES}
_GROQ_KEY_NAME = re.compile(r"^GRO[QP]_API_KEY\d*$")


def read(name: str) -> Any:
    """The parsed value of one registered variable."""
    return BY_NAME[name].read()


def read_var(name: str, parse: Parser, default: Any) -> Any:
    """Read a variable by name with one of the parse rules (the API's named helpers)."""
    raw = os.environ.get(name)
    return default if raw is None else parse(raw, default, name)


def retrieval_env_values() -> dict[str, Any]:
    """The retrieval knobs as ``{RetrievalConfig field: value}``."""
    return {v.field: v.read() for v in VARIABLES if v.field is not None}


def retrieval_env_overrides() -> dict[str, Any]:
    """The retrieval knobs set in the environment, as ``{"retrieval.<field>": value}``.

    Only variables that are set count: these are the overrides an experiment
    config load applies on top of its static defaults, and what the manifest
    records as ``env_overrides``.
    """
    return {
        f"retrieval.{v.field}": v.read()
        for v in VARIABLES
        if v.field is not None and v.name in os.environ
    }


def solr_core_override(lake: str) -> str | None:
    value = os.environ.get(f"LAKEGEN_SOLR_CORE_{lake.strip().upper()}", "").strip()
    return value or None


def pneuma_override(lake: str) -> tuple[str | None, str | None]:
    """The lake's (base URL, index name) overrides; None where the environment is silent."""
    prefix = f"LAKEGEN_PNEUMA_{lake.strip().casefold().upper()}"
    return os.environ.get(f"{prefix}_BASE_URL"), os.environ.get(f"{prefix}_INDEX_NAME")


def together_api_key() -> str:
    return (os.environ.get("TOGETHERAI_API_KEY") or os.environ.get("TOGETHER_API_KEY") or "").strip()


def groq_api_keys() -> tuple[str, ...]:
    return tuple(
        os.environ[name].strip()
        for name in sorted(os.environ)
        if _GROQ_KEY_NAME.match(name) and os.environ[name].strip()
    )


def jev_api_key() -> str | None:
    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    return None


@dataclass(frozen=True)
class Settings:
    """A snapshot of the infrastructure and operational variables."""

    state_dir: str | None
    job_dir_name: str | None
    solr_base_url: str
    oci_config_file: str
    oci_profile: str
    oci_compartment_id: str | None
    oci_service_endpoint: str | None
    together_api_key: str
    together_base_url: str
    together_temperature: float
    together_parallel_tool_calls: bool
    together_recover_leaks: bool
    ollama_base_url: str
    groq_api_keys: tuple[str, ...]
    groq_offload: bool
    groq_ledger: str | None
    jev_api_key: str | None
    workflow_workers: int
    batch_pause_every: int
    batch_pause_seconds: float
    batch_tokens_per_minute: float
    batch_tokens_per_minute_file: str
    split_model: str
    split_share: float
    dashboard_token: str
    selection_cardinality_experiment: bool
    enable_tabpfn: bool

    def solr_core(self, lake: str) -> str:
        """The Solr core that serves a lake: its override, else its own name."""
        return solr_core_override(lake) or lake


def get_settings() -> Settings:
    """Read the environment now; nothing is cached."""
    return Settings(
        state_dir=read("LAKEGEN_STATE_DIR"),
        job_dir_name=read("LAKEGEN_JOB_DIR"),
        solr_base_url=read("SOLR_BASE_URL"),
        oci_config_file=read("OCI_CONFIG_FILE"),
        oci_profile=read("OCI_PROFILE"),
        oci_compartment_id=read("OCI_COMPARTMENT_ID"),
        oci_service_endpoint=read("OCI_SERVICE_ENDPOINT"),
        together_api_key=together_api_key(),
        together_base_url=read("LAKEGEN_TOGETHER_BASE_URL"),
        together_temperature=read("LAKEGEN_TOGETHER_TEMPERATURE"),
        together_parallel_tool_calls=read("LAKEGEN_TOGETHER_PARALLEL_TOOL_CALLS"),
        together_recover_leaks=read("LAKEGEN_TOGETHER_RECOVER_LEAKS"),
        ollama_base_url=read("LAKEGEN_OLLAMA_BASE_URL"),
        groq_api_keys=groq_api_keys(),
        groq_offload=read("LAKEGEN_GROQ_OFFLOAD"),
        groq_ledger=read("LAKEGEN_GROQ_LEDGER"),
        jev_api_key=jev_api_key(),
        workflow_workers=read("LAKEGEN_WORKFLOW_WORKERS"),
        batch_pause_every=read("LAKEGEN_BATCH_PAUSE_EVERY"),
        batch_pause_seconds=read("LAKEGEN_BATCH_PAUSE_SECONDS"),
        batch_tokens_per_minute=read("LAKEGEN_BATCH_TOKENS_PER_MINUTE"),
        batch_tokens_per_minute_file=read("LAKEGEN_BATCH_TOKENS_PER_MINUTE_FILE"),
        split_model=read("LAKEGEN_SPLIT_MODEL"),
        split_share=read("LAKEGEN_SPLIT_SHARE"),
        dashboard_token=read("LAKEGEN_DASHBOARD_TOKEN"),
        selection_cardinality_experiment=read("LAKEGEN_SELECTION_CARDINALITY_EXPERIMENT"),
        enable_tabpfn=read("LAKEGEN_ENABLE_TABPFN"),
    )


# --- documentation ----------------------------------------------------------

GROUP_TITLES = {
    "paths": "Runtime state",
    "solr": "Solr",
    "pneuma": "Pneuma",
    "providers": "Model providers",
    "api": "HTTP API",
    "switches": "Behaviour switches",
    "retrieval": "Retrieval defaults",
}


def _rows(group: str) -> Iterator[str]:
    for family in FAMILIES:
        if family.group == group:
            yield f"| `{family.pattern}` | — | {family.help} | {family.read_at} |"
    for variable in VARIABLES:
        if variable.group == group:
            default = "unset" if variable.default is None else f"`{variable.default!r}`"
            yield f"| `{variable.name}` | {default} | {variable.help} | {variable.read_at} |"


def render_markdown() -> str:
    """The reference of every environment variable, generated from the registry."""
    lines = [
        "# Environment variables",
        "",
        "Generated from `lakegen.config.settings` by `scripts/gen_config_docs.py`; do not edit by hand.",
        "`.env` in the repository root is loaded at start-up by the API, the CLI, the UI and the",
        "retrieval benchmark; variables already set in the environment win over it.",
        "",
    ]
    for group, title in GROUP_TITLES.items():
        lines += [f"## {title}", "", "| Variable | Default | Meaning | Read |", "|---|---|---|---|"]
        lines += list(_rows(group))
        lines.append("")
    return "\n".join(lines)
