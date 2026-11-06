"""Stored configurations are read as recorded: no validation, no default filling."""

from __future__ import annotations

import json
from pathlib import Path

from lakegen.config.experiment import ExperimentConfig
from lakegen.config.recorded import load_recorded_config, schema_keys

ROOT = Path(__file__).resolve().parents[2]
STORED = json.loads(
    (ROOT / "tests" / "golden" / "fixtures" / "configs" / "stored_job_configs.json").read_text(
        encoding="utf-8"
    )
)
# The keyword cascade's size (removed 2026-10-27) and Jev's settings (removed 2026-10-29).
RETIRED_KEYS = {
    "discovery.keyword_cascade_size",
    "discovery.jev_model",
    "discovery.jev_max_candidates",
    "discovery.jev_threshold",
    "discovery.jev_reject_floor",
    "discovery.initial_candidates",
    "discovery.expansion_size",
    "discovery.max_expansions",
    "discovery.max_search_attempts",
    "discovery.initial_shortlist_size",
    "discovery.max_inspected_candidates",
    "discovery.max_inspections_per_file",
    "discovery.search_after_inspection",
    "discovery.keyword_search",
    "discovery.state_board",
    "discovery.compact_history",
    "tool_access",
    "orchestrated_memory_enabled",
}


def test_schema_keys_cover_every_field_including_nested_blocks():
    keys = schema_keys()
    assert "retrieval.top_k" in keys and "discovery.selector" in keys and "gates.plan" in keys
    assert "benchmark.path" in keys and "experiment_id" in keys
    assert len(keys) == len(set(keys))
    dumped = ExperimentConfig().model_dump(mode="json")
    assert set(load_recorded_config(dumped).values) == set(keys)


def test_a_current_config_is_fully_recorded():
    recorded = load_recorded_config(ExperimentConfig(experiment_id="x").model_dump(mode="json"))
    assert recorded.unknown == {} and recorded.not_recorded == ()
    assert recorded.get("experiment_id") == "x"
    assert recorded.with_todays_defaults().experiment_id == "x"


def test_legacy_keys_are_kept_apart_and_missing_fields_are_not_filled():
    recorded = load_recorded_config(
        {
            "experiment_id": "old",
            "planner_enabled": True,
            "discovery": {"parallel_inspection_enabled": True, "max_zero_result_retries": 3},
            "retrieval": {"mode": "keyword"},
        }
    )
    assert recorded.unknown == {
        "planner_enabled": True,
        "discovery.parallel_inspection_enabled": True,
    }
    assert recorded.get("discovery.max_zero_result_retries") == 3
    assert (
        "retrieval.rerank_model" in recorded.not_recorded
        and "discovery.selector" in recorded.not_recorded
    )
    assert "discovery.max_zero_result_retries" not in recorded.not_recorded


def test_the_stored_job_configs_load_as_recorded():
    missing_selector = 0
    for entry in STORED.values():
        recorded = load_recorded_config(entry["config"])
        assert set(recorded.unknown) <= RETIRED_KEYS, entry["jobs"]
        if recorded.not_recorded:
            assert recorded.not_recorded == ("discovery.selector",), entry["jobs"]
            missing_selector += 1
        recorded.with_todays_defaults()  # every stored config still re-runs under today's schema
    assert missing_selector == 45  # the jobs that ran before the selector setting existed
