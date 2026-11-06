"""The environment registry reproduces the rules the code read the environment with."""

from __future__ import annotations

from pathlib import Path

import pytest

from lakegen.config import settings
from lakegen.config.settings import (
    SCRUBBED_PREFIXES,
    VARIABLES,
    flag,
    get_settings,
    is_one,
    non_negative_float_or_default,
    not_disabled,
    not_zero,
    optional_int,
    positive_int_or_default,
    render_markdown,
    retrieval_env_overrides,
    retrieval_env_values,
    share,
)

ROOT = Path(__file__).resolve().parents[2]


def test_every_registered_name_is_scrubbed_by_the_test_suite():
    for variable in VARIABLES:
        assert variable.name.startswith(SCRUBBED_PREFIXES), variable.name
    assert len({v.name for v in VARIABLES}) == len(VARIABLES)


@pytest.mark.parametrize(("raw", "expected"), [("3", 3), ("abc", 1), ("0", 1), ("-2", 1)])
def test_positive_int_keeps_the_default_on_bad_input(raw, expected):
    assert positive_int_or_default(raw, 1) == expected


@pytest.mark.parametrize(
    ("raw", "expected"), [("2.5", 2.5), ("", 7.0), ("  ", 7.0), ("x", 7.0), ("-1", 7.0)]
)
def test_non_negative_float_keeps_the_default_on_bad_input(raw, expected):
    assert non_negative_float_or_default(raw, 7.0) == expected


def test_share_is_capped_at_one():
    assert share("1.7", 0.0) == 1.0
    assert share("0.2", 0.0) == 0.2


@pytest.mark.parametrize("raw", ["1", "true", "YES", " yes "])
def test_flag_accepts_one_true_and_yes(raw):
    assert flag(raw, False) is True


def test_flag_variants_differ_exactly_as_the_readers_did():
    assert flag("on", False) is False
    assert not_disabled("", True) is True and not_disabled("No", True) is False
    assert not_zero("0", True) is False and not_zero("", True) is True
    assert is_one("1", False) is True and is_one("true", False) is False


def test_optional_int_reads_blank_as_no_cap():
    assert optional_int("", None) is None
    assert optional_int(" 12 ", None) == 12


def test_defaults_match_the_documented_ones(tmp_path):
    found = get_settings()
    # The suite itself points the state root at a temporary directory.
    assert found.state_dir == str(tmp_path / "state")
    assert found.solr_base_url == "http://localhost:8983/solr"
    assert found.workflow_workers == 1
    assert found.split_share == 0.0 and found.split_model == ""
    assert found.groq_offload is False
    assert found.groq_api_keys == () and found.together_api_key == ""
    assert found.job_dir_name is None
    knobs = retrieval_env_values()
    assert knobs["mode"] == "keyword" and knobs["top_k"] == 20
    assert knobs["rerank_model"] == "cohere.rerank-v4.0-fast"
    assert knobs["pneuma_index_name"] == "lakegen-cohere-v4-1024"
    assert knobs["scan_workers"] == 16 and knobs["grep_max_files"] is None
    assert len(knobs) == 40


def test_values_are_read_at_call_time(monkeypatch):
    monkeypatch.setenv("LAKEGEN_WORKFLOW_WORKERS", "4")
    monkeypatch.setenv("LAKEGEN_RETRIEVAL_TOP_K", "7")
    assert get_settings().workflow_workers == 4
    assert retrieval_env_values()["top_k"] == 7


def test_lake_overrides(monkeypatch):
    assert get_settings().solr_core("uk") == "uk"
    monkeypatch.setenv("LAKEGEN_SOLR_CORE_UK", " uk_fixed ")
    assert get_settings().solr_core("uk") == "uk_fixed"
    monkeypatch.setenv("LAKEGEN_SOLR_CORE_UK", "")
    assert get_settings().solr_core("uk") == "uk"
    assert settings.pneuma_override("valencia") == (None, None)
    monkeypatch.setenv("LAKEGEN_PNEUMA_VALENCIA_BASE_URL", "http://localhost:8771")
    assert settings.pneuma_override("Valencia") == ("http://localhost:8771", None)


def test_key_families(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "a")
    monkeypatch.setenv("GROP_API_KEY2", " b ")
    monkeypatch.setenv("GROQ_API_KEY3", "")
    assert settings.groq_api_keys() == ("b", "a")  # name order, blanks skipped
    monkeypatch.setenv("TOGETHER_API_KEY", "fallback")
    assert settings.together_api_key() == "fallback"
    monkeypatch.setenv("TOGETHERAI_API_KEY", " primary ")
    assert settings.together_api_key() == "primary"


def test_json_object_variables_take_an_object_or_nothing():
    assert settings.json_object("", {}, "X") == {}
    assert settings.json_object('{"reasoning_effort": "high"}', {}, "X") == {
        "reasoning_effort": "high"
    }
    with pytest.raises(ValueError, match="DEDICATE_OCI_HEADERS is not valid JSON"):
        settings.json_object("{", {}, "DEDICATE_OCI_HEADERS")
    with pytest.raises(ValueError, match="DEDICATE_OCI_EXTRA_BODY must be a JSON object"):
        settings.json_object("[1]", {}, "DEDICATE_OCI_EXTRA_BODY")


def test_dedicated_endpoint_settings(monkeypatch):
    assert settings.dedicated_model("DEDICATE_OCI_MODEL") == ""
    blank = get_settings()
    assert (blank.dedicated_api_key, blank.dedicated_base_url) == ("", "")
    assert (blank.dedicated_headers, blank.dedicated_extra_body) == ({}, {})
    monkeypatch.setenv("DEDICATE_OCI_MODEL", " ocid1.endpoint ")
    monkeypatch.setenv("DEDICATE_OCI_API_KEY", " key ")
    monkeypatch.setenv(
        "DEDICATE_OCI_BASE_URL", "https://host/20231130/actions/v1/chat/completions/"
    )
    monkeypatch.setenv("DEDICATE_OCI_HEADERS", '{"x-compartment": "c"}')
    monkeypatch.setenv("DEDICATE_OCI_EXTRA_BODY", '{"reasoning_effort": "low"}')
    assert settings.dedicated_model("DEDICATE_OCI_MODEL") == "ocid1.endpoint"
    got = get_settings()
    assert got.dedicated_api_key == "key"
    assert (
        got.dedicated_base_url == "https://host/20231130/actions/v1"
    )  # what the clients append to
    assert got.dedicated_headers == {"x-compartment": "c"}
    assert got.dedicated_extra_body == {"reasoning_effort": "low"}


def test_configuration_docs_are_current():
    generated = render_markdown()
    assert all(variable.name in generated for variable in VARIABLES)
    assert (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8") == generated, (
        "run scripts/gen_config_docs.py"
    )


def test_only_set_variables_count_as_overrides(monkeypatch):
    assert retrieval_env_overrides() == {}
    monkeypatch.setenv("LAKEGEN_RETRIEVAL_TOP_K", "7")
    monkeypatch.setenv("LAKEGEN_RERANK_MODEL", "")
    assert retrieval_env_overrides() == {"retrieval.top_k": 7, "retrieval.rerank_model": ""}
