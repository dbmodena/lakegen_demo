"""The lake and model registries hold what the code used to hard-code."""

from __future__ import annotations

import pytest

from lakegen.config import registry
from lakegen.config.registry import (
    get_lake,
    get_model,
    groq_models,
    lake_names,
    model_names,
    models_by_provider,
    pneuma_routes,
)
from lakegen.core.config import BASE_DIR

LAKES = ("nyc", "valencia", "bologna", "paris", "uk", "kramabench", "target_bird", "chicago")
MODELS = (
    "openai.gpt-oss-120b", "meta.llama-3.3-70b-instruct", "ollama.gpt-oss-20b",
    "ollama.qwen3.8-27b", "together.gpt-oss-120b",
)


def test_lakes_in_the_order_the_interfaces_list_them():
    assert lake_names() == LAKES
    assert get_lake("nyc").title == "New York City Open Data portal"
    assert get_lake("target_bird").title == "BIRD databases (TARGET benchmark)"


def test_pneuma_routes_match_the_services():
    assert pneuma_routes() == {
        "nyc": ("http://localhost:8765", "lakegen-cohere-v4-1024"),
        "uk": ("http://localhost:8766", "lakegen-cohere-v4-1024"),
        "valencia": ("http://localhost:8771", "lakegen-cohere-v4-1024"),
        "paris": ("http://localhost:8772", "lakegen-cohere-v4-1024"),
        "bologna": ("http://localhost:8773", "lakegen-cohere-v4-1024"),
        "kramabench": ("http://localhost:8781", "lakegen-cohere-v4-1024"),
        "target_bird": ("http://localhost:8782", "lakegen-cohere-v4-1024"),
        "chicago": ("http://localhost:8783", "lakegen-cohere-v4-1024"),
    }


def test_environment_overrides_apply_when_asked_not_when_loaded(monkeypatch):
    lake = get_lake("uk")
    assert lake.solr_core == "uk" and lake.pneuma_route() == ("http://localhost:8766", "lakegen-cohere-v4-1024")
    monkeypatch.setenv("LAKEGEN_SOLR_CORE_UK", "uk_fixed")
    monkeypatch.setenv("LAKEGEN_PNEUMA_UK_BASE_URL", "http://localhost:8767")
    assert lake.solr_core == "uk_fixed"
    assert lake.pneuma_route() == ("http://localhost:8767", "lakegen-cohere-v4-1024")


def test_lake_paths_follow_the_rules_the_code_used():
    uk, nyc = get_lake("uk"), get_lake("nyc")
    assert uk.dataset_roots == ("clean_datasets", "datasets") and nyc.dataset_roots == ("datasets",)
    assert uk.metadata_files[0] == "metadata_retrieved_cleaned.json"
    assert nyc.metadata_files == ("metadata_retrieved_only.json", "metadata.json")
    assert nyc.blend_db_path == BASE_DIR / "data" / "blend_nyc.db"
    assert nyc.statistics_path == BASE_DIR / "data" / "nyc" / "metadata" / "datasets_metadata.csv"
    assert nyc.starters_path == BASE_DIR / "queries" / "generated_queries_new_york.json"
    assert uk.reference_questions_path == BASE_DIR / "benchmark" / "generated_queries_semantic_uk.json"
    assert get_lake("chicago").starters_path is None


def test_tables_dir_prefers_a_directory_that_holds_tables(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "BASE_DIR", tmp_path)
    lake = get_lake("uk")
    (tmp_path / "data" / "uk" / "clean_datasets" / "parquets").mkdir(parents=True)
    (tmp_path / "data" / "uk" / "datasets" / "parquets").mkdir(parents=True)
    (tmp_path / "data" / "uk" / "datasets" / "parquets" / "a.parquet").write_bytes(b"")
    assert lake.tables_dir == tmp_path / "data" / "uk" / "datasets" / "parquets"
    (tmp_path / "data" / "uk" / "clean_datasets" / "parquets" / "b.parquet").write_bytes(b"")
    assert lake.tables_dir == tmp_path / "data" / "uk" / "clean_datasets" / "parquets"
    assert get_lake("paris").tables_dir == tmp_path / "data" / "paris" / "datasets" / "parquets"
    (tmp_path / "data" / "uk" / "metadata").mkdir()
    assert lake.metadata_path.name == "metadata_retrieved_cleaned.json"
    (tmp_path / "data" / "uk" / "metadata" / "metadata.json").write_text("{}")
    assert lake.metadata_path.name == "metadata.json"


def test_unknown_names_list_the_known_ones():
    with pytest.raises(KeyError, match="known: nyc, valencia"):
        get_lake("mars")
    with pytest.raises(KeyError, match="known: openai.gpt-oss-120b"):
        get_model("gpt-5")


def test_models_and_their_provider_ids():
    assert model_names() == MODELS
    assert models_by_provider("ollama") == {"ollama.gpt-oss-20b": "gpt-oss:20b", "ollama.qwen3.8-27b": "qwen3.8:27b"}
    assert models_by_provider("together") == {"together.gpt-oss-120b": "openai/gpt-oss-120b"}
    assert models_by_provider("oci") == {
        "openai.gpt-oss-120b": "openai.gpt-oss-120b", "meta.llama-3.3-70b-instruct": "meta.llama-3.3-70b-instruct",
    }
    assert groq_models() == {"openai.gpt-oss-120b": "openai/gpt-oss-120b"}


def test_model_limits_are_the_ones_the_clients_were_built_with():
    gpt_oss, llama = get_model("openai.gpt-oss-120b"), get_model("meta.llama-3.3-70b-instruct")
    assert (gpt_oss.context_window, gpt_oss.max_output_tokens) == (128_000, 16_000)
    assert (llama.context_window, llama.max_output_tokens) == (None, 4_000)
    local = get_model("ollama.qwen3.8-27b")
    assert (local.context_window, local.max_output_tokens, local.timeout_seconds) == (32_768, 16_000, 600)
    hosted = get_model("together.gpt-oss-120b")
    assert (hosted.context_window, hosted.max_output_tokens, hosted.timeout_seconds) == (128_000, 16_000, 300)
    assert all(model.temperature == 0.1 for model in registry.models())
