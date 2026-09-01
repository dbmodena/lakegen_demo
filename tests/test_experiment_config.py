import builtins
import json

import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from lakegen.experiment_config import ExperimentConfig, load_experiment_config
from lakegen.manifest import create_manifest, persist_manifest, redact_secrets
from lakegen.runner import ExperimentRunner
from lakegen.ui.state import RuntimeSettings
from src.api import _resolve_api_config
from src.cli import resolve_cli_experiment
from lakegen.retrieval import RetrievalConfig
from lakegen.service import run_question


def test_default_config_matches_existing_interactive_workflow(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("LAKEGEN_"):
            monkeypatch.delenv(name, raising=False)

    config = ExperimentConfig()

    assert config.discovery_architecture == "unified"
    assert config.tool_access == "agentic"
    assert config.retrieval.mode == "keyword"
    assert config.retrieval.top_k == 10
    assert config.planner_enabled is False
    assert not any(config.reviewers.model_dump().values())
    assert config.max_revision_rounds == 3
    assert config.coder_context_level == "full"
    assert config.automatic_test_coder is False
    assert config.require_semantic_plan is True
    assert config.interaction_mode == "human_gated"
    assert config.gates.model_dump() == {
        "keywords": True,
        "datasets": True,
        "plan": False,
        "result": False,
    }


def test_cli_defaults_keep_historical_values_despite_retrieval_environment(monkeypatch):
    monkeypatch.setenv("LAKEGEN_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("LAKEGEN_RETRIEVAL_TOP_K", "99")

    config = resolve_cli_experiment()

    assert config.retrieval.mode == "keyword"
    assert config.retrieval.top_k == 10


def test_yaml_json_and_cli_overrides_resolve_identically(tmp_path):
    payload = {
        "experiment_id": "thesis-01",
        "core": "bologna",
        "discovery_architecture": "divided",
        "retrieval": {"mode": "hybrid", "top_k": 20},
    }
    yaml_path = tmp_path / "experiment.yaml"
    json_path = tmp_path / "experiment.json"
    yaml_path.write_text(
        "experiment_id: thesis-01\ncore: bologna\ndiscovery_architecture: divided\n"
        "retrieval:\n  mode: hybrid\n  top_k: 20\n",
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    yaml_config = load_experiment_config(yaml_path)
    json_config = load_experiment_config(json_path)
    cli_config = resolve_cli_experiment(
        config_path=json_path,
        set_values=["retrieval.top_k=20"],
    )

    assert yaml_config == json_config == cli_config


@pytest.mark.parametrize(
    "update",
    [
        {"reviewers": {"dataset": True}},
        {"max_revision_rounds": 4},
        {"gates": {"result": True}},
    ],
)
def test_unimplemented_combinations_are_rejected(update):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(update)


def test_semantic_planner_is_supported():
    assert ExperimentConfig(planner_enabled=True).planner_enabled is True


@pytest.mark.parametrize("level", ["full", "schema_only", "minimal"])
def test_coder_context_levels_are_supported(level):
    assert ExperimentConfig(coder_context_level=level).coder_context_level == level


def test_automatic_coder_context_sweep_is_supported():
    assert ExperimentConfig(automatic_test_coder=True).automatic_test_coder is True


def test_orchestrated_context_is_supported_and_unknown_tool_access_is_rejected():
    assert (
        ExperimentConfig(tool_access="orchestrated_context").tool_access
        == "orchestrated_context"
    )
    with pytest.raises(ValidationError):
        ExperimentConfig(tool_access="not-a-mode")


def test_code_gate_is_removed_from_the_schema():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"gates": {"code": True}})


def test_manifest_contains_all_experimental_variables_and_is_immutable(tmp_path):
    config = ExperimentConfig(experiment_id="manifest-test", seed=42)
    manifest = create_manifest(
        config,
        base_dir=tmp_path,
        question="How many parks?",
        question_id="q-1",
        run_id="run-1",
    )
    path = persist_manifest(manifest, tmp_path / "manifests")
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert stored["experiment_id"] == "manifest-test"
    assert stored["question_id"] == "q-1"
    assert stored["run_id"] == "run-1"
    assert stored["model"] == config.model
    assert stored["core"] == stored["dataset"] == config.core
    assert stored["seed"] == 42
    assert stored["representation_version"] == "metadata-v1"
    assert stored["retrieval_parameters"] == config.retrieval.model_dump(mode="json")
    assert stored["resolved_config"] == config.model_dump(mode="json")
    assert "code" not in stored["resolved_config"]["gates"]
    with pytest.raises(FileExistsError):
        persist_manifest(manifest, tmp_path / "manifests")
    with pytest.raises(ValidationError):
        manifest.seed = 2


def test_secrets_are_redacted_and_environment_is_not_serialized(tmp_path, monkeypatch):
    monkeypatch.setenv("OCI_API_KEY", "super-secret")
    config = ExperimentConfig()
    manifest = create_manifest(config, base_dir=tmp_path, question="Question?")
    serialized = json.dumps(manifest.model_dump(mode="json"))

    assert "super-secret" not in serialized
    assert redact_secrets({"api_key": "abc", "nested": {"password": "def"}}) == {
        "api_key": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
    }
    sanitized_url = redact_secrets(
        "https://user:pass@example.test/v1?api_key=abc&mode=fast"
    )
    assert "user" not in sanitized_url and "pass" not in sanitized_url
    assert "abc" not in sanitized_url and "mode=fast" in sanitized_url


def test_api_and_ui_translate_settings_to_canonical_config():
    api_config = _resolve_api_config(
        core="bologna",
        model="openai.gpt-oss-120b",
        retrieval_mode="hybrid",
        top_k=25,
        hybrid_alpha=0.3,
        candidate_multiplier=4,
    )
    ui_runtime = RuntimeSettings.from_chainlit_settings(
        {
            "model_name": "openai.gpt-oss-120b",
            "retrieval_mode": "hybrid",
            "use_unified_agent": True,
        },
        solr_core="bologna",
    )

    assert api_config.core == ui_runtime.experiment.core == "bologna"
    assert api_config.discovery_architecture == ui_runtime.experiment.discovery_architecture
    assert api_config.retrieval.mode == ui_runtime.experiment.retrieval.mode == "hybrid"
    assert api_config.interaction_mode == "autonomous"
    assert ui_runtime.experiment.interaction_mode == "human_gated"


def test_api_inline_values_are_only_replaced_by_explicit_overrides():
    config = _resolve_api_config(
        core="nyc",
        model="openai.gpt-oss-120b",
        retrieval_mode="keyword",
        top_k=30,
        hybrid_alpha=0.5,
        candidate_multiplier=5,
        config_data={
            "core": "bologna",
            "retrieval": {"mode": "hybrid", "top_k": 20},
        },
        explicit_fields={"top_k"},
    )

    assert config.core == "bologna"
    assert config.retrieval.mode == "hybrid"
    assert config.retrieval.top_k == 30
    assert config.interaction_mode == "autonomous"


def test_autonomous_runner_never_calls_input(monkeypatch):
    config = ExperimentConfig(interaction_mode="autonomous")
    runtime = object()

    def forbidden_input(*_args, **_kwargs):
        raise AssertionError("input() must not be called")

    monkeypatch.setattr(builtins, "input", forbidden_input)
    result = ExperimentRunner(config).run(
        "Question?",
        runtime_factory=lambda **_kwargs: runtime,
        executor=lambda question, actual_runtime, *, question_id, log_context: {
            "question": question,
            "runtime": actual_runtime,
            "question_id": question_id,
        },
    )

    assert result == {
        "question": "Question?",
        "runtime": runtime,
        "question_id": None,
    }


def test_divided_autonomous_runner_executes_real_separate_phases(monkeypatch, tmp_path):
    config = ExperimentConfig(
        discovery_architecture="divided",
        interaction_mode="autonomous",
    )
    runtime = SimpleNamespace(
        model_name=config.model,
        solr_core=config.core,
        csv_dir=tmp_path,
        portal_name="NYC",
        retrieval=RetrievalConfig(),
        experiment=config,
    )
    (tmp_path / "table.csv").write_text("value\n42\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["table.csv"])
    monkeypatch.setattr(
        "lakegen.service.phase12_agent",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unified must not run")),
    )
    monkeypatch.setattr(
        "lakegen.service.phase1_generate_keywords",
        lambda **_kwargs: (["value"], "raw", 2, "keyword reasoning"),
    )
    monkeypatch.setattr(
        "lakegen.service.phase2_select_tables",
        lambda **_kwargs: (
            ["table.csv"], ["table.csv"], {}, "selected", "tool trace", 3
        ),
    )
    generated = SimpleNamespace(
        tokens=4,
        clean_code="print(42)",
        code_raw="print(42)",
        rejected_reason="",
        error=None,
        raw_result="42",
    )
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute", lambda *_args, **_kwargs: generated
    )
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("The answer is 42.", 5)
    )
    monkeypatch.setattr("lakegen.service.persist_manifest", lambda *_args: tmp_path)
    logged = {}
    monkeypatch.setattr(
        "lakegen.service.save_experiment_log", lambda **kwargs: logged.update(kwargs)
    )
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("Question?", runtime)

    assert result.status == "completed"
    assert result.tables == ["table.csv"]
    assert result.tokens == {"p1_p2": 5, "p3": 4, "p4": 5}
    assert logged["tokens_phase1"] == 2
    assert logged["tokens_phase2"] == 3
    assert {call["phase"] for call in result.llm_calls} == {
        "discovery", "code", "result"
    }
    assert result.phase_metrics["total"]["latency_seconds"] >= 0
