from types import SimpleNamespace

from lakegen.experiment_config import load_experiment_config
from lakegen.manifest import create_manifest
from lakegen.phases.phase3 import phase3_generate_code
from lakegen.reproducibility import initialize_reproducibility


def test_local_generators_repeat_for_same_seed_and_differ_for_other_seed():
    first = initialize_reproducibility(17)
    second = initialize_reproducibility(17)
    other = initialize_reproducibility(18)

    assert [first.python_random.random() for _ in range(4)] == [
        second.python_random.random() for _ in range(4)
    ]
    assert first.numpy_rng.random(4).tolist() == second.numpy_rng.random(4).tolist()
    assert first.python_random.random() != other.python_random.random()
    assert first.numpy_rng.random() != other.numpy_rng.random()


def test_runs_do_not_share_mutable_rng_state():
    consumed = initialize_reproducibility(9)
    untouched = initialize_reproducibility(9)
    consumed.python_random.random()
    consumed.numpy_rng.random()

    fresh = initialize_reproducibility(9)
    assert untouched.python_random.random() == fresh.python_random.random()
    assert untouched.numpy_rng.random() == fresh.numpy_rng.random()


def test_seed_comes_from_resolved_configuration_and_manifest(tmp_path):
    config = load_experiment_config(data_override={"seed": 73})
    manifest = create_manifest(config, base_dir=tmp_path, question="Question?")

    assert config.seed == 73
    assert manifest.seed == 73
    assert manifest.reproducibility == {
        "configured_seed": 73,
        "effective_seed": 73,
        "initialized_components": [
            "python_random_local_generator",
            "numpy_local_generator",
        ],
        "seed_applied_to": [],
        "instructions_provided_to": [],
        "generated_code_seed_instruction_provided": False,
        "generated_code_seed_usage_verified": False,
        "llm_provider_seed_supported": False,
        "llm_provider_seed_applied": False,
        "deterministic_llm_generation": False,
        "uncontrolled_components": ["oci_llm_generation"],
    }


def test_coder_receives_effective_seed_without_hardcoded_42(monkeypatch, tmp_path):
    table = tmp_path / "table.csv"
    table.write_text("value\n1\n", encoding="utf-8")
    captured = {}

    class PromptManager:
        def render(self, _name, prompt, **kwargs):
            return "system" if prompt == "system_prompt" else "initial"

    class LLM:
        temperature = 0.1

        def stream_chat(self, messages, **_kwargs):
            captured["prompt"] = messages[1].content
            return iter([SimpleNamespace(
                delta="print(1)", additional_kwargs={}, raw=None
            )])

    monkeypatch.setattr("lakegen.phases.phase3._detect_tabpfn_intent", lambda _q: "prediction")
    monkeypatch.setattr("lakegen.phases.phase3._tabpfn_enabled", lambda: True)

    instruction_events = []
    code, _tokens = phase3_generate_code(
        "predict", [table.name], [table.name], {}, "", LLM(), PromptManager(),
        tmp_path, seed=73, stream_reasoning=False,
        seed_instruction_recorder=lambda: instruction_events.append("provided"),
    )

    assert code == "print(1)"
    assert "effective seed for this run is 73" in captured["prompt"]
    assert "random_state=73" in captured["prompt"]
    assert "random_state=42" not in captured["prompt"]
    assert instruction_events == ["provided"]


def test_error_before_prompt_does_not_record_seed_instruction(tmp_path):
    instruction_events = []

    try:
        phase3_generate_code(
            "question", ["missing.csv"], [], {}, "", object(), object(),
            tmp_path,
            seed=73,
            seed_instruction_recorder=lambda: instruction_events.append("provided"),
        )
    except (FileNotFoundError, AttributeError):
        pass

    assert instruction_events == []


def test_telemetry_distinguishes_instruction_from_verified_usage():
    context = initialize_reproducibility(73)
    telemetry = context.telemetry(generated_code_seed_instruction_provided=True)

    assert telemetry["initialized_components"] == [
        "python_random_local_generator",
        "numpy_local_generator",
    ]
    assert telemetry["seed_applied_to"] == []
    assert telemetry["instructions_provided_to"] == ["code_generator"]
    assert telemetry["generated_code_seed_instruction_provided"] is True
    assert telemetry["generated_code_seed_usage_verified"] is False
    assert telemetry["deterministic_llm_generation"] is False
    for value in (
        telemetry["initialized_components"],
        telemetry["seed_applied_to"],
        telemetry["instructions_provided_to"],
        telemetry["uncontrolled_components"],
    ):
        assert len(value) == len(set(value))
