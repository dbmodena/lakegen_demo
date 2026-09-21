import json
from types import SimpleNamespace

from lakegen.semantic_code_judge import judge_semantic_code_result


class FakePromptManager:
    def __init__(self):
        self.calls = []

    def render(self, _agent, prompt_type, **kwargs):
        self.calls.append((prompt_type, kwargs))
        return f"Stage: {prompt_type}"


class FakeLlm:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.token_usage_total = 0

    def chat(self, _messages):
        message = SimpleNamespace(content=next(self.contents), additional_kwargs={})
        return SimpleNamespace(message=message, raw={"usage": {"total_tokens": 17}})


def pipeline_responses(*, final="alternative_correct", critic_supported=True,
                       assessment="verified", hardcoding="none",
                       proof=True):
    source_proof = {
        "source": "newer.parquet",
        "version_or_scope_difference": "newer snapshot",
        "causal_explanation": "the newer snapshot contains two more rows",
    } if proof else {"source": "", "version_or_scope_difference": "",
                     "causal_explanation": ""}
    item = {
        "requirement_id": "R1", "status": assessment,
        "failure_basis": (
            "necessary_code_contradiction" if assessment == "failed" else "none"
        ),
        "evidence_source": "generated_code", "evidence": "code counts rows",
    }
    return [
        json.dumps({"requirements": [{
            "id": "R1", "type": "measure", "description": "count records",
            "evidence_scope": "computation", "essential": True,
        }], "ambiguities": []}),
        json.dumps({"requirement_analysis": [{
            "requirement_id": "R1", "status": "supported",
            "evidence": "code counts rows",
        }], "hardcoding_risk": hardcoding, "observed_operations": ["count"],
            "concerns": []}),
        json.dumps({"proposed_disposition": final,
                    "requirement_assessments": [item],
                    "alternative_source_proof": source_proof,
                    "rationale": "evidence supports the count"}),
        json.dumps({"verdict_supported": critic_supported, "objections": [],
                    "unsupported_claims": []}),
    ]


def run_judge(llm, prompt_manager=None, deterministic=None):
    return judge_semantic_code_result(
        question="How many records?", expected_description="A count",
        reference_result=10, selected_tables=["newer.parquet"],
        selected_metadata={"newer.parquet": {"description": "Updated data"}},
        generated_code="result = len(df)", generated_result=12,
        deterministic_evaluation=deterministic or {"exact_result_match": False},
        llm=llm, prompt_manager=prompt_manager or FakePromptManager(),
    )


def test_semantic_judge_accepts_fully_evidenced_alternative():
    judgment, tokens = run_judge(FakeLlm(pipeline_responses()))

    assert judgment["disposition"] == "alternative_correct"
    assert judgment["all_requirements_verified"] is True
    assert judgment["requirements"][0]["evidence_scope"] == "computation"
    assert judgment["confidence"] == 1.0
    assert tokens == 68


def test_semantic_judge_downgrades_missing_alternative_proof():
    judgment, _ = run_judge(FakeLlm(pipeline_responses(proof=False)))

    assert judgment["requested_disposition"] == "alternative_correct"
    assert judgment["disposition"] == "indeterminate"
    assert "alternative-source proof is incomplete" in judgment["rationale"]


def test_semantic_judge_rejects_proof_from_unselected_source():
    responses = pipeline_responses()
    proposed = json.loads(responses[2])
    proposed["alternative_source_proof"]["source"] = "unselected.parquet"
    responses[2] = json.dumps(proposed)

    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["disposition"] == "indeterminate"
    assert "alternative-source proof is incomplete" in judgment["rationale"]


def test_semantic_judge_downgrades_unverified_requirement():
    judgment, _ = run_judge(
        FakeLlm(pipeline_responses(assessment="unknown"))
    )

    assert judgment["disposition"] == "indeterminate"
    assert judgment["requirements_missing"] == ["R1"]


def test_semantic_judge_downgrades_unresolved_critic():
    responses = pipeline_responses(critic_supported=False)
    responses[3] = json.dumps({
        "verdict_supported": False,
        "objections": [{"objection_id": "O1", "requirement_id": "R1", "severity": "major",
                         "reason": "the newer source claim is unsupported"}],
        "unsupported_claims": ["newer snapshot"],
    })
    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["disposition"] == "indeterminate"
    assert "critic downgraded R1" in judgment["rationale"]


def test_critic_can_invalidate_unsupported_failed_assessment():
    responses = pipeline_responses(final="incorrect", assessment="failed")
    responses[3] = json.dumps({
        "verdict_supported": False,
        "objections": [{
            "objection_id": "O1", "requirement_id": "R1", "severity": "major",
            "reason": "the supplied evidence does not prove a failure",
        }],
        "unsupported_claims": [],
    })

    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["requirement_assessments"][0]["status"] == "unknown"
    assert judgment["disposition"] == "indeterminate"


def test_semantic_judge_keeps_hardcoding_risk_diagnostic_only():
    for risk in ("possible", "high", "unknown"):
        judgment, _ = run_judge(
            FakeLlm(pipeline_responses(hardcoding=risk))
        )

        assert judgment["disposition"] == "alternative_correct"
        assert judgment["code_analysis"]["hardcoding_risk"] == risk


def test_semantic_judge_does_not_let_an_extra_llm_overrule_critic():
    responses = pipeline_responses(critic_supported=False)
    responses[3] = json.dumps({
        "verdict_supported": False,
        "objections": [{"objection_id": "O1", "requirement_id": "R1",
                         "severity": "major", "reason": "check the count"}],
        "unsupported_claims": [],
    })
    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["disposition"] == "indeterminate"


def test_semantic_judge_ignores_llm_disposition_when_evidence_is_verified():
    judgment, _ = run_judge(FakeLlm(pipeline_responses(final="incorrect")))

    assert judgment["requested_disposition"] == "incorrect"
    assert judgment["disposition"] == "alternative_correct"


def test_output_only_failure_is_normalized_to_partially_correct():
    responses = pipeline_responses(final="incorrect")
    requirements = json.loads(responses[0])
    requirements["requirements"].append({
        "id": "R2", "type": "completeness", "evidence_scope": "output",
        "description": "include both averages", "essential": True,
    })
    responses[0] = json.dumps(requirements)
    for index in (2,):
        payload = json.loads(responses[index])
        payload["requirement_assessments"].append({
            "requirement_id": "R2", "status": "failed",
            "failure_basis": "observed_result",
            "evidence_source": "generated_code", "evidence": "averages omitted",
        })
        responses[index] = json.dumps(payload)

    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["requested_disposition"] == "incorrect"
    assert judgment["disposition"] == "partially_correct"


def test_source_identity_failure_is_normalized_to_incorrect():
    responses = pipeline_responses(final="partially_correct")
    requirements = json.loads(responses[0])
    requirements["requirements"][0]["evidence_scope"] = "source_identity"
    responses[0] = json.dumps(requirements)
    for index in (2,):
        payload = json.loads(responses[index])
        payload["requirement_assessments"][0].update({
            "status": "failed", "failure_basis": "necessary_code_contradiction",
            "evidence": "wrong source",
        })
        responses[index] = json.dumps(payload)

    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["requested_disposition"] == "partially_correct"
    assert judgment["disposition"] == "incorrect"


def test_semantic_judge_fails_closed_on_invalid_response():
    judgment, tokens = run_judge(FakeLlm(["not json", "still not json"]))

    assert judgment["disposition"] == "indeterminate"
    assert judgment["judge_error"].startswith("JSONDecodeError:")
    assert tokens == 0


def test_code_analysis_is_gold_blind_and_comparison_omits_result_types():
    manager = FakePromptManager()
    deterministic = {
        "expected_result_type": "table", "result_type_match": False,
        "exact_result_match": False, "representation_equivalent_match": True,
        "requirement_checks": {"result_type": False, "row_count": True},
        "row_f1": 1.0,
    }
    run_judge(FakeLlm(pipeline_responses()), manager, deterministic)

    calls = dict(manager.calls)
    code_kwargs = calls["code_analysis"]
    assert "reference_result_preview" not in code_kwargs
    assert "deterministic_comparison" not in code_kwargs
    comparison = json.loads(calls["evidence_judgment"]["deterministic_comparison"])
    assert "expected_result_type" not in comparison
    assert "result_type_match" not in comparison
    assert "result_type" not in comparison["requirement_checks"]


def test_verified_status_without_evidence_is_changed_to_unknown():
    responses = pipeline_responses()
    proposed_payload = json.loads(responses[2])
    proposed_payload["requirement_assessments"][0]["evidence"] = ""
    responses[2] = json.dumps(proposed_payload)
    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["disposition"] == "indeterminate"
    assert judgment["requirement_assessments"][0]["status"] == "unknown"


def test_hypothetical_failure_without_basis_is_changed_to_unknown():
    responses = pipeline_responses(final="incorrect", assessment="failed")
    proposed = json.loads(responses[2])
    proposed["requirement_assessments"][0]["failure_basis"] = "none"
    proposed["requirement_assessments"][0]["evidence"] = "a tie could happen"
    responses[2] = json.dumps(proposed)

    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["requirement_assessments"][0]["status"] == "unknown"
    assert judgment["disposition"] == "indeterminate"


def test_observed_failure_with_basis_is_preserved():
    responses = pipeline_responses(final="incorrect", assessment="failed")
    proposed = json.loads(responses[2])
    proposed["requirement_assessments"][0]["failure_basis"] = "observed_result"
    proposed["requirement_assessments"][0]["evidence"] = "actual result has two rows"
    responses[2] = json.dumps(proposed)

    judgment, _ = run_judge(FakeLlm(responses))

    assert judgment["requirement_assessments"][0]["status"] == "failed"
    assert judgment["disposition"] == "incorrect"
