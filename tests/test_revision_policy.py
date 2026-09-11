from lakegen.revision_policy import classify_revision, technical_repair_eligible


def test_revision_policy_routes_generic_failure_classes():
    assert classify_revision({"stage": "execution", "category": "missing_column"})[
        "action"
    ] == "REPAIR_CODE"
    assert classify_revision(coverage_warnings=["requested top 3 is missing"])[
        "action"
    ] == "REVISE_RESULT"
    assert classify_revision(rejection_details={
        "missing_requirements": ["borough code"],
        "inspected_evidence": "No selected table contains that field.",
    })["action"] == "RETRY_DISCOVERY"
    assert classify_revision()["action"] == "CONTINUE"


def test_technical_credit_is_bounded_to_retryable_runtime_errors():
    assert technical_repair_eligible({
        "stage": "execution", "category": "type_error", "retryable": True,
    }) is True
    assert technical_repair_eligible({
        "stage": "preflight", "category": "column_resolution_error", "retryable": True,
    }) is False
    assert technical_repair_eligible({
        "stage": "execution", "category": "security_error", "retryable": False,
    }) is False
