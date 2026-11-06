Judge every requirement separately after considering the gold-blind code analysis.
Plausibility is not evidence. Metadata saying a source is new or updated does not
prove differing values correct. An alternative value needs a concrete source,
version/scope difference, causal operation, and evidence for every requirement.
Otherwise mark it unknown and prefer indeterminate. A verified item needs non-empty
concrete evidence. Do not enforce serialization shape.
Metadata is sufficient evidence for source_identity requirements, but metadata
alone never proves that a differing computed value is correct.
Judge the observed execution. Mark a requirement failed only when the supplied
result exhibits the violation or the code necessarily contradicts the requirement
for this execution. Hypothetical ties, nulls, missing categories, or future inputs
are concerns, not failures. Accept semantically equivalent operations: max/idxmax
may replace descending sort plus top-1, and nlargest/head after sorting may implement
top-k. A failed assessment must declare failure_basis as observed_result or
necessary_code_contradiction; otherwise use unknown.
Return JSON only:
{"proposed_disposition":"alternative_correct|partially_correct|incorrect|indeterminate","requirement_assessments":[{"requirement_id":"R1","status":"verified|failed|unknown","failure_basis":"observed_result|necessary_code_contradiction|none","evidence_source":"generated_code|code_analysis|result_comparison|metadata|gold_comparison","evidence":"..."}],"alternative_source_proof":{"source":"","version_or_scope_difference":"","causal_explanation":""},"rationale":"..."}

REQUIREMENTS: {{ requirements }}
GOLD-BLIND CODE ANALYSIS: {{ code_analysis }}
GOLD RESULT PREVIEW: {{ reference_result_preview }}
GENERATED RESULT PREVIEW: {{ generated_result_preview }}
DETERMINISTIC FULL-RESULT COMPARISON: {{ deterministic_comparison }}
SELECTED TABLE METADATA: {{ selected_metadata }}
