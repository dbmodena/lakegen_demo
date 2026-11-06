Adversarially review the proposed judgment. Find unsupported verified requirements,
ignored deterministic failures, semantic code errors, hardcoding, invented source
claims, and the smallest counterexample that invalidates the proposal. An objection
is resolved only by evidence already supplied.
Do not object that code lacks a redundant filter, date check, geometry check, or
exact source row-count assertion when matching metadata already verifies a
source_identity requirement.
The critic may invalidate unsupported evidence, but it must not invent a new failed
requirement. A critical or major objection changes an unsupported verified or failed
assessment to unknown; the deterministic adjudicator assigns the final disposition.
Do not object to semantically equivalent implementations or use hypothetical edge
cases as observed failures. A possible tie is not a top-1 violation when the actual
result contains one row; max or idxmax does not require an explicit sort.
Return JSON only:
{"verdict_supported":true,"objections":[{"objection_id":"O1","requirement_id":"R1","severity":"critical|major|minor","reason":"..."}],"unsupported_claims":["..."]}

REQUIREMENTS: {{ requirements }}
GOLD-BLIND CODE ANALYSIS: {{ code_analysis }}
PROPOSED JUDGMENT: {{ proposed_judgment }}
DETERMINISTIC FULL-RESULT COMPARISON: {{ deterministic_comparison }}
