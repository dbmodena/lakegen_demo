Extract an atomic checklist of only the semantic outcomes explicitly required by
the question. Split genuinely independent filters,
periods, scope, grouping, measures, aggregation, ordering, limits, units, and
completeness requirements. Do not invent implementation steps, source-authority
requirements, exact output labels, zero-valued categories, sorting before a top-1
result, or filters already guaranteed by the named input snapshot. A method is a
requirement only when the question explicitly requires that method. Mark optional
presentation details non-essential and report ambiguities instead of resolving them
by creating requirements. Never make scalar/table/list/JSON/DataFrame shape,
single-row wrapping, or exact column labels an essential requirement. Preserve a
content requirement when a representation actually omits one of the values the
question requests.
Classify each requirement by evidence scope: source_identity when a named dataset,
edition, organisation, period, geography, or feature type identifies the selected
source; row_operation when rows from a broader source must be filtered;
computation for transformations and aggregation; output for returned content.
Before returning, remove semantic duplicates. Never turn a descriptive fact from
the expected-result description (for example "the 43 areas") into a separate
row-count assertion unless the question explicitly asks to count, validate, or
return that cardinality. A named layer or population such as "Faughan network
contribution categories" identifies the source; it does not imply an additional
row filter unless the selected source contains multiple named networks.
Selected tables and metadata are context for choosing evidence_scope only. Never
copy a selected filename, identifier, title, period, organisation, or layer name
into a requirement unless that fact is already required by the question or expected
description. State source requirements independently of the candidate actually
selected; a later stage will decide whether the candidate satisfies them.
Return JSON only:
{"requirements":[{"id":"R1","type":"filter|scope|grouping|measure|aggregation|ordering|limit|unit|completeness|other","evidence_scope":"source_identity|row_operation|computation|output","description":"...","essential":true}],"ambiguities":["..."]}

QUESTION: {{ question }}
SELECTED TABLES: {{ selected_tables }}
SELECTED TABLE METADATA: {{ selected_metadata }}
