Analyze what the generated code actually computes without seeing the gold result.
For every requirement, report positive support, contradiction, or unknown. Absence
of an obvious bug is not evidence. Detect hardcoding, missing filters, wrong
aggregation or granularity, unsafe joins, and incomplete output. A supplied local
dataset path is not answer hardcoding; hardcoding means embedding answer values or
logic that bypasses the requested computation.
For source_identity requirements, matching selected-table metadata is positive
evidence and no redundant code filter or row-count assertion is required. For
example, a table identified as the December 2021 boundaries, the Faughan network
layer, or a dated organogram already establishes that source identity. Require a
code filter only for row_operation requirements over a broader source.
Return JSON only:
{"requirement_analysis":[{"requirement_id":"R1","status":"supported|contradicted|unknown","evidence":"concrete code operation or reason"}],"hardcoding_risk":"none|possible|high","observed_operations":["..."],"concerns":["..."]}

REQUIREMENTS: {{ requirements }}
SELECTED TABLES: {{ selected_tables }}
SELECTED TABLE METADATA: {{ selected_metadata }}
GENERATED CODE: {{ generated_code }}
