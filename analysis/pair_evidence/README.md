# Pair evidence: evaluating `check_join_union`

Reproduces the evidence behind `src/lakegen/agent_tools/join_keys.py` and `union_mapping.py` (the
measured replacement for the old name-only join/union rule). Nothing here runs in CI.

```
PORTAL=uk  PYTHONPATH=src:. python analysis/pair_evidence/eval_joins.py     # ~1 min
PORTAL=uk  PYTHONPATH=src:. python analysis/pair_evidence/eval_unions.py    # ~6 min
PORTAL=nyc PYTHONPATH=src:. python analysis/pair_evidence/eval_joins.py     # ~12 min
PORTAL=nyc PYTHONPATH=src:. python analysis/pair_evidence/eval_unions.py    # ~5 min
```

Needs OrQa's generated benchmark (`$ORQA_DATA_DIR`, default `/home/bilel/gits/orqa/data`, folder
`<portal>/candidates_discovery/`) and the LakeGen tables for the portal.

## What they measure

- `eval_joins.py`: OrQa's labelled joins (candidate relationships + `.merge(...)` calls of
  judge-approved queries). A key "reproduces" the labelled join when an inner join on it yields the same
  matched row pairs (>= 95%). Baseline = the previous rule (the best COMA pair is the key).
- `eval_unions.py`: OrQa's labelled unions + tables that queries actually `pd.concat`, against random
  negatives, comparing the previous rule (average COMA score >= 0.5) with the tiered mapping.
- `orqa_labels.py`: the label extraction (AST for the executed code). SQL joins are not extracted.

## Numbers at the time of writing

| | previous rule | measured |
|---|---|---|
| UK labelled joins accepted / top-1 reproduces / best-of-3 | - / 20 of 42 / 25 of 42 | 42 of 56 / 25 / 37 |
| NYC labelled joins accepted / top-1 / best-of-3 | - / 106 of 223 / 145 of 223 | 223 of 292 / 125 / 161 |
| random lake pairs called a union (UK / NYC) | 20.7% / 18.8% | 1.7% / 0.4% (UNION tier) |
| OrQa-labelled unions found, any tier (UK / NYC) | 100% / 98.7% | 100% / 98.1% |

## Read with care

- OrQa's labels come from an LLM plus the same name matcher, so recall against them is partly
  circular; executed queries are the independent evidence (about 50 executed merges).
- The thresholds (80% / 50% coverage, <= 2 shared values, 50% match rate) are judgment calls chosen
  while looking at these UK and NYC pairs; the SUBSET UNION tier was added after seeing NYC, so NYC
  is not a clean holdout for it.
- Accepted "negatives" are often genuine same-schema files OrQa never labelled. Look at them.
