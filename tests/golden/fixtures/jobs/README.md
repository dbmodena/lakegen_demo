# Stored API jobs, trimmed

`sample/` is an API job directory as `lakegen.experiments.analysis` reads one: for each
job its `<id>.json` (status, timestamps, `settings.resolved_config`,
`provider_switches`, `requeues`), its `<id>.questions.json` and its
`<id>.results.jsonl`. The two jobs come from the benchmark-lake grid of
2026-10-01/02 (`/data/bilel/lakegen/jobs/lakegen_jobs_bench`), cut to four
results each, with every result trimmed to the fields the analysis reads
(ranking, tables, tokens, elapsed time, the scalar code-evaluation fields, a
few manifest fields) and every question to its gold tables:

- `f438db7b…` (`bench-llm-target_bird-hybrid`): the model and judge were
  switched from OCI to Together by hand after result 83 and the Pneuma
  entity model seven minutes later; the four results are numbers 0, 1, 82
  and 83 of the job, so the switches fall between the last two.
- `3687ab4f…` (`bench-llm-chicago-pneuma`): results were removed and
  re-queued by hand, and an 80/20 Together/OCI split marks each result with
  the model it ran on; two results of each model.
