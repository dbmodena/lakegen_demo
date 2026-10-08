# Changelog

Behaviour changes made during the refactor, by date. Everything else on the
branch moves code without changing what a run does.

## 2026-10-08

- **Keyword memory keeps the questions' retrieval memories.** Persisting a
  run's keywords replaced the whole scope of the memory file, which dropped
  every question's retrieval memory stored under it. The keywords now merge
  into the scope (`lakegen.keyword_memory.persist_keyword_memory`).
- **The keyword memory survives concurrent writers, and each experiment
  has its own.** Several API processes shared one memory file under a
  process-local lock, so one process's read-modify-write could overwrite
  another's. Every read and write now holds a lock on a file beside the
  memory (`<name>.lock`, with `fcntl.flock`), and a run of an experiment
  remembers in `runs/<experiment>/memory.json`; runs outside an experiment
  (the default id) keep `.lakegen_keyword_memory.json`.
- **The coder sweep's rejection carries the coder's per-table judgment.**
  With `automatic_test_coder`, a full-context rejection returned to
  discovery without excluding the tables the coder proved unusable or
  carrying the ones it found useful, as the direct coder did. Both paths
  now apply the judgment (`CodingStage`/`CoderSweep` share it); the
  sweep's recorded variants gain `rejection_keep_tables` and
  `rejection_skip_tables`.
- **Batch table metrics pair each question with its own result.** The
  API's end-of-batch metrics paired questions and results by position, so
  a batch whose results were re-queued by hand scored questions against
  other questions' results. `lakegen.analysis.metrics.batch_cases` now
  pairs by source path and id (the last result of a re-queued question
  counts), and by position only for entries that carry no source.

## 2026-10-09

- **The divided judge's payload is read whole.** The judge's answer parser
  matched `FINAL_PAYLOAD:` up to the first closing brace, so a payload that
  nested a selection plan (every fallback after a stall or an agent error)
  was cut short: its reason read "No reasoning provided." and its tables
  were replaced by the top two candidates. The parser now decodes one
  complete JSON object (`lakegen.discovery.candidates.parse_table_selector_response`).
- **The coder's tools name the tools the coder has.** The coder gets
  `inspect_data`, `run_analysis` and `reject_data`, but the messages its
  tools returned told it to call `inspect_table`, `run_code`,
  `inspect_result`, `set_analysis_contract`, `plan_conflict` and
  `reject_tables`, none of which it could call. Every message, next action
  and required action now names an offered tool.
- **The NYC and UK lakes route Pneuma to our own services.** `config/lakes.yaml`
  sent `nyc` to port 8765 and `uk` to 8766, another user's servers, so a
  run outside an experiment spec (the chat, `lakegen ask`, a plain API
  call) queried them while our services listened on 8774 and 8767; only
  the specs and `LAKEGEN_PNEUMA_<LAKE>_BASE_URL` reached ours. The lakes
  now route to 8774 and 8767, `config/services.yaml` no longer needs its
  port override, and the example configs name the same routes.
- **A run's generated code lives under its manifest's id.** The engine
  named the coding directory with a fresh uuid unrelated to the manifest
  it wrote for the same run, so neither could be found from the other.
  The directory is now `coding/<manifest run_id>/`.
