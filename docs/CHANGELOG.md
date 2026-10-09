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

- **Same-named numbering that lines up is offered as a join by position.**
  Two columns of consecutive numbers (`sezione` 1..449 and 1..446) were
  always rejected as row counters. When both sides carry the column under
  one name (not `id`, `objectid` or another row counter) and at least 90%
  of each side's rows find their number on the other, the pair is now a JOIN
  candidate, ranked last, with a note to check the number means the same in
  both tables. On the 5 portals' join conditions this finds one more
  (77% to 78%); on 400 unrelated portal pairs it raised one more false
  alarm.
- **The join check finds foreign keys that the column names give away.**
  Valentine scores `alignment_id` against `id` low and keeps only the best
  partner of each column, so such a pair was never measured. The check now
  also proposes pairs whose names share a word (`link_to_member` /
  `member_id`) or where one column is a generic key and the other names its
  table (`alignment_id` / `id` of `superhero_alignment`)
  (`matching/name_cue.py`). Each proposal is counted on the whole key
  columns at call time and kept only when at least 90% of one side's rows
  find a partner (no index, no offline step). On the benchmark join
  conditions the BIRD ones found went from 70% to 98% (the true key in the
  first three from 70% to 93%); the portals' found rate stayed at 77%.
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
- **The chat's starter questions and the UK reference sidebar read files
  the repository holds.** `config/lakes.yaml` named `queries/*.json` and
  `benchmark/generated_queries_semantic_uk.json`, none of which exist in
  the checkout, so the chat showed only the built-in starters and the UK
  sidebar nothing. Every lake now takes its starters from its curated
  `benchmark/100q_<lake>.json`, and the UK sidebar reads that file's cases
  (the generated-queries shape is still read when a file holds it).
- **The example config's benchmark path is in the repository.**
  `config/experiment.example.yaml` named a file in another user's home
  directory as its retrieval-benchmark input; it now names
  `benchmark/100q_nyc.json`, as the UK examples name theirs.
- **The join/union check counts the join over every row and says when it is empty or
  explodes.** `check_join_union` counted shared keys and the join size on
  at most 20,000 sampled rows per table, so a many-to-many key of two big tables
  was reported 4 to 100 times too small (808 million rows read as 197 million),
  and a rare shared value could be missed. The best keys are now counted over all
  the rows of their key columns (a large sample, marked `~`, for tables beyond
  16 million cells), and each key states its inner, left and outer join rows,
  whether the merge is EMPTY, SAFE, a FANOUT (more rows than the larger table) or
  EXPLODES (the rule of the sandbox's `CARTESIAN_PRODUCT_DETECTED` guard, now one
  constant in `lakegen.domain.merge_limits`), and how many rows the join has once
  both tables are aggregated to the key. It also reports what a plain `pd.merge`
  does on the columns as stored (nothing, an error between text and numbers, or
  millions of rows from missing keys matching each other). The shared repeated
  columns (a year, a district) are listed and measured together as the grain.
  Measured on 127 reference merges of the OrQa lakes: the size class agrees with
  the real pandas merge on 82 of 82, the plain-merge size on 79 of 79, and the
  key columns of 91% of the references appear in the report (61% before).
- **Valentine also matches renamed columns on their values.** Columns that find no
  name partner are scored on their values with COMA's instance matcher, so a key
  renamed `zona_2001` or `col_3` is a candidate; it is kept only when the
  measured overlap confirms it. On 220 real pairs with perturbed names the share of
  true keys that reach the threshold rises from 78% to 94% for a mix of renamings.
  Equal scores in the union alignment now go to the identical name.
