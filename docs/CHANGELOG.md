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
