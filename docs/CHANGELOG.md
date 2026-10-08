# Changelog

Behaviour changes made during the refactor, by date. Everything else on the
branch moves code without changing what a run does.

## 2026-10-08

- **Keyword memory keeps the questions' retrieval memories.** Persisting a
  run's keywords replaced the whole scope of the memory file, which dropped
  every question's retrieval memory stored under it. The keywords now merge
  into the scope (`lakegen.keyword_memory.persist_keyword_memory`).
