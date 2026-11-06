# Changelog

Behaviour changes made during the refactor, by date. Everything else on the
branch moves code without changing what a run does.

## 2026-11-02

- **A third dedicated endpoint, and the model picker names what each one
  serves.** `config/models.yaml` adds `dedicated.qwen3.8-27b` (provider
  `oci_dedicated`, `model_env: DEDICATE_OCI_MODEL_3`): Qwen3.8 27B on an OCI
  Dedicated AI Cluster endpoint, beside the two gpt-oss-20b endpoints. Every
  model entry now has a `title`, the label the webapp's model picker shows
  (`MODEL_ITEMS`, label -> name, as the retriever picker already did), so
  the three endpoints read "gpt-oss-20b (OCI dedicated endpoint 1)",
  "... endpoint 2" and "Qwen3.8 27B (OCI dedicated endpoint 3)" instead of
  their names; a `ModelSpec` without a title keeps its name. The CLI, the API
  and the experiment configs still take the names.
- **The table selector offers the other selections that also answer.** A
  generic question ("how many parishes were there in 2017?") is answered by
  several tables of the lake alike: the same boundaries at three resolutions,
  the same list in two languages, the same subject with or without some areas
  the question does not name. After the contrast and drop checks, the new
  alternatives check (`selectors/memgpt/alternatives.py`) shows the selection
  beside the selections one change away from it (a selected table swapped for
  one of its closest look-alikes, or one of several tables left out, at most
  twelve) and asks once which of them hold every requirement; the model keeps
  them best first with a line each on what differs. The selection itself never
  moves: the coder runs on it, and the alternatives go to the person asking.
  The plan records them under `alternatives` (`pool`, `offered` with the
  tables, the change and the note), the trace as `[alternatives]`, the chat
  shows them as a step and under the selected tables in the selection
  summary, and the batch metrics add `AnyOfSelection` (the selection or an
  offered alternative is the gold) beside `ExactSelection`. Measured in
  experiments/alternatives_lab (arm `vet`, 228 dev questions): the gold is the
  selection or an offered alternative in 0.754 of the questions against 0.684
  for the selection alone (16 of 72 misses recovered, of the 23 one change
  away), 0.5 alternatives per question, three quarters accepted by an LLM
  judge, 0.8 calls and 2.3k tokens per question. Copies of a selected table
  found on the files added nothing, so no file is read. The forced-choice
  tool of the checks is now named after its function (`choose` for the
  contrast and drop checks as before, `keep` here).

## 2026-10-30

- **TabPFN is gone.** The coder no longer detects a prediction, forecasting,
  classification, regression or causal intent in the question, and no
  longer appends the TabPFN rules (or the "backend disabled" note) to its
  prompt: every question gets the same prompt. The `LAKEGEN_ENABLE_TABPFN`
  switch, its `enable_tabpfn` line in the manifest's `switches` block and
  the commented-out `tabpfn` dependency are removed; the `code_generator`
  system prompt names only scikit-learn where it mentioned both.
- **The package is laid out in seven areas.** `src/lakegen` is now
  `entrypoints/` (api, cli, webapp: the Chainlit app moved from `src/app.py`
  to `entrypoints/webapp/app.py`, the `src/api.py` shim is gone),
  `pipeline/` (the engine, stages, strategies, `run/`, `wiring/`, and the
  phases `discovery/`, `coding/` with its `tools/`, `evaluation/`, `answer`,
  plus `evidence/` and `observability/`), `retrieval/` (with `solr/` and
  `pneuma/` inside), `llm/` (providers, embeddings, rerank, agent runner,
  the prompt renderer), `config/`, `datastructures/` (was `domain/`) and
  `utils/` (table files, naming rules, table ids, merge limits). The prompts
  are files at the repository root (`prompts/<agent>/*.md`,
  `LAKEGEN_PROMPTS_DIR` points elsewhere), no longer package data. Names
  say what things do: `write_keywords`, `select_tables`, `discover_tables`
  and `write_and_run_code` replace the phase-numbered functions,
  `CoderTools`/`CoderState`/`CoderResult` the `Phase3*`/`P3State` names,
  `AgentStall`, `RetrievalService`, `SolrClient`, `PneumaHttpClient`,
  `DuckDBSemanticRetriever`, `MemGPTSelector`/`MemGPTAgent` their longer
  forms; `csv_log` is `experiment_log`, `judge` is `table_inspection`,
  `ledger` is `requirement_ledger`, `bootstrap` is `startup`, `files` is
  `config_files`. The experiment log writes its paths with `Path`. Dead code
  removed: the thinking capture of the old architect,
  `requirement_ledger_block_message`, two unused inspection helpers,
  `ensure_project_paths`. The import contracts and the tests' tree follow
  the areas. Runs, logs and stored configurations are unaffected.
- **The chat's UK reference-benchmark sidebar is gone.** The element sidebar
  that listed the UK benchmark's questions with their gold tables and expected
  results (`lakegen.interfaces.chainlit.benchmark_sidebar`), its
  `reference_questions` lake setting (`config/lakes.yaml`,
  `Lake.reference_questions_path`) and its two i18n keys are removed; the
  starter questions stay. Kept locally in `legacy/uk_benchmark_sidebar/`.
- **The unified agent is the MemGPT selector with a search tool.** The
  unified architecture (`discovery.architecture: unified`) now runs the
  divided architecture's MemGPT agent (`lakegen.workflow.discovery.unified`)
  with a `search` tool and one rule the system keeps about it: search is
  offered while there is nothing to work from and withdrawn once a search has
  found candidates; it comes back when a requirement stays open after the
  agent has inspected what it found (at most 2 such searches, 4 in all). The
  first search is production's retrieval with the query's words as keywords;
  a later one is the retrieval service's new `search_for` (OR over the words,
  KNN over the query, reciprocal-rank fusion, rerank). The contrast and drop
  checks run after it confirms, and its selection reaches the coder like the
  divided round's. Measured in experiments/unified_lab (228 dev questions,
  gpt-oss-20b): exact selections 0.750 against 0.566 for the agent it
  replaces (49 fixed / 7 broken, p < 0.001) and 0.768 for the divided round
  (6 / 10, within rerun noise), at 72 s per question against 78 and 97. The
  chat shows its tool calls as steps, as the divided selector's.
  Gone with the old architect: its tools package (`lakegen.tools.unified`:
  search_tables with its ban memory, inspect_columns, expand_candidates,
  check_join_union, the confirmation ledger, the semantic planner and the
  coder brief), the context editor (state board, compact history), the
  keyword and retrieval memory files (`.lakegen_keyword_memory.json`,
  `runs/<experiment>/memory.json`, `memory_scope`), the carried inspections,
  and the dials only it read: `discovery.budget.*` except
  `max_zero_result_retries`, `discovery.keyword_search`,
  `discovery.context`. Stored configurations still load (those settings are
  dropped). The run trace's selection records lose `semantic_plan_failure`,
  `initial_stall_reason`, `recovery_started`, `recovery_stop_reason`,
  `semantic_planner_attempts` and `semantic_draft_present`, the discovery
  record `expansion_requirements`. `phase12_agent` takes `lake` and
  `tool_callback` and no longer `retrieval_memory`, `retrieval_observer`,
  `require_semantic_plan` or `discovery_config`; the round state is
  `lakegen.workflow.discovery.state.DiscoveryState` (was
  `lakegen.tools.unified.P12State`). The hint a rejected unified round hands
  the next no longer says the kept tables' schemas are pre-loaded (they are
  retrieved first, and inspected again). The old code, its tests and a
  recovery plan are kept locally in `legacy/unified_architect/` (`RECOVERY.md`).

## 2026-10-29

- **The orchestrated discovery mode is gone.** `tool_access=orchestrated_context`
  (the code retrieved and prepared a context, and the agent selected from it in
  one tool-free turn), its memory (`orchestrated_memory_enabled`), the two
  orchestrated strategies, the context preparer (`discovery/orchestrated.py`,
  `discovery/context.py`), `SelectorKind.LLM` and the run trace's
  prepared-context telemetry (`configured_tool_access`, `execution_path`,
  `prepared_candidate_count`, `prepared_context_utf8_bytes`,
  `empty_context_retries`, the three `*_error` fields) are removed. Discovery
  is agentic: the strategy is chosen by the architecture alone. Stored
  configurations still load: an `agentic` `tool_access` and a false
  `orchestrated_memory_enabled` are dropped (every stored job has them); a
  configuration that asks for the orchestrated mode is refused. The run
  trace's telemetry block keeps its `tool_access` key so old and new logs read
  the same way. The code, its tests and goldens, and a recovery plan are kept
  locally, out of git, in `legacy/orchestrated/` (`RECOVERY.md`).
- **Batches run on the experiment harness's batch server.** LakeGen's API
  answers one question (`/health`, `/v1/query`); the batch endpoints
  (`/v1/batches`, `/v1/batches/files`, `/v1/batches/{job_id}`,
  `/v1/benchmarks`), the batch runner, its job store and the report metrics
  moved with the rest of `lakegen.experiments` to `experiment_runner/batches`.
  The batch server (`python -m experiment_runner.batches`, or
  `uvicorn experiment_runner.batches.server:app`, the `api` service of
  `config/services.yaml`) includes LakeGen's API, so specs and clients use the
  same URLs. A submission's configuration and the limit on concurrent
  questions are LakeGen's (`lakegen.interfaces.api.submission`), shared by both.
- **Jev is gone.** TypeSafe's Jev selector (`selectors/jev.py`,
  `jev_service.py`), its preset, its `discovery.jev_*` settings
  (`discovery.selector.model`, `max_candidates`, `threshold`, `reject_floor`),
  its API key (`TYPESAFE_API_KEY`, `JEV_API_KEY`) and its fallbacks are
  removed: the divided architecture selects with the MemGPT agent only (the
  orchestrated context, until its removal below, with its tool-free turn). As with `llm`, the
  value `jev` stays in `discovery.selector` so the configurations of the runs
  made with it still load (stored jobs' `jev_*` settings are dropped on load,
  as the cascade's size is); a run that asks for it is refused. The grid specs
  that ran it (`bench_grid`, `qwen_grid`) and the paper's analysis of those
  runs keep naming it.
- **The experiment harness is `experiment_runner/`, outside LakeGen.** Specs,
  the run store, executors, services, the legacy job index, the analysis of
  runs, the runs dashboard and the retrieval-only benchmark moved out of
  `lakegen` into a package at the repository root, which imports LakeGen and
  is never imported by it (an import contract). `lakegen exp|runs|services|bench`
  are now `python -m experiment_runner plan|run|status|index-legacy|services|bench`,
  and the dashboard is `python -m experiment_runner.dashboard` (as in
  `config/services.yaml`). `lakegen.experiments` keeps what the API's batches
  need: `batch`, `jobs`, and the report's `metrics` and `results` (formerly
  under `analysis/`); the benchmark's metrics log, which batches write, is in
  `lakegen.retrieval.evaluation`. `src/cli.py`, which only aliased
  `lakegen.interfaces.cli.ask`, is gone. `.gitignore` ignored every folder
  named `experiments`, so new files in `src/lakegen/experiments` were left out
  of git; it now ignores the lab folder at the root only.
- **Config files have one layout.** Every config file is in the nested layout
  of `config/base.yaml` and starts with `schema_version: 2`; a flat file
  (`core:`, `discovery_architecture:` ...) is refused with an error naming the
  layout to use. `--set` and the specs' `set`, grid and per-value keys take the
  nested paths (`retrieval.hybrid.alpha`, `lake`); a flat spelling is refused
  with its nested path. The flat form stays internal: the API still accepts it,
  manifests record it and stored runs load as before. The seven
  `config/examples/experiment.example*.yaml` files are gone (each was the base
  plus one retrieval preset); `lakegen config show` prints the nested layout by
  default (`--format flat` for the internal record) and lists only the
  settings it cannot carry that differ from their defaults. The golden
  fixtures were converted with `to_v2`; their resolved configs differ only in
  settings of mode blocks their own retrieval mode never reads, which now keep
  the defaults.
- **The prompts are Markdown.** Each agent's prompts are files in its folder,
  `workflow/prompts/<agent>/<prompt>.md`, instead of keys of one YAML file;
  the rendered text is unchanged (checked template by template). The prompt
  manager only reads and renders them.
- **The Solr cores are built from `solr_builder/`.** Indexing, the table
  representation, the UK document rules and the embedding command moved out of
  `lakegen` into a package at the repository root, with the two core builders
  (`build_portal_core`, `build_uk_fixed`) that lived as untracked local scripts.
  `lakegen index` is now `python -m solr_builder.index`; `lakegen.retrieval`
  no longer re-exports the indexer. The builders run as modules, take the Solr
  URL from the settings, and `build_portal_core` needs `--orqa-src` (it had a
  path on one machine as default). An import contract keeps LakeGen from
  importing the builder.
- **The Pneuma engine lives in `pneuma_engine/`, apart from LakeGen.** The
  index builder and the HTTP service moved out of `scripts/pneuma/` into a
  package at the repository root, run from there inside `.venv-pneuma`:
  `python -m pneuma_engine.server`, `python -m pneuma_engine.bootstrap`,
  `pneuma_engine/run_bootstrap.sh` and `pneuma_engine/install.sh` (see its
  README). LakeGen keeps only the client side in
  `lakegen.integrations.pneuma`, and an import contract fails the gate if it
  ever imports the engine. The engine logs what it printed before (same lines,
  on stdout); the server also logs each failed query with its traceback, and
  the bootstrap rejects an out-of-range option with a usage error instead of a
  traceback. `config/services.yaml` names each lake's index folder relative to
  the new `LAKEGEN_PNEUMA_INDEX_DIR` (`pneuma-out` by default) instead of a
  path on one machine, and the bootstrap wrapper writes to `pneuma-out/<lake>`
  by default, not `/data/pneuma/<lake>`.
- **The Pneuma retriever works without an injected client again.** Since the
  move of the retrieval modes (1eb9cac), `PneumaRetriever` read
  `client.create_pneuma_client` from its own `client=None` argument, so every
  `pneuma` and `pneuma_seeker` run built by `TableRetrievalService` failed
  with `AttributeError` before querying the service.
- **Errors that were swallowed silently are logged.** Every `except` that fell
  back without a trace now logs (debug for expected fallbacks, warning for the
  rest); the fallbacks are unchanged. Diagnostic prints became warnings: a
  stream error or repetition loop in the coder and keyword writer, a failed
  keyword count, a Solr error, an unavailable selector, a crashed cell-scan
  worker, an i18n formatting error. The streamed model output and the
  `[phase…]` progress lines are still printed. A failed batch is logged with
  its traceback. `assert` statements became explicit checks (`RuntimeError`;
  the Chainlit driver keeps `AssertionError`, which its workflow does not
  mistake for a lake without tables).
- **The whole tree meets one bar, in the gate and in CI.** `scripts/check.sh`
  runs ruff (lint and `format --check`), bandit, the import contracts, the
  suite against the baseline and an import smoke on every file, and
  `.gitlab-ci.yml` runs its `lint` and `tests` stages; `scripts/ruff_legacy.txt`
  is gone. Public code has Google-style docstrings (pydocstyle and pydoclint
  in ruff, `tests/test_docstrings.py`); the docstrings and pydantic schemas a
  model reads as tool descriptions are unchanged, byte for byte.
- **Repository hygiene.** `tests/` is no longer gitignored, so new test files
  can be committed; the `scripts/` exceptions of `.gitignore` name the grouped
  folders, so ruff checks them. The Chicago, KramaBench and TARGET question
  sets are files instead of symlinks into `data/`, so a clone holds them. The
  test of `experiments/retrieval_lab` moved next to it, since that folder is
  not versioned. Unused dependencies are gone (bitarray, dotenv, thefuzz,
  timeout-decorator, nest-asyncio, pytest-cov) and the directly imported ones
  are declared (requests, openai, numpy); `./install.sh` runs `uv sync
  --locked`. `CoderLifecycle` is a `StrEnum`, whose `str()` is its value.

## 2026-10-27

- **The keyword writer retries a search that matches no table, and never
  repeats one.** In the divided architecture each proposal of the keyword
  writer is now counted, before retrieval runs, with the strict-AND query that
  keyword retrieval runs (hybrid: its lexical side). A proposal that matches
  no table is banned as its smallest failing word sets (found with match
  counts, as the unified agent already did) and goes back to the writer in the
  same conversation, with those sets and each word's count; a proposal
  containing a banned set is refused without reaching Solr. At most
  `discovery.budget.max_zero_result_retries` (2) rewrites follow, so three
  searches. A refusal spends none of them, but a fourth refusal in a round
  ends it too. Past either, the round goes on with the last proposal: hybrid
  answers from its semantic side, and in keyword mode the run ends with no
  table instead of starting another round. The banned sets carry to the run's
  later rounds (the writer's first message names them) but are neither loaded
  from nor written to the keyword memory file, so one question's search never
  depends on another's. Before, an empty search went back to a fresh writer
  with a flat list of words to avoid and spent one of the three discovery
  rounds, and in hybrid mode an empty lexical side went unnoticed. Each round
  records its proposals under `keyword_searches` (matched, empty, refused or
  unchecked).
- **The retrieval intent may name any number of concepts.** The cap of two
  concepts (`RetrievalIntent.concepts`, "at most two" in the writer's prompt)
  is gone. It bounded concepts, not the words strict AND needs together: of
  the searches with one or two concepts since 2026-09-17, 946 of 2,253 held a
  multi-word concept, up to six words. An answer with a third concept failed
  validation and the round searched with no keywords. The prompt still asks
  for concise dataset subjects; a search too narrow to match is what the retry
  loop rewrites. Unlike the loop, this changes the first proposal of a round.
- **A hybrid search whose keyword side matched nothing earns a retry
  everywhere.** The unified agent's search budget counts such a search as
  empty (it found its tables through the semantic side alone), and the
  unified orchestrated path, with its memory on, retries it as it retries an
  empty keyword search, banning the smallest failing sets.
- **The keyword cascade is gone.** `discovery.keyword_search: cascade` (the
  unified agent's ranked AND word sets, run until one found a table) and
  `keyword_cascade_size` are removed; the default is `single`, which hybrid
  mode already ran. The value `cascade` still loads, so past configurations
  do, and is refused in keyword mode, where it ran; a stored job's
  `keyword_cascade_size` is dropped on load. The experiment specs pin
  `cascade` as their day's default, and `experiments/keyword_relax`, which
  replays the cascade, no longer runs.

## 2026-10-26

- **Inspection profiles and cards carry less noise.** A profile no longer
  samples geometry blobs, GIS ids and audit columns (`geo_shape`,
  `geo_point_2d`, `the_geom`, `OBJECTID`, `GlobalID`, `created_user`,
  `last_edited_date`...): they are named on one line, and a date range of
  `last_edited_date` no longer counts as the table's period. A text column
  whose values are all distinct (an id, free text) shows "one per row" instead
  of four meaningless "most frequent" values. These were 11% of the profile
  text in the overnight runs. A card in STATE now says each requirement once:
  `shows R2 in columns (nb_inscrit)` names the column, the separate "words in
  column names / values / title" facts are gone, and the line is built from
  whole facts within 600 characters instead of being cut mid-word at 420. The
  card's data (`Card.facts`, `Card.shows`) and the plan are unchanged apart from
  the column named in `by`. Three switches default to the current behaviour for
  the experiments in experiments/coverage_lab: `Memory.card_negatives` (the
  "not in any inspected value" period lines and "words found nowhere"),
  `Memory.slim_inspected` (inspected candidates listed by title only) and
  `MemgptAgent.queue_gate` (confirm refused for an uninspected alternative).
- **The table selector leaves out a selected table that holds nothing.** On
  the questions that need several tables the MemGPT selector's other common
  miss was the extra table beside the right ones (the polling sectors next to
  the results, a second year's copy). After the contrast check, with two or
  more tables selected, the new drop check (`selectors/memgpt/drop.py`) shows
  the whole selection and the selection without each table, the tables
  lettered by the requirements they show rather than by their window number,
  and asks twice with the options in two orders; the table is left out only
  when both answers name the same drop. Measured in experiments/coverage_lab
  after the contrast check on the 188 dev questions: 19 selections fixed and
  none broken over five samples in three candidate orders (p 0.001 and
  0.008), two calls and about 3.6k tokens on the questions with two or more
  selected tables. The plan records it under `drop`, the trace as `[drop]`.
- **The table selector keeps every name a question lists.** The MemGPT
  selector tracks the things a question asks about one by one (five sensors,
  three party lists) as their own requirements, but only those its
  `items.named_items` recognised as a list in the question; it lost names with a
  month word in them ("salvador Abril" was taken for a period), names sharing a
  head word ("Cádiz 3" after "Cádiz 16" was matched from the first "Cádiz" and
  fell out of the list), and the whole list when the model passed it as a dict
  or nested lists. Now a month word is a period only with a day or a year
  beside it, each name is matched at its tightest stretch, and the names are
  read in any shape (`items.listed_names`). The coverage lab
  (`experiments/coverage_lab`) measured the repair on the 188 dev questions:
  nothing broken, the five-sensor question fixed, and no change to any of the
  decision lab's 256 recorded define calls.

## 2026-10-24

- **The OCI dedicated endpoints are models of the pipeline.** `config/models.yaml`
  lists `dedicated.gpt-oss-20b` and `dedicated.gpt-oss-20b-2` (provider
  `oci_dedicated`, `integrations/llm/providers/oci_dedicated.py`): the Together
  client pointed at an OCI Dedicated AI Cluster endpoint through its
  OpenAI-compatible route, so `lakegen ask --model dedicated.gpt-oss-20b`, the
  webapp's model picker, an experiment config and the API's split model can all
  run on them. The endpoint's model id is its OCID, which belongs to one
  deployment, so each entry names the `.env` variable that holds it
  (`model_env: DEDICATE_OCI_MODEL`); the key, base URL, extra headers and extra
  body (`{"reasoning_effort": ...}`) are the `DEDICATE_OCI_*` variables, now in
  the settings registry (and so scrubbed in tests and listed in
  `docs/configuration.md`). The limits come from `models.yaml` like every other
  model's: the selector lab's `DEDICATE_OCI_CONTEXT_WINDOW`, `_MAX_TOKENS` and
  `_TEMPERATURE` stay lab-only. A `ModelSpec` now carries the entry's other keys
  in `options`, for its provider. The Pneuma-Seeker entity extraction takes a
  `dedicated.*` entity model on the same route (`pneuma/entities.py`).
  `config/presets/model/dedicated_gpt_oss_20b.yaml` runs the pipeline and the
  code judge on the first endpoint.
- **The table selector reads whole titles, the dataset's title, the file's
  other name and the publisher.** The MemGPT selector's candidate list, STATE
  cards, selection line and contrast check no longer cut a title (they cut at
  90, 60, 50 and 100 characters; on the decision lab's 400 questions the cut
  hid what separates a gold table from a look-alike for 9 Valencia sensor
  tables and 13 UK boundary files). Under the title a candidate shows
  `dataset:` (the title of the dataset a file belongs to, when the indexed
  title is the file's own name, as on data.gov.uk where 4,545 files are titled
  "CSV"; read at query time from the lake's CKAN package list,
  `catalogue.lake_titles`), `also titled:` (the file's own name when it says
  something the title does not, Valencia's Valencian names) and `publisher:`
  (with the ODS portals' responsible entity). The semantic branch's field
  list now fetches the publisher and the second titles, so a table only that
  branch found has them too; list-valued Solr fields are joined instead of
  spelled as a Python list. The read-only block gains a CATALOGUE line that
  explains those lines and, per portal, the note `catalogue:` of
  `config/lakes.yaml` (what the title fields are, who the publisher is; the
  five portals have one). `SelectionRequest.lake` and
  `phase2_select_tables(lake=)` carry the lake's name there. Description cuts
  (150 in the list, 200 in the contrast check) are unchanged. The webapp's and
  the CLI's candidate list and step titles name a table the same way (`CSV
  (dataset: ...)`). The golden snapshots record the new `lake` argument of the
  phase-2 call; nothing else in them changed.
- **A turn without a tool call is asked again with one required, and a reply
  the transport cut is retried.** The MemGPT selector's agent asks the model
  again with `tool_choice="required"` when a turn brings no call that the
  recovery parser can read, on clients that declare `tool_choice_supported`
  (the Together client, so Together and the dedicated endpoints); a provider
  failure counts as the provider's, not as a turn without a call, and is
  retried up to three times. `TogetherLLM` treats a reply or stream chunk the
  OpenAI client could not decode (`JSONDecodeError`, seen under load as a
  stream cut mid-chunk) as a transport failure and retries it with backoff.
  The recovery parser also reads `{"<tool>": {...}}` and `tool(key=value)`
  written as text. Verified on 100 paired questions (decision lab, run
  `choice1`): provider-error turns 6-10 → 0, selections unchanged within rerun
  noise.

## 2026-10-22

- **The webapp shows the MemGPT selector's tool calls one by one.** Each call
  is a collapsed step under "Phase 2 - Table Selection" ("Used inspect · #1,
  #2", "Used select · #1 Vivienda libre (precio metro cuadrado) → R1, R2", "Used
  ⚠ confirm · refused"), whose body is written for a person: tables by title
  instead of file name, the requirements as a list with their coverage, each
  inspected table's rows and the requirements it shows above its profile, the
  measured joins and unions one per line, the contrast check's alternatives
  and choice. The first step lists the candidates retrieval returned (those
  shown first and those an expansion reveals), the last one the full trace;
  the step's summary gives the selected tables by title and the reasoning
  instead of the raw trace. The calls reach the webapp as
  `domain.types.ToolCall` through `StageHooks.tool_call` ->
  `phase2_select_tables(tool_callback=)` -> `SelectionRequest.tool_callback`;
  the formatting is `selectors/memgpt/display.py`. Without a tool channel (the
  CLI, the service) each call is one line of the stream instead of the raw
  result cut at 160 characters. The UI goldens of the divided scenarios record
  the new `tool_callback` argument of the selection call; nothing else changed
  in them.
- **The MemGPT selector's inspections show what the whole table holds, and the
  portal's own place needs no table.** The shared profile is read from the
  first 500 rows: most text columns showed only their type, numeric columns
  only their type, years kept as integers (`Any`, `exercice_comptable`) were
  never measured, and m/d/yy dates were not read. The cards built from it
  stated false negatives ("words found nowhere: debra, coupar" for the table
  whose `Traveller Name` holds Debra Coupar; "2021: not in any inspected value
  or date range" for the table whose `Any` is 2021), and the selector rejected,
  or added a table, on them. An inspection now adds, from the whole table (at
  most 200,000 rows; `selectors/memgpt/evidence.py`): integer years as a list of
  values, numeric ranges, the number of distinct values and the most frequent
  ones of text columns, the step of date columns, m/d/yy dates, and the
  question's and requirements' words found in the text values, which the cards
  count as values; named items and periods are scanned in the text values of
  every inspected table. A requirement that names nothing but the portal's own
  place ("location: Valencia", the named item "Valencia") is waived when the
  requirements are defined, with the reason: no card could show it, so it stayed
  open and the selector added a table that mentions the city to close it. The
  place is the portal title before "Open Data" (benchmark lakes have none).
  `MemgptAgent.evidence_profiles` and `MemgptAgent.waive_portal_place` switch
  them off. Validated in `experiments/profile_lab` (400 questions, gpt-oss-20b):
  exact selections with copies counted 0.782 against 0.700 run side by side
  (53 questions fixed, 20 broken, McNemar p < 0.001), rejections 4.8% -> 0.7%,
  8.4 calls per question instead of 9.0, 8% more prompt tokens. That run put 50
  concurrent requests on one endpoint and the old selector lost about 5 points
  under that load: against the same selector run at normal load the gain is
  +3.0 points (34 fixed, 22 broken).
- **The MemGPT selector checks its selection against look-alike tables.** Once
  it holds a selection, the contrast check
  (`lakegen.workflow.discovery.selectors.memgpt.contrast`) builds the
  alternatives that swap one selected table for one of its three closest
  look-alikes (title words, indexed columns, the same dataset; at most seven),
  profiles the ones never inspected, and makes one call that shows the tables
  side by side and what each alternative changes. The model keeps the
  selection unless another alternative clearly holds a requirement it does
  not; a failed call keeps it. The plan records the check (`contrast`) and
  the trace says what it did. Validated in `experiments/sibling_lab` (arm
  `contrast`): 6 questions fixed and none broken over 478, +1.6 points of
  exact selections on the 250 held-out questions, 0.9 calls per question.
  `MemgptSelector.agent_cls` names the agent the selector runs, so an
  experiment can run a variant through the same selector.

## 2026-10-21

- **The MemGPT table selector replaces the divided judge, and divided +
  memgpt is the default pipeline.** The judge agent saw 10 of the 20
  retrieved candidates, inspected at most 8 and made at most 15 tool calls,
  so on a question needing six tables it could select four and run out of
  calls before inspecting the other two. Its replacement
  (`lakegen.workflow.discovery.selectors.memgpt`) is an agent with a
  MemGPT-style memory: a read-only block (instructions, tools, question, the
  requirements it defines first, the previous attempt's feedback), a STATE
  the system writes from its tool calls (the selection with the requirements
  each table covers, a card per inspected table, the tables still worth
  inspecting, measured joins and unions), and a FIFO of recent turns behind a
  summary. It may expand twice and inspect four tables per action; confirm
  and reject are refused while a requirement is open or STATE still queues a
  table; the periods and named items of a question become requirements that
  need evidence. Validated in `experiments/selector_lab` (arm `memgpt2b`,
  gpt-oss-20b, frozen contexts of the five lakes): exact selections 0.66 vs
  the judge's 0.38 on 50 questions, 0.54 vs 0.30 on 127.
- The defaults are now `discovery.architecture: divided` and
  `discovery.selector: memgpt` (`config/base.yaml`, the schema, the CLI, which
  gains `--unified`, and the UI switch); the unified agent stays available.
  `presets/selector/llm.yaml` is gone. `llm` still loads, so the
  configurations and specs of past runs keep their meaning, but a divided
  agentic run that asks for it is refused; `rest_run` and `portal_grid2`
  now name the architecture (and selector) their runs used.
- Jev falls back to the MemGPT selector instead of the judge. When the MemGPT
  selector cannot select (no LLM, a failure, a run that ends empty), the
  round asks for new keywords. A rejection carries the tables that show a
  requirement into the next attempt. The orchestrated context still selects
  with its tool-free turn (memgpt needs tools); a round whose retrieval found
  nothing now records selector `none` and no selector call.

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
