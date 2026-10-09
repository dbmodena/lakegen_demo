# Contributing to LakeGen

## The package

Everything is under `src/lakegen`, one package per concern, each with an
`__init__.py` that says what it is for:

| Package | Holds |
|---|---|
| `domain/` | pure types: the table-id normalisers, the discovery and code outcomes, the result models (`results`), the shared errors and types |
| `config/` | the environment registry (`settings`), the lake and model registries, the v2 configuration files and presets, `ExperimentConfig` (`experiment`), the retrieval options (`retrieval`), the runtime settings, the state paths (`paths`, `locations`), the bootstrap |
| `lakes/` | a lake's tables on disk (`table_io`) and its catalogue entries |
| `llm/` | the chat models: one provider module each (`providers/`), the tool-call fixes, token accounting, the Groq offload, `get_llm`, the agent runner (`agents/`) |
| `retrieval/` | one module per retrieval strategy, the registered modes, the service that composes them, the Pneuma adapter (`pneuma/`), the cell scan, the run log, the benchmark |
| `discovery/` | phases 1 and 2, which tables answer the question: the keyword writer and its memory, the divided judge, the unified agent and its tools (`tools/`), the orchestrated paths and context, the requirement ledger, the table selectors (`selectors/`) |
| `coding/` | phase 3, the coder agent and its tools (`tools/`): the plan grounding, the contract, the sandbox, the result checks, the coder's context, sweep and attempts |
| `evaluation/` | judging a code result: the comparison with the reference, the semantic judge, the output checks, the reference execution |
| `answer/` | phase 4, the answer written from the result |
| `pipeline/` | the engine that runs one question (a generator of stages and gates), its strategies, stages, budgets, state, replays, the service and the runner that call it |
| `observability/` | the recorder that turns a run into its `QueryResult`, the CSV log, the manifest, reproducibility, tracing |
| `experiments/` | specs, the run store, executors, services |
| `analysis/` | the runs of the store as a table, the metrics, the readers of older records |
| `interfaces/` | the only importers of web and UI frameworks: the API, the `lakegen` command, the Chainlit driver, the dashboard |
| `prompts/` | the prompt templates and their manager |

The packages run from the shared layers up in the order of a question:
`discovery`, `coding`, `evaluation`, `answer`, composed by `pipeline`.

The old entry files (`src/api.py`, `src/cli.py`, `src/app.py`) and the old
module paths (`lakegen.phases.*`, `lakegen.core.*`, `lakegen.agent_tools.*`,
`lakegen.ui.*`, `lakegen.agents.*` and the top-level modules such as
`lakegen.service`, `lakegen.manifest` or `lakegen.experiment_config`) still work: each aliases or
re-exports its new home, for the experiment scripts that import them.

### The rules the gate enforces

- **Dependencies point down.** `interfaces → experiments → pipeline → coding →
  discovery → answer → retrieval → llm → lakes → config → domain`; only
  `interfaces` imports chainlit and fastapi; `observability` is imported by the
  pipeline and the interfaces only. The contracts are in `pyproject.toml`
  (`[tool.importlinter]`) and `lint-imports` checks them.
- **The environment is read in one place**, `lakegen.config.settings`; ruff bans
  `os.environ` elsewhere.
- **The quality bar** (`pyproject.toml`, `[tool.ruff]`) applies to every new or
  rewritten file: functions of at most 60 statements and complexity 12, sorted
  imports, modern annotations. Files that predate the refactor are listed in
  `scripts/ruff_legacy.txt` and get only pyflakes' real errors until they are
  rewritten; take a file off the list when you rewrite it.
- **Behaviour changes are listed.** `tests/golden/` pins what the service, the
  CLI and the Chainlit driver do; a commit either moves code (goldens
  unchanged) or changes behaviour (goldens re-recorded with
  `pytest --update-golden`, the change noted in `docs/CHANGELOG.md`).

Run `scripts/check.sh` before committing (the pre-commit hook in
`scripts/hooks` runs it; `git config core.hooksPath scripts/hooks`). Pass
`--update-baseline` when tests were added or renamed.

## Adding things

Each extension point is a protocol and a registry: a configuration value names
a registered kind, and an unregistered value fails with the registered ones
listed.

### A retrieval mode

1. Add the value to `RetrievalMode` in `lakegen/retrieval/config.py`, with its
   traits (`fuses`, `uses_duckdb`, `uses_embeddings`, ...).
2. Write the retriever in its own module, and the mode that composes it from
   the service's `Branches`:

   ```python
   # lakegen/retrieval/modes.py
   @register(RetrievalMode.TITLE_ONLY)
   class TitleOnlyMode:
       def __init__(self, branches: Branches) -> None:
           self.keyword = branches.require("keyword")

       def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
           hits = self.keyword.retrieve(list(request.keywords), top_k=request.top_k, q_op="AND")
           return RetrievalResult(hits)
   ```

   A mode that needs a branch the service does not prepare puts it in
   `Branches.extras` from `TableRetrievalService.__init__`.
3. Configure it: `retrieval.mode: title_only`. Add a case to
   `tests/test_retrieval_registry.py`.

### A table selector

1. Add the value to `TableSelector` in `lakegen/config/experiment.py`.
2. Register a class with `kind`, `title` and `select(request) -> Selection` in
   `lakegen/discovery/selectors/`; raise `SelectorUnavailable` when it cannot
   run, and the architecture falls back to its own LLM judge:

   ```python
   # lakegen/discovery/selectors/first.py
   @register(TableSelector.FIRST)
   class FirstSelector:
       kind = TableSelector.FIRST
       title = "First-candidate selector"

       def __init__(self, discovery: DiscoveryConfig) -> None: ...

       def select(self, request: SelectionRequest) -> Selection:
           chosen = request.candidates[:1]
           return Selection(selected=chosen, reasoning="The top candidate.", trace="--- First ---")
   ```

   Import the module at the bottom of `selectors/__init__.py`.
3. Configure it: `discovery.selector.kind: first` (the divided architecture
   only, which has a separate selection step).

### A discovery strategy

1. Add the value to `DiscoveryArchitecture` or `ToolAccess`.
2. Register a class in `lakegen/pipeline/strategies.py` with `architecture`,
   `tool_access`, `writes_keywords_first`, `discover(stage, keyword_round)` and
   `after_round(stage, outcome, attempt)`; the `DiscoveryStage` it receives
   holds the run, its state and the shared helpers (the gated keyword round,
   the seeded agent state, the rejection bookkeeping). The agents it calls are
   dependencies of the run (`PipelineDeps`), so tests can script them.
3. Configure it: `discovery.architecture` and `discovery.tool_access`.

### An LLM provider or model

- A model of a known provider is one entry in `config/models.yaml` (provider,
  provider model id, context window, output limit, timeout, Groq id).
- A new provider is a module in `lakegen/llm/providers/` with a class holding
  `name` and `build(spec, settings, callback_manager) -> LLM`, decorated
  `@register("name")` and imported at the bottom of `providers/__init__.py`;
  the client should keep the token accounting the others have
  (`token_usage_total`).

### A lake

One entry in `config/lakes.yaml` (title, Pneuma route, starter questions; the
data layout is `data/<lake>/` with the defaults at the top of the file), then
the Solr core and, for semantic retrieval, `lakegen index --core
<lake>`. Configure it with `lake: <name>` (`core:` in the flat form).

### An experiment

A spec in `experiments/specs/` (see the existing ones): the base files and
presets, the fixed `set` values, the `grid`, the question set, the
`experiment_id` pattern, the execution backend and the required services.
`lakegen exp plan` shows the cells before anything runs.

### An analysis

A function over `lakegen.analysis.load_runs(experiment)` (one row per stored
result, with the effective model and the provider switches applied) or over
`runs_of_job` for one job; the metrics of a batch are in
`lakegen.analysis.metrics`. Readers of records that predate the run store are
in `lakegen.analysis.sources`.

### A judge

The semantic code judge is `lakegen.evaluation.semantic_judge` (wrapped by
`SemanticJudge` in the pipeline); it is configured by model, not yet by kind.
A registry like the others is the way to add a second one.

## Tests

- `tests/golden/`: the pipeline's behaviour per surface and scenario
  (`harness.py` scripts the phases with fakes and records every call).
- `tests/test_pipeline_*.py`: the engine, its gates, strategies, budgets,
  adapters, replays.
- `tests/needs_testing_grounds/`: tests of code that lives on the
  `testing-grounds` branch; not collected here.

Tests patch the module that defines a collaborator (for example
`lakegen.retrieval.pneuma.seeker.scan_many`), never a re-export.
