# LakeGen

LakeGen answers natural-language questions over collections of tables. It finds
the relevant tables, generates and executes the analysis code, and produces a
textual answer.

LakeGen can be used through:

- a Chainlit web interface;
- the `lakegen` command (one question step by step, experiments, services, benchmarks);
- an HTTP API for single questions (batches of questions run on the
  experiment harness's batch server, which serves the same API);
- standalone retrieval benchmarks.

The code is one package, `src/lakegen`, laid out as described in
[Code layout](#code-layout) below; `docs/CHANGELOG.md` records the behaviour
changes of the 2026 refactor and `docs/configuration.md` every environment
variable.

## Code layout

The package is `src/lakegen`, in seven areas; an import may only go down this
list. The prompts (`prompts/`, one Markdown file per prompt, a folder per
agent) and the configuration (`config/`) are files at the repository root.

| Area | Holds |
|---|---|
| `entrypoints/` | the only importers of web and UI frameworks: `api/` (FastAPI), `cli/` (the `lakegen` command), `webapp/` (the Chainlit app) |
| `pipeline/` | one question through the phases: the engine, its stages and strategies, `run/` (context, state, events, budgets), `wiring/`; `discovery/` (keywords, the divided round, the unified agent, the MemGPT selector), `coding/` (the coder and its `tools/`), `evaluation/` (judging results), `answer`; `evidence/` (table inspection, join and union evidence, requirement ledger); `observability/` (recorder, logs, manifest) |
| `retrieval/` | the service that composes the modes, the registry, the run log, the benchmark, `modes/` (keyword, semantic, hybrid, DuckDB, Pneuma ranking), `solr/` (client, catalogue entries), `pneuma/` (service client, Seeker, cell scan) |
| `llm/` | the chat models (`providers/`), embeddings, rerank, token accounting, the agent runner, the prompt renderer |
| `config/` | settings, the lake and model registries, `ExperimentConfig`, the config files, paths |
| `datastructures/` | the records the areas exchange: results, outcomes, retrieval hits, errors |
| `utils/` | general-purpose helpers: table files, naming rules, table ids, merge limits |

Outside the package: `pneuma_engine/` (the Pneuma retrieval engine, with
its own environment: see [its README](pneuma_engine/README.md)), `solr_builder/`
(what builds the Solr cores and their embeddings), `experiment_runner/` (the
experiment harness: specs, the batch server, the run store, services,
analysis, the dashboard and the retrieval benchmark), `config/`
(YAML registries, presets and `examples/`), `benchmark/` (question sets),
`scripts/` (the gate, builders, job and analysis tools), `experiments/`
(working directories and `specs/`), `tests/` (one folder per package;
`golden/` holds the snapshots, fixtures and the baseline list).
`scripts/check.sh` is the gate every commit passes, and what CI
(`.gitlab-ci.yml`) runs: ruff (lint and format), bandit and the import
contracts of `pyproject.toml`, then the tests against the baseline and an
import smoke.

To add a retrieval mode, register a class in `retrieval/modes/` with
`retrieval.registry.register`. A table selector is registered in
`workflow/discovery/selectors/`, a discovery architecture in
`workflow/pipeline/strategies.py`, an LLM provider in
`integrations/llm/providers/`, and a lake or a model is one entry in
`config/lakes.yaml` or `config/models.yaml`.

## 1. Requirements

- Python 3.11-3.13;
- [`uv`](https://docs.astral.sh/uv/);
- Apache Solr with the required cores and metadata;
- OCI Generative AI credentials;
- OCI embeddings (default), or optional Ollama with `bge-m3` when using semantic or hybrid retrieval.

Install the Python dependencies from the project root:

```bash
./install.sh
```

It runs `uv sync --locked`, which creates `.venv` from `uv.lock` with the
tools of the gate.

This installs the package in editable mode and puts the `lakegen` command in
the virtual environment (`.venv/bin/lakegen`; `uv run lakegen ...` also works).

## 2. Configuration

### OCI Generative AI

LakeGen reads OCI credentials from `~/.oci/config`. The selected profile must
include the compartment ID:

```ini
[DEFAULT]
user=...
fingerprint=...
tenancy=...
region=eu-frankfurt-1
key_file=/path/to/oci_api_key.pem
oci_compartment_id=...
```

The application supports these OCI models:

- `openai.gpt-oss-120b`;
- `meta.llama-3.3-70b-instruct`.

### Solr

By default, LakeGen connects to `http://localhost:8983/solr`. Set a different
address when necessary:

```bash
export SOLR_BASE_URL=http://localhost:8983/solr
```

The supported Solr cores and data portals are `nyc`, `valencia`, `bologna`,
`paris`, and `uk`.

A portal and its Solr core share one name, and the data directory, Pneuma
route and database path are all derived from it. To serve a portal from a
differently named core, set `LAKEGEN_SOLR_CORE_<PORTAL>`. Only the core that is
queried changes; everything derived from the portal name stays as it is:

```bash
export LAKEGEN_SOLR_CORE_UK=uk_fixed   # portal "uk" is queried from core uk_fixed
```

The Solr builder (below) ignores this and writes to the core it is given.

#### Building the cores

Everything that writes a core lives in `solr_builder/` at the repository root;
LakeGen itself only queries the cores, and an import contract keeps it that
way. Its commands run from the repository root with LakeGen's environment, dry
run first:

- `python -m solr_builder.build_portal_core` builds a portal from a CKAN/ODS
  crawl into a new, empty core (it needs `--orqa-src`, an orqa checkout's
  `src`, for orqa's metadata normalizer);
- `python -m solr_builder.build_uk_fixed` rebuilds the `uk_fixed` core from
  `metadata_retrieved_cleaned.json` and the parquet files. Compared with the
  first UK build it names each table by package title plus resource name (not
  the resource name alone, which is often the literal "CSV"), stores
  descriptions as plain text, and keeps `columns.name`, `columns.label` and
  `columns.type` aligned with the parquet schema (a column with an empty name is
  stored as DuckDB's `C<index>`);
- `python -m solr_builder.index` writes a core's table embeddings (see
  [Enable semantic and hybrid retrieval](#enable-semantic-and-hybrid-retrieval)).

```bash
.venv/bin/python -m solr_builder.build_uk_fixed                # dry run: diff against the core
.venv/bin/python -m solr_builder.build_uk_fixed --apply        # write (backs up first, then verifies)
.venv/bin/python -m solr_builder.build_uk_fixed --apply --embed  # also OCI cohere.embed-v4.0 vectors
.venv/bin/python -m solr_builder.build_uk_fixed --verify       # re-run the checks only
.venv/bin/python -m solr_builder.build_uk_fixed --restore backups/uk_fixed_build/before.jsonl.gz
```

The builders never overwrite a production core: `build_uk_fixed` refuses them
by name and never creates a core, `build_portal_core` refuses a core that
already holds documents.

### Experiment configuration

Every setting, with its accepted values, is in `config/base.yaml`. An
experiment file extends it and changes what differs, usually by adding
presets from `config/presets/` (a model, a retrieval mode, a selector, a coder
setup):

```yaml
schema_version: 2
extends: [base.yaml, presets/retrieval/hybrid_rrf.yaml, presets/model/together_gpt_oss_120b.yaml]
experiment_id: uk-hybrid-together
lake: uk
```

Config files all use this nested layout; `--set` and the specs name their
keys the same way (`retrieval.hybrid.alpha=0.5`). `benchmark.path` names the
input file of the retrieval-only benchmark (see section 7).

The main environment variables are:

- `SOLR_BASE_URL`: Solr base address;
- `OCI_PROFILE`: OCI profile to use;
- `OCI_COMPARTMENT_ID`: OCI compartment override;
- `OCI_SERVICE_ENDPOINT`: optional OCI service endpoint;
- `LAKEGEN_EMBEDDING_BASE_URL`: embedding service address.

## 3. Run LakeGen

### Web interface

```bash
uv run chainlit run src/lakegen/entrypoints/webapp/app.py
```

Open the address displayed in the terminal, then select the portal, model, and
retrieval mode. Do not use `-w` or `--watch`: generated Python files would
cause Chainlit to restart the active chat.

### Command-line interface

`lakegen ask` runs one question step by step in the terminal (the keywords
and the tables are reviewed before the code runs); with a configuration whose
`interaction_mode` is `autonomous` it runs without questions, as the API does.
Run a question with the default settings:

```bash
lakegen ask "Which districts have the highest number of parks?"
```

Select a portal and retrieval mode:

```bash
lakegen ask \
  --core nyc \
  --retrieval-mode hybrid \
  "Which districts have the highest number of parks?"
```

Run with an experiment configuration:

```bash
lakegen ask \
  --config my_experiment.yaml \
  "The question to analyze"
```

Individual configuration values can be overridden with `--set`:

```bash
lakegen ask \
  --config my_experiment.yaml \
  --set retrieval.top_k=20 \
  "The question to analyze"
```

List every option with `lakegen ask --help`; `lakegen --help` lists the other
command (`config`). Experiments, services and the retrieval benchmark are run
with `python -m experiment_runner` (see [Experiments as specs](#experiments-as-specs)).

Configuration files are checked and compared with:

```bash
lakegen config show my_experiment.yaml            # every setting, resolved
lakegen config validate my_experiment.yaml other.yaml
lakegen config diff my_experiment.yaml other.yaml
```

### HTTP API

Start the API locally:

```bash
python -m lakegen.entrypoints.api --host 127.0.0.1 --port 8000
```

(`uvicorn lakegen.entrypoints.api.app:app` starts the same application.)

By default the API executes one workflow at a time. To execute up to two
independent questions concurrently (on the batch server, batch questions
included), set the limit before starting Uvicorn:

```bash
LAKEGEN_WORKFLOW_WORKERS=2 python -m lakegen.entrypoints.api --port 8000
```

Each workflow can itself use `scan_workers` processes for retrieval. Start with
`2` workflow workers and reduce `scan_workers` in the experiment configuration
if the host becomes CPU- or memory-bound; raising both limits blindly can make
the total work slower.

Open <http://127.0.0.1:8000/docs> for interactive documentation and endpoint
testing. Check that the service is running with:

```bash
curl http://127.0.0.1:8000/health
```

Submit one question:

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Which districts have the highest number of parks?",
    "core": "nyc",
    "retrieval_mode": "keyword"
  }'
```

## 4. Batch execution

Batches run on the batch server of the experiment harness
(`experiment_runner/batches`). It serves LakeGen's API as well (`/health`,
`/v1/query`), adds the batch endpoints below, and resumes the jobs a previous
process left unfinished:

```bash
.venv/bin/python -m experiment_runner.batches --host 127.0.0.1 --port 8000
```

Submit a JSON file containing questions:

```bash
curl -X POST \
  'http://127.0.0.1:8000/v1/batches?core=nyc&retrieval_mode=keyword' \
  -H 'Content-Type: application/json' \
  --data-binary @queries/generated_queries_nyc.json
```

The response contains a `job_id`. Use it to read the status and results:

```bash
curl http://127.0.0.1:8000/v1/batches/JOB_ID
```

To check only the status without downloading partial results:

```bash
curl 'http://127.0.0.1:8000/v1/batches/JOB_ID?include_results=false'
```

A configuration and a question file can also be uploaded separately:

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/v1/batches/files \
  -F 'config=@config/base.yaml' \
  -F 'questions=@queries/generated_queries_nyc.json'
```

Batch jobs run with the `LAKEGEN_WORKFLOW_WORKERS` limit. Their state and
results are stored in `.lakegen_jobs/`, so they remain available after an API
restart.

For a large JSON collection, this client splits it into independent durable API
jobs and submits those jobs concurrently:

```bash
.venv/bin/python scripts/jobs/submit_parallel_batches.py benchmark/50q_nyc.json \
  --chunk-size 10 --parallelism 2 --wait
```

`--parallelism` controls only request submission; actual execution is bounded
by `LAKEGEN_WORKFLOW_WORKERS`. The tool accepts a root JSON list or an envelope
with `questions`, `queries`, or `cases`, and preserves any inline `config`.

### Experiments as specs

A grid of runs is an experiment spec in `experiments/specs/`: the base
configuration files, the values that vary (`grid`), the question set per lake,
how to execute (through a running API or in-process) and the services it needs.
The experiment harness, `experiment_runner/` at the repository root, runs them
(LakeGen itself never imports it):
`python -m experiment_runner plan SPEC` lists the cells and which already ran,
`python -m experiment_runner run SPEC` submits the rest (it can be restarted: a
cell in the run store is skipped), `python -m experiment_runner status SPEC`
follows them. The run store under the state root (`runs/<experiment>/`)
records what each cell submitted;
`experiment_runner.analysis.runs.load_runs("bench-grid")` reads every result of
an experiment as one table, and `python -m experiment_runner.dashboard` shows
the rates live. `python -m experiment_runner services status|up|down` manages
the services of `config/services.yaml` (Pneuma per lake, Ollama, the API,
dashboards).

## 5. Retrieval modes

The workflow defaults to `retrieval.top_k=20` and the divided architecture: a
keyword writer, then the MemGPT table selector, which sees 10 candidates first,
can reveal 5 more twice and inspects up to 4 per action before it may confirm
(`discovery.architecture: unified` runs the same agent with a search tool: it
searches the catalogue itself, selects from what it finds, and may search again
when a requirement stays open after it has inspected what it found). In
keyword and hybrid mode the writer's keywords are counted before retrieval with
the same strict-AND query: a proposal no table matches goes back to the writer
with its smallest failing word sets, which stay banned for the run (a proposal
containing one is refused unsearched), at most
`discovery.budget.max_zero_result_retries` (2) times. Explicit
configuration, environment settings where supported, and request/CLI overrides
remain authoritative; the resolved configuration stored with each run records
the effective value. Config files are loaded only when supplied.

- `keyword`: BM25 lexical search through Solr; this is the default mode.
- `semantic`: vector search using the complete question.
- `hybrid`: combines lexical and semantic results.
- `duckdb_agentic`: searches local Parquet files without a Solr index.
- `hybrid_duckdb_semantic`, shown as "hybrid (duckdb + semantic)": fuses the
  `duckdb_agentic` ranking with semantic search by `fusion_method`, as
  `hybrid` does: `weighted` min-max normalizes each branch and weights DuckDB
  by `alpha`, `rrf` uses `rrf_k`. A table missing from a branch scores 0
  there. Each DuckDB hit is resolved to the catalog document of its Parquet
  file before fusion. It needs the vector index and the local Parquet
  directory. `hybrid` is a separate mode and is unchanged.
  `config/presets/retrieval/hybrid_duckdb_semantic.yaml` on the base runs it with
  `rrf` and the Cohere reranker, the configuration measured in
  `experiments/rerank_lab` (100 UK questions: Recall@10 0.869, Hit@1 0.55,
  against 0.822 and 0.49 unreranked).

`keyword`, `semantic`, `hybrid` and `hybrid_duckdb_semantic` results then go
through a second-stage reranker (`rerank_model`, default `cohere.rerank-v4.0-fast` on OCI): it
reorders the top `rerank_depth` (20) tables and leaves the rest in retrieval
order, adding about 0.1 s per search. On the 100-question UK benchmark it
raised Recall@10 and Hit@1 for all four (`experiments/rerank_lab`); deeper
pools were not better. If the reranker fails, the retrieval order is kept
and the error is recorded on the retrieval run. `rerank_model: null` (or an
empty `LAKEGEN_RERANK_MODEL`) turns it off. `duckdb_agentic` and the Pneuma
modes are not reranked. It is on by default in the experiment configuration and the
environment settings, so `--config` benchmark runs rerank too;
`RetrievalConfig()` built in code does not.
`pneuma_seeker`'s content stage is weighted by `pneuma_content_weight`, with its
title, column-name and cell evidence weighted by `pneuma_table_name_weight`,
`pneuma_column_name_weight` and `pneuma_cell_weight`. A family
weighted `0` is **not computed**: it cannot change a ranking, so the work is
skipped rather than performed and multiplied away. In particular,
`pneuma_content_weight: 0` makes `pneuma_seeker` exactly equal to `pneuma`.

- `pneuma`: uses a separately prepared Pneuma index and service.
  `pneuma_value_scan_weight` (default `0`, off) adds an optional value scan:
  the tables Pneuma returned are scanned with DuckDB for the agent's entities
  (else its keywords), matched as whole words in every column cast to text
  within the `duckdb_max_columns_per_file` / `duckdb_max_scan_rows_per_file`
  bounds, and reranked by value evidence weighted against Pneuma's order. It
  is Pneuma-Seeker's content search restricted to Pneuma's own pool. On the
  100-question UK benchmark it was neutral (weights 0.1-0.25) to slightly
  harmful (0.5), since sibling tables in a pool usually share the same
  values; it costs about 0.2 s per query.
- `pneuma_seeker`: Pneuma augmented as in the *Pneuma-Seeker* paper (§5.3),
  "Pneuma + content search + table enumeration". The content search follows
  the paper's reference implementation. Under this mode the discovery prompts
  ask for entities with its extraction rules (specific, named, canonical strings
  written as the question writes them, never general concepts); a caller that
  passes none at all, such as the retrieval benchmark, can have them extracted
  per question with the reference's own prompt by setting `pneuma_entity_model`
  (default `null`, off). Each entity
  is scanned for across the local Parquet with its case-insensitive,
  word-bounded regex: `art` does not match `Department`, and `New York` matches
  `NEW-YORK`. Per entity and table, `pneuma_table_name_weight` x a match in the
  table's catalog title + `pneuma_column_name_weight` x matching column names +
  `pneuma_cell_weight` x matching cells (3, 2 and 1 by default, as upstream) is
  damped by `log(1 + raw)` and normalized per entity; a table's
  content score is its mean over the entities times the fraction it matched,
  fused with Pneuma's own ranking weighted by `pneuma_content_weight`. A
  question naming no entity gets no content search, so its ranking is Pneuma's.
  The entities a hit matched are appended to its description. Tables whose
  identifiers form a family (`water_body_testing_2020` -> `..._\d{4}`) are then
  enumerated, which is what plain top-k retrieval cannot do. Knowing
  deviations: Pneuma's service returns ranks but no scores, so the fusion uses
  `1/rank`; the fusion is the paper's weighted combination, where the reference
  code only fills the top-k slots Pneuma leaves over (`pneuma_content_fusion:
  fill` selects that merge, with only entity-matching tables filling a slot);
  matching cells are counted
  rather than regex occurrences; a table's name is its catalog title, matched
  once per table, because the lakes name files by opaque id; and enumeration
  runs automatically rather than as an agent action.
  Set `pneuma_enumerate_tables: false` to reproduce the paper's middle ablation
  arm. `config/presets/retrieval/pneuma_seeker_reference.yaml` is the reference
  configuration the grids ran: `fill`, extracted entities, and no automatic
  enumeration. It still differs in one
  respect the service decides: our Pneuma always runs its LLM relevance judge,
  which the reference leaves off. The cell scan reads every row and every column of every file, one column
  and one chunk of rows at a time so a worker's memory stays bounded; only
  `grep_max_files` can cut it short. Note
  that both shipped cores name tables by opaque id
  (`43nn-pn8j`), which no family pattern matches, so enumeration contributes
  nothing there; it needs a lake with semantic filenames.

### Enable semantic and hybrid retrieval

Semantic and hybrid retrieval default to OCI `cohere.embed-v4.0` using the same
`~/.oci/config`, profile, compartment and endpoint as the LLM. No Ollama service
is required. `OCI_CONFIG_FILE`, `OCI_PROFILE`, `OCI_COMPARTMENT_ID`, and
`OCI_SERVICE_ENDPOINT` are honored. Requests use `SEARCH_QUERY` for questions
and `SEARCH_DOCUMENT` for indexed tables, with 1024-dimensional float vectors
([OCI request documentation](https://docs.oracle.com/en-us/iaas/tools/python/2.155.1/api/generative_ai_inference/models/oci.generative_ai_inference.models.EmbedTextDetails.html)).

Existing `bge-m3` vectors must be regenerated with OCI before using the new
model: equal dimensions do not make different embedding spaces compatible.
The embedding model and representation version must match those used to index
the Solr documents. Model metadata filters prevent mixing these spaces.

For an existing Ollama index, explicitly set `embedding_model: bge-m3` in the
experiment YAML (or `LAKEGEN_EMBEDDING_MODEL=bge-m3` for CLI indexing). Ollama
uses `embedding_base_url`, defaulting to `http://localhost:11434`; this setting
is ignored for OCI model identifiers (`cohere.*` or `ocid1.*`).

Before changing a Solr core, run the indexer without `--apply`. This validates
the operation without writing documents:

```bash
.venv/bin/python -m solr_builder.index \
  --core nyc \
  --ensure-schema \
  --metadata-source backups/nyc-solr-ready-metadata.json
```

Only after reviewing the dry run, apply the update and create a backup:

```bash
.venv/bin/python -m solr_builder.index \
  --core nyc \
  --ensure-schema \
  --metadata-source backups/nyc-solr-ready-metadata.json \
  --backup backups/nyc-before-metadata-v1.json \
  --apply
```

Replace the example metadata and backup paths with files appropriate for the
selected core.

## 6. Create benchmark files

`scripts/build/build_benchmark.py` does not generate new questions. It selects valid,
successful cases from an existing generated-query JSON file, regardless of the
source engine, and writes a
deterministic benchmark sample.

Create the default 100-question NYC benchmark:

```bash
uv run python scripts/build/build_benchmark.py --dataset nyc
```

This reads `queries/generated_queries_nyc.json` and writes
`benchmark/100q_nyc.json`.

Create the UK benchmark:

```bash
uv run python scripts/build/build_benchmark.py --dataset uk
```

This reads `queries/generated_queries_uk.json` and writes
`benchmark/100q_uk.json`.

Use custom paths, sample size, and seed when needed:

```bash
uv run python scripts/build/build_benchmark.py \
  --dataset nyc \
  --input queries/generated_queries_nyc.json \
  --output benchmark/50q_nyc.json \
  --count 50 \
  --seed 42
```

The input must contain enough successful Pandas cases to satisfy the requested
difficulty and single-table/multi-table distribution.

## 7. Run benchmarks

### Complete pipeline through the API

List the benchmark files available in `benchmark/`:

```bash
curl http://127.0.0.1:8000/v1/benchmarks
```

Queue a complete LakeGen run for a benchmark:

```bash
curl -X POST \
  'http://127.0.0.1:8000/v1/benchmarks/100q_nyc.json/batches?core=nyc&retrieval_mode=hybrid&experiment_id=nyc-hybrid'
```

Use the returned `job_id` with `/v1/batches/JOB_ID` to monitor the run.

### Retrieval-only benchmark

This compares the retrieval modes without running the agents, code generation,
or answer synthesis. Every case is retrieved once per mode, and the ranking is
scored against the case's gold tables with Hit@k, Recall@k, MRR, and nDCG@k for
k = 1, 5, and 10.

#### Input files

Two input formats are accepted and detected automatically:

- **Curated benchmark**, such as `benchmark/100q_nyc.json` (see section 6): a
  list of cases, or an object with a `cases` list. Each case needs `question`,
  `keywords`, and `relevant_table_ids`.
- **Generated-queries file**, such as `generated_queries_semantic.json`, as
  written by the query generator. Its questions are nested as
  `ENGINE -> table scope -> group -> record`, and they become cases as follows:
  - only records with `status: success` become cases;
  - the gold tables are the record's `tables`, whose aliases (`Table_0`) are
    resolved to table ids through the group's `_meta.tables`. An alias that does
    not resolve stops the load with an error naming the record, for example
    `PANDAS/multi_table/mt_0/0`;
  - the search keywords are the record's `question_keywords`, the same choice
    `scripts/build/build_benchmark.py` makes, so a question is searched identically in a
    sample built from the same file;
  - the case id is the record's `client_id`;
  - the generator sometimes asks the same question over the same tables once
    for Pandas and once for SQL. Such a question counts once, so it does not
    weigh twice in the averages.

#### Run with a file

```bash
.venv/bin/python -m experiment_runner bench \
  benchmark/100q_nyc.json \
  --core nyc \
  --output logs/nyc_retrieval_benchmark.json
```

The same command accepts a generated-queries file in place of
`benchmark/100q_nyc.json`. Without `--config`, every mode starts from the
default retrieval settings.

#### Run from an experiment configuration

Set the input file in the `benchmark` section of the experiment configuration:

```yaml
benchmark:
  path: /path/to/orqa/data/nyc/candidates_discovery/generated_queries_semantic.json
```

Then pass `--config` instead of a file:

```bash
.venv/bin/python -m experiment_runner bench \
  --config my_experiment.yaml \
  --output logs/nyc_generated_retrieval_benchmark.json
```

The configuration supplies:

- `benchmark.path`: the input file, curated or generated. A relative path is
  resolved from the working directory;
- `core`: the Solr core to query;
- `retrieval`: the settings every mode starts from, such as the embedding
  model, the Pneuma service, and the grep weights.

Command-line values take precedence: a positional file replaces
`benchmark.path`, and `--core`, `--top-k`, and `--candidate-multiplier` replace
the configured values. Only this command reads `benchmark.path`; the CLI,
Chainlit, and API workflows ignore it.

#### Modes, options, and outputs

Every retrieval mode is run unless `--modes` names some. `semantic`, `hybrid`
and `hybrid_duckdb_semantic` need the vector index from section 5, and `pneuma`
and `pneuma_seeker` need the Pneuma service from section 8. `hybrid` and `hybrid_duckdb_semantic` each run once per `--alphas`
value (default `0.25 0.5 0.75`), plus once with reciprocal rank fusion.

- `--table-dir`: the local Parquet directory. `duckdb_agentic`,
  `hybrid_duckdb_semantic` and `pneuma_seeker` need it, and without it they are
  recorded as skipped. When it is given, every gold table must exist in it,
  otherwise the run fails without writing the report. Each hit is then scored
  under the local file it maps to (as the workflow maps it), since gold ids are
  file names: UK files are `<dataset_id>___<resource_id>`, which a Solr
  `resource_id` alone never equals.
- `--output`: the JSON report, with every case's ranking and the mean metrics
  of each mode.
- `--metrics-log`: the CSV file the mean metrics are appended to (default
  `logs/retrieval_benchmarks_log.csv`).

Run the module with `--help` for the complete list.

#### Known limitation: keyword search

The keyword search requires every keyword to match (`q.op=AND` in
`lakegen/retrieval/solr/client.py`). With the five to seven keywords a question carries, most
searches return no table at all. In September 2026, on the NYC core, 189 of the
223 generated cases and 90 of the 100 cases of `benchmark/100q_nyc.json` got an
empty keyword ranking, for a Hit@10 of 0.04 and 0.03. All of their gold tables
are present in the core, so these scores reflect the all-keywords rule rather
than missing data. Keep this in mind when comparing modes that use the keyword
search, including `hybrid`.

### Portal grid with a live dashboard

`experiments/portal_grid/` runs every retrieval mode over the Valencia, Paris
and Bologna benchmarks (`benchmark/100q_<portal>.json`, built from orqa's
generated queries; questions are in the portal's language) through the complete
pipeline, and shows the results live:

```bash
PYTHONPATH=.:src .venv/bin/python experiments/portal_grid/make_parquet.py   # once: Parquet mirrors (the DuckDB branch reads only Parquet)
experiments/portal_grid/start_services.sh          # Pneuma per portal (8771-8773), API on 8002, dashboard on 8790, Cloudflare tunnel
PYTHONPATH=.:src .venv/bin/python experiments/portal_grid/run_grid.py      # 3 portals x 5 modes as API batches
experiments/portal_grid/stop_services.sh
```

`run_grid.py` submits one API batch per portal and mode with the example
configuration as inline config (`--set key=value` overrides it, `--count N`
limits the cases, `--modes`/`--portals` narrow the grid). The start script
prints the dashboard URL with its token; the page refreshes every 10 s with
Hit@k, Recall@10, MRR, nDCG@10, full coverage, selection hit/recall/precision/
exact, answered, code-execution, reference-match, time and token figures per
portal and mode, a per-question outcome grid, and the latest finished questions.
Each lake's Pneuma route is in `config/lakes.yaml`;
`LAKEGEN_PNEUMA_<LAKE>_BASE_URL` overrides one.

## 8. Pneuma retrieval (optional)

The `pneuma` and `pneuma_seeker` modes query a Pneuma service. The engine
behind it, the index builder and the HTTP service, lives in
[`pneuma_engine/`](pneuma_engine/README.md) with an environment of its own
(`pneuma_engine/install.sh`), because Pneuma's dependencies conflict with
LakeGen's. LakeGen only calls the service, from
`src/lakegen/retrieval/pneuma/client.py`.

Each lake's route (URL and index name) is in `config/lakes.yaml`;
`LAKEGEN_PNEUMA_<LAKE>_BASE_URL` and `LAKEGEN_PNEUMA_<LAKE>_INDEX_NAME`
override it. `python -m experiment_runner services up pneuma:<lake>` starts a lake's service from
`config/services.yaml`, with the index folders under
`LAKEGEN_PNEUMA_INDEX_DIR` (`pneuma-out` by default). Then select `pneuma` as
the retrieval mode; for another port, set `retrieval.pneuma_base_url` in the
experiment configuration.

## 9. Tests and outputs

Run the test suite with:

```bash
uv run pytest
```

It also runs the Pneuma engine's tests (`pneuma_engine/tests`). The suite is
hermetic (every test writes under a temporary state root).
`scripts/check.sh` is the commit gate, and CI (`.gitlab-ci.yml`) runs its two
stages: `lint` (ruff check and `ruff format --check` on the whole tree,
bandit, the import contracts of `pyproject.toml`) and `tests` (the suite
against `tests/golden/BASELINE.txt`, the golden snapshots in `tests/golden/`,
which `pytest --update-golden` re-records for a listed behaviour change only,
and an import smoke test). The pre-commit hook in `scripts/hooks` runs it
(`git config core.hooksPath scripts/hooks`).

Code follows Google-style docstrings (ruff's pydocstyle and pydoclint checks,
plus `tests/test_docstrings.py`): a one-line docstring when a function's name
and signature say it all, otherwise `Args:`, `Returns:` and `Raises:`
sections.

The main generated outputs are:

- `logs/experiments_log.csv`: CLI and Chainlit runs;
- `logs/api_experiments_log.csv`: API runs;
- `logs/retrieval_rankings.jsonl`: ordered retrieval candidates and scores;
- `logs/retrieval_benchmarks_log.csv`: benchmark metrics;
- `.lakegen_jobs/`: the batch server's job state and results;
- `runs/`: the run store of the experiment specs, and each experiment's keyword memory;
- `coding/`: Python programs generated while answering questions.

All of them live under the state root (`LAKEGEN_STATE_DIR`, the repository by
default); see `docs/configuration.md`.

For every experiment parameter, see `config/base.yaml` and the `--help` option
of the relevant command.
