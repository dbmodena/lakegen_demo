# LakeGen

LakeGen answers natural-language questions over collections of tables. It finds
the relevant tables, generates and executes the analysis code, and produces a
textual answer.

LakeGen can be used through:

- a Chainlit web interface;
- the `lakegen` command (one question step by step, experiments, services, benchmarks);
- an HTTP API for single questions and batches;
- standalone retrieval benchmarks.

The code is one package, `src/lakegen`, whose layout and rules are described in
[CONTRIBUTING.md](CONTRIBUTING.md); `docs/CHANGELOG.md` records the behaviour
changes of the 2026 refactor and `docs/configuration.md` every environment
variable.

## 1. Requirements

- Python 3.11-3.13;
- [`uv`](https://docs.astral.sh/uv/);
- Apache Solr with the required cores and metadata;
- OCI Generative AI credentials;
- OCI embeddings (default), or optional Ollama with `bge-m3` when using semantic or hybrid retrieval.

Install the Python dependencies from the project root:

```bash
uv sync
```

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

`index_retrieval.py` ignores this and writes to the core it is given.

#### Rebuilding the corrected UK core

`scripts/build_uk_fixed.py` rebuilds the `uk_fixed` core from
`metadata_retrieved_cleaned.json` and the parquet files. Compared with the
first UK build it names each table by package title plus resource name (not the
resource name alone, which is often the literal "CSV"), stores descriptions as
plain text, and keeps `columns.name`, `columns.label` and `columns.type`
aligned with the parquet schema (a column with an empty name is stored as
DuckDB's `C<index>`).

```bash
python scripts/build_uk_fixed.py                # dry run: diff against the core
python scripts/build_uk_fixed.py --apply        # write (backs up first, then verifies)
python scripts/build_uk_fixed.py --apply --embed  # also OCI cohere.embed-v4.0 vectors
python scripts/build_uk_fixed.py --verify       # re-run the checks only
python scripts/build_uk_fixed.py --restore backups/uk_fixed_build/before.jsonl.gz
```

It never creates a core and refuses the production cores. `scripts/` and
`tests/` are git-ignored in this repository, so add the script with
`git add -f`.

### Experiment configuration

All settings and their accepted values are documented in
`config/experiment.example.yaml`. Copy or edit that file to define a repeatable
experiment. Its `benchmark.path` setting names the input file of the
retrieval-only benchmark (see section 7).

The main environment variables are:

- `SOLR_BASE_URL`: Solr base address;
- `OCI_PROFILE`: OCI profile to use;
- `OCI_COMPARTMENT_ID`: OCI compartment override;
- `OCI_SERVICE_ENDPOINT`: optional OCI service endpoint;
- `LAKEGEN_EMBEDDING_BASE_URL`: embedding service address.

## 3. Run LakeGen

### Web interface

```bash
uv run chainlit run src/app.py
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
  --config config/experiment.example.yaml \
  "The question to analyze"
```

Individual configuration values can be overridden with `--set`:

```bash
lakegen ask \
  --config config/experiment.example.yaml \
  --set retrieval.top_k=20 \
  "The question to analyze"
```

List every option with `lakegen ask --help`; `lakegen --help` lists the other
commands (`config`, `exp`, `runs`, `services`, `bench`). `python src/cli.py`
still runs the same command.

Configuration files are checked and compared with:

```bash
lakegen config show config/experiment.example.yaml --format v2
lakegen config validate config/*.yaml
lakegen config diff config/experiment.example_uk_keyword.yaml config/experiment.example_uk_semantic.yaml
```

### HTTP API

Start the API locally:

```bash
python -m lakegen.interfaces.api --host 127.0.0.1 --port 8000
```

(`uvicorn lakegen.interfaces.api.app:app` and the old `uvicorn src.api:app`
start the same application.)

By default the API executes one workflow at a time. To execute up to two
independent questions or batch jobs concurrently, set the limit before starting
Uvicorn:

```bash
LAKEGEN_WORKFLOW_WORKERS=2 python -m lakegen.interfaces.api --port 8000
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
  -F 'config=@config/experiment.example.yaml' \
  -F 'questions=@queries/generated_queries_nyc.json'
```

Batch jobs run with the `LAKEGEN_WORKFLOW_WORKERS` limit. Their state and
results are stored in `.lakegen_jobs/`, so they remain available after an API
restart.

For a large JSON collection, this client splits it into independent durable API
jobs and submits those jobs concurrently:

```bash
.venv/bin/python scripts/submit_parallel_batches.py benchmark/50q_nyc.json \
  --chunk-size 10 --parallelism 2 --wait
```

`--parallelism` controls only request submission; actual execution is bounded
by `LAKEGEN_WORKFLOW_WORKERS`. The tool accepts a root JSON list or an envelope
with `questions`, `queries`, or `cases`, and preserves any inline `config`.

### Experiments as specs

A grid of runs is an experiment spec in `experiments/specs/`: the base
configuration files, the values that vary (`grid`), the question set per lake,
how to execute (through a running API or in-process) and the services it needs.
`lakegen exp plan SPEC` lists the cells and which already ran, `lakegen exp run
SPEC` submits the rest (it can be restarted: a cell in the run store is
skipped), `lakegen exp status SPEC` follows them. The run store under the state
root (`runs/<experiment>/`) records what each cell submitted;
`lakegen.experiments.analysis.load_runs("bench-grid")` reads every result of an experiment
as one table, and `python -m lakegen.interfaces.dashboard` shows the rates
live. `lakegen services status|up|down` manages the services of
`config/services.yaml` (Pneuma per lake, Ollama, the API, dashboards).

## 5. Retrieval modes

The workflow defaults to `retrieval.top_k=20`. Unified discovery initially shows
10 candidates and can reveal up to 5 more in one guided expansion. Explicit
configuration, environment settings where supported, and request/CLI overrides
remain authoritative; the resolved configuration stored with each run records
the effective value. Example YAML files are loaded only when supplied.

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
  `config/experiment.example_uk_hybrid_duckdb_semantic.yaml` runs it with
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
  arm. `config/experiment.example_uk_pneuma_seeker.yaml` configures the
  reference retriever itself: k=10, a 5x Pneuma pool, alpha 0.5, `fill`,
  extracted entities, and no automatic enumeration. It still differs in one
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
lakegen index \
  --core nyc \
  --ensure-schema \
  --metadata-source backups/nyc-solr-ready-metadata.json
```

Only after reviewing the dry run, apply the update and create a backup:

```bash
lakegen index \
  --core nyc \
  --ensure-schema \
  --metadata-source backups/nyc-solr-ready-metadata.json \
  --backup backups/nyc-before-metadata-v1.json \
  --apply
```

Replace the example metadata and backup paths with files appropriate for the
selected core.

## 6. Create benchmark files

`build_benchmark.py` does not generate new questions. It selects valid,
successful cases from an existing generated-query JSON file, regardless of the
source engine, and writes a
deterministic benchmark sample.

Create the default 100-question NYC benchmark:

```bash
uv run python build_benchmark.py --dataset nyc
```

This reads `queries/generated_queries_nyc.json` and writes
`benchmark/100q_nyc.json`.

Create the UK benchmark:

```bash
uv run python build_benchmark.py --dataset uk
```

This reads `queries/generated_queries_uk.json` and writes
`benchmark/100q_uk.json`.

Use custom paths, sample size, and seed when needed:

```bash
uv run python build_benchmark.py \
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
    `build_benchmark.py` makes, so a question is searched identically in a
    sample built from the same file;
  - the case id is the record's `client_id`;
  - the generator sometimes asks the same question over the same tables once
    for Pandas and once for SQL. Such a question counts once, so it does not
    weigh twice in the averages.

#### Run with a file

```bash
PYTHONPATH=src:. uv run python -m lakegen.retrieval.benchmark \
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
PYTHONPATH=src:. uv run python -m lakegen.retrieval.benchmark \
  --config config/experiment.example.yaml \
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
`lakegen/integrations/solr/client.py`). With the five to seven keywords a question carries, most
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
`PNEUMA_PORTAL_ROUTES` in `lakegen/retrieval/config.py` maps the three portals
to their Pneuma ports; `LAKEGEN_PNEUMA_<PORTAL>_BASE_URL` overrides one.

## 8. Pneuma retrieval (optional)

Pneuma uses a separate Python 3.12 environment because its dependencies
conflict with LakeGen's main environment:

```bash
uv venv --python 3.12 .venv-pneuma
uv pip install --python .venv-pneuma/bin/python -r requirements-pneuma.txt
```

Prepare an index for a portal:

```bash
bash scripts_pneuma/run_pneuma_bootstrap_monitored.sh nyc
```

By default Pneuma uses the configured OpenAI-compatible Ollama endpoint. To
offload both summarization and embedding generation to OCI Generative AI, use
the same OCI profile and compartment configured for LakeGen:

```bash
PNEUMA_PROVIDER=oci bash scripts_pneuma/run_pneuma_bootstrap_monitored.sh uk \
  --oci-llm-model openai.gpt-oss-20b \
  --oci-embedding-model cohere.embed-v4.0
```

`OCI_PROFILE`, `OCI_CONFIG_FILE`, `OCI_COMPARTMENT_ID`, and
`OCI_SERVICE_ENDPOINT` are honored. The selected provider and model identifiers
are recorded in `inference-provider.json`; credentials are never copied there.
Use `--summary-limit N --skip-metadata --skip-index` for a resumable OCI canary
that processes only the next `N` pending tables.

Start the service using the generated index path:

```bash
.venv-pneuma/bin/python scripts_pneuma/pneuma_server.py \
  --out-path /data/pneuma/nyc \
  --port 8774
```

For an OCI-built index, use the identical embedding model at query time:

```bash
.venv-pneuma/bin/python scripts_pneuma/pneuma_server.py \
  --provider oci \
  --oci-llm-model openai.gpt-oss-20b \
  --oci-embedding-model cohere.embed-v4.0 \
  --out-path /data/pneuma/uk \
  --port 8767
```

LakeGen routes Pneuma automatically by the selected Solr core:

- `nyc` uses `/data/pneuma/nyc`, served on port `8774`;
- `uk` uses `/data/pneuma/uk`, served on port `8767`;
- both use the index name `lakegen-cohere-v4-1024`.

The routes are internal defaults rather than extra experiment fields. They can
still be overridden, when needed, with
`LAKEGEN_PNEUMA_NYC_BASE_URL`, `LAKEGEN_PNEUMA_NYC_INDEX_NAME`,
`LAKEGEN_PNEUMA_UK_BASE_URL`, and `LAKEGEN_PNEUMA_UK_INDEX_NAME`.

Then select `pneuma` as the retrieval mode. For another port, update
`retrieval.pneuma_base_url` in the experiment configuration.

## 9. Tests and outputs

Run the test suite with:

```bash
uv run pytest
```

The suite is hermetic (every test writes under a temporary state root).
`scripts/check.sh` is the commit gate: the suite against `tests/BASELINE.txt`,
the golden snapshots in `tests/golden/` (`pytest --update-golden` re-records
them, for a listed behaviour change only), ruff on the staged files, the
import contracts of `pyproject.toml` and an import smoke test; the pre-commit
hook in `scripts/hooks` runs it (`git config core.hooksPath scripts/hooks`).

The main generated outputs are:

- `logs/experiments_log.csv`: CLI and Chainlit runs;
- `logs/api_experiments_log.csv`: API runs;
- `logs/retrieval_rankings.jsonl`: ordered retrieval candidates and scores;
- `logs/retrieval_benchmarks_log.csv`: benchmark metrics;
- `.lakegen_jobs/`: API batch state and results;
- `runs/`: the run store of the experiment specs, and each experiment's keyword memory;
- `coding/`: Python programs generated while answering questions.

All of them live under the state root (`LAKEGEN_STATE_DIR`, the repository by
default); see `docs/configuration.md`.

For detailed experiment parameters, use `config/experiment.example.yaml` and
the `--help` option of the relevant command.
