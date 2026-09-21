# LakeGen

LakeGen answers natural-language questions over collections of tables. It finds
the relevant tables, generates and executes the analysis code, and produces a
textual answer.

LakeGen can be used through:

- a Chainlit web interface;
- a command-line interface;
- an HTTP API for single questions and batches;
- standalone retrieval benchmarks.

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

Run a question with the default settings:

```bash
uv run python src/cli.py "Which districts have the highest number of parks?"
```

Select a portal and retrieval mode:

```bash
uv run python src/cli.py \
  --core nyc \
  --retrieval-mode hybrid \
  "Which districts have the highest number of parks?"
```

Run with an experiment configuration:

```bash
uv run python src/cli.py \
  --config config/experiment.example.yaml \
  "The question to analyze"
```

Individual configuration values can be overridden with `--set`:

```bash
uv run python src/cli.py \
  --config config/experiment.example.yaml \
  --set retrieval.top_k=20 \
  "The question to analyze"
```

List every CLI option with:

```bash
uv run python src/cli.py --help
```

### HTTP API

Start the API locally:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

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

Batch jobs run sequentially. Their state and results are stored in
`.lakegen_jobs/`, so they remain available after an API restart.

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
Each modality's evidence families are weighted by `grep_value_weight` (cell
matches) and `grep_metadata_weight` (filename, column names, catalog fields),
and `pneuma_seeker`'s content stage by `pneuma_content_weight`, with its
title, column-name and cell evidence weighted by `pneuma_table_name_weight`,
`pneuma_column_name_weight` and `pneuma_cell_weight`. A family
weighted `0` is **not computed**: it cannot change a ranking, so the work is
skipped rather than performed and multiplied away. Weighting values `0` turns
grep into a metadata-only search that reads no cells at all; weighting metadata
`0` is what `grep_values` does by mode. Setting both to `0` is rejected, and
`pneuma_content_weight: 0` makes `pneuma_seeker` exactly equal to `pneuma`.

All three local-Parquet modalities (`grep`, `grep_values`, `pneuma_seeker`) scan
**every file in the lake by default**. The per-file scan is an independent
bounded read, so it runs as a parallel map over `scan_workers` worker processes
(default 16): on a 2,673-file/179 GB lake an exhaustive `grep_values` search
takes ~10s end to end, against ~52s scanning one file at a time. Every polars
query runs in those workers, never in the application process: polars has
segfaulted under this load, and a worker crash now costs a retried batch instead
of the app. Setting `grep_max_files` reintroduces a cut for deliberately
cheap runs, but a cut is an *approximation, not just a saving*: it changes which
files are read and the IDF (or per-keyword maximum) the survivors are scored
with, so a capped run is not comparable with a full one. Such runs mark
themselves `truncated` in their evidence.

- `grep`: regex search over local Parquet files without a Solr index. Unlike
  `duckdb_agentic` it casts every column to text, so matches in numeric and
  temporal columns count too, and it reads one column at a time through the
  streaming engine so a multi-GB table stays within bounded memory. Ranking is
  TF-IDF over the query terms, with per-file frequency normalized by row count.
  Filenames, column names, and the catalog contribute alongside the cells.
- `grep_values`: the same retriever restricted to cell values. Filenames,
  column names, and catalog metadata never select or score a file, so a table
  is found only where the question's terms appear in its data. The discovery
  prompts ask the model for a list of values likely to be stored in the rows
  (category labels, place names, codes, years) instead of dataset topics, and
  the search uses exactly those values, each matched whole: nothing is split,
  aliased, or added from the question. The catalog is
  still read for the title, description, and tags a hit displays, which leaves
  the ranking as the only difference from `grep`. Where a lake exceeds
  `grep_max_files` there is no free signal left to order it by, so a bounded
  prefix read of every file replaces the metadata prefilter and
  `grep_probe_rows_per_file` becomes the knob that trades cost for reach into
  long tables; a lake under that cap is scanned whole.
- `pneuma`: uses a separately prepared Pneuma index and service.
- `pneuma_seeker`: Pneuma augmented as in the *Pneuma-Seeker* paper (§5.3),
  "Pneuma + content search + table enumeration". The content search follows
  the paper's reference implementation. Under this mode the discovery prompts
  ask for entities with its extraction rules (specific, named, canonical strings
  written as the question writes them, never general concepts), and each entity
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
  code only fills the top-k slots Pneuma leaves over; matching cells are counted
  rather than regex occurrences; a table's name is its catalog title, matched
  once per table, because the lakes name files by opaque id; and enumeration
  runs automatically rather than as an agent action.
  Set `pneuma_enumerate_tables: false` to reproduce the paper's middle ablation
  arm. The cell scan reads every row and every column of every file, one column
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
uv run python index_retrieval.py \
  --core nyc \
  --ensure-schema \
  --metadata-source backups/nyc-solr-ready-metadata.json
```

Only after reviewing the dry run, apply the update and create a backup:

```bash
uv run python index_retrieval.py \
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

Every retrieval mode is run. `semantic` and `hybrid` need the vector index from
section 5, and `pneuma` and `pneuma_seeker` need the Pneuma service from
section 8. Hybrid runs once per `--alphas` value (default `0.25 0.5 0.75`),
plus once with reciprocal rank fusion.

- `--table-dir`: the local Parquet directory. `grep`, `grep_values`,
  `duckdb_agentic`, and `pneuma_seeker` need it, and without it they are
  recorded as skipped. When it is given, every gold table must exist in it,
  otherwise the run fails without writing the report.
- `--output`: the JSON report, with every case's ranking and the mean metrics
  of each mode.
- `--metrics-log`: the CSV file the mean metrics are appended to (default
  `logs/retrieval_benchmarks_log.csv`).

Run the module with `--help` for the complete list.

#### Known limitation: keyword search

The keyword search requires every keyword to match (`q.op=AND` in
`src/client_solr.py`). With the five to seven keywords a question carries, most
searches return no table at all. In September 2026, on the NYC core, 189 of the
223 generated cases and 90 of the 100 cases of `benchmark/100q_nyc.json` got an
empty keyword ranking, for a Hit@10 of 0.04 and 0.03. All of their gold tables
are present in the core, so these scores reflect the all-keywords rule rather
than missing data. Keep this in mind when comparing modes that use the keyword
search, including `hybrid`.

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
  --port 8765
```

For an OCI-built index, use the identical embedding model at query time:

```bash
.venv-pneuma/bin/python scripts_pneuma/pneuma_server.py \
  --provider oci \
  --oci-llm-model openai.gpt-oss-20b \
  --oci-embedding-model cohere.embed-v4.0 \
  --out-path /data/pneuma/uk \
  --port 8765
```

LakeGen routes Pneuma automatically by the selected Solr core:

- `nyc` uses `/data/pneuma/nyc`, served on port `8765`;
- `uk` uses `/data/pneuma/uk`, served on port `8766`;
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

The main generated outputs are:

- `logs/experiments_log.csv`: CLI and Chainlit runs;
- `logs/api_experiments_log.csv`: API runs;
- `logs/retrieval_rankings.jsonl`: ordered retrieval candidates and scores;
- `logs/retrieval_benchmarks_log.csv`: benchmark metrics;
- `.lakegen_jobs/`: API batch state and results;
- `coding/`: Python programs generated while answering questions.

For detailed experiment parameters, use `config/experiment.example.yaml` and
the `--help` option of the relevant command.
