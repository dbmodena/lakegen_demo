ciao
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
- Ollama with `bge-m3` when using semantic or hybrid retrieval.

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
experiment.

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

- `keyword`: BM25 lexical search through Solr; this is the default mode.
- `semantic`: vector search using the complete question.
- `hybrid`: combines lexical and semantic results.
- `duckdb_agentic`: searches local Parquet files without a Solr index.
- `pneuma`: uses a separately prepared Pneuma index and service.

### Enable semantic and hybrid retrieval

Install the default embedding model in Ollama:

```bash
ollama pull bge-m3
```

Ollama is expected at `http://localhost:11434`. The embedding model and
representation version must match those used to index the Solr documents.

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
successful Pandas cases from an existing generated-query JSON file and writes a
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

This compares keyword, semantic, and hybrid retrieval without running the
agents, code generation, or answer synthesis:

```bash
PYTHONPATH=src:. uv run python -m lakegen.retrieval.benchmark \
  benchmark/100q_nyc.json \
  --core nyc \
  --output logs/nyc_retrieval_benchmark.json
```

Useful optional arguments include `--top-k`, `--candidate-multiplier`,
`--alphas`, and `--table-dir`. Run the module with `--help` for the complete
list.

## 8. Pneuma retrieval (optional)

Pneuma uses a separate Python 3.12 environment because its dependencies
conflict with LakeGen's main environment:

```bash
uv venv --python 3.12 .venv-pneuma
uv pip install --python .venv-pneuma/bin/python -r requirements-pneuma.txt
```

Prepare an index for a portal:

```bash
bash scripts/run_pneuma_bootstrap_monitored.sh nyc
```

By default Pneuma uses the configured OpenAI-compatible Ollama endpoint. To
offload both summarization and embedding generation to OCI Generative AI, use
the same OCI profile and compartment configured for LakeGen:

```bash
PNEUMA_PROVIDER=oci bash scripts/run_pneuma_bootstrap_monitored.sh uk \
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
.venv-pneuma/bin/python scripts/pneuma_server.py \
  --out-path /data/pneuma/nyc \
  --port 8765
```

For an OCI-built index, use the identical embedding model at query time:

```bash
.venv-pneuma/bin/python scripts/pneuma_server.py \
  --provider oci \
  --oci-llm-model openai.gpt-oss-20b \
  --oci-embedding-model cohere.embed-v4.0 \
  --out-path /data/pneuma/uk \
  --port 8765
```

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
