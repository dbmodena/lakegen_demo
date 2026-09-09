# LakeGen

LakeGen lets users query collections of tables using natural language. It
identifies the relevant tables, generates and executes the required analysis
code, and returns a textual answer. It can be used through a web interface, a
CLI, or an HTTP API.

## Installation

Requirements:

- Python 3.11-3.13 and `uv`;
- a Solr instance with the required cores and metadata already loaded;
- OCI Generative AI credentials.

Install the dependencies from the project root:

```bash
uv sync
```

OCI credentials are read from `~/.oci/config`. The profile must also include
the compartment ID:

```ini
[DEFAULT]
user=...
fingerprint=...
tenancy=...
region=eu-frankfurt-1
key_file=/path/to/oci_api_key.pem
oci_compartment_id=...
```

By default, Solr is expected at `http://localhost:8983/solr`. To use a
different address:

```bash
export SOLR_BASE_URL=http://localhost:8983/solr
```

## Web interface

```bash
uv run chainlit run src/app.py
```

Open the address displayed in the terminal, then select the data portal, model,
and retrieval mode. Do not use `-w` or `--watch`: generated Python files would
cause Chainlit to restart the active chat.

## Command-line interface

```bash
uv run python src/cli.py "Which districts have the highest number of parks?"
```

Example with options:

```bash
uv run python src/cli.py \
  --core nyc \
  --retrieval-mode hybrid \
  "Which districts have the highest number of parks?"
```

Run `uv run python src/cli.py --help` to list all available options.

A YAML or JSON configuration file can also be provided:

```bash
uv run python src/cli.py \
  --config config/experiment.example.yaml \
  "The question to analyze"
```

## HTTP API

Start the server:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Interactive documentation and endpoint testing are available at
<http://127.0.0.1:8000/docs>.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Which districts have the highest number of parks?",
    "core": "nyc",
    "retrieval_mode": "keyword"
  }'
```

The API also supports batch processing. All endpoints and their fields are
described in the interactive documentation.

## Table retrieval

LakeGen supports five retrieval modes:

- `keyword`: lexical BM25 search through Solr; this is the default mode;
- `semantic`: vector search based on the complete question;
- `hybrid`: a combination of lexical and semantic search;
- `duckdb_agentic`: direct search over local Parquet files;
- `pneuma`: retrieval through a separate Pneuma service.

The `semantic` and `hybrid` modes require Ollama and the same embedding model
used to index the documents. The default model is `bge-m3`:

```bash
ollama pull bge-m3
```

The `pneuma` mode requires a prepared index and a running Pneuma service.

## Configuration and output

The available portals are `nyc`, `valencia`, `bologna`, `paris`, and `uk`.
The OCI models available in the application are `openai.gpt-oss-120b` and
`meta.llama-3.3-70b-instruct`.

All configuration options and their default values are documented in
`config/experiment.example.yaml`. The main environment variables are:

- `SOLR_BASE_URL`: Solr base address;
- `OCI_PROFILE`: OCI profile to use;
- `OCI_COMPARTMENT_ID`: OCI compartment override;
- `LAKEGEN_EMBEDDING_BASE_URL`: embedding service address.

Results are written to `logs/`. Jobs submitted through the API are stored in
`.lakegen_jobs/` and remain available after the server restarts.
