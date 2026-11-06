# Pneuma retrieval engine

The index builder and the HTTP query service behind LakeGen's `pneuma` and
`pneuma_seeker` retrieval modes, around the MIT-licensed
[Pneuma 0.0.4](https://pypi.org/project/pneuma/). LakeGen never imports this
package: it sends questions to the service over HTTP
(`src/lakegen/integrations/pneuma/client.py`).

| Module | Holds |
|---|---|
| `server.py` | the HTTP service: `GET /health`, `POST /query` |
| `bootstrap.py` | builds or resumes one lake's index |
| `indexing.py` | patches to Pneuma's indexing: bounded DuckDB reads, robust embeddings, batched Chroma inserts |
| `metadata.py` | the catalogue text registered as each table's context |
| `judge.py` | the schema-constrained LLM relevance judge (paper §5.1) |
| `narration.py` | concurrent column narration, capped row samples |
| `oci_client.py` | an OpenAI-shaped client backed by OCI Generative AI |

## Install

The engine has its own environment: Pneuma's Chroma stack needs NumPy < 2,
LakeGen needs NumPy >= 2.

```bash
pneuma_engine/install.sh            # creates .venv-pneuma at the repository root
```

`PNEUMA_VENV` changes where, `PNEUMA_PYTHON` the Python version (with uv), and
`PNEUMA_CONSTRAINTS=pneuma_engine/constraints.txt` pins the CUDA 12.4 torch
wheel of the GPU host.

## Build an index

```bash
pneuma_engine/run_bootstrap.sh nyc
```

It reads `data/<lake>/` (override with `--table-dir` and `--metadata`), writes
the index to `pneuma-out/<lake>` (`PNEUMA_OUT_PATH`), and logs to
`bootstrap.log` and its memory use to `bootstrap.resources.log` there. Every
step resumes where an interrupted build stopped.

By default Pneuma uses an OpenAI-compatible endpoint (a local Ollama). To run
both summaries and embeddings on OCI Generative AI, with the OCI profile LakeGen
uses:

```bash
PNEUMA_PROVIDER=oci pneuma_engine/run_bootstrap.sh uk \
  --oci-llm-model openai.gpt-oss-20b \
  --oci-embedding-model cohere.embed-v4.0
```

`OCI_PROFILE`, `OCI_CONFIG_FILE`, `OCI_COMPARTMENT_ID` and
`OCI_SERVICE_ENDPOINT` are honoured. The provider and model names are recorded
in `inference-provider.json` next to the index; credentials never are. Use
`--summary-limit N --skip-metadata --skip-index` for a canary that processes
only the next `N` pending tables. `python -m pneuma_engine.bootstrap --help`
lists every option.

## Serve an index

From the repository root:

```bash
.venv-pneuma/bin/python -m pneuma_engine.server --out-path pneuma-out/nyc --port 8774
```

An OCI-built index needs the same embedding model at query time:

```bash
.venv-pneuma/bin/python -m pneuma_engine.server \
  --provider oci \
  --oci-llm-model openai.gpt-oss-20b \
  --oci-embedding-model cohere.embed-v4.0 \
  --out-path pneuma-out/uk \
  --port 8767
```

`--chat-base-url` sends the judge's chat requests to an OpenAI-compatible
endpoint (such as a local Ollama) while embeddings stay on OCI.

LakeGen's experiment harness can also start one service per lake:
`python -m experiment_runner services up pneuma:<lake>`
runs the template in `config/services.yaml`, whose index folders are relative
to `LAKEGEN_PNEUMA_INDEX_DIR` (`pneuma-out` by default).

## Reach it from LakeGen

Each lake's Pneuma route (URL and index name) is in `config/lakes.yaml`;
`LAKEGEN_PNEUMA_<LAKE>_BASE_URL` and `LAKEGEN_PNEUMA_<LAKE>_INDEX_NAME`
override it. Then select the `pneuma` or `pneuma_seeker` retrieval mode.

## Tests

The tests need no Pneuma installation; they run with LakeGen's suite
(`pytest` from the repository root collects `pneuma_engine/tests`).
