# Environment variables

Generated from `lakegen.config.settings` by `scripts/gen_config_docs.py`; do not edit by hand.
`.env` in the repository root is loaded at start-up by the API, the CLI, the UI and the
retrieval benchmark; variables already set in the environment win over it.

## Runtime state

| Variable | Default | Meaning | Read |
|---|---|---|---|
| `LAKEGEN_STATE_DIR` | unset | Root of the runtime state (coding runs, logs, manifests, jobs, keyword memory); the repository root when unset. | import of core.config, and every call-time path |
| `LAKEGEN_JOB_DIR` | unset | Name of the API job folder under the state root (.lakegen_jobs when unset), so an API instance can own its jobs. | import of api |

## Solr

| Variable | Default | Meaning | Read |
|---|---|---|---|
| `LAKEGEN_SOLR_CORE_<LAKE>` | — | Serves a lake from a differently named Solr core; blank means the lake's own name. | call |
| `SOLR_BASE_URL` | `'http://localhost:8983/solr'` | Base URL of the Solr service. | every Solr client; the retrieval benchmark's --solr-base-url default |

## Pneuma

| Variable | Default | Meaning | Read |
|---|---|---|---|
| `LAKEGEN_PNEUMA_<LAKE>_BASE_URL` | — | The lake's Pneuma service URL. | call |
| `LAKEGEN_PNEUMA_<LAKE>_INDEX_NAME` | — | The lake's Pneuma index name. | call |

## Model providers

| Variable | Default | Meaning | Read |
|---|---|---|---|
| `TOGETHERAI_API_KEY, TOGETHER_API_KEY` | — | Together AI key; the first non-empty one wins. | call |
| `GROQ_API_KEY*, GROP_API_KEY*` | — | Groq keys for the offload, every variable matching GRO[QP]_API_KEY<digits>, in name order. | call |
| `TYPESAFE_API_KEY, JEV_API_KEY` | — | TypeSafe key for the Jev table selector; the first set one wins. | call |
| `OCI_CONFIG_FILE` | `'~/.oci/config'` | Path of the OCI configuration file. | call |
| `OCI_PROFILE` | `'DEFAULT'` | Profile of the OCI configuration file. | call |
| `OCI_COMPARTMENT_ID` | unset | Compartment for OCI Generative AI; the profile's oci_compartment_id when unset. | call |
| `OCI_SERVICE_ENDPOINT` | unset | OCI Generative AI endpoint; derived from the profile's region when unset. | call |
| `LAKEGEN_TOGETHER_BASE_URL` | `'https://api.together.xyz/v1'` | Base URL of the Together AI API. | call |
| `LAKEGEN_TOGETHER_TEMPERATURE` | `0.1` | Sampling temperature of the Together models. | call |
| `LAKEGEN_TOGETHER_PARALLEL_TOOL_CALLS` | `True` | Let Together's gpt-oss make parallel tool calls (0 turns it off). | call |
| `LAKEGEN_TOGETHER_RECOVER_LEAKS` | `True` | Recover tool calls that Together's gpt-oss writes as plain text (0 turns it off). | call |
| `LAKEGEN_OLLAMA_BASE_URL` | `'http://localhost:11434'` | Base URL of the local Ollama server for the ollama.* models. | call |
| `LAKEGEN_GROQ_OFFLOAD` | `False` | Send small tool-free gpt-oss-120b calls to Groq's free tier (1, true or yes). | call |
| `LAKEGEN_GROQ_LEDGER` | unset | Path of the Groq call ledger; logs/groq_calls.jsonl under the state root when unset. | first Groq call |

## HTTP API

| Variable | Default | Meaning | Read |
|---|---|---|---|
| `LAKEGEN_WORKFLOW_WORKERS` | `1` | Questions the API runs at once. | import of api |
| `LAKEGEN_BATCH_PAUSE_EVERY` | `0` | After this many finished questions a batch sleeps LAKEGEN_BATCH_PAUSE_SECONDS; 0 disables it. | import of api |
| `LAKEGEN_BATCH_PAUSE_SECONDS` | `0.0` | Length of the periodic batch pause. | import of api |
| `LAKEGEN_BATCH_TOKENS_PER_MINUTE` | `0.0` | Token rate a batch is held to by sleeping after each question; 0 disables it. | import of api |
| `LAKEGEN_BATCH_TOKENS_PER_MINUTE_FILE` | `''` | A file whose number overrides the token rate, re-read before every rest. | import of api |
| `LAKEGEN_SPLIT_MODEL` | `''` | A second model that runs a share of every batch's questions. | import of api |
| `LAKEGEN_SPLIT_SHARE` | `0.0` | The share of questions (0 to 1) that run on LAKEGEN_SPLIT_MODEL. | import of api |

## Behaviour switches

| Variable | Default | Meaning | Read |
|---|---|---|---|
| `LAKEGEN_SELECTION_CARDINALITY_EXPERIMENT` | `False` | Adds the experimental table-count check to the unified architect's prompt (1). | call |
| `LAKEGEN_ENABLE_TABPFN` | `False` | Lets generated analyses use the optional TabPFN stack (1, true, yes or on). | call |

## Retrieval defaults

| Variable | Default | Meaning | Read |
|---|---|---|---|
| `LAKEGEN_RETRIEVAL_MODE` | `'keyword'` | Retrieval mode. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_RETRIEVAL_TOP_K` | `20` | Ranked pool size. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_HYBRID_ALPHA` | `0.5` | Lexical weight of weighted hybrid fusion. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_CANDIDATE_MULTIPLIER` | `5` | Per-branch pool size as a multiple of top_k. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_REPRESENTATION_VERSION` | `'metadata-v1'` | Version of the indexed representation. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_EMBEDDING_MODEL` | `'cohere.embed-v4.0'` | Embedding model of semantic retrieval. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_EMBEDDING_BASE_URL` | `'http://localhost:11434'` | Ollama URL for Ollama embedding models. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_VECTOR_FIELD` | `'table_embedding'` | Solr dense-vector field. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_BM25_QUERY_FIELDS` | unset | BM25 fields and boosts in Solr qf format. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_MISSING_SIGNAL_POLICY` | `'zero'` | How hybrid fusion treats a signal missing from one branch. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_FUSION_METHOD` | `'weighted'` | Hybrid fusion method: weighted or rrf. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_RRF_K` | `60` | RRF smoothing constant. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_INDEX_NAME` | `'lakegen-cohere-v4-1024'` | Pneuma index name when the lake has no route. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_BASE_URL` | `'http://localhost:8767'` | Pneuma service URL when the lake has no route. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_TIMEOUT_SECONDS` | `120.0` | Pneuma request timeout. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_CONTENT_WEIGHT` | `0.5` | Pneuma-Seeker: weight of content search against Pneuma's ranking. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_CONTENT_FUSION` | `'weighted'` | Pneuma-Seeker: weighted or fill. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_ENTITY_MODEL` | unset | Pneuma-Seeker: chat model that extracts entities when the caller gives none. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_ENUMERATE_TABLES` | `True` | Pneuma-Seeker: pull in tables whose names form a family (0, false or no turns it off). | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_TABLE_NAME_WEIGHT` | `3.0` | Pneuma-Seeker: weight of a title match. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_COLUMN_NAME_WEIGHT` | `2.0` | Pneuma-Seeker: weight of a column-name match. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_CELL_WEIGHT` | `1.0` | Pneuma-Seeker: weight of a cell match. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_PNEUMA_VALUE_SCAN_WEIGHT` | `0.0` | Plain pneuma: weight of a value scan of the returned tables; 0 scans nothing. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_DUCKDB_MAX_FILES` | `250` | duckdb_agentic: files scanned. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_DUCKDB_MAX_COLUMNS_PER_FILE` | `40` | duckdb_agentic: columns scanned per file. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_DUCKDB_SAMPLE_ROWS` | `3` | duckdb_agentic: sample rows per file. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_DUCKDB_MAX_SCAN_ROWS_PER_FILE` | `100000` | duckdb_agentic: rows scanned per file. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_DUCKDB_PROBE_FILES` | `25` | duckdb_agentic: following files probed for value-only terms. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_DUCKDB_PROBE_ROWS_PER_FILE` | `1000` | duckdb_agentic: rows per probed file. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_MAX_FILES` | unset | Pneuma-Seeker content scan: file cap; blank scans the whole lake. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_MAX_COLUMNS_PER_FILE` | `40` | Content scan: columns per file. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_MAX_SCAN_ROWS_PER_FILE` | `100000` | Content scan: rows per file. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_SAMPLE_ROWS` | `3` | Content scan: sample rows. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_PROBE_FILES` | `25` | Content scan: probed files. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_SCAN_WORKERS` | `16` | Worker processes of the content scan. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_VALUE_WEIGHT` | `1.0` | Content scan: weight of value evidence. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_METADATA_WEIGHT` | `1.0` | Content scan: weight of metadata evidence. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_GREP_PROBE_ROWS_PER_FILE` | `1000` | Content scan: rows per probed file. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_RERANK_MODEL` | `'cohere.rerank-v4.0-fast'` | Second-stage reranker; empty turns reranking off. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
| `LAKEGEN_RERANK_DEPTH` | `20` | How many top hits the reranker reorders. | RetrievalConfig.from_env: every experiment config load, the CLI and the UI |
