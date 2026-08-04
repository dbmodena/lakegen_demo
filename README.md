# LakeGen

## OCI Generative AI

LakeGen uses OCI Generative AI for text generation and reads credentials from
the standard `~/.oci/config` file. The selected profile must contain the normal
OCI API-key fields plus the LakeGen compartment field:

```ini
[DEFAULT]
user=...
fingerprint=...
tenancy=...
region=eu-frankfurt-1
key_file=/absolute/path/to/oci_api_key.pem
oci_compartment_id=...
```

Keep the config and private key outside the repository and restrict their
permissions:

```bash
chmod 700 ~/.oci
chmod 600 ~/.oci/config ~/.oci/oci_api_key.pem
```

Optional environment overrides are `OCI_CONFIG_FILE`, `OCI_PROFILE`,
`OCI_COMPARTMENT_ID`, and `OCI_SERVICE_ENDPOINT`.

The Chainlit, CLI, and HTTP API model selector exposes:

- `openai.gpt-oss-120b` (default)
- `meta.llama-3.3-70b-instruct`

## Chainlit UI

Start the interactive application without file watching:

```bash
uv run chainlit run src/app.py
```

Do not add `-w` or `--watch` during normal use. LakeGen generates Python files
under `coding/` while answering a question; Chainlit's watcher would detect
those files, restart the server, and discard the active chat before the final
answer is displayed.

## HTTP API

Avvia l'API in locale:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

La documentazione OpenAPI interattiva è disponibile su
`http://127.0.0.1:8000/docs`.

### Domanda singola

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Which districts have the most parks?",
    "core": "nyc",
    "model": "openai.gpt-oss-120b"
  }'
```

### File JSON di domande

L'endpoint batch accetta direttamente il contenuto dei file in `queries_old`,
oltre ai formati più semplici `[{"question": "..."}]` e
`{"questions": ["...", "..."]}`.

```bash
curl -X POST \
  'http://127.0.0.1:8000/v1/batches?core=nyc&model=openai.gpt-oss-120b' \
  -H 'Content-Type: application/json' \
  --data-binary @queries_old/generated_queries_new_york.json
```

La risposta contiene un `job_id`. Controlla avanzamento e risultati con:

```bash
curl http://127.0.0.1:8000/v1/batches/JOB_ID
```

Per leggere solo lo stato, senza trasferire i risultati già prodotti:

```bash
curl 'http://127.0.0.1:8000/v1/batches/JOB_ID?include_results=false'
```

I batch sono eseguiti sequenzialmente. Stato e risultati JSONL vengono salvati in
`.lakegen_jobs/`, così le risposte già completate restano leggibili dopo un
riavvio dell'API. L'API va esposta su una rete pubblica solo dopo aver aggiunto un
livello di autenticazione e limiti di traffico.

Il riepilogo CSV delle esecuzioni API viene scritto in
`logs/api_experiments_log.csv`; CLI e Chainlit continuano a usare
`logs/experiments_log.csv`.
