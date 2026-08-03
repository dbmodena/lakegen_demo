# LakeGen

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
    "model": "qwen3.6:27b"
  }'
```

### File JSON di domande

L'endpoint batch accetta direttamente il contenuto dei file in `queries_old`,
oltre ai formati più semplici `[{"question": "..."}]` e
`{"questions": ["...", "..."]}`.

```bash
curl -X POST \
  'http://127.0.0.1:8000/v1/batches?core=nyc&model=qwen3.6:27b' \
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
