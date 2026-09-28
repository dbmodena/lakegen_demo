# LakeGen — risultati sperimentali della tesi

Questa cartella contiene i risultati aggregati degli esperimenti della tesi,
pronti per scrivere il capitolo dei risultati e produrre tabelle e figure.
I dati grezzi (un JSON per domanda, circa 1,6 GB) non sono versionati: restano
sul server in `.lakegen_jobs*/` e si rigenerano con gli script indicati sotto.

## Il sistema in breve

LakeGen risponde a domande in linguaggio naturale su un data lake di tabelle
open data. Per ogni domanda: (1) **retrieval** delle tabelle candidate da un
indice Solr, (2) **selezione** delle tabelle da parte di un LLM, (3) un
**coder** LLM scrive ed esegue codice pandas, (4) il risultato viene confrontato
con un risultato di riferimento.

## Disegno sperimentale (fattoriale)

| Fattore | Livelli | Chiave nei CSV |
|---|---|---|
| Portale (core) | NYC (~2.700 tabelle), UK (~15.000 tabelle) | `core` |
| Retrieval | BM25 (lessicale), Denso (KNN su embedding), Ibrido (fusione pesata, alpha 0,25), RRF (Reciprocal Rank Fusion) | `retrieval` |
| Modello | GPT-OSS 120B, Llama 3.3 70B (entrambi su OCI) | `model` / `model_label` |
| Interazione | Agentica (il modello usa tool di ricerca/ispezione), Orchestrata (il sistema prepara il contesto, il modello sceglie senza tool) | `access` |
| Contesto del coder | full (schema + righe di esempio), schema_only, minimal (solo comando di caricamento) | `coder_context` |

2 × 4 × 2 × 2 = 32 configurazioni. Le tre varianti del coder girano **sulla
stessa selezione di tabelle** in ogni configurazione, quindi l'effetto del
contesto del coder è isolato dal resto.

**Benchmark.** Domande generate automaticamente (con codice e risultato di
riferimento) da un altro sistema: `benchmark/381q_nyc.json`, `benchmark/414q_uk.json`.
Le prime 100 domande di ciascuno sono un sottoinsieme con 20 domande multi-table
e 80 single-table, difficoltà proporzionali. Il resto (281 NYC, 314 UK) è lo
"stage 2".

**Cosa è stato eseguito.**
- 32 configurazioni sulle prime 100 domande (agentico e orchestrato): cartella `100q/`.
- 16 configurazioni agentiche sul benchmark completo (381/414): cartella `full/`
  (aggiunta a fine esecuzione).
- 4 configurazioni agentiche ripetute identiche sulle prime 100 domande, per
  misurare la variabilità tra due run: cartella `repetition/`.

## Metriche

Tutte le metriche sono **end-to-end**: una domanda bloccata prima del coder conta
come fallimento. Valori in [0, 1] nei CSV, più alto = meglio.

- `exact_selection`: le tabelle selezionate coincidono esattamente con le tabelle gold.
- `selection_hit` / `selection_recall`: almeno una gold selezionata / quota di gold selezionate.
- `exact_result_match`: il risultato coincide con il riferimento (confronto severo).
- `lenient_result_match`: confronto sul contenuto, tollera nomi di colonna, maiuscole,
  scala percentuale (×100) e colonne extra; le righe devono coincidere.
- `supported_correct`: risultato corretto o equivalente e supportato da evidenza
  (giudizio di un LLM con verifica).
- `execution_success`: il codice generato è stato eseguito senza errori.
- Metriche di retrieval (solo in `summary.md`): Recall@10, Hit@5, MRR del ranking.

## File

### `100q/summary.md`
Report leggibile, per portale: tabella principale (32 configurazioni, coder full),
effetti principali di ogni fattore con intervallo di confidenza al 95%, confronto
dei contesti del coder, interazione modello × modalità.

### `100q/effects.csv`
Una riga per effetto stimato. Colonne: `core`, `factor` (retrieval, model, access,
coder_context), `comparison` (un livello, es. `Denso`, oppure una differenza
appaiata, es. `Agentica - Orchestrata`), `outcome` (metrica), `mean`,
`ci_low`, `ci_high` (IC 95% bootstrap sulle domande, 2000 ripetizioni),
`questions`. Una differenza è "significativa" se l'intervallo non contiene lo 0.

### `100q/per_question_all.csv` (9.600 righe)
Una riga per (configurazione, domanda, contesto del coder). È la fonte per
qualsiasi grafico o analisi aggiuntiva. Colonne principali:
`experiment_id`, `core`, `model_label`, `access`, `retrieval`, `coder_context`,
`question_id`, `gold_table_count` (1 = single-table, >1 = multi-table),
`exact_selection`, `selection_recall`, `exact_result_match`,
`lenient_result_match`, `supported_correct`, `execution_success`,
`status` (completed / failed / blocked_by_selection / rejected),
`elapsed_seconds`, `tokens_total` (comprende le tre varianti del coder),
`coder_ran` (0 = domanda fermata prima del coder).
Per le metriche di selezione usare solo le righe con `coder_context == "full"`
(la selezione è la stessa per i tre contesti).

### `repetition/repetition_comparison.csv`
Per 4 configurazioni (migliore per modello e portale): metrica nella run
originale e nella ripetizione, differenza in punti percentuali, quota di domande
con lo stesso esito (`agreement`), domande corrette solo in una delle due run.

### `logs/`
Log delle tre esecuzioni (`suite*.log`: invio e completamento dei batch) e file
di stato con gli id dei job sul server.

## Risultati principali (100 domande, coder full)

- **Orchestrato > agentico** con retrieval denso/ibrido/RRF, su entrambi i portali
  e con entrambi i modelli, a circa metà del costo (tempo e token). Differenza
  media agentico − orchestrato: NYC exact selection −21 pp [−27, −16], exact match
  −8 pp [−12, −4]; UK exact selection −13 pp [−18, −7], exact match −3 pp (n.s.).
- **Eccezione: BM25 orchestrato su UK** crolla (Recall@10 ~41% contro ~70%
  in agentico): senza poter riformulare, la ricerca in AND stretto spesso non
  trova tabelle.
- **Modello:** GPT-OSS seleziona le tabelle molto meglio di Llama (+15/+20 pp);
  sull'accuratezza finale il vantaggio è piccolo (NYC exact +7 pp, UK ~0).
- **Llama** beneficia moltissimo dell'orchestrato nella selezione (es. NYC
  ibrido: 21% → 63%): usa male i tool, ma sceglie bene da una lista.
- **Contesto del coder:** full ≈ schema_only; minimal perde 4–7 pp.
- **Ripetizioni:** due run identiche differiscono fino a ±5 pp su 100 domande
  (81–96% delle domande con lo stesso esito). Differenze di pochi punti vanno
  considerate non distinguibili dal rumore.

## Limiti da dichiarare

1. **Benchmark sintetico:** domande e riferimenti generati automaticamente; alcuni
   riferimenti potrebbero essere errati o ambigui.
2. **Un'esecuzione per configurazione** (tranne le 4 ripetute); i modelli non sono
   deterministici (OCI non supporta un seed; temperatura 0,1).
3. **Ricerca densa senza ente:** il campo `publisher` non era incluso nei campi
   restituiti dalla ricerca KNN, quindi con denso/ibrido/RRF né il modello né il
   reranker vedono l'ente che pubblica la tabella (con BM25 sì). Lasciato
   invariato per tutte le run per coerenza.
4. **Titoli UK generici:** molte risorse UK hanno titoli come "2019-20"; il titolo
   del pacchetto non è indicizzato.
5. **Correzioni durante le run** (valgono per tutti i modelli):
   - parser del selettore orchestrato: accetta nomi di tabella senza `.parquet`
     (Llama li ometteva); applicata prima delle run orchestrate valide;
   - `OverflowError` con interi molto grandi nella validazione del risultato del
     coder: una domanda NYC (`6555f2d2`) fallita per questo in 4 run agentiche Llama;
   - limite di memoria di 12 GB per script generato (8 GB per le ripetizioni),
     introdotto dopo che un join molti-a-molti aveva saturato la RAM del server;
     gli script oltre il limite falliscono con MemoryError invece di essere uccisi.
6. **Keyword memory:** ogni configurazione ha una memoria delle ricerche fallite
   che si accumula tra le domande della stessa run (non tra configurazioni).

## Rigenerare i report

```
.venv/bin/python analysis/compare_runs.py      # scrive reports/thesis/{100q,full}/
```
