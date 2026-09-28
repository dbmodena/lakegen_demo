# Confronto delle configurazioni — 100q

Run trovate: 32 / 32.

## Core NYC

16 run, 100 domande. Valori end-to-end in percentuale con intervallo di confidenza al 95% (bootstrap sulle domande, 2000 ripetizioni).

### 1. Tabella principale (contesto coder: full)

| Retrieval | Modello | Interazione | Recall@10 | Hit@5 | MRR | Exact selection | Exact match | Lenient match | Supported | Tempo medio | Token medi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | GPT-OSS 120B | Agentica | 58.7% | 64.0% | 58.3% | 40.0% | 16.0% | 23.0% | 16.0% | 220s | 143,633 |
| BM25 | GPT-OSS 120B | Orchestrata | 59.8% | 61.0% | 55.4% | 51.0% | 29.0% | 36.0% | 29.0% | 108s | 71,570 |
| BM25 | Llama 3.3 70B | Agentica | 60.0% | 64.0% | 56.8% | 37.0% | 14.0% | 20.0% | 15.0% | 273s | 142,103 |
| BM25 | Llama 3.3 70B | Orchestrata | 58.5% | 60.0% | 55.5% | 45.0% | 20.0% | 29.0% | 21.0% | 183s | 109,064 |
| Denso | GPT-OSS 120B | Agentica | 95.5% | 96.0% | 81.8% | 60.0% | 27.0% | 37.0% | 29.0% | 162s | 100,449 |
| Denso | GPT-OSS 120B | Orchestrata | 95.5% | 96.0% | 81.8% | 70.0% | 33.0% | 42.0% | 33.0% | 141s | 85,050 |
| Denso | Llama 3.3 70B | Agentica | 95.5% | 96.0% | 81.8% | 25.0% | 21.0% | 26.0% | 21.0% | 340s | 195,498 |
| Denso | Llama 3.3 70B | Orchestrata | 93.5% | 94.0% | 79.8% | 57.0% | 23.0% | 31.0% | 23.0% | 225s | 138,514 |
| Ibrido | GPT-OSS 120B | Agentica | 98.5% | 99.0% | 84.4% | 67.0% | 24.0% | 36.0% | 25.0% | 175s | 102,879 |
| Ibrido | GPT-OSS 120B | Orchestrata | 98.5% | 99.0% | 84.2% | 73.0% | 35.0% | 44.0% | 35.0% | 143s | 85,745 |
| Ibrido | Llama 3.3 70B | Agentica | 98.5% | 99.0% | 84.2% | 21.0% | 16.0% | 24.0% | 17.0% | 328s | 171,875 |
| Ibrido | Llama 3.3 70B | Orchestrata | 96.5% | 97.0% | 81.7% | 63.0% | 28.0% | 37.0% | 28.0% | 238s | 140,425 |
| RRF | GPT-OSS 120B | Agentica | 98.5% | 99.0% | 83.3% | 56.0% | 25.0% | 36.0% | 26.0% | 184s | 113,264 |
| RRF | GPT-OSS 120B | Orchestrata | 96.5% | 97.0% | 82.8% | 74.0% | 31.0% | 46.0% | 31.0% | 139s | 88,253 |
| RRF | Llama 3.3 70B | Agentica | 98.5% | 99.0% | 84.6% | 22.0% | 18.0% | 29.0% | 18.0% | 335s | 170,809 |
| RRF | Llama 3.3 70B | Orchestrata | 96.5% | 96.0% | 82.3% | 63.0% | 27.0% | 36.0% | 27.0% | 253s | 152,721 |

I token comprendono le tre varianti del coder, eseguite in ogni run. Le metriche di retrieval sono le medie del batch, senza intervallo.

### 2. Effetti principali (contesto coder: full)

#### Retrieval

| Retrieval | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| BM25 | 43.2% [36.0, 50.7] | 19.8% [14.0, 25.8] | 27.0% [20.8, 34.0] | 20.2% [14.5, 26.2] |
| Denso | 53.0% [45.8, 60.2] | 26.0% [19.5, 33.2] | 34.0% [26.5, 42.0] | 26.5% [19.8, 33.8] |
| Ibrido | 56.0% [48.8, 63.2] | 25.8% [19.2, 32.8] | 35.2% [27.8, 43.2] | 26.2% [19.5, 33.5] |
| RRF | 53.8% [46.8, 60.8] | 25.2% [19.2, 32.0] | 36.8% [29.5, 44.5] | 25.5% [19.5, 32.2] |

Differenze appaiate in punti percentuali (* = intervallo che non contiene lo zero):

| Confronto | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| BM25 − Denso | -9.8 [-15.5, -3.5] * | -6.2 [-10.8, -1.5] * | -7.0 [-12.0, -1.8] * | -6.2 [-11.0, -1.5] * |
| BM25 − Ibrido | -12.8 [-17.8, -7.5] * | -6.0 [-10.2, -2.0] * | -8.2 [-13.0, -3.8] * | -6.0 [-10.0, -2.0] * |
| BM25 − RRF | -10.5 [-15.8, -5.0] * | -5.5 [-9.5, -1.5] * | -9.8 [-14.5, -5.2] * | -5.2 [-9.5, -1.2] * |
| Denso − Ibrido | -3.0 [-7.0, +0.8] | +0.2 [-3.2, +3.5] | -1.2 [-5.0, +2.2] | +0.2 [-3.2, +3.5] |
| Denso − RRF | -0.8 [-4.5, +2.8] | +0.8 [-2.8, +4.2] | -2.8 [-6.2, +0.5] | +1.0 [-2.5, +4.5] |
| Ibrido − RRF | +2.2 [-0.5, +5.0] | +0.5 [-2.5, +3.5] | -1.5 [-4.8, +1.2] | +0.8 [-2.5, +3.8] |

#### Modello

| Modello | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| GPT-OSS 120B | 61.4% [53.6, 68.5] | 27.5% [20.8, 34.8] | 37.5% [29.9, 45.9] | 28.0% [21.2, 35.5] |
| Llama 3.3 70B | 41.6% [34.9, 48.4] | 20.9% [15.0, 27.3] | 29.0% [22.1, 36.1] | 21.2% [15.4, 27.5] |

Differenze appaiate in punti percentuali (* = intervallo che non contiene lo zero):

| Confronto | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| GPT-OSS 120B − Llama 3.3 70B | +19.8 [+14.6, +24.2] * | +6.6 [+1.2, +12.0] * | +8.5 [+3.1, +14.0] * | +6.8 [+1.4, +12.2] * |

#### Interazione

| Interazione | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| Agentica | 41.0% [34.5, 47.5] | 20.1% [14.8, 25.6] | 28.9% [22.4, 35.8] | 20.9% [15.2, 26.8] |
| Orchestrata | 62.0% [54.1, 69.8] | 28.2% [21.5, 35.8] | 37.6% [29.8, 46.0] | 28.4% [21.6, 35.9] |

Differenze appaiate in punti percentuali (* = intervallo che non contiene lo zero):

| Confronto | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| Agentica − Orchestrata | -21.0 [-26.8, -15.5] * | -8.1 [-12.2, -4.2] * | -8.8 [-13.4, -4.4] * | -7.5 [-11.6, -3.4] * |

### 3. Contesto del coder

Le tre varianti usano la stessa selezione di tabelle, quindi il confronto isola l'effetto del contesto. Condizionato = solo domande in cui il coder è partito (come nelle metriche del batch).

| Modello | Contesto | Exact match | Lenient match | Supported | Exact condizionato |
|---|---:|---:|---:|---:|---:|
| GPT-OSS 120B | full | 27.5% [20.8, 34.8] | 37.5% [29.9, 45.9] | 28.0% [21.2, 35.5] | 30.1% |
| GPT-OSS 120B | schema_only | 26.9% [20.2, 33.9] | 35.6% [28.2, 43.5] | 27.3% [20.6, 34.4] | 29.3% |
| GPT-OSS 120B | minimal | 24.9% [18.9, 31.2] | 33.0% [26.4, 40.0] | 25.4% [19.2, 31.8] | 27.5% |
| Llama 3.3 70B | full | 20.9% [15.0, 27.3] | 29.0% [22.1, 36.1] | 21.2% [15.4, 27.5] | 23.1% |
| Llama 3.3 70B | schema_only | 19.6% [13.8, 26.1] | 26.0% [19.4, 33.5] | 20.0% [14.1, 26.5] | 21.2% |
| Llama 3.3 70B | minimal | 12.1% [7.8, 16.8] | 16.6% [11.5, 22.5] | 12.6% [8.2, 17.5] | 13.7% |
| Tutti | full | 24.2% [18.2, 30.4] | 33.2% [26.5, 40.2] | 24.6% [18.6, 30.9] | 26.9% |
| Tutti | schema_only | 23.2% [17.7, 29.1] | 30.8% [24.4, 37.6] | 23.6% [18.1, 29.7] | 25.8% |
| Tutti | minimal | 18.5% [14.1, 23.2] | 24.8% [19.5, 30.7] | 19.0% [14.5, 23.9] | 20.7% |

| Modello | Confronto | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| GPT-OSS 120B | full − schema_only | +0.6 [-1.5, +2.8] | +1.9 [-0.8, +4.6] | +0.8 [-1.4, +2.9] |
| GPT-OSS 120B | full − minimal | +2.6 [-0.5, +6.0] | +4.5 [+1.0, +8.1] * | +2.6 [-0.6, +6.0] |
| Llama 3.3 70B | full − schema_only | +1.2 [-2.0, +4.4] | +3.0 [-0.6, +6.6] | +1.2 [-2.0, +4.5] |
| Llama 3.3 70B | full − minimal | +8.8 [+4.9, +12.6] * | +12.4 [+7.6, +17.1] * | +8.6 [+4.8, +12.5] * |
| Tutti | full − schema_only | +0.9 [-1.1, +2.9] | +2.4 [-0.1, +4.9] | +1.0 [-1.1, +3.0] |
| Tutti | full − minimal | +5.7 [+3.1, +8.4] * | +8.4 [+5.2, +11.8] * | +5.6 [+3.0, +8.4] * |

### 4. Interazione modello × modalità (exact match, full)

| Modello | Agentica | Orchestrata |
|---|---:|---:|
| GPT-OSS 120B | 23.0% [16.5, 30.0] | 32.0% [24.0, 40.2] |
| Llama 3.3 70B | 17.2% [11.2, 23.8] | 24.5% [17.0, 32.5] |

## Core UK

16 run, 100 domande. Valori end-to-end in percentuale con intervallo di confidenza al 95% (bootstrap sulle domande, 2000 ripetizioni).

### 1. Tabella principale (contesto coder: full)

| Retrieval | Modello | Interazione | Recall@10 | Hit@5 | MRR | Exact selection | Exact match | Lenient match | Supported | Tempo medio | Token medi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | GPT-OSS 120B | Agentica | 68.7% | 73.0% | 57.3% | 46.0% | 20.0% | 35.0% | 26.0% | 217s | 134,503 |
| BM25 | GPT-OSS 120B | Orchestrata | 40.5% | 40.0% | 33.5% | 32.0% | 11.0% | 25.0% | 15.0% | 96s | 64,284 |
| BM25 | Llama 3.3 70B | Agentica | 71.7% | 79.0% | 62.1% | 38.0% | 19.0% | 26.0% | 21.0% | 277s | 121,624 |
| BM25 | Llama 3.3 70B | Orchestrata | 41.1% | 45.0% | 34.9% | 22.0% | 12.0% | 19.0% | 12.0% | 190s | 93,886 |
| Denso | GPT-OSS 120B | Agentica | 82.2% | 80.0% | 57.9% | 35.0% | 17.0% | 30.0% | 19.0% | 210s | 127,915 |
| Denso | GPT-OSS 120B | Orchestrata | 82.2% | 80.0% | 57.9% | 50.0% | 19.0% | 36.0% | 22.0% | 137s | 78,835 |
| Denso | Llama 3.3 70B | Agentica | 82.2% | 80.0% | 57.9% | 14.0% | 13.0% | 21.0% | 14.0% | 340s | 155,548 |
| Denso | Llama 3.3 70B | Orchestrata | 82.2% | 80.0% | 57.9% | 42.0% | 25.0% | 30.0% | 25.0% | 262s | 136,351 |
| Ibrido | GPT-OSS 120B | Agentica | 90.0% | 85.0% | 62.4% | 40.0% | 17.0% | 30.0% | 19.0% | 212s | 126,810 |
| Ibrido | GPT-OSS 120B | Orchestrata | 88.0% | 83.0% | 60.8% | 56.0% | 21.0% | 37.0% | 28.0% | 148s | 85,989 |
| Ibrido | Llama 3.3 70B | Agentica | 86.5% | 83.0% | 60.9% | 13.0% | 17.0% | 22.0% | 18.0% | 338s | 151,088 |
| Ibrido | Llama 3.3 70B | Orchestrata | 87.5% | 84.0% | 60.8% | 46.0% | 25.0% | 31.0% | 25.0% | 323s | 162,704 |
| RRF | GPT-OSS 120B | Agentica | 86.2% | 84.0% | 62.4% | 41.0% | 18.0% | 33.0% | 20.0% | 208s | 123,968 |
| RRF | GPT-OSS 120B | Orchestrata | 84.5% | 82.0% | 62.6% | 53.0% | 23.0% | 40.0% | 29.0% | 143s | 89,364 |
| RRF | Llama 3.3 70B | Agentica | 88.5% | 85.0% | 63.0% | 17.0% | 13.0% | 21.0% | 14.0% | 347s | 157,693 |
| RRF | Llama 3.3 70B | Orchestrata | 83.2% | 82.0% | 59.6% | 45.0% | 25.0% | 32.0% | 25.0% | 250s | 126,423 |

I token comprendono le tre varianti del coder, eseguite in ogni run. Le metriche di retrieval sono le medie del batch, senza intervallo.

### 2. Effetti principali (contesto coder: full)

#### Retrieval

| Retrieval | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| BM25 | 34.5% [28.2, 41.2] | 15.5% [10.5, 21.0] | 26.2% [20.0, 32.8] | 18.5% [13.0, 24.8] |
| Denso | 35.2% [28.5, 42.2] | 18.5% [13.0, 24.5] | 29.2% [22.2, 36.5] | 20.0% [14.2, 26.8] |
| Ibrido | 38.8% [32.0, 45.8] | 20.0% [14.0, 26.5] | 30.0% [23.2, 37.2] | 22.5% [16.2, 29.2] |
| RRF | 39.0% [32.2, 45.8] | 19.8% [14.2, 25.8] | 31.5% [24.5, 38.8] | 22.0% [16.0, 28.2] |

Differenze appaiate in punti percentuali (* = intervallo che non contiene lo zero):

| Confronto | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| BM25 − Denso | -0.8 [-7.2, +5.8] | -3.0 [-7.8, +2.2] | -3.0 [-8.8, +2.8] | -1.5 [-6.8, +4.2] |
| BM25 − Ibrido | -4.2 [-10.0, +1.2] | -4.5 [-8.8, -0.5] * | -3.8 [-8.5, +1.2] | -4.0 [-8.5, +0.5] |
| BM25 − RRF | -4.5 [-9.5, +0.2] | -4.2 [-8.0, -0.8] * | -5.2 [-9.8, -1.0] * | -3.5 [-7.5, +0.2] |
| Denso − Ibrido | -3.5 [-8.0, +0.8] | -1.5 [-5.5, +2.0] | -0.8 [-5.0, +3.2] | -2.5 [-6.5, +1.0] |
| Denso − RRF | -3.8 [-9.0, +1.2] | -1.2 [-5.0, +2.2] | -2.2 [-6.8, +1.8] | -2.0 [-6.0, +2.0] |
| Ibrido − RRF | -0.2 [-3.5, +3.0] | +0.2 [-2.8, +3.5] | -1.5 [-5.2, +2.5] | +0.5 [-2.8, +4.0] |

#### Modello

| Modello | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| GPT-OSS 120B | 44.1% [37.2, 51.1] | 18.2% [12.9, 24.2] | 33.2% [26.1, 40.4] | 22.2% [16.5, 28.7] |
| Llama 3.3 70B | 29.6% [23.9, 35.5] | 18.6% [13.2, 24.8] | 25.2% [19.0, 32.1] | 19.2% [13.6, 25.6] |

Differenze appaiate in punti percentuali (* = intervallo che non contiene lo zero):

| Confronto | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| GPT-OSS 120B − Llama 3.3 70B | +14.5 [+9.8, +19.1] * | -0.4 [-5.0, +4.8] | +8.0 [+3.0, +13.4] * | +3.0 [-1.5, +7.8] |

#### Interazione

| Interazione | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| Agentica | 30.5% [24.9, 36.4] | 16.8% [11.6, 22.5] | 27.3% [21.0, 33.8] | 18.9% [13.8, 24.9] |
| Orchestrata | 43.2% [35.8, 51.1] | 20.1% [14.5, 26.0] | 31.2% [24.5, 38.1] | 22.6% [16.6, 29.0] |

Differenze appaiate in punti percentuali (* = intervallo che non contiene lo zero):

| Confronto | Exact selection | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| Agentica − Orchestrata | -12.8 [-18.2, -7.2] * | -3.4 [-7.1, +0.6] | -4.0 [-8.2, +0.5] | -3.8 [-7.6, +0.2] |

### 3. Contesto del coder

Le tre varianti usano la stessa selezione di tabelle, quindi il confronto isola l'effetto del contesto. Condizionato = solo domande in cui il coder è partito (come nelle metriche del batch).

| Modello | Contesto | Exact match | Lenient match | Supported | Exact condizionato |
|---|---:|---:|---:|---:|---:|
| GPT-OSS 120B | full | 18.2% [12.9, 24.2] | 33.2% [26.1, 40.4] | 22.2% [16.5, 28.7] | 20.8% |
| GPT-OSS 120B | schema_only | 17.2% [12.1, 23.1] | 32.5% [25.5, 39.6] | 20.0% [14.4, 26.5] | 19.8% |
| GPT-OSS 120B | minimal | 16.5% [11.5, 21.9] | 31.2% [24.6, 38.1] | 19.9% [14.4, 26.0] | 19.0% |
| Llama 3.3 70B | full | 18.6% [13.2, 24.8] | 25.2% [19.0, 32.1] | 19.2% [13.6, 25.6] | 21.7% |
| Llama 3.3 70B | schema_only | 15.1% [9.8, 20.9] | 21.4% [15.2, 28.0] | 15.4% [10.1, 21.2] | 17.4% |
| Llama 3.3 70B | minimal | 10.5% [6.4, 14.9] | 15.8% [10.8, 21.4] | 11.0% [6.5, 15.6] | 11.7% |
| Tutti | full | 18.4% [13.5, 23.9] | 29.2% [23.1, 35.5] | 20.8% [15.6, 26.6] | 21.1% |
| Tutti | schema_only | 16.2% [11.6, 21.2] | 26.9% [20.9, 33.2] | 17.7% [13.0, 23.2] | 18.6% |
| Tutti | minimal | 13.5% [9.8, 17.5] | 23.5% [18.4, 28.9] | 15.4% [11.3, 20.0] | 15.7% |

| Modello | Confronto | Exact match | Lenient match | Supported |
|---|---:|---:|---:|---:|
| GPT-OSS 120B | full − schema_only | +1.0 [-0.9, +2.9] | +0.8 [-0.9, +2.2] | +2.2 [+0.5, +3.9] * |
| GPT-OSS 120B | full − minimal | +1.8 [-0.2, +3.9] | +2.0 [-0.5, +4.2] | +2.4 [+0.5, +4.4] * |
| Llama 3.3 70B | full − schema_only | +3.5 [+0.0, +6.9] | +3.9 [+0.1, +7.4] * | +3.9 [+0.5, +7.1] * |
| Llama 3.3 70B | full − minimal | +8.1 [+4.0, +12.9] * | +9.5 [+5.0, +14.6] * | +8.2 [+4.2, +12.9] * |
| Tutti | full − schema_only | +2.2 [+0.1, +4.2] * | +2.3 [+0.2, +4.3] * | +3.1 [+1.1, +4.9] * |
| Tutti | full − minimal | +4.9 [+2.6, +7.6] * | +5.8 [+3.1, +8.7] * | +5.3 [+3.1, +7.8] * |

### 4. Interazione modello × modalità (exact match, full)

| Modello | Agentica | Orchestrata |
|---|---:|---:|
| GPT-OSS 120B | 18.0% [12.0, 24.8] | 18.5% [12.8, 24.8] |
| Llama 3.3 70B | 15.5% [10.0, 21.8] | 21.8% [15.2, 29.0] |

## Run usate

| Experiment | Domande | Job |
|---|---:|---:|
| nyc_bm25_gptoss_agentic | 100 | `f32fbc45216c4f5c80ac1087071f7e1b` |
| nyc_bm25_gptoss_orchestrated | 100 | `26cd59bda70b4a6492541518c5eed377` |
| nyc_bm25_llama_agentic | 100 | `a32dc95f38d84801878ff4d47495b556` |
| nyc_bm25_llama_orchestrated | 100 | `a072598955384b6cbcc9ab08a17a431c` |
| nyc_dense_gptoss_agentic | 100 | `26ae65e46cbc4336a60c0ab20c2bc087` |
| nyc_dense_gptoss_orchestrated | 100 | `ed35b22deb7f46e1886d97a608df4678` |
| nyc_dense_llama_agentic | 100 | `3ac276f96c35431eb1e2dcddcbd51a6a` |
| nyc_dense_llama_orchestrated | 100 | `3f89d75538174833bccfb04bb507e92d` |
| nyc_hybrid_gptoss_agentic | 100 | `92bd0319fd164b599dce50cb822af708` |
| nyc_hybrid_gptoss_orchestrated | 100 | `b4848f30c425436c98d6df7fb071d2d3` |
| nyc_hybrid_llama_agentic | 100 | `1206c702445248a08a1f531ad3b08ba9` |
| nyc_hybrid_llama_orchestrated | 100 | `8f75bac8d6ed4f0084f9cbf853daf3e8` |
| nyc_rrf_gptoss_agentic | 100 | `983f392c189f4eacbd91d88a03155557` |
| nyc_rrf_gptoss_orchestrated | 100 | `a35adaf62d47493dab841966ec1f11a0` |
| nyc_rrf_llama_agentic | 100 | `a4b042ec3bd44951bbaa1b62cd9fec5e` |
| nyc_rrf_llama_orchestrated | 100 | `1220358954b946c7b6b9cd65e8acc2f9` |
| uk_bm25_gptoss_agentic | 100 | `19babe97520b4a52b7131db3e5a6c83f` |
| uk_bm25_gptoss_orchestrated | 100 | `288c1944f81d43bead81bcba30c2a610` |
| uk_bm25_llama_agentic | 100 | `4d0048034c2141b3adb71b6208214f70` |
| uk_bm25_llama_orchestrated | 100 | `72c780fd797746e3a8a27511ddadba1f` |
| uk_dense_gptoss_agentic | 100 | `55b731f6a2d5463b9a277174b1d53469` |
| uk_dense_gptoss_orchestrated | 100 | `2058636e98ef4455af0f7ff3959434f7` |
| uk_dense_llama_agentic | 100 | `6538dc2f69494751acd9bb0384c7bdf8` |
| uk_dense_llama_orchestrated | 100 | `c05e40218fb34fab8c8a29ea101a454e` |
| uk_hybrid_gptoss_agentic | 100 | `39b922578b9449d7bc2c525455fa5126` |
| uk_hybrid_gptoss_orchestrated | 100 | `0022c0af502645dd84f44f5a18f34ce3` |
| uk_hybrid_llama_agentic | 100 | `0926e174e7194e02b4c58bdaa597aed0` |
| uk_hybrid_llama_orchestrated | 100 | `9ae4613c943247439d79b14b2cca31c3` |
| uk_rrf_gptoss_agentic | 100 | `f883444d514d4e69a78773e6c40c1747` |
| uk_rrf_gptoss_orchestrated | 100 | `a1f4f7e45e564b00b176ac4120686591` |
| uk_rrf_llama_agentic | 100 | `f6408d0bccc14e9eaa022770070e4bbd` |
| uk_rrf_llama_orchestrated | 100 | `54ce6506380047a0a4ca992f290c39a0` |

