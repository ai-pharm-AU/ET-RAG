# Combined results: 3 runs (mean ± SD across runs)

## All 40 questions (operational failures scored 0)

| Agent | SC | MC | SA | LA | Avg |
|---|---|---|---|---|---|
| Full Context | 0.90 ± 0.00 | 0.32 ± 0.02 | 0.80 ± 0.01 | 0.70 ± 0.01 | 0.68 ± 0.00 |
| Cosine RAG | 0.93 ± 0.06 | 0.40 ± 0.02 | 0.83 ± 0.01 | 0.87 ± 0.01 | 0.76 ± 0.01 |
| ET-RAG | 0.90 ± 0.00 | 0.61 ± 0.06 | 0.89 ± 0.00 | 0.86 ± 0.00 | 0.82 ± 0.01 |

## Excluding items not answerable from the corpus (LA10, LA4, LA5, MC3, MC4, SC9)

| Agent | SC | MC | SA | LA | Avg |
|---|---|---|---|---|---|
| Full Context | 1.00 ± 0.00 | 0.38 ± 0.02 | 0.80 ± 0.01 | 0.90 ± 0.00 | 0.77 ± 0.01 |
| Cosine RAG | 0.93 ± 0.06 | 0.45 ± 0.02 | 0.83 ± 0.01 | 0.88 ± 0.01 | 0.77 ± 0.01 |
| ET-RAG | 0.89 ± 0.00 | 0.59 ± 0.07 | 0.89 ± 0.00 | 0.88 ± 0.00 | 0.81 ± 0.02 |

## Multiple choice exact matches per run (of 10)

| Agent | Runs |
|---|---|
| Full Context | 1 / 1 / 1 |
| Cosine RAG | 2 / 2 / 2 |
| ET-RAG | 5 / 4 / 4 |

## Operational failures (all runs)

| Agent | Failures |
|---|---|
| Full Context | 0 |
| Cosine RAG | 0 |
| ET-RAG | 0 |
