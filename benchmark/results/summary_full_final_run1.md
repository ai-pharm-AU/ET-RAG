## Primary analysis (operational failures scored 0)

| Agent | SC | MC | SA | LA | Avg |
|---|---|---|---|---|---|
| Full Context | 0.90 | 0.32 | 0.79 | 0.69 | 0.68 |
| Cosine RAG | 1.00 | 0.38 | 0.83 | 0.85 | 0.77 |
| ET-RAG | 0.90 | 0.68 | 0.89 | 0.86 | 0.83 |

## SA dimension scores

| Metric | Full Context | Cosine RAG | ET-RAG |
|---|---|---|---|
| Completeness | 0.76 | 0.74 | 0.83 |
| Accuracy | 0.79 | 0.84 | 0.90 |
| Relevance | 0.87 | 0.97 | 0.99 |
| Overall | 0.79 | 0.83 | 0.89 |

## LA dimension scores

| Metric | Full Context | Cosine RAG | ET-RAG |
|---|---|---|---|
| Completeness | 0.65 | 0.80 | 0.81 |
| Accuracy | 0.70 | 0.86 | 0.87 |
| Relevance | 0.78 | 0.94 | 0.96 |
| Overall | 0.69 | 0.85 | 0.86 |

## Multiple choice exact matches

| Agent | Exact |
|---|---|
| Full Context | 1/10 |
| Cosine RAG | 2/10 |
| ET-RAG | 5/10 |

## Operational failures and unparsed choices

| Agent | API failures | Unparsed |
|---|---|---|
| Full Context | 0 | 0 |
| Cosine RAG | 0 | 0 |
| ET-RAG | 0 | 0 |
