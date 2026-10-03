## Primary analysis (operational failures scored 0)

| Agent | SC | MC | SA | LA | Avg |
|---|---|---|---|---|---|
| Full Context | 0.90 | 0.31 | 0.80 | 0.71 | 0.68 |
| Cosine RAG | 0.90 | 0.39 | 0.83 | 0.87 | 0.75 |
| ET-RAG | 0.90 | 0.57 | 0.89 | 0.86 | 0.81 |

## SA dimension scores

| Metric | Full Context | Cosine RAG | ET-RAG |
|---|---|---|---|
| Completeness | 0.76 | 0.75 | 0.84 |
| Accuracy | 0.81 | 0.84 | 0.90 |
| Relevance | 0.88 | 0.97 | 0.98 |
| Overall | 0.80 | 0.83 | 0.89 |

## LA dimension scores

| Metric | Full Context | Cosine RAG | ET-RAG |
|---|---|---|---|
| Completeness | 0.67 | 0.82 | 0.80 |
| Accuracy | 0.71 | 0.88 | 0.87 |
| Relevance | 0.78 | 0.97 | 0.97 |
| Overall | 0.71 | 0.87 | 0.86 |

## Multiple choice exact matches

| Agent | Exact |
|---|---|
| Full Context | 1/10 |
| Cosine RAG | 2/10 |
| ET-RAG | 4/10 |

## Operational failures and unparsed choices

| Agent | API failures | Unparsed |
|---|---|---|
| Full Context | 0 | 0 |
| Cosine RAG | 0 | 0 |
| ET-RAG | 0 | 0 |
