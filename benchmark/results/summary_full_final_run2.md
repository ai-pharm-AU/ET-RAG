## Primary analysis (operational failures scored 0)

| Agent | SC | MC | SA | LA | Avg |
|---|---|---|---|---|---|
| Full Context | 0.90 | 0.34 | 0.80 | 0.69 | 0.68 |
| Cosine RAG | 0.90 | 0.42 | 0.84 | 0.88 | 0.76 |
| ET-RAG | 0.90 | 0.57 | 0.90 | 0.87 | 0.81 |

## SA dimension scores

| Metric | Full Context | Cosine RAG | ET-RAG |
|---|---|---|---|
| Completeness | 0.76 | 0.75 | 0.85 |
| Accuracy | 0.80 | 0.86 | 0.90 |
| Relevance | 0.88 | 0.97 | 0.99 |
| Overall | 0.80 | 0.84 | 0.90 |

## LA dimension scores

| Metric | Full Context | Cosine RAG | ET-RAG |
|---|---|---|---|
| Completeness | 0.65 | 0.83 | 0.81 |
| Accuracy | 0.70 | 0.88 | 0.88 |
| Relevance | 0.78 | 0.97 | 0.97 |
| Overall | 0.69 | 0.88 | 0.87 |

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
