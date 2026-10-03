# ET-RAG benchmark (40 questions × 3 agents)

Reproduces the manuscript's main evaluation (Tables 2–4, 6–8): the Full Context,
Cosine RAG, and ET-RAG agents answer the 40 benchmark questions from the 10
review papers in `review papers/`, and the answers are scored as described in
the Evaluation section.

The agents are the ones in `test.py`, which the Streamlit app
(`INTEGRATED_MULTI_AGENT_COMPLETE.py`) and the ablation study
(`ablation/run_ablation.py`) also run.

| Agent | Model | Context |
|---|---|---|
| Full Context | `gemini-3.8-flash` | complete extracted text of all papers, no retrieval |
| Cosine RAG | `gpt-4o-mini` | top-15 chunks by cosine similarity (query + synonym expansion) |
| ET-RAG | `gpt-4o-mini` | top-25 chunks by 0.5·similarity + 0.3·study-design + 0.2·recency, plus top-3 abstracts and one passage per answer option (ablation configuration A3) |

All agents use temperature 0.1. Chunks are 2,000 characters with 400 overlap,
embedded with `text-embedding-3-small`.

## Run

```bash
# from the repository root; needs OPENAI_API_KEY and GOOGLE_API_KEY in .env
python benchmark/run_benchmark.py run --score             # all 40 questions, then score
python benchmark/run_benchmark.py run --types sc --limit 2  # a quick subset
python benchmark/run_benchmark.py score path/to/answers.csv # score an existing CSV
```

`run` appends one row per question to the answers CSV, so rerunning with the
same `--output` resumes an interrupted run.

The app's batch mode produces the same answers: upload the papers, paste the
questions, and use **Download Full Results as CSV** when the batch finishes.
`score` accepts that CSV and matches each row to its benchmark question by text.

## Scoring

- **Single choice:** correct or incorrect. The forestry control (SC5) is
  correct only when the agent says the topic is not covered.
- **Multiple choice:** strict partial credit. Selecting any incorrect option
  scores 0; otherwise the score is (correct options selected) / (correct
  options). Exact-set matches are reported separately.
- **Short and long answer:** a GPT-4o-mini judge scores completeness, accuracy,
  and relevance against the reference answer; overall = 0.4·C + 0.4·A + 0.2·R.
- API errors are **operational failures**. They score 0 in the primary
  analysis and are excluded in the secondary matched-question analysis.

## Answer keys

- Single and multiple choice: `Questions for alz bot.docx`
- Short and long answer: `reference_answers_open_ended.json`, the full-length
  reference answers (also readable in `ANSWER_KEY_SHORT_LONG.md`)

## Outputs (`results/`)

- `answers_<name>.csv`: every agent's raw answer, success flag, confidence, sources
- `scored_<name>.csv`: per-question, per-agent prediction and score
- `summary_<name>.md`: the tables above, by agent and question type
