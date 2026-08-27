# A0–A3 ET-RAG Ablation

This folder contains controlled multiple-choice and optional single-choice
ablations built entirely on one configurable ET-RAG agent. Multiple-choice
remains the default. A0 is that ET-RAG pipeline reduced to cosine-only
retrieval, and exactly one ET-RAG component is enabled at each later step. The
study does not switch between separately implemented agents.

| Configuration | Retrieval score | Hybrid context | Per-option retrieval | Quality check |
|---|---|---:|---:|---:|
| A0 | 1.0 Similarity | No | No | No |
| A1 | 0.7 Similarity + 0.3 Evidence | No | No | No |
| A2 | 0.5 Similarity + 0.3 Evidence + 0.2 Time | No | No | No |
| A3 | Same as A2 | Yes | No | No |

All four configurations directly call `agent_etrag` in `test.py`; the ablation
runner does not contain a second ET-RAG implementation. It supplies only the
controlled retrieval weights and component switches. Every condition uses the
same PDFs, chunks, OpenAI embedding model, GPT-4o-mini model, temperature
(`0.0`), top-chunk limit, and ET-RAG answer prompt. A1 adds an evidence-focused
candidate query and chunk-level study-design signal. A2 retains A1 and adds a
query anchored to the newest years present in the corpus plus graduated recency
scoring. A3 retains A2 and adds focused abstracts from top-ranked sources plus
one terminology passage per option. A3 also applies a non-regression evidence
gate: supplementary context may resolve an option only through direct evidence
with the same role, direction, and polarity; topical similarity and multi-hop
inference cannot change the primary chunk-based decision. Per-option semantic
retrieval and the second-pass quality check are not part of this study.

At startup, the script verifies that A1–A3 introduce exactly one component per
step and that A2/A3's `0.5/0.3/0.2` retrieval weights exactly match
`ETRAG_WEIGHTS` in `test.py`.

## Validate without API calls

```powershell
E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py --validate-only
```

## Run all configurations

Configure `OPENAI_API_KEY` in the same shell, then run:

```powershell
$env:OPENAI_API_KEY = "your-key"
E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py
```

Run the same A0-A3 configurations on the DOCX `Single Choice Questions`
section without changing or overwriting multiple-choice outputs:

```powershell
E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py --question-type single-choice
```

Validate that mode without API calls:

```powershell
E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py --question-type single-choice --validate-only
```

The 10 questions are parsed from the DOCX `Multiple Choice Questions` section.
Correctness uses strict partial credit: selecting any unsupported option gives
zero; otherwise, the score is the proportion of correct options selected.
`NONE_SELECTED`, `UNPARSED`, `NOT_COVERED`, `ERROR`, and empty predictions
score zero. Exact matches remain available separately in the raw `is_correct`
audit column.

The default is five repeated rounds (`--rounds 5`) so every configuration has
five observations per question for mean ± sample SD. This performs 200 answer
generation calls. Use `--rounds 1` for a quick single trial; its SD values are
reported as zero. The existing
`faiss_index_etrag_evaluation` cache is reused when its manifest matches the PDF
corpus.

Recalculate strict partial-credit scores and all reports from existing A0–A3
raw TSV files without making API calls:

```powershell
E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py --refresh-reports
```

For existing single-choice TSVs, add `--question-type single-choice` to that
command. Single-choice correctness is exact match, including `NOT_COVERED`.

## Outputs

- `configurations.tsv`: the fixed A0–A3 configuration matrix.
- `results/A0_*_multiple_choice_results.tsv` through
  `results/A3_*_multiple_choice_results.tsv`: detailed tab-delimited
  per-round, per-question results.
- `results/A0_*_multiple_choice_question_summary.tsv` through
  `results/A3_*_multiple_choice_question_summary.tsv`: one tab-delimited presentation table per
  configuration with `question_index`, `correctness`, `confidence`, and
  `execution_time`. Values use mean ± sample SD with one-decimal precision.
- `results/ablation_multiple_choice_all_results.tsv`: all configurations combined.
- `results/ablation_multiple_choice_summary.tsv`: exact-match accuracy, strict
  partial-credit correctness, confidence, and execution time by configuration.
- `results/A0_*_single_choice_results.tsv` through
  `results/A3_*_single_choice_results.tsv`: single-choice raw results using
  exact-match correctness.
- `results/A0_*_single_choice_question_summary.tsv` through
  `results/A3_*_single_choice_question_summary.tsv`: single-choice mean ± SD
  tables.
- `results/ablation_single_choice_all_results.tsv` and
  `results/ablation_single_choice_summary.tsv`: combined and overall
  single-choice reports.

Progress is written after every completed model response. Existing result files
are protected; use `--overwrite` only when intentionally replacing a run.

## Evidence-weighting note

The current corpus directory consists of review papers and the shared loader
assigns `study_type="review"` to every source. Earlier code therefore gave all
chunks the same evidence score. The revised A1 retains that paper-level type but
also detects whether an individual retrieved chunk explicitly discusses a
meta-analysis, systematic review, randomized trial, cohort, or lower-level
design. This is a ranking signal for evidence discussed by the review; it does
not relabel a review article as a primary trial.

The added components are nested by design, but a monotonic score increase is an
empirical hypothesis rather than a guaranteed property. Confirm it across new
rounds and, ideally, a held-out question set rather than tuning to one result
file alone.
