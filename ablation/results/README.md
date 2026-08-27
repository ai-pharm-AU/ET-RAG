# Generated ablation results

Running `ablation/run_ablation.py` with `OPENAI_API_KEY` configured evaluates
the DOCX multiple-choice section and writes the A0–A3 per-round TSV files, one
four-column question-summary TSV for every configuration, the combined TSV,
and the configuration summary to this directory. Result files are not
pre-populated because they must contain real GPT-4o-mini responses rather than
mocked or fabricated answers.

Files whose names do not contain `multiple_choice` are earlier single-choice
outputs. The updated runner uses distinct filenames and preserves those
historical results.

Multiple-choice `correctness` uses strict partial credit: any unsupported
selection scores zero; otherwise the score is the fraction of correct options
selected. `UNPARSED`, `NOT_COVERED`, `ERROR`, and empty predictions score zero.
