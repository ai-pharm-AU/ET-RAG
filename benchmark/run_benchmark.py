"""
ET-RAG manuscript benchmark: 40 questions x 3 agents, scored as in the paper.

Two steps, each usable on its own:

  run    Answer all 40 benchmark questions with the three agents headlessly
         (same code path as the Streamlit app) and write an answers CSV.
  score  Score an answers CSV -- from `run`, or the CSV downloaded from the app
         after pasting the questions -- against the answer keys.

Examples (from the repository root):
  python benchmark/run_benchmark.py run
  python benchmark/run_benchmark.py run --types sc,mc --limit 2
  python benchmark/run_benchmark.py score benchmark/results/answers_<stamp>.csv
  python benchmark/run_benchmark.py run --score          # run, then score

Scoring (manuscript "Evaluation" section):
  single choice    correct / incorrect; the forestry control (Q5) is correct
                   only when the agent reports that it is not covered
  multiple choice  strict partial credit: any incorrect option scores 0,
                   otherwise (correct options selected) / (correct options)
  short / long     GPT-4o-mini judge against the reference answer:
                   0.4 completeness + 0.4 accuracy + 0.2 relevance
Operational failures (API errors) are scored 0 in the primary analysis and
excluded in the secondary matched-question analysis.
"""

import argparse
import difflib
import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
OPEN_ENDED_KEY = BENCHMARK_DIR / "reference_answers_open_ended.json"
QUESTIONS_DOCX = REPO_ROOT / "Questions for alz bot.docx"

AGENTS = (
    ("agent1", "agent1_full_context", "Full Context"),
    ("agent2", "agent2_cosine_rag", "Cosine RAG"),
    ("agent3", "agent3_etrag", "ET-RAG"),
)
TYPE_ORDER = ("single_choice", "multiple_choice", "short_answer", "long_answer")
TYPE_LABELS = {"single_choice": "SC", "multiple_choice": "MC", "short_answer": "SA", "long_answer": "LA"}
JUDGE_MODEL = "gpt-4o-mini"


def load_core():
    """Import test.py as 'etrag_core' (the name 'test' is also a stdlib package)."""
    module = sys.modules.get("etrag_core")
    if module is None:
        spec = importlib.util.spec_from_file_location("etrag_core", REPO_ROOT / "test.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["etrag_core"] = module
        spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Benchmark questions and keys
# ----------------------------------------------------------------------------

def _format_choice_question(item):
    options = "\n".join(f"{letter}. {text}" for letter, text in sorted(item["options"].items()))
    return f"{item['question']}\n{options}"


def load_benchmark(core):
    """Return the 40 questions, each with its type, ID, prompt text, and key."""
    questions = []
    for number, item in enumerate(core.extract_docx_single_choice_questions(QUESTIONS_DOCX), 1):
        key = item["answer_key"].strip().upper().replace(" ", "_")
        questions.append({
            "key_id": f"SC{number}", "question_type": "single_choice",
            "question_text": _format_choice_question(item),
            "key": "NOT_COVERED" if key == "NOT_COVERED" else key,
        })
    for number, item in enumerate(core.extract_docx_multiple_choice_questions(QUESTIONS_DOCX), 1):
        questions.append({
            "key_id": f"MC{number}", "question_type": "multiple_choice",
            "question_text": _format_choice_question(item),
            "key": ", ".join(sorted(item["answer_keys"])),
        })
    open_ended = json.loads(OPEN_ENDED_KEY.read_text(encoding="utf-8"))
    for question_type in ("short_answer", "long_answer"):
        for item in open_ended[question_type]:
            questions.append({
                "key_id": item["id"], "question_type": question_type,
                "question_text": item["question"], "key": item["reference_answer"],
            })
    if len(questions) != 40:
        raise RuntimeError(f"Expected 40 benchmark questions, found {len(questions)}")
    return questions


# ----------------------------------------------------------------------------
# run: answer all questions with the three agents
# ----------------------------------------------------------------------------

def run_agents(args):
    core = load_core()
    questions = load_benchmark(core)
    wanted_types = {t for t in TYPE_ORDER if TYPE_LABELS[t].lower() in args.types.lower().split(",")}
    questions = [q for q in questions if q["question_type"] in wanted_types]
    if args.limit:
        by_type = {}
        for question in questions:
            by_type.setdefault(question["question_type"], []).append(question)
        questions = [q for group in by_type.values() for q in group[:args.limit]]

    output_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"answers_{datetime.now():%Y%m%d_%H%M%S}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output_path.exists():
        done = set(pd.read_csv(output_path)["key_id"])
        print(f"Resuming {output_path.name}: {len(done)} questions already answered")

    pdf_files, paper_metadata, raw_texts, chunks = core._load_evaluation_papers(args.papers_dir)
    vector_store = core._load_or_build_evaluation_index(
        pdf_files, chunks, args.index_cache, rebuild=args.rebuild_index)

    print(f"\nAnswering {len(questions)} questions with 3 agents "
          f"({core.FULL_CONTEXT_MODEL} / gpt-4o-mini / gpt-4o-mini, T={core.TEMPERATURE})")
    for position, question in enumerate(questions, 1):
        if question["key_id"] in done:
            continue
        started = time.time()
        r1, r2, r3 = core.run_three_agents(
            question["question_text"], question["question_type"],
            paper_metadata, raw_texts, vector_store, vector_store)
        row = core.build_result_row(position, question["question_text"], question["question_type"],
                                    r1, r2, r3, elapsed=time.time() - started)
        row = {"key_id": question["key_id"], **row}
        pd.DataFrame([row]).to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
        status = " ".join(f"{label}={'ok' if row[f'{agent}_success'] else 'FAIL'}"
                          for agent, _, label in AGENTS)
        print(f"[{position:2}/{len(questions)}] {question['key_id']:5} {row['response_time_sec']:5.1f}s  {status}")
        time.sleep(args.pause)

    print(f"\nAnswers saved: {output_path}")
    return output_path


# ----------------------------------------------------------------------------
# score: grade an answers CSV
# ----------------------------------------------------------------------------

_NOT_COVERED = re.compile(r"\bnot[_ ]covered\b|not found in the provided", re.IGNORECASE)


def parse_single_choice(answer):
    """Return A-D, NOT_COVERED, or UNPARSED from any agent's single-choice answer."""
    answer = str(answer)
    # A selected letter wins over "not covered", which explanations often use
    # for individual rejected options. The response's opening comes first.
    opening = re.match(r"\W*(?:(?:final\s+)?answer\s*[:\-]?\s*)?\**\(?([A-D])(?![A-Za-z])", answer,
                       re.IGNORECASE)
    if opening:
        return opening.group(1).upper()
    patterns = (
        r"^\W*(?:final\s+)?answer\s*[:\-]\s*\**\(?([A-D])(?![A-Za-z])",
        r"\b(?:correct\s+)?answer\s+is\s*[:\-]?\s*\**\(?([A-D])(?![A-Za-z])",
        r"\b(?:option|choice)\s+\**([A-D])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return "NOT_COVERED" if _NOT_COVERED.search(answer) else "UNPARSED"


def _letters(text):
    return sorted(set(re.findall(r"\b([A-D])\b", text.upper())))


# A letter list such as "A, B, and C" at the start of a string; option text that
# follows it (which may contain letters, e.g. "Vitamin D") is ignored.
_LETTER_LIST = re.compile(
    r"\W*(?:(?:final\s+)?answer\s*[:\-]?\s*)?\**\s*"
    r"([A-D](?:\s*(?:,\s*(?:and|&)?|and|&)\s*[A-D])*)(?![A-Za-z])",
    re.IGNORECASE,
)


_PURE_LETTER_LINE = re.compile(
    r"\W*(?:answer\s*[:\-]?\s*)?\**\s*[A-D](?:\s*(?:,\s*(?:and|&)?|and|&)\s*[A-D])*\s*\**[\s\.]*$",
    re.IGNORECASE,
)
_VERDICT = re.compile(r"\b(not\s+supported|not\s+covered|unsupported|supported)\b", re.IGNORECASE)


def parse_multiple_choice(answer):
    """Return a set of letters, or 'NOT_COVERED'/'NONE'/'UNPARSED', from any agent's answer."""
    answer = str(answer)
    answer_lines = re.findall(r"^\W*(?:final\s+)?answer\s*[:\-]\s*([^\n]+)$", answer,
                              re.IGNORECASE | re.MULTILINE)
    for line in reversed(answer_lines):
        if re.match(r"\W*(?:none|no options?)\b", line, re.IGNORECASE):
            return "NONE"
        letter_list = _LETTER_LIST.match(line)
        if letter_list:
            return _letters(letter_list.group(1))
    # An opening line that is only a letter list ("B, C") is the agent's selection.
    first_line = next((line for line in answer.splitlines() if line.strip()), "")
    if _PURE_LETTER_LINE.match(first_line):
        return _letters(first_line)
    # Otherwise read per-option verdict lines, e.g. "A. Tau - Supported" or
    # "B: NOT SUPPORTED"; the first verdict word on the line decides.
    verdicts = {}
    for line in answer.splitlines():
        option = re.match(r"\W*([A-D])(?![A-Za-z])", line)
        verdict = _VERDICT.search(line)
        if option and verdict and option.group(1).upper() not in verdicts:
            verdicts[option.group(1).upper()] = verdict.group(1).lower() == "supported"
    if len(verdicts) >= 2:
        return sorted(letter for letter, supported in verdicts.items() if supported) or "NONE"
    leading = _LETTER_LIST.match(answer)
    if leading:
        return _letters(leading.group(1))
    return "NOT_COVERED" if _NOT_COVERED.search(answer) else "UNPARSED"


def score_multiple_choice(prediction, key_letters):
    """Strict partial credit, plus whether the selected set matches the key exactly."""
    if not isinstance(prediction, list) or not prediction:
        return 0.0, False
    if any(letter not in key_letters for letter in prediction):
        return 0.0, False
    return len(prediction) / len(key_letters), prediction == key_letters


def judge_open_ended(llm, question, answer, reference, question_type):
    """GPT-4o-mini judge from the manuscript (0.4 completeness + 0.4 accuracy + 0.2 relevance)."""
    prompt = f"""You are an expert evaluator comparing an agent's answer against a benchmark answer.

QUESTION: {question}

BENCHMARK ANSWER:
{reference}

AGENT ANSWER:
{answer}

Score each dimension from 0.0 to 1.0 using this scale:
- 1.0 = Excellent — covers all key points with high accuracy
- 0.8-0.9 = Good — covers most key points, minor omissions
- 0.6-0.7 = Adequate — covers some key points, notable gaps
- 0.4-0.5 = Partial — misses significant content but has some correct info
- 0.2-0.3 = Poor — mostly incomplete or inaccurate
- 0.0-0.1 = Absent — no relevant content or completely wrong

Dimensions:
1. COMPLETENESS: What fraction of the benchmark's key points does the answer cover?
2. ACCURACY: Is the information factually correct compared to the benchmark?
3. RELEVANCE: Does the answer stay on topic and address the question directly?

Return ONLY valid JSON with scores to 2 decimal places:
{{"completeness": 0.00, "accuracy": 0.00, "relevance": 0.00}}"""
    for attempt in range(4):
        try:
            text = llm.invoke(prompt).content
            scores = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
            scores = {name: float(scores[name]) for name in ("completeness", "accuracy", "relevance")}
            scores["overall"] = round(0.4 * scores["completeness"] + 0.4 * scores["accuracy"]
                                      + 0.2 * scores["relevance"], 4)
            return scores
        except Exception as error:
            if attempt == 3:
                raise RuntimeError(f"judge failed: {error}") from error
            time.sleep(5 * (attempt + 1))


def match_rows_to_keys(answers, questions):
    """Attach each answered row to its benchmark question (by key_id, else by text)."""
    by_id = {q["key_id"]: q for q in questions}
    stems = {q["key_id"]: q["question_text"].split("\n")[0].lower() for q in questions}
    matched = []
    for _, row in answers.iterrows():
        key_id = row.get("key_id")
        if not isinstance(key_id, str) or key_id not in by_id:
            text = str(row["question_text"]).lower()
            text_stem = re.split(r"\s[A-D][\.\)]\s", text)[0]
            key_id, best = None, 0.0
            for candidate, stem in stems.items():
                ratio = difflib.SequenceMatcher(None, text_stem, stem).ratio()
                if ratio > best:
                    key_id, best = candidate, ratio
            if best < 0.6:
                print(f"  ! could not match question: {text[:80]}")
                continue
        matched.append((row, by_id[key_id]))
    return matched


def score_answers(answers_path, output_dir=None):
    core = load_core()
    from langchain_openai import ChatOpenAI

    answers_path = Path(answers_path)
    answers = pd.read_csv(answers_path)
    questions = load_benchmark(core)
    judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0.0, max_tokens=100)

    records = []
    for row, question in match_rows_to_keys(answers, questions):
        question_type = question["question_type"]
        for agent, answer_column, label in AGENTS:
            answer = "" if pd.isna(row.get(answer_column)) else str(row[answer_column])
            success = bool(row.get(f"{agent}_success", not answer.startswith("Error")))
            failed = (not success) or answer.startswith("Error") or not answer.strip()
            record = {
                "key_id": question["key_id"], "question_type": question_type, "agent": label,
                "operational_failure": failed, "prediction": "", "key": question["key"] if question_type
                in ("single_choice", "multiple_choice") else "", "score": 0.0, "exact_match": "",
                "completeness": "", "accuracy": "", "relevance": "", "answer": answer,
            }
            if question_type == "single_choice":
                prediction = "ERROR" if failed else parse_single_choice(answer)
                record.update(prediction=prediction, score=float(prediction == question["key"]))
            elif question_type == "multiple_choice":
                prediction = "ERROR" if failed else parse_multiple_choice(answer)
                key_letters = _letters(question["key"])
                score, exact = score_multiple_choice(prediction, key_letters)
                record.update(prediction=", ".join(prediction) if isinstance(prediction, list) else prediction,
                              score=score, exact_match=exact)
            elif not failed:
                # The judge scores every delivered answer; an answer that wrongly
                # claims "not covered" earns low completeness from the judge.
                scores = judge_open_ended(judge, question["question_text"], answer, question["key"], question_type)
                record.update(score=scores["overall"], completeness=scores["completeness"],
                              accuracy=scores["accuracy"], relevance=scores["relevance"])
            records.append(record)
        print(f"  scored {question['key_id']}")

    scored = pd.DataFrame(records)
    output_dir = Path(output_dir) if output_dir else answers_path.parent
    stem = answers_path.stem.replace("answers_", "")
    scored_path = output_dir / f"scored_{stem}.csv"
    scored.to_csv(scored_path, index=False)
    summary = summarize(scored)
    summary_path = output_dir / f"summary_{stem}.md"
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Scored answers: {scored_path}\nSummary:        {summary_path}")
    return scored_path


def _table(rows, header):
    """Render [(label, [cell, ...]), ...] as a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + label + " | " + " | ".join(values) + " |" for label, values in rows]
    return "\n".join(lines)


def summarize(scored):
    """Markdown tables matching the manuscript's Tables 2, 5-style counts, and 7."""
    agents = [label for _, _, label in AGENTS]
    present = [t for t in TYPE_ORDER if t in set(scored["question_type"])]
    means = scored.groupby(["agent", "question_type"])["score"].mean()

    rows = []
    for agent in agents:
        values = [f"{means.get((agent, t), float('nan')):.2f}" for t in present]
        category_means = [means.get((agent, t)) for t in present]
        values.append(f"{sum(category_means) / len(category_means):.2f}")
        rows.append((agent, values))
    out = ["## Primary analysis (operational failures scored 0)", "",
           _table(rows, ["Agent"] + [TYPE_LABELS[t] for t in present] + ["Avg"]), ""]

    # Secondary: open-ended questions where no agent had an operational failure.
    failed_ids = set(scored.loc[scored["operational_failure"], "key_id"])
    open_types = [t for t in ("short_answer", "long_answer") if t in present]
    if failed_ids and open_types:
        matched = scored[~scored["key_id"].isin(failed_ids)]
        matched_means = matched.groupby(["agent", "question_type"])["score"].mean()
        rows = [(agent, [f"{matched_means.get((agent, t), float('nan')):.2f}" for t in open_types])
                for agent in agents]
        out += ["## Secondary analysis (excluding questions with any operational failure)", "",
                _table(rows, ["Agent"] + [TYPE_LABELS[t] for t in open_types]), ""]

    if open_types:
        judged = scored[scored["question_type"].isin(open_types)].copy()
        for column in ("completeness", "accuracy", "relevance"):
            judged[column] = pd.to_numeric(judged[column], errors="coerce").fillna(0.0)
        for question_type in open_types:
            subset = judged[judged["question_type"] == question_type]
            dims = subset.groupby("agent")[["completeness", "accuracy", "relevance", "score"]].mean()
            rows = [(metric.title() if metric != "score" else "Overall",
                     [f"{dims.loc[agent, metric]:.2f}" for agent in agents])
                    for metric in ("completeness", "accuracy", "relevance", "score")]
            out += [f"## {TYPE_LABELS[question_type]} dimension scores", "",
                    _table(rows, ["Metric"] + agents), ""]

    if "multiple_choice" in present:
        mc = scored[scored["question_type"] == "multiple_choice"]
        rows = [(agent, [f"{int((mc[mc['agent'] == agent]['exact_match'] == True).sum())}/"
                         f"{int((mc['agent'] == agent).sum())}"]) for agent in agents]
        out += ["## Multiple choice exact matches", "", _table(rows, ["Agent", "Exact"]), ""]

    unparsed = scored[scored["prediction"].isin(["UNPARSED"])]
    failures = scored.groupby("agent")["operational_failure"].sum()
    out += ["## Operational failures and unparsed choices", "",
            _table([(agent, [str(int(failures.get(agent, 0))),
                             str(int((unparsed["agent"] == agent).sum()))]) for agent in agents],
                   ["Agent", "API failures", "Unparsed"]), ""]
    return "\n".join(out)


def rekey_runs(scored_paths, overrides_path):
    """Re-score only the overridden items of each scored run against a revised key.

    Choice items are re-scored from the stored prediction (no API calls);
    open-ended items are re-judged against the revised reference answer. Every
    other row is copied unchanged. Writes <name>_corpuskey.csv next to each input.
    """
    from langchain_openai import ChatOpenAI

    core = load_core()
    overrides = json.loads(Path(overrides_path).read_text(encoding="utf-8"))["items"]
    questions = {q["key_id"]: q for q in load_benchmark(core)}
    judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0.0, max_tokens=100)
    outputs = []
    for path in map(Path, scored_paths):
        scored = pd.read_csv(path, dtype={"key": str, "prediction": str, "exact_match": object})
        for index, row in scored[scored["key_id"].isin(overrides)].iterrows():
            item = overrides[row["key_id"]]
            if row["operational_failure"]:
                continue
            if item["question_type"] == "single_choice":
                scored.at[index, "key"] = item["corpus_key"]
                scored.at[index, "score"] = float(row["prediction"] == item["corpus_key"])
            elif item["question_type"] == "multiple_choice":
                key_letters = _letters(item["corpus_key"])
                prediction = str(row["prediction"])
                if not key_letters:  # the papers support no option
                    correct = prediction in ("NONE", "NOT_COVERED")
                    score, exact = float(correct), correct
                elif prediction in ("NONE", "NOT_COVERED", "UNPARSED", "ERROR"):
                    score, exact = 0.0, False
                else:
                    score, exact = score_multiple_choice(_letters(prediction), key_letters)
                scored.at[index, "key"] = item["corpus_key"]
                scored.at[index, "score"] = score
                scored.at[index, "exact_match"] = exact
            else:
                result = judge_open_ended(judge, questions[row["key_id"]]["question_text"],
                                          str(row["answer"]), item["corpus_key"], item["question_type"])
                for column in ("completeness", "accuracy", "relevance"):
                    scored.at[index, column] = result[column]
                scored.at[index, "score"] = result["overall"]
        output = path.with_name(f"{path.stem}_corpuskey.csv")
        scored.to_csv(output, index=False)
        outputs.append(output)
        print(f"  re-keyed {path.name} -> {output.name}")
    return outputs


def combine_runs(scored_paths, exclude=(), output=None):
    """Mean +/- SD across repeated runs of each agent's category scores."""
    frames = []
    for run_number, path in enumerate(scored_paths, 1):
        frame = pd.read_csv(path)
        frames.append(frame.assign(run=run_number))
    scored = pd.concat(frames, ignore_index=True)
    agents = [label for _, _, label in AGENTS]
    present = [t for t in TYPE_ORDER if t in set(scored["question_type"])]

    def table(data, title):
        # Category score per run, then mean and SD of those run-level scores.
        per_run = data.groupby(["run", "agent", "question_type"])["score"].mean().unstack()
        per_run = per_run.reindex(columns=present)
        per_run["Avg"] = per_run.mean(axis=1)
        stats = per_run.groupby(level="agent").agg(["mean", "std"])
        columns = present + ["Avg"]
        rows = [(agent, [f"{stats.loc[agent, (c, 'mean')]:.2f} ± {stats.loc[agent, (c, 'std')]:.2f}"
                         for c in columns]) for agent in agents]
        return [f"## {title}", "", _table(rows, ["Agent"] + [TYPE_LABELS.get(c, c) for c in columns]), ""]

    out = [f"# Combined results: {len(scored_paths)} runs (mean ± SD across runs)", ""]
    out += table(scored, "All 40 questions (operational failures scored 0)")
    if exclude:
        kept = scored[~scored["key_id"].isin(exclude)]
        out += table(kept, f"Excluding items not answerable from the corpus ({', '.join(sorted(exclude))})")

    mc = scored[scored["question_type"] == "multiple_choice"]
    if not mc.empty:
        exact = mc.assign(exact=mc["exact_match"].astype(str) == "True").groupby(["agent", "run"])["exact"].sum()
        rows = [(agent, [" / ".join(str(int(v)) for v in exact.loc[agent].tolist())]) for agent in agents]
        out += ["## Multiple choice exact matches per run (of 10)", "", _table(rows, ["Agent", "Runs"]), ""]

    failures = scored.groupby("agent")["operational_failure"].sum()
    out += ["## Operational failures (all runs)", "",
            _table([(agent, [str(int(failures.get(agent, 0)))]) for agent in agents], ["Agent", "Failures"]), ""]

    per_question = scored.pivot_table(index="key_id", columns="agent", values="score", aggfunc="mean")
    per_question = per_question.reindex(columns=agents)
    output = Path(output) if output else Path(scored_paths[0]).parent / "combined_summary.md"
    per_question.round(3).to_csv(output.with_suffix(".per_question.csv"))
    output.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"Combined summary: {output}\nPer-question means: {output.with_suffix('.per_question.csv')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="answer the benchmark with all three agents")
    run_parser.add_argument("--types", default="sc,mc,sa,la", help="comma list of sc,mc,sa,la")
    run_parser.add_argument("--limit", type=int, default=None, help="first N questions of each type")
    run_parser.add_argument("--output", default=None, help="answers CSV (an existing file is resumed)")
    run_parser.add_argument("--papers-dir", default=str(REPO_ROOT / "review papers"))
    run_parser.add_argument("--index-cache", default=str(REPO_ROOT / "faiss_index_etrag_evaluation"))
    run_parser.add_argument("--rebuild-index", action="store_true")
    run_parser.add_argument("--pause", type=float, default=2.0, help="seconds between questions")
    run_parser.add_argument("--score", action="store_true", help="score the answers when done")

    score_parser = commands.add_parser("score", help="score an answers CSV (from run or the app)")
    score_parser.add_argument("answers_csv")
    score_parser.add_argument("--output-dir", default=None)

    combine_parser = commands.add_parser("combine", help="mean ± SD across several scored runs")
    combine_parser.add_argument("scored_csvs", nargs="+")
    combine_parser.add_argument("--exclude", default="", help="comma list of key IDs, e.g. SC9,MC3")
    combine_parser.add_argument("--output", default=None)

    rekey_parser = commands.add_parser("rekey", help="re-score runs with corpus-grounded key overrides")
    rekey_parser.add_argument("scored_csvs", nargs="+")
    rekey_parser.add_argument("--overrides", default=str(BENCHMARK_DIR / "corpus_grounded_key.json"))

    args = parser.parse_args()
    if args.command == "rekey":
        rekey_runs(args.scored_csvs, args.overrides)
        return
    if args.command == "combine":
        exclude = [key for key in args.exclude.split(",") if key]
        combine_runs(args.scored_csvs, exclude, args.output)
        return
    if args.command == "run":
        answers_path = run_agents(args)
        if args.score:
            score_answers(answers_path)
    else:
        score_answers(args.answers_csv, args.output_dir)


if __name__ == "__main__":
    main()
