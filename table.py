r"""Create tab-delimited tables from every ET-RAG round-results export.

The generated table contains exactly five columns:
1. question_index
2. correct_answer
3. etrag_answer
4. confidence
5. execution_time_sec

Run:
    E:\anaconda3_64\envs\py312\python.exe table.py

The default command discovers every file matching:
    etrag_single_choice_results_round_*.csv

and creates the corresponding:
    etrag_single_choice_table_round_*.tsv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
ROUND_RESULTS_PATTERN = "etrag_single_choice_results_round_*.csv"
DEFAULT_QUESTION_STATS_TSV = WORKSPACE / "etrag_single_choice_question_statistics.tsv"
DEFAULT_QUESTION_SUMMARY_TSV = WORKSPACE / "etrag_single_choice_question_summary.tsv"


def _format_correct_answer(row):
    """Combine an answer letter and its text into one readable table value."""
    answer_key = str(row["correct_answer_key"]).strip()
    answer_text = str(row["correct_answer_text"]).strip()

    # Standard choices are shown as, for example, "B. Amyloid-beta (Aβ)".
    if answer_key.upper() in {"A", "B", "C", "D"}:
        return f"{answer_key.upper()}. {answer_text}"

    # Control questions use NOT_COVERED rather than an A-D option.
    return answer_text if answer_text and answer_text.lower() != "nan" else answer_key


def _default_output_path(input_csv):
    """Derive the matching round-table TSV name from a round-results CSV."""
    input_csv = Path(input_csv)
    output_name = input_csv.name.replace(
        "etrag_single_choice_results_",
        "etrag_single_choice_table_",
        1,
    )
    return input_csv.with_name(Path(output_name).with_suffix(".tsv").name)


def create_etrag_results_table(input_csv, output_tsv=None):
    """Parse the ET-RAG export, print the table, and save it as tab-delimited text."""
    input_csv = Path(input_csv).expanduser().resolve()
    output_tsv = (
        Path(output_tsv).expanduser().resolve()
        if output_tsv
        else _default_output_path(input_csv).resolve()
    )

    if not input_csv.is_file():
        raise FileNotFoundError(
            f"ET-RAG results file was not found: {input_csv}\n"
            "Run test.py first to generate an ET-RAG round-results export."
        )

    results = pd.read_csv(input_csv)
    required_columns = {
        "question_number",
        "correct_answer_key",
        "correct_answer_text",
        "etrag_answer",
        "etrag_confidence",
        "response_time_sec",
    }
    missing_columns = sorted(required_columns.difference(results.columns))
    if missing_columns:
        raise ValueError(
            "The ET-RAG results file is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Construct the requested table in the exact requested column order.
    table = pd.DataFrame({
        "question_index": pd.to_numeric(results["question_number"], errors="raise").astype(int),
        "correct_answer": results.apply(_format_correct_answer, axis=1),
        "etrag_answer": results["etrag_answer"].fillna("").astype(str),
        "confidence": pd.to_numeric(results["etrag_confidence"], errors="raise"),
        "execution_time_sec": pd.to_numeric(results["response_time_sec"], errors="raise"),
    })

    # Ensure the rows remain ordered by their question index.
    table = table.sort_values("question_index").reset_index(drop=True)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8-sig")

    print("=" * 78)
    print("ET-RAG SINGLE-CHOICE RESULTS TABLE")
    print("=" * 78)
    print(
        table.to_string(
            index=False,
            formatters={
                "confidence": lambda value: f"{value:.2%}",
                "execution_time_sec": lambda value: f"{value:.2f}",
            },
        )
    )
    print(f"\nRows exported: {len(table)}")
    print(f"Saved tab-delimited table: {output_tsv}")
    return table


def _round_number(path):
    """Return the numeric round for natural sorting (1, 2, ..., 10)."""
    match = re.search(r"_round_(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else float("inf")


def create_question_statistics(
    input_files,
    output_tsv=DEFAULT_QUESTION_STATS_TSV,
    summary_output_tsv=DEFAULT_QUESTION_SUMMARY_TSV,
):
    """Calculate per-question confidence/time means and sample deviations."""
    output_tsv = Path(output_tsv).expanduser().resolve()
    frames = []

    for input_file in input_files:
        frame = pd.read_csv(input_file)
        required_columns = {
            "question_number",
            "is_correct",
            "etrag_confidence",
            "response_time_sec",
        }
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            raise ValueError(
                f"{Path(input_file).name} is missing required columns: "
                + ", ".join(missing_columns)
            )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["question_number"] = pd.to_numeric(
        combined["question_number"], errors="raise"
    ).astype(int)
    combined["etrag_confidence"] = pd.to_numeric(
        combined["etrag_confidence"], errors="raise"
    )
    combined["response_time_sec"] = pd.to_numeric(
        combined["response_time_sec"], errors="raise"
    )
    # Normalize CSV boolean values to 1 (correct) and 0 (incorrect).
    correctness_map = {
        True: 1.0,
        False: 0.0,
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
    }
    combined["correctness"] = combined["is_correct"].map(
        lambda value: correctness_map.get(
            value if isinstance(value, bool) else str(value).strip().lower()
        )
    )
    if combined["correctness"].isna().any():
        raise ValueError("The is_correct column contains unrecognized values.")

    # pandas std() uses the sample standard deviation (N-1), appropriate for
    # summarizing repeated experimental rounds.
    statistics = (
        combined.groupby("question_number", as_index=False)
        .agg(
            correctness_mean=("correctness", "mean"),
            correctness_std=("correctness", "std"),
            confidence_mean=("etrag_confidence", "mean"),
            confidence_std=("etrag_confidence", "std"),
            execution_time_mean_sec=("response_time_sec", "mean"),
            execution_time_std_sec=("response_time_sec", "std"),
        )
        .rename(columns={"question_number": "question_index"})
        .sort_values("question_index")
        .reset_index(drop=True)
    )

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8-sig")

    # Create a compact presentation table with mean and SD in the same cell.
    compact_summary = pd.DataFrame({
        "question_index": statistics["question_index"],
        "correctness": statistics.apply(
            lambda row: (
                f"{row['correctness_mean']:.1%} ± "
                f"{row['correctness_std']:.1%}"
            ),
            axis=1,
        ),
        "confidence": statistics.apply(
            lambda row: (
                f"{row['confidence_mean']:.1%} ± "
                f"{row['confidence_std']:.1%}"
            ),
            axis=1,
        ),
        "execution_time_sec": statistics.apply(
            lambda row: (
                f"{row['execution_time_mean_sec']:.1f} ± "
                f"{row['execution_time_std_sec']:.1f}"
            ),
            axis=1,
        ),
    })
    summary_output_tsv = Path(summary_output_tsv).expanduser().resolve()
    summary_output_tsv.parent.mkdir(parents=True, exist_ok=True)
    compact_summary.to_csv(
        summary_output_tsv,
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 78)
    print("PER-QUESTION STATISTICS ACROSS ALL ROUNDS")
    print("=" * 78)
    print(
        statistics.to_string(
            index=False,
            formatters={
                "correctness_mean": lambda value: f"{value:.3%}",
                "correctness_std": lambda value: f"{value:.3%}",
                "confidence_mean": lambda value: f"{value:.3%}",
                "confidence_std": lambda value: f"{value:.3%}",
                "execution_time_mean_sec": lambda value: f"{value:.3f}",
                "execution_time_std_sec": lambda value: f"{value:.3f}",
            },
        )
    )
    print(f"\nRounds included: {len(input_files)}")
    print(f"Saved statistics: {output_tsv}")
    print("\nCOMPACT MEAN ± SD SUMMARY")
    print(compact_summary.to_string(index=False))
    print(f"\nSaved compact summary: {summary_output_tsv}")
    return statistics


def create_all_round_tables(
    directory=WORKSPACE,
    statistics_output=DEFAULT_QUESTION_STATS_TSV,
    summary_output=DEFAULT_QUESTION_SUMMARY_TSV,
):
    """Discover every round-results CSV and create a matching TSV table."""
    directory = Path(directory).expanduser().resolve()
    input_files = sorted(directory.glob(ROUND_RESULTS_PATTERN), key=_round_number)
    if not input_files:
        raise FileNotFoundError(
            f"No files matching {ROUND_RESULTS_PATTERN!r} were found in: {directory}"
        )

    generated_files = []
    for input_csv in input_files:
        output_tsv = _default_output_path(input_csv)
        print(f"\nProcessing: {input_csv.name}")
        create_etrag_results_table(input_csv, output_tsv)
        generated_files.append(output_tsv.resolve())

    print("\n" + "=" * 78)
    print(f"Completed {len(generated_files)} ET-RAG round table(s):")
    for generated_file in generated_files:
        print(f"- {generated_file}")
    create_question_statistics(input_files, statistics_output, summary_output)
    return generated_files


def parse_arguments():
    """Support all-round batch mode or an optional single-file conversion."""
    parser = argparse.ArgumentParser(
        description="Create a five-column table from ET-RAG results."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="optional single ET-RAG results CSV; omit to process every round",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional TSV path used with --input; otherwise inferred from its round",
    )
    parser.add_argument(
        "--directory",
        default=str(WORKSPACE),
        help="directory searched for all round-results CSV files",
    )
    parser.add_argument(
        "--stats-output",
        default=str(DEFAULT_QUESTION_STATS_TSV),
        help="tab-delimited output for per-question cross-round statistics",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_QUESTION_SUMMARY_TSV),
        help="compact TSV with mean ± SD correctness, confidence, and time",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        if arguments.output and not arguments.input:
            raise ValueError("--output can only be used together with --input")
        if arguments.input:
            create_etrag_results_table(arguments.input, arguments.output)
        else:
            create_all_round_tables(
                arguments.directory,
                arguments.stats_output,
                arguments.summary_output,
            )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Could not create ET-RAG table: {error}")
