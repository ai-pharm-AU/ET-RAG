r"""Run the A0-A3 ET-RAG ablation on the 10 multiple-choice questions.

Live run (requires OPENAI_API_KEY):
    E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py

Single-choice run (the existing multiple-choice mode remains the default):
    E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py --question-type single-choice

Configuration/corpus validation without API calls:
    E:\anaconda3_64\envs\py312\python.exe ablation\run_ablation.py --validate-only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Reuse the tested question parser, PDF loader, metadata weights, and FAISS
# cache from test.py while keeping every ablation output in this folder.
ABLATION_DIR = Path(__file__).resolve().parent
WORKSPACE = ABLATION_DIR.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import test as app  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


MODEL_NAME = "gpt-4o-mini"
AGENT_FAMILY = "ET-RAG"
DEFAULT_ROUNDS = 5
DEFAULT_DOCX = app.DEFAULT_QUESTIONS_DOCX
DEFAULT_PAPERS_DIR = app.DEFAULT_REVIEW_PAPERS_DIR
DEFAULT_INDEX_CACHE = app.DEFAULT_ETRAG_EVAL_CACHE
DEFAULT_OUTPUT_DIR = ABLATION_DIR / "results"
CONFIGURATION_FILE = ABLATION_DIR / "configurations.tsv"


@dataclass(frozen=True)
class AblationConfiguration:
    """One controlled ablation configuration."""

    config_id: str
    name: str
    similarity_weight: float
    evidence_weight: float
    temporal_weight: float
    hybrid_context: bool
    per_option_retrieval: bool
    quality_check: bool
    purpose: str

    @property
    def retrieval_score_label(self) -> str:
        parts = []
        if self.similarity_weight:
            parts.append(f"{self.similarity_weight:.1f} Sim")
        if self.evidence_weight:
            parts.append(f"{self.evidence_weight:.1f} Evidence")
        if self.temporal_weight:
            parts.append(f"{self.temporal_weight:.1f} Time")
        return " + ".join(parts)

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")

    @property
    def enabled_components(self) -> str:
        """Describe the ET-RAG components enabled for this configuration."""
        components = ["cosine retrieval"]
        if self.evidence_weight > 0:
            components.append("evidence quality")
        if self.temporal_weight > 0:
            components.append("temporal recency")
        if self.hybrid_context:
            components.append("hybrid paper context")
        if self.per_option_retrieval:
            components.append("per-option retrieval")
        if self.quality_check:
            components.append("quality check")
        return " + ".join(components)


CONFIGURATIONS = [
    AblationConfiguration(
        "A0",
        "Cosine baseline",
        1.0,
        0.0,
        0.0,
        False,
        False,
        False,
        "Reference baseline",
    ),
    AblationConfiguration(
        "A1",
        "+ Evidence quality",
        0.7,
        0.3,
        0.0,
        False,
        False,
        False,
        "Test evidence-quality weighting",
    ),
    AblationConfiguration(
        "A2",
        "+ Temporal recency",
        0.5,
        0.3,
        0.2,
        False,
        False,
        False,
        "Test added temporal information",
    ),
    AblationConfiguration(
        "A3",
        "+ Hybrid context",
        0.5,
        0.3,
        0.2,
        True,
        False,
        False,
        "Test paper-level contextual augmentation",
    ),
]


def validate_configuration_sequence() -> None:
    """Ensure A0-A3 are nested versions of the same ET-RAG agent."""
    expected_ids = [f"A{index}" for index in range(4)]
    actual_ids = [configuration.config_id for configuration in CONFIGURATIONS]
    if actual_ids != expected_ids:
        raise ValueError(f"Expected configuration IDs {expected_ids}, got {actual_ids}")

    expected_components = [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (True, True, False, False, False),
        (True, True, True, False, False),
    ]
    actual_components = [
        (
            configuration.evidence_weight > 0,
            configuration.temporal_weight > 0,
            configuration.hybrid_context,
            configuration.per_option_retrieval,
            configuration.quality_check,
        )
        for configuration in CONFIGURATIONS
    ]
    if actual_components != expected_components:
        raise ValueError(
            "Ablation components are not introduced one at a time: "
            f"{actual_components}"
        )

    for configuration in CONFIGURATIONS:
        weight_total = (
            configuration.similarity_weight
            + configuration.evidence_weight
            + configuration.temporal_weight
        )
        if not np.isclose(weight_total, 1.0):
            raise ValueError(
                f"{configuration.config_id} retrieval weights sum to {weight_total}, not 1.0"
            )

    # A2 and A3 use the scoring weights declared by ET-RAG in test.py. A3 then
    # adds hybrid paper context; later augmentation components remain disabled.
    weighted_etrag = CONFIGURATIONS[2]
    expected_etrag_weights = (
        float(app.ETRAG_WEIGHTS["cosine"]),
        float(app.ETRAG_WEIGHTS["evidence"]),
        float(app.ETRAG_WEIGHTS["temporal"]),
    )
    actual_etrag_weights = (
        weighted_etrag.similarity_weight,
        weighted_etrag.evidence_weight,
        weighted_etrag.temporal_weight,
    )
    if not np.allclose(actual_etrag_weights, expected_etrag_weights):
        raise ValueError(
            "A2/A3 weights must match test.py ETRAG_WEIGHTS: "
            f"expected {expected_etrag_weights}, got {actual_etrag_weights}"
        )
    if any(
        configuration.per_option_retrieval or configuration.quality_check
        for configuration in CONFIGURATIONS
    ):
        raise ValueError(
            "Per-option retrieval and quality checking must remain disabled "
            "in the A0-A3 study"
        )


def write_configuration_table(path: Path = CONFIGURATION_FILE) -> Path:
    """Save the active A0-A3 configuration matrix as TSV."""
    rows = []
    for configuration in CONFIGURATIONS:
        rows.append({
            "agent_family": AGENT_FAMILY,
            "configuration": f"{configuration.config_id}: {configuration.name}",
            "retrieval_score": configuration.retrieval_score_label,
            "enabled_components": configuration.enabled_components,
            "hybrid_paper_context": configuration.hybrid_context,
            "per_option_retrieval": configuration.per_option_retrieval,
            "quality_check": configuration.quality_check,
            "purpose": configuration.purpose,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding="utf-8-sig")
    return path


def format_question(item: dict[str, Any]) -> str:
    """Combine a question and its A-D choices for retrieval and generation."""
    option_lines = [
        f"{letter}. {text}"
        for letter, text in sorted(item["options"].items())
    ]
    return item["question"] + "\n" + "\n".join(option_lines)


def normalize_expected_answers(answer_keys: list[str] | str) -> str:
    """Normalize the ground-truth answer set to ordered comma-delimited keys."""
    return app._normalize_multiple_choice_keys(answer_keys)


def calculate_strict_partial_credit(
    correct_answer_keys: list[str] | str,
    predicted_answer_keys: list[str] | str,
) -> float:
    """Score multiple-choice answers while penalizing unsupported selections.

    Selecting any incorrect option gives zero. Otherwise, the score is the
    proportion of correct options selected. NONE_SELECTED, UNPARSED,
    NOT_COVERED, ERROR, and an empty prediction therefore receive zero.
    """
    correct_keys = set(re.findall(r"\b([A-D])\b", str(correct_answer_keys).upper()))
    predicted_keys = set(
        re.findall(r"\b([A-D])\b", str(predicted_answer_keys).upper())
    )
    if not correct_keys:
        raise ValueError(
            f"No valid correct answer keys were supplied: {correct_answer_keys!r}"
        )
    if not predicted_keys or not predicted_keys.issubset(correct_keys):
        return 0.0
    return len(predicted_keys) / len(correct_keys)


# [2026-08-27 SINGLE-CHOICE ABLATION ADDITION]
# The current multiple-choice scoring and output functions below are retained
# unchanged. These helpers add a separate exact-match single-choice path.
def normalize_single_choice_answer(answer_key: str) -> str:
    """Normalize A-D or NOT_COVERED for single-choice comparison."""
    return app._normalize_expected_answer(str(answer_key))


def calculate_single_choice_correctness(
    correct_answer_key: str,
    predicted_answer_key: str,
) -> float:
    """Return 1.0 for an exact single-choice match and 0.0 otherwise."""
    expected = normalize_single_choice_answer(correct_answer_key)
    predicted = normalize_single_choice_answer(predicted_answer_key)
    return float(predicted == expected)


def _single_choice_result_path(
    output_dir: Path,
    configuration: AblationConfiguration,
) -> Path:
    """Return a raw single-choice TSV path distinct from existing outputs."""
    return (
        output_dir
        / f"{configuration.config_id}_{configuration.slug}_single_choice_results.tsv"
    )


def _single_choice_question_summary_path(
    output_dir: Path,
    configuration: AblationConfiguration,
) -> Path:
    """Return a per-question single-choice summary TSV path."""
    return (
        output_dir
        / f"{configuration.config_id}_{configuration.slug}_single_choice_question_summary.tsv"
    )


def _result_path(output_dir: Path, configuration: AblationConfiguration) -> Path:
    return (
        output_dir
        / f"{configuration.config_id}_{configuration.slug}_multiple_choice_results.tsv"
    )


def _question_summary_path(
    output_dir: Path,
    configuration: AblationConfiguration,
) -> Path:
    """Return the per-question mean ± SD table path for one configuration."""
    return (
        output_dir
        / f"{configuration.config_id}_{configuration.slug}_multiple_choice_question_summary.tsv"
    )


def save_progress(
    rows_by_configuration: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Save every completed row so long API runs are recoverable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for configuration in CONFIGURATIONS:
        rows = rows_by_configuration[configuration.config_id]
        if rows:
            pd.DataFrame(rows).to_csv(
                _result_path(output_dir, configuration),
                sep="\t",
                index=False,
                encoding="utf-8-sig",
            )
            combined_rows.extend(rows)
    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(
            output_dir / "ablation_multiple_choice_all_results.tsv",
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )


def create_summary(all_rows: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    """Create one partial-credit/confidence/time summary per configuration."""
    frame = pd.DataFrame(all_rows)
    summary = (
        frame.groupby(["config_order", "config_id", "configuration"], as_index=False)
        .agg(
            exact_matches=("is_correct", "sum"),
            total_responses=("is_correct", "size"),
            exact_match_accuracy=("is_correct", "mean"),
            correctness_mean=("correctness_score", "mean"),
            correctness_sd=("correctness_score", "std"),
            confidence_mean=("confidence", "mean"),
            confidence_sd=("confidence", "std"),
            execution_time_mean_sec=("execution_time_sec", "mean"),
            execution_time_sd_sec=("execution_time_sec", "std"),
        )
        .sort_values("config_order")
        .drop(columns=["config_order"])
    )
    summary.to_csv(
        output_dir / "ablation_multiple_choice_summary.tsv",
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def create_question_summary_tables(
    all_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write one four-column question summary TSV for every A0-A3 setting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows).copy()
    required_columns = {
        "config_id",
        "question_number",
        "correctness_score",
        "confidence",
        "execution_time_sec",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Cannot create question summary tables; missing columns: "
            + ", ".join(missing_columns)
        )

    # pandas std() uses N-1; a one-round run is reported as zero SD.
    frame["correctness_score"] = pd.to_numeric(
        frame["correctness_score"],
        errors="raise",
    )
    output_paths: dict[str, Path] = {}
    for configuration in CONFIGURATIONS:
        configuration_rows = frame[frame["config_id"] == configuration.config_id]
        if configuration_rows.empty:
            raise ValueError(
                f"No result rows are available for {configuration.config_id}"
            )
        statistics = (
            configuration_rows.groupby("question_number", as_index=False)
            .agg(
                correctness_mean=("correctness_score", "mean"),
                correctness_sd=("correctness_score", "std"),
                confidence_mean=("confidence", "mean"),
                confidence_sd=("confidence", "std"),
                execution_time_mean=("execution_time_sec", "mean"),
                execution_time_sd=("execution_time_sec", "std"),
            )
            .fillna(0.0)
            .sort_values("question_number")
        )

        compact_table = pd.DataFrame({
            "question_index": statistics["question_number"].astype(int),
            "correctness": statistics.apply(
                lambda row: (
                    f"{row['correctness_mean']:.1%} ± "
                    f"{row['correctness_sd']:.1%}"
                ),
                axis=1,
            ),
            "confidence": statistics.apply(
                lambda row: (
                    f"{row['confidence_mean']:.1%} ± "
                    f"{row['confidence_sd']:.1%}"
                ),
                axis=1,
            ),
            "execution_time": statistics.apply(
                lambda row: (
                    f"{row['execution_time_mean']:.1f} ± "
                    f"{row['execution_time_sd']:.1f} (s)"
                ),
                axis=1,
            ),
        })
        output_path = _question_summary_path(output_dir, configuration)
        compact_table.to_csv(
            output_path,
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )
        output_paths[configuration.config_id] = output_path

    return output_paths


# [2026-08-27 SINGLE-CHOICE TSV REPORTING]
# These functions deliberately use ``single_choice`` filenames, leaving every
# existing multiple-choice raw result and summary untouched.
def save_single_choice_progress(
    rows_by_configuration: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Save completed single-choice rows after every model response."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for configuration in CONFIGURATIONS:
        rows = rows_by_configuration[configuration.config_id]
        if rows:
            pd.DataFrame(rows).to_csv(
                _single_choice_result_path(output_dir, configuration),
                sep="\t",
                index=False,
                encoding="utf-8-sig",
            )
            combined_rows.extend(rows)
    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(
            output_dir / "ablation_single_choice_all_results.tsv",
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )


def create_single_choice_summary(
    all_rows: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Create the A0-A3 exact-match/confidence/time single-choice summary."""
    frame = pd.DataFrame(all_rows)
    summary = (
        frame.groupby(["config_order", "config_id", "configuration"], as_index=False)
        .agg(
            exact_matches=("is_correct", "sum"),
            total_responses=("is_correct", "size"),
            exact_match_accuracy=("is_correct", "mean"),
            correctness_mean=("correctness_score", "mean"),
            correctness_sd=("correctness_score", "std"),
            confidence_mean=("confidence", "mean"),
            confidence_sd=("confidence", "std"),
            execution_time_mean_sec=("execution_time_sec", "mean"),
            execution_time_sd_sec=("execution_time_sec", "std"),
        )
        .fillna(0.0)
        .sort_values("config_order")
        .drop(columns=["config_order"])
    )
    summary.to_csv(
        output_dir / "ablation_single_choice_summary.tsv",
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def create_single_choice_question_summary_tables(
    all_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write one four-column single-choice question summary per setting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows).copy()
    required_columns = {
        "config_id",
        "question_number",
        "correctness_score",
        "confidence",
        "execution_time_sec",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Cannot create single-choice question summaries; missing columns: "
            + ", ".join(missing_columns)
        )

    frame["correctness_score"] = pd.to_numeric(
        frame["correctness_score"],
        errors="raise",
    )
    output_paths: dict[str, Path] = {}
    for configuration in CONFIGURATIONS:
        configuration_rows = frame[frame["config_id"] == configuration.config_id]
        if configuration_rows.empty:
            raise ValueError(
                f"No single-choice rows are available for {configuration.config_id}"
            )
        statistics = (
            configuration_rows.groupby("question_number", as_index=False)
            .agg(
                correctness_mean=("correctness_score", "mean"),
                correctness_sd=("correctness_score", "std"),
                confidence_mean=("confidence", "mean"),
                confidence_sd=("confidence", "std"),
                execution_time_mean=("execution_time_sec", "mean"),
                execution_time_sd=("execution_time_sec", "std"),
            )
            .fillna(0.0)
            .sort_values("question_number")
        )

        compact_table = pd.DataFrame({
            "question_index": statistics["question_number"].astype(int),
            "correctness": statistics.apply(
                lambda row: (
                    f"{row['correctness_mean']:.1%} ± "
                    f"{row['correctness_sd']:.1%}"
                ),
                axis=1,
            ),
            "confidence": statistics.apply(
                lambda row: (
                    f"{row['confidence_mean']:.1%} ± "
                    f"{row['confidence_sd']:.1%}"
                ),
                axis=1,
            ),
            "execution_time": statistics.apply(
                lambda row: (
                    f"{row['execution_time_mean']:.1f} ± "
                    f"{row['execution_time_sd']:.1f} (s)"
                ),
                axis=1,
            ),
        })
        output_path = _single_choice_question_summary_path(
            output_dir,
            configuration,
        )
        compact_table.to_csv(
            output_path,
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )
        output_paths[configuration.config_id] = output_path

    return output_paths


def refresh_existing_result_reports(output_dir: Path) -> pd.DataFrame:
    """Backfill strict partial credit and rebuild reports without API calls."""
    output_dir = output_dir.expanduser().resolve()
    frames = []
    for configuration in CONFIGURATIONS:
        result_path = _result_path(output_dir, configuration)
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Raw result TSV is missing for {configuration.config_id}: {result_path}"
            )

        frame = pd.read_csv(result_path, sep="\t", encoding="utf-8-sig")
        required_columns = {"correct_answer_keys", "predicted_answer_keys"}
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"{result_path.name} is missing columns: {', '.join(missing_columns)}"
            )

        frame["correctness_score"] = frame.apply(
            lambda row: calculate_strict_partial_credit(
                row["correct_answer_keys"],
                row["predicted_answer_keys"],
            ),
            axis=1,
        )
        frame["correctness_scheme"] = "strict_partial_credit"
        frame["is_correct"] = np.isclose(frame["correctness_score"], 1.0)

        # Keep the scoring audit fields beside the answer-key comparison.
        ordered_columns = [
            column
            for column in frame.columns
            if column not in {"correctness_score", "correctness_scheme", "is_correct"}
        ]
        prediction_index = ordered_columns.index("predicted_answer_keys") + 1
        ordered_columns[prediction_index:prediction_index] = [
            "correctness_score",
            "correctness_scheme",
            "is_correct",
        ]
        frame = frame[ordered_columns]
        frame.to_csv(
            result_path,
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        output_dir / "ablation_multiple_choice_all_results.tsv",
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )
    all_rows = combined.to_dict(orient="records")
    summary = create_summary(all_rows, output_dir)
    question_paths = create_question_summary_tables(all_rows, output_dir)

    print("\nSTRICT PARTIAL-CREDIT REPORTS REFRESHED")
    print(f"Configurations: {len(CONFIGURATIONS)}")
    print(f"Raw responses: {len(combined)}")
    print(f"Overall summary: {output_dir / 'ablation_multiple_choice_summary.tsv'}")
    for config_id, path in question_paths.items():
        print(f"- {config_id}: {path}")
    return summary


def refresh_single_choice_result_reports(output_dir: Path) -> pd.DataFrame:
    """Recalculate exact-match single-choice reports without API calls."""
    output_dir = output_dir.expanduser().resolve()
    frames = []
    for configuration in CONFIGURATIONS:
        result_path = _single_choice_result_path(output_dir, configuration)
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Raw single-choice TSV is missing for "
                f"{configuration.config_id}: {result_path}"
            )

        frame = pd.read_csv(result_path, sep="\t", encoding="utf-8-sig")
        required_columns = {"correct_answer_keys", "predicted_answer_keys"}
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"{result_path.name} is missing columns: {', '.join(missing_columns)}"
            )

        frame["correctness_score"] = frame.apply(
            lambda row: calculate_single_choice_correctness(
                row["correct_answer_keys"],
                row["predicted_answer_keys"],
            ),
            axis=1,
        )
        frame["correctness_scheme"] = "exact_match"
        frame["is_correct"] = np.isclose(frame["correctness_score"], 1.0)
        frame.to_csv(
            result_path,
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        output_dir / "ablation_single_choice_all_results.tsv",
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )
    all_rows = combined.to_dict(orient="records")
    summary = create_single_choice_summary(all_rows, output_dir)
    question_paths = create_single_choice_question_summary_tables(
        all_rows,
        output_dir,
    )

    print("\nSINGLE-CHOICE EXACT-MATCH REPORTS REFRESHED")
    print(f"Configurations: {len(CONFIGURATIONS)}")
    print(f"Raw responses: {len(combined)}")
    print(f"Overall summary: {output_dir / 'ablation_single_choice_summary.tsv'}")
    for config_id, path in question_paths.items():
        print(f"- {config_id}: {path}")
    return summary


def ensure_output_targets_available(output_dir: Path, overwrite: bool) -> None:
    """Protect prior ablation results unless overwrite was explicitly requested."""
    targets = [
        _result_path(output_dir, configuration)
        for configuration in CONFIGURATIONS
    ]
    targets.extend(
        _question_summary_path(output_dir, configuration)
        for configuration in CONFIGURATIONS
    )
    targets.extend([
        output_dir / "ablation_multiple_choice_all_results.tsv",
        output_dir / "ablation_multiple_choice_summary.tsv",
    ])
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise ValueError(
            "Ablation result files already exist. Use --overwrite to replace them:\n"
            + formatted
        )


def ensure_single_choice_output_targets_available(
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Protect existing single-choice results independently of other outputs."""
    targets = [
        _single_choice_result_path(output_dir, configuration)
        for configuration in CONFIGURATIONS
    ]
    targets.extend(
        _single_choice_question_summary_path(output_dir, configuration)
        for configuration in CONFIGURATIONS
    )
    targets.extend([
        output_dir / "ablation_single_choice_all_results.tsv",
        output_dir / "ablation_single_choice_summary.tsv",
    ])
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise ValueError(
            "Single-choice ablation files already exist. Use --overwrite "
            "to replace them:\n" + formatted
        )


def run_ablation(
    docx_path: Path,
    papers_dir: Path,
    index_cache: Path,
    output_dir: Path,
    limit: int | None,
    overwrite: bool,
    rounds: int = DEFAULT_ROUNDS,
) -> pd.DataFrame:
    """Run repeated A0-A3 trials and save raw and mean ± SD results."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in this shell or create "
            "a workspace .env file before running the live ablation."
        )

    validate_configuration_sequence()
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    write_configuration_table()
    ensure_output_targets_available(output_dir, overwrite)

    questions = app.extract_docx_multiple_choice_questions(docx_path)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise ValueError("No multiple-choice questions were extracted from the DOCX.")

    pdf_files, paper_metadata, raw_texts, chunks = app._load_evaluation_papers(
        papers_dir
    )
    vector_store = app._load_or_build_evaluation_index(
        pdf_files,
        chunks,
        index_cache,
        rebuild=False,
    )
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=512,
    )

    rows_by_configuration = {
        configuration.config_id: [] for configuration in CONFIGURATIONS
    }

    print("\n" + "=" * 78)
    print("A0-A3 CONFIGURABLE ET-RAG MULTIPLE-CHOICE ABLATION")
    print("=" * 78)
    print(f"Questions: {len(questions)}")
    print(f"Rounds: {rounds}")
    print(f"Papers: {len(pdf_files)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {output_dir}")

    for round_number in range(1, rounds + 1):
        print(f"\n[ABLATION ROUND {round_number}/{rounds}]")
        for question_number, item in enumerate(questions, 1):
            question = format_question(item)

            for config_order, configuration in enumerate(CONFIGURATIONS):
                # Directly execute the ET-RAG agent defined in test.py. The
                # runner only supplies controlled A0-A3 component settings.
                agent_result = app.agent_etrag(
                    question,
                    vector_store=vector_store,
                    question_type="multiple_choice",
                    paper_metadata=paper_metadata,
                    raw_texts=raw_texts,
                    retrieval_weights={
                        "cosine": configuration.similarity_weight,
                        "evidence": configuration.evidence_weight,
                        "temporal": configuration.temporal_weight,
                    },
                    use_hybrid_context=configuration.hybrid_context,
                    per_option_retrieval=configuration.per_option_retrieval,
                    quality_check=configuration.quality_check,
                    options=item["options"],
                    llm=llm,
                )
                if not agent_result.get("success"):
                    raise RuntimeError(
                        f"test.py agent_etrag failed in round {round_number}, "
                        f"question {question_number}, {configuration.config_id}: "
                        f"{agent_result.get('answer', 'unknown error')}"
                    )

                predicted = app._extract_predicted_multiple_choice(
                    agent_result["answer"]
                )
                expected = normalize_expected_answers(item["answer_keys"])
                correctness_score = calculate_strict_partial_credit(
                    expected,
                    predicted,
                )
                confidence = agent_result["confidence"]
                execution_time = agent_result["execution_time_sec"]
                row = {
                    "round_number": round_number,
                    "config_order": config_order,
                    "agent_family": AGENT_FAMILY,
                    "invocation_source": agent_result["invocation_source"],
                    "config_id": configuration.config_id,
                    "configuration": configuration.name,
                    "enabled_components": configuration.enabled_components,
                    "similarity_weight": configuration.similarity_weight,
                    "evidence_weight": configuration.evidence_weight,
                    "temporal_weight": configuration.temporal_weight,
                    "question_number": question_number,
                    "question": item["question"],
                    "options": " | ".join(
                        f"{letter}. {text}"
                        for letter, text in sorted(item["options"].items())
                    ),
                    "correct_answer_keys": expected,
                    "correct_answer_texts": " | ".join(item["answer_texts"]),
                    "predicted_answer_keys": predicted,
                    "correctness_score": round(correctness_score, 6),
                    "correctness_scheme": "strict_partial_credit",
                    "is_correct": np.isclose(correctness_score, 1.0),
                    "confidence": round(confidence, 6),
                    "execution_time_sec": round(execution_time, 3),
                    "base_retrieval_time_sec": round(
                        agent_result["base_retrieval_time_sec"],
                        3,
                    ),
                    "option_retrieval_time_sec": round(
                        agent_result["option_retrieval_time_sec"],
                        3,
                    ),
                    "candidate_count": agent_result["candidate_count"],
                    "top_chunk_count": agent_result["top_chunk_count"],
                    "mean_top_retrieval_score": round(
                        agent_result["mean_top_retrieval_score"],
                        6,
                    ),
                    # [2026-08-27 NESTED COMPONENT DIAGNOSTICS]
                    # These columns make it possible to verify that A1, A2,
                    # and A3 really add focused information instead of merely
                    # changing a configuration label.
                    "mean_top_cosine_score": round(
                        agent_result["mean_top_cosine_score"],
                        6,
                    ),
                    "mean_top_evidence_score": round(
                        agent_result["mean_top_evidence_score"],
                        6,
                    ),
                    "mean_top_temporal_score": round(
                        agent_result["mean_top_temporal_score"],
                        6,
                    ),
                    "evidence_query_candidates_added": agent_result[
                        "evidence_query_candidates_added"
                    ],
                    "temporal_query_candidates_added": agent_result[
                        "temporal_query_candidates_added"
                    ],
                    "hybrid_context_file_count": agent_result[
                        "hybrid_context_file_count"
                    ],
                    "hybrid_option_passage_count": agent_result[
                        "hybrid_option_passage_count"
                    ],
                    "hybrid_context_char_count": agent_result[
                        "hybrid_context_char_count"
                    ],
                    "hybrid_non_regression_guard": agent_result[
                        "hybrid_non_regression_guard"
                    ],
                    "hybrid_context": configuration.hybrid_context,
                    "per_option_retrieval": configuration.per_option_retrieval,
                    "quality_check": configuration.quality_check,
                    "quality_changed": agent_result["quality_changed"],
                    "source_files": " | ".join(agent_result["files_used"]),
                    "draft_answer": agent_result["draft_answer"],
                    "final_answer": agent_result["answer"],
                }
                rows_by_configuration[configuration.config_id].append(row)
                save_progress(rows_by_configuration, output_dir)

                status = (
                    "FULL"
                    if row["is_correct"]
                    else "PARTIAL" if correctness_score > 0 else "ZERO"
                )
                print(
                    f"R{round_number} Q{question_number:02d} "
                    f"{configuration.config_id}: {status} "
                    f"predicted={predicted} expected={expected} "
                    f"score={correctness_score:.1%} "
                    f"confidence={confidence:.1%} time={execution_time:.1f}s"
                )

    all_rows = [
        row
        for configuration in CONFIGURATIONS
        for row in rows_by_configuration[configuration.config_id]
    ]
    summary = create_summary(all_rows, output_dir)
    question_summary_paths = create_question_summary_tables(all_rows, output_dir)
    print("\n" + "=" * 78)
    print("ABLATION SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("\nPer-configuration question tables:")
    for config_id, path in question_summary_paths.items():
        print(f"- {config_id}: {path}")
    print(f"\nSaved results in: {output_dir}")
    return summary


# [2026-08-27 SINGLE-CHOICE ABLATION RUNNER]
# ``run_ablation`` above remains the original multiple-choice workflow. This
# separate runner directly calls the same configurable test.agent_etrag agent
# and writes only single-choice-specific TSV filenames.
def run_single_choice_ablation(
    docx_path: Path,
    papers_dir: Path,
    index_cache: Path,
    output_dir: Path,
    limit: int | None,
    overwrite: bool,
    rounds: int = DEFAULT_ROUNDS,
) -> pd.DataFrame:
    """Run repeated A0-A3 trials on the DOCX single-choice questions."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in this shell or create "
            "a workspace .env file before running the live ablation."
        )

    validate_configuration_sequence()
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    write_configuration_table()
    ensure_single_choice_output_targets_available(output_dir, overwrite)

    questions = app.extract_docx_single_choice_questions(docx_path)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise ValueError("No single-choice questions were extracted from the DOCX.")

    pdf_files, paper_metadata, raw_texts, chunks = app._load_evaluation_papers(
        papers_dir
    )
    vector_store = app._load_or_build_evaluation_index(
        pdf_files,
        chunks,
        index_cache,
        rebuild=False,
    )
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=512,
    )
    rows_by_configuration = {
        configuration.config_id: [] for configuration in CONFIGURATIONS
    }

    print("\n" + "=" * 78)
    print("A0-A3 CONFIGURABLE ET-RAG SINGLE-CHOICE ABLATION")
    print("=" * 78)
    print(f"Questions: {len(questions)}")
    print(f"Rounds: {rounds}")
    print(f"Papers: {len(pdf_files)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {output_dir}")

    for round_number in range(1, rounds + 1):
        print(f"\n[SINGLE-CHOICE ABLATION ROUND {round_number}/{rounds}]")
        for question_number, item in enumerate(questions, 1):
            question = format_question(item)

            for config_order, configuration in enumerate(CONFIGURATIONS):
                agent_result = app.agent_etrag(
                    question,
                    vector_store=vector_store,
                    question_type="single_choice",
                    paper_metadata=paper_metadata,
                    raw_texts=raw_texts,
                    retrieval_weights={
                        "cosine": configuration.similarity_weight,
                        "evidence": configuration.evidence_weight,
                        "temporal": configuration.temporal_weight,
                    },
                    use_hybrid_context=configuration.hybrid_context,
                    per_option_retrieval=configuration.per_option_retrieval,
                    quality_check=configuration.quality_check,
                    options=item["options"],
                    llm=llm,
                )
                if not agent_result.get("success"):
                    raise RuntimeError(
                        f"test.py agent_etrag failed in round {round_number}, "
                        f"single-choice question {question_number}, "
                        f"{configuration.config_id}: "
                        f"{agent_result.get('answer', 'unknown error')}"
                    )

                predicted = app._extract_predicted_single_choice(
                    agent_result["answer"]
                )
                expected = normalize_single_choice_answer(item["answer_key"])
                correctness_score = calculate_single_choice_correctness(
                    expected,
                    predicted,
                )
                confidence = agent_result["confidence"]
                execution_time = agent_result["execution_time_sec"]
                row = {
                    "round_number": round_number,
                    "config_order": config_order,
                    "agent_family": AGENT_FAMILY,
                    "invocation_source": agent_result["invocation_source"],
                    "question_type": "single_choice",
                    "config_id": configuration.config_id,
                    "configuration": configuration.name,
                    "enabled_components": configuration.enabled_components,
                    "similarity_weight": configuration.similarity_weight,
                    "evidence_weight": configuration.evidence_weight,
                    "temporal_weight": configuration.temporal_weight,
                    "question_number": question_number,
                    "question": item["question"],
                    "options": " | ".join(
                        f"{letter}. {text}"
                        for letter, text in sorted(item["options"].items())
                    ),
                    "correct_answer_keys": expected,
                    "correct_answer_texts": item["answer_text"],
                    "predicted_answer_keys": predicted,
                    "correctness_score": round(correctness_score, 6),
                    "correctness_scheme": "exact_match",
                    "is_correct": np.isclose(correctness_score, 1.0),
                    "confidence": round(confidence, 6),
                    "execution_time_sec": round(execution_time, 3),
                    "base_retrieval_time_sec": round(
                        agent_result["base_retrieval_time_sec"],
                        3,
                    ),
                    "option_retrieval_time_sec": round(
                        agent_result["option_retrieval_time_sec"],
                        3,
                    ),
                    "candidate_count": agent_result["candidate_count"],
                    "top_chunk_count": agent_result["top_chunk_count"],
                    "mean_top_retrieval_score": round(
                        agent_result["mean_top_retrieval_score"],
                        6,
                    ),
                    "mean_top_cosine_score": round(
                        agent_result["mean_top_cosine_score"],
                        6,
                    ),
                    "mean_top_evidence_score": round(
                        agent_result["mean_top_evidence_score"],
                        6,
                    ),
                    "mean_top_temporal_score": round(
                        agent_result["mean_top_temporal_score"],
                        6,
                    ),
                    "evidence_query_candidates_added": agent_result[
                        "evidence_query_candidates_added"
                    ],
                    "temporal_query_candidates_added": agent_result[
                        "temporal_query_candidates_added"
                    ],
                    "hybrid_context_file_count": agent_result[
                        "hybrid_context_file_count"
                    ],
                    "hybrid_option_passage_count": agent_result[
                        "hybrid_option_passage_count"
                    ],
                    "hybrid_context_char_count": agent_result[
                        "hybrid_context_char_count"
                    ],
                    "hybrid_non_regression_guard": agent_result[
                        "hybrid_non_regression_guard"
                    ],
                    "hybrid_context": configuration.hybrid_context,
                    "per_option_retrieval": configuration.per_option_retrieval,
                    "quality_check": configuration.quality_check,
                    "quality_changed": agent_result["quality_changed"],
                    "source_files": " | ".join(agent_result["files_used"]),
                    "draft_answer": agent_result["draft_answer"],
                    "final_answer": agent_result["answer"],
                }
                rows_by_configuration[configuration.config_id].append(row)
                save_single_choice_progress(rows_by_configuration, output_dir)

                status = "CORRECT" if row["is_correct"] else "INCORRECT"
                print(
                    f"R{round_number} Q{question_number:02d} "
                    f"{configuration.config_id}: {status} "
                    f"predicted={predicted} expected={expected} "
                    f"confidence={confidence:.1%} time={execution_time:.1f}s"
                )

    all_rows = [
        row
        for configuration in CONFIGURATIONS
        for row in rows_by_configuration[configuration.config_id]
    ]
    summary = create_single_choice_summary(all_rows, output_dir)
    question_summary_paths = create_single_choice_question_summary_tables(
        all_rows,
        output_dir,
    )
    print("\n" + "=" * 78)
    print("SINGLE-CHOICE ABLATION SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("\nPer-configuration single-choice question tables:")
    for config_id, path in question_summary_paths.items():
        print(f"- {config_id}: {path}")
    print(f"\nSaved single-choice results in: {output_dir}")
    return summary


def validate_without_api(docx_path: Path, papers_dir: Path) -> None:
    """Validate questions, corpus, chunks, and configuration isolation."""
    validate_configuration_sequence()
    configuration_path = write_configuration_table()
    questions = app.extract_docx_multiple_choice_questions(docx_path)
    pdf_files, paper_metadata, raw_texts, chunks = app._load_evaluation_papers(
        papers_dir
    )
    if len(questions) != 10:
        raise ValueError(f"Expected 10 multiple-choice questions, found {len(questions)}")
    if not chunks or not raw_texts or not paper_metadata:
        raise ValueError("ET-RAG corpus chunks, paper metadata, or raw texts are empty")

    required_agent_parameters = {
        "retrieval_weights",
        "use_hybrid_context",
        "per_option_retrieval",
        "quality_check",
        "options",
    }
    available_agent_parameters = set(app.agent_etrag.__code__.co_varnames)
    if not required_agent_parameters.issubset(available_agent_parameters):
        missing = sorted(required_agent_parameters - available_agent_parameters)
        raise ValueError(f"test.py agent_etrag is missing ablation controls: {missing}")

    print("\nAblation validation: PASS")
    print(f"Agent family: {AGENT_FAMILY} (one configurable pipeline)")
    print("Direct invocation: test.agent_etrag")
    print(f"Configurations: {len(CONFIGURATIONS)}")
    print(f"Multiple-choice questions: {len(questions)}")
    print(f"Papers: {len(pdf_files)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Configuration table: {configuration_path}")
    # [2026-08-27 EVIDENCE-QUALITY NOTE UPDATE]
    # ORIGINAL CODE (retained as requested):
    # print(
    #     "Methodological note: the current corpus loader classifies every file "
    #     "in 'review papers' as study_type='review'; therefore A1 evidence "
    #     "weights may be uniform unless source study types are enriched."
    # )
    print(
        "Methodological note: source papers remain classified as reviews, but "
        "A1 now also scores explicit study-design evidence discussed inside "
        "each chunk and adds an evidence-focused candidate query."
    )


def validate_single_choice_without_api(
    docx_path: Path,
    papers_dir: Path,
) -> None:
    """Validate the additive single-choice mode without OpenAI API calls."""
    validate_configuration_sequence()
    configuration_path = write_configuration_table()
    questions = app.extract_docx_single_choice_questions(docx_path)
    # [2026-08-27 FAST SINGLE-CHOICE VALIDATION]
    # Parsing and chunking all 1,200+ corpus chunks made --validate-only take
    # several minutes. The live runner still performs the full loader. Here we
    # verify PDF presence and exercise the exact shared chunk helper locally.
    # ORIGINAL CODE (retained as requested):
    # pdf_files, paper_metadata, raw_texts, chunks = app._load_evaluation_papers(
    #     papers_dir
    # )
    papers_dir = papers_dir.expanduser().resolve()
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF papers were found in: {papers_dir}")
    chunks = app._build_example_chunks(
        "Abstract Validation text. Introduction Validation body.",
        {
            "title": "Validation paper",
            "year": "2026",
            "study_type": "review",
        },
        "validation.pdf",
    )
    if len(questions) != 10:
        raise ValueError(
            f"Expected 10 single-choice questions, found {len(questions)}"
        )
    if not chunks or "SOURCE_FILE: validation.pdf" not in chunks[0]:
        raise ValueError("ET-RAG shared chunk helper failed validation")

    for question_number, item in enumerate(questions, 1):
        if len(item.get("options", {})) != 4:
            raise ValueError(
                f"Single-choice question {question_number} does not have A-D options"
            )
        normalized_key = normalize_single_choice_answer(item["answer_key"])
        if normalized_key not in {"A", "B", "C", "D", "NOT_COVERED"}:
            raise ValueError(
                f"Invalid single-choice key for question {question_number}: "
                f"{item['answer_key']!r}"
            )

    print("\nSingle-choice ablation validation: PASS")
    print(f"Agent family: {AGENT_FAMILY} (one configurable pipeline)")
    print("Direct invocation: test.agent_etrag")
    print(f"Configurations: {len(CONFIGURATIONS)}")
    print(f"Single-choice questions: {len(questions)}")
    print(f"Papers: {len(pdf_files)}")
    print(f"Chunk-helper probe: {len(chunks)} chunk")
    print(f"Configuration table: {configuration_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled A0-A3 ET-RAG choice-question ablation."
    )
    # [2026-08-27 SINGLE-CHOICE CLI OPTION]
    # Default remains multiple-choice, preserving every existing command.
    # ORIGINAL BEHAVIOR (retained as requested): multiple-choice only.
    parser.add_argument(
        "--question-type",
        choices=("multiple-choice", "single-choice"),
        default="multiple-choice",
        help="question section to evaluate (default: multiple-choice)",
    )
    parser.add_argument("--docx", default=str(DEFAULT_DOCX))
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--index-cache", default=str(DEFAULT_INDEX_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"number of repeated trials used for mean ± SD (default: {DEFAULT_ROUNDS})",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-reports",
        action="store_true",
        help=(
            "recalculate existing A0-A3 TSVs for the selected question type "
            "and regenerate reports without API calls"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate configurations/questions/corpus without API calls",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    try:
        if arguments.refresh_reports:
            if arguments.question_type == "single-choice":
                refresh_single_choice_result_reports(Path(arguments.output_dir))
            else:
                # ORIGINAL MULTIPLE-CHOICE CALL (retained and still active):
                # refresh_existing_result_reports(Path(arguments.output_dir))
                refresh_existing_result_reports(Path(arguments.output_dir))
        elif arguments.validate_only:
            if arguments.question_type == "single-choice":
                validate_single_choice_without_api(
                    Path(arguments.docx),
                    Path(arguments.papers_dir),
                )
            else:
                # ORIGINAL MULTIPLE-CHOICE VALIDATION (retained and active):
                validate_without_api(
                    Path(arguments.docx),
                    Path(arguments.papers_dir),
                )
        else:
            if arguments.question_type == "single-choice":
                run_single_choice_ablation(
                    docx_path=Path(arguments.docx),
                    papers_dir=Path(arguments.papers_dir),
                    index_cache=Path(arguments.index_cache),
                    output_dir=Path(arguments.output_dir),
                    limit=arguments.limit,
                    overwrite=arguments.overwrite,
                    rounds=arguments.rounds,
                )
            else:
                # ORIGINAL MULTIPLE-CHOICE RUNNER (retained and still active):
                run_ablation(
                    docx_path=Path(arguments.docx),
                    papers_dir=Path(arguments.papers_dir),
                    index_cache=Path(arguments.index_cache),
                    output_dir=Path(arguments.output_dir),
                    limit=arguments.limit,
                    overwrite=arguments.overwrite,
                    rounds=arguments.rounds,
                )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Ablation could not start: {error}")


if __name__ == "__main__":
    main()
