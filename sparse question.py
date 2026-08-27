r"""Extract and print single-choice answers from Questions for alz bot.docx.

Run from the command line with:
    E:\anaconda3_64\envs\py312\python.exe text.py

An alternative DOCX file can be supplied with:
    E:\anaconda3_64\envs\py312\python.exe text.py --docx "path\to\questions.docx"
"""

import argparse
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


# Default Word document requested for this example.
DEFAULT_DOCX_PATH = Path(
    r"C:\Users\yzlco\Desktop\chatbot\code\Questions for alz bot.docx"
)

# Namespace used by text elements inside Microsoft Word document.xml files.
WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def read_docx_lines(docx_path):
    """Extract visible lines from a DOCX using Python's standard library."""
    docx_path = Path(docx_path).expanduser().resolve()
    if not docx_path.is_file():
        raise FileNotFoundError(f"DOCX file was not found: {docx_path}")

    # DOCX is a ZIP archive; the main document text is in word/document.xml.
    with zipfile.ZipFile(docx_path) as docx_archive:
        document_xml = docx_archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraph_tag = f"{{{WORD_NAMESPACE}}}p"
    text_tag = f"{{{WORD_NAMESPACE}}}t"
    break_tags = {
        f"{{{WORD_NAMESPACE}}}br",
        f"{{{WORD_NAMESPACE}}}cr",
    }

    lines = []
    for paragraph in root.iter(paragraph_tag):
        parts = []
        for element in paragraph.iter():
            if element.tag == text_tag and element.text:
                parts.append(element.text)
            elif element.tag in break_tags:
                parts.append("\n")

        # Manual Word line breaks may place an option and answer in one XML
        # paragraph. Split those into individual logical lines.
        for line in "".join(parts).splitlines():
            cleaned_line = " ".join(line.split())
            if cleaned_line:
                lines.append(cleaned_line)

    return lines


def extract_single_choice_questions(docx_path=DEFAULT_DOCX_PATH):
    """Extract questions, options, and correct answers from one DOCX section."""
    questions = []
    current_question = None
    inside_single_choice_section = False

    for line in read_docx_lines(docx_path):
        # Begin at the Single Choice heading and stop before Multiple Choice.
        if re.match(r"^Single Choice Questions", line, re.IGNORECASE):
            inside_single_choice_section = True
            continue
        if re.match(r"^Multiple Choice Questions", line, re.IGNORECASE):
            break
        if not inside_single_choice_section:
            continue

        option_match = re.match(r"^([A-D])[\.)]\s*(.+)$", line)
        answer_match = re.match(r"^Correct Answer:\s*(.+)$", line, re.IGNORECASE)

        if option_match and current_question:
            option_letter, option_text = option_match.groups()
            current_question["options"][option_letter] = option_text.strip()

        elif answer_match and current_question:
            answer_key = answer_match.group(1).strip()
            current_question["answer_key"] = answer_key
            current_question["answer_text"] = current_question["options"].get(
                answer_key.upper(),
                answer_key,
            )
            questions.append(current_question)
            current_question = None

        else:
            # A non-option line inside this section is the next question.
            current_question = {
                "question": line,
                "options": {},
                "answer_key": None,
                "answer_text": None,
            }

    return questions


def print_single_choice_answers(docx_path=DEFAULT_DOCX_PATH):
    """Print each question and its correct answer in the command line."""
    questions = extract_single_choice_questions(docx_path)

    print("=" * 78)
    print("SINGLE-CHOICE ANSWERS FROM QUESTIONS FOR ALZ BOT")
    print("=" * 78)
    print(f"Source: {Path(docx_path).resolve()}")
    print(f"Questions extracted: {len(questions)}\n")

    for question_number, item in enumerate(questions, 1):
        answer_key = item["answer_key"]
        answer_text = item["answer_text"]

        if answer_key and answer_key.upper() in item["options"]:
            formatted_answer = f"{answer_key.upper()}. {answer_text}"
        else:
            formatted_answer = answer_text or "Answer not found"

        print(f"{question_number}. {item['question']}")
        print(f"   Correct answer: {formatted_answer}\n")

    return questions


def parse_arguments():
    """Read an optional alternative DOCX path from the command line."""
    parser = argparse.ArgumentParser(
        description="Print single-choice questions and answers from a DOCX file."
    )
    parser.add_argument(
        "--docx",
        default=str(DEFAULT_DOCX_PATH),
        help="path to the questions DOCX file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    print_single_choice_answers(arguments.docx)
