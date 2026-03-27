# Multi-Agent Research Assistant for Medical Literature Analysis

A production-ready Streamlit application that uses **3 AI agents** to answer questions about uploaded medical research papers, with emphasis on evidence-based medicine principles.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## What It Does

Upload your research papers (PDFs), then ask questions — either one at a time or paste 40+ questions at once. Three independent AI agents analyze the papers and a fourth GPT layer synthesizes their answers into one clear, cited response with consensus evaluation.

**Zero hallucination guarantee** — agents only use content from your uploaded papers. Irrelevant questions are correctly identified as "not covered."

## The 3 Agents

| Agent | Model | Approach | Strength |
|-------|-------|----------|----------|
| **Full Context** | Google Gemini 2.0 Flash | Sends entire corpus to LLM | Cross-paper synthesis, complete context |
| **Cosine RAG** | OpenAI GPT-4o-mini | FAISS vector similarity search | Fast, scalable, specific facts |
| **ET-RAG** (Novel) | OpenAI GPT-4o-mini | Evidence + Temporal weighted retrieval | Prioritizes high-quality, recent evidence |

### ET-RAG Scoring Formula

```
Score = 0.5 × Cosine Similarity + 0.3 × Evidence Quality + 0.2 × Temporal Recency
```

Evidence quality follows the **GRADE framework**:
- Meta-analysis (1.0) > Systematic Review (0.95) > RCT (0.9) > Cohort (0.75) > Case Report (0.35)

## Features

- **Single Question Mode** — Ask one question, get detailed responses from all 3 agents + synthesized answer
- **Batch Mode** — Paste 40+ questions (any format), GPT parses them, all 3 agents answer each one
- **Smart Question Parsing** — Handles numbered lists, bullet points, multi-line MCQs with options, section headers
- **Auto Question Type Detection** — Single choice, multiple choice, short answer, long answer (detected from section headers)
- **GPT-Powered Synthesis** — Reads all 3 agent responses, evaluates correctness, produces one polished answer
- **Consensus Evaluation** — STRONG / MAJORITY / SPLIT based on actual content analysis, not keyword matching
- **Medical Synonym Expansion** — "sleep disorder" auto-expands to include insomnia, OSA, narcolepsy, etc.
- **Hybrid Retrieval** — Original + expanded query, 30 candidates each, deduplicated, top-15 selected
- **Source Attribution** — Citations in [Paper #, Page #] format
- **CSV Export** — Download batch results with all agent responses, confidence scores, and consensus

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ai-pharm-AU/CHATBOT.git
cd CHATBOT
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=AIza...
```

You need:
- **OpenAI API key** — for GPT-4o-mini (Cosine RAG, ET-RAG, synthesis, parsing)
- **Google API key** — for Gemini 2.0 Flash (Full Context agent)

### 3. Run

```bash
streamlit run INTEGRATED_MULTI_AGENT_COMPLETE.py
```

### 4. Use

1. Upload research papers (PDF) in the sidebar
2. Click "Process Papers" — extracts metadata, creates embeddings, builds FAISS indices
3. Ask a single question in the chat, OR paste multiple questions (numbered or newline-separated)
4. View each agent's individual response + synthesized final answer with consensus

## Batch Question Format

Paste questions in any format — the GPT parser handles it:

```
Single Choice Questions (10)

1. Which protein is a hallmark of AD?
   A. α-synuclein  B. Amyloid-beta  C. Huntingtin  D. Dopamine

2. What sleep disorder is linked to AD risk?
   A. Narcolepsy  B. Insomnia  C. OSA  D. Sleepwalking

Short Answer Questions (10)

21. What is the role of Aβ-PET in AD diagnosis?
22. How does hearing loss influence AD?

Long Answer Questions (10)

31. Discuss the bidirectional relationship between sleep and AD.
```

## Architecture

```
User Input
    │
    ├── GPT-4o-mini (Question Parser)
    │       → Extracts questions + types from raw text
    │
    ├── For each question:
    │   ├── Agent 1: Gemini 2.0 Flash (Full Context)
    │   ├── Agent 2: GPT-4o-mini (Cosine RAG via FAISS)
    │   ├── Agent 3: GPT-4o-mini (ET-RAG via FAISS + GRADE + Temporal)
    │   │
    │   └── GPT-4o-mini (Synthesizer)
    │           → Evaluates each agent's correctness
    │           → Produces final answer with citations
    │           → Determines consensus (STRONG/MAJORITY/SPLIT)
    │
    └── Display: Per-agent answers + Synthesis + Consensus
```

## Configuration

All tunable parameters are at the top of the file:

```python
TEMPERATURE = 0.1          # Low for deterministic responses
MAX_TOKENS = 2048
TOP_K_CHUNKS = 15          # Chunks sent to LLM
RETRIEVAL_CANDIDATES = 30  # Candidates before reranking
CHUNK_SIZE = 2000          # Characters per chunk
CHUNK_OVERLAP = 400        # Overlap for context preservation
```

## Evaluation Results (Proof of Concept: Alzheimer's Disease)

Tested on 40 questions (10 single-choice, 10 multiple-choice, 10 short-answer, 10 long-answer) across 10 peer-reviewed papers.

| Metric | Value |
|--------|-------|
| Strong Consensus (3/3 agree) | 70% |
| Majority Consensus (2/3 agree) | 22% |
| Average Confidence | 74% |
| Hallucination Rate | 0% |
| Control Question (irrelevant topic) | Correctly identified as "not covered" |
| Avg Response Time per Question | ~33s |

## Tech Stack

- **Python 3.12** / **Streamlit** — UI framework
- **OpenAI GPT-4o-mini** — RAG agents, synthesis, question parsing
- **Google Gemini 2.0 Flash** — Full context analysis
- **FAISS** — Vector similarity search
- **LangChain** — LLM orchestration
- **OpenAI text-embedding-3-small** — 1536-dim embeddings
- **PyPDF2** — PDF text extraction

## File Structure

```
CHATBOT/
├── INTEGRATED_MULTI_AGENT_COMPLETE.py  # Entire application (single file)
├── requirements.txt                     # Python dependencies
├── .env.example                         # API key template
├── .env                                 # Your API keys (not in git)
├── .gitignore
└── README.md
```

## Academic Context

- Research project for academic poster presentation
- Methodology adapted from Martinez-Garcia et al. (2025) *Transplantation Proceedings*
- Novel contribution: **ET-RAG** — first RAG system integrating GRADE evidence hierarchy + temporal weighting
- Target journals: JMIR, JAMIA, NPJ Digital Medicine

## Contributors

- **Revanth Reddy Palem**
- **Ruchith**

## License

MIT License
