# ET-RAG: A Multi-Agent RAG Framework for Biomedical Literature Analysis

A Streamlit application that uses **3 AI agents** to answer research questions from uploaded medical papers. Each agent receives a differently assembled portion of the extracted text; their answers are presented side by side so the user can see where the agents agree and where they diverge.

## What It Does

Upload research papers as PDFs, then ask questions. Three agents analyze different views of the extracted text, and each returns an answer grounded in the content made available to it. A synthesis layer combines the three responses into a single cited answer with a consensus indicator.

- Paste one question or forty — the system auto-detects question count and type.
- Irrelevant questions are identified as "not covered" (zero hallucination).
- Every answer cites the source paper by title and year.

## The 3 Agents

| Agent | Strategy | How It Works |
|-------|----------|-------------|
| **Paper Prefix** | Fixed leading excerpt | Reads only the first 8,000 characters of each paper's extracted text in one prompt. This is not a summary: later sections are unavailable to this agent. |
| **Cosine RAG** | Chunk retrieval | Embeds the question, retrieves the top-25 most similar chunks from a FAISS index. Fast and precise for specific facts. |
| **ET-RAG** | Hybrid (novel) | Retrieves chunks, reranks them by evidence quality + recency, then adds the first 15,000 extracted characters of each paper as supplementary prefixes. |

All three agents use **GPT-4o-mini** as the language model. They differ only in how they assemble the context window, so any performance difference is attributable to the retrieval strategy.

### ET-RAG Scoring

Each retrieved chunk receives a composite score:

```
S(c) = 0.5 * cosine_similarity + 0.3 * evidence_weight + 0.2 * temporal_weight
```

**Evidence weights** follow the GRADE hierarchy:

| Study Type | Weight |
|-----------|--------|
| Meta-analysis | 1.00 |
| Systematic review | 0.95 |
| Randomized controlled trial | 0.90 |
| Cohort study | 0.75 |
| Case-control study | 0.65 |
| Case series | 0.45 |
| Case report | 0.35 |

**Temporal weights** reflect knowledge recency:
- Published within 3 years: 1.0
- 3–7 years: 0.85
- Older than 7 years: 0.6

## Quick Start

```bash
git clone https://github.com/ai-pharm-AU/CHATBOT.git
cd CHATBOT
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

Run:

```bash
streamlit run INTEGRATED_MULTI_AGENT_COMPLETE.py
```

Then upload PDFs in the sidebar, click "Process Papers," and start asking questions.

## How It Works

```
User uploads PDFs
    → Text extracted, chunked (2000 chars, 400 overlap)
    → Embedded with OpenAI text-embedding-3-small (1536 dims)
    → Stored in FAISS index

User asks a question
    → Medical synonym expansion (15 term families)
    → Three agents run in parallel:
        Agent 1: Paper Prefix (first 8,000 extracted characters per paper; no retrieval or summarization)
        Agent 2: Cosine RAG (multi-query retrieval, top-25 chunks)
        Agent 3: ET-RAG (retrieval + reranking + leading paper prefixes)
    → Synthesis layer combines responses
    → User sees: each agent's answer + final synthesized answer + consensus
```

## Evaluation

Tested on 40 questions across 4 types using 10 Alzheimer's disease papers (2021–2025). All agents used GPT-4o-mini at temperature 0.0.

| Question Type | Paper Prefix | Cosine RAG | ET-RAG |
|--------------|-------------|------------|--------|
| Single Choice (10) | 90% | 90% | **90%** |
| Multiple Choice (10) | 0.58 | 0.33 | **0.59** |
| Short Answer (10) | 0.89 | 0.63 | **0.91** |
| Long Answer (10) | 0.81 | 0.58 | **0.85** |
| **Combined** | 0.80 | 0.61 | **0.81** |

- Hallucination rate: **0%** (control question correctly rejected)
- ET-RAG reranking overhead: **< 5%** of total query time

## Configuration

Key parameters at the top of `INTEGRATED_MULTI_AGENT_COMPLETE.py`:

```python
TEMPERATURE = 0.0       # Deterministic output
MAX_TOKENS = 2048
TOP_K_CHUNKS = 20       # Chunks sent to LLM
RETRIEVAL_CANDIDATES = 40
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
PAPER_PREFIX_CHAR_LIMIT = 8_000  # Leading excerpt, not a summary
ETRAG_PREFIX_CHAR_LIMIT = 15_000
```

## Tech Stack

- Python 3.12 / Streamlit
- OpenAI GPT-4o-mini — all three agents + synthesis
- Google Gemini 2.0 Flash — metadata extraction only
- FAISS — vector similarity search
- LangChain — LLM orchestration
- OpenAI text-embedding-3-small — 1536-dim embeddings
- PyPDF2 — PDF text extraction

## File Structure

```
CHATBOT/
├── INTEGRATED_MULTI_AGENT_COMPLETE.py  # Entire application
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Contributors

- **Revanth Reddy Palem** — Auburn University
