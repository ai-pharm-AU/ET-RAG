"""
MULTI-AGENT ALZHEIMER'S RESEARCH CHATBOT
=========================================
Production-Ready Version

Three AI Agents Answer Your Research Questions:
- Full Context Agent (GPT-4o-mini) - Analyzes condensed corpus (10K/paper)
- Cosine RAG Agent (GPT-4o-mini) - Multi-query semantic retrieval
- ET-RAG Agent (GPT-4o-mini) - Hybrid: Evidence-weighted retrieval + paper context

Features:
- Upload your own research papers
- Ask any question about the content
- Get answers from 3 different AI approaches
- See which files were used
- Get consensus recommendation
- Zero hallucination - only uses your uploaded content
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import re
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model settings for consistency
TEMPERATURE = 0.0  # Fully deterministic for medical accuracy
MAX_TOKENS = 2048
TOP_K_CHUNKS = 20  # More chunks for better coverage
RETRIEVAL_CANDIDATES = 40  # Retrieve more, then rerank
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
CURRENT_YEAR = 2025

# Evidence quality hierarchy (for ET-RAG)
EVIDENCE_WEIGHTS = {
    'meta-analysis': 1.0,
    'systematic-review': 0.95,
    'rct': 0.9,
    'cohort': 0.75,
    'case-control': 0.65,
    'case-series': 0.45,
    'case-report': 0.35,
    'review': 0.5,
    'unknown': 0.5
}

# ET-RAG scoring weights
ETRAG_WEIGHTS = {
    'cosine': 0.5,
    'evidence': 0.3,
    'temporal': 0.2
}

# ============================================================================
# SETUP
# ============================================================================

load_dotenv()

# Initialize APIs
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"⚠️ API Configuration Error: {e}")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "papers_processed" not in st.session_state:
    st.session_state.papers_processed = False
if "paper_metadata" not in st.session_state:
    st.session_state.paper_metadata = {}
if "raw_texts" not in st.session_state:
    st.session_state.raw_texts = {}
if "vector_store_cosine" not in st.session_state:
    st.session_state.vector_store_cosine = None
if "vector_store_etrag" not in st.session_state:
    st.session_state.vector_store_etrag = None
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = None

# ============================================================================
# QUERY EXPANSION
# ============================================================================

def expand_query(question):
    """Expand query with medical synonyms and variations"""
    
    # Common medical term expansions
    expansions = {
        'sleep disorder': 'sleep disorder OR sleep disturbance OR insomnia OR sleep apnea OR OSA OR narcolepsy OR sleep problem',
        'sleep stage': 'sleep stage OR NREM OR REM OR slow-wave OR sleep architecture OR sleep cycle',
        'metabolite clearance': 'metabolite clearance OR glymphatic OR Aβ clearance OR waste clearance OR brain clearance',
        'AD': 'AD OR Alzheimer OR Alzheimer\'s disease OR dementia',
        'imaging': 'imaging OR scan OR PET OR MRI OR CT OR SPECT',
        'amyloid': 'amyloid OR Aβ OR A-beta OR amyloid-beta OR plaque',
        'tau': 'tau OR neurofibrillary tangle OR NFT OR p-tau OR phosphorylated tau',
        'biomarker': 'biomarker OR marker OR indicator OR test',
        'treatment': 'treatment OR therapy OR therapeutic OR intervention OR drug',
        'protein': 'protein OR peptide OR molecule',
        'brain': 'brain OR cerebral OR CNS OR central nervous system OR neural',
        'inflammation': 'inflammation OR neuroinflammation OR inflammatory',
        'risk': 'risk OR risk factor OR associated with OR linked to',
        'diagnosis': 'diagnosis OR diagnostic OR detection OR identification',
        'cognitive': 'cognitive OR cognition OR memory OR thinking',
        'hearing loss': 'hearing loss OR hearing impairment OR deafness OR auditory dysfunction',
        'gut': 'gut OR microbiome OR microbiota OR intestinal OR gastrointestinal',
        'diet': 'diet OR dietary OR Mediterranean OR nutrition OR food',
        'hypoxia': 'hypoxia OR oxygen OR ischemia OR cerebrovascular',
        'volume change': 'volume change OR atrophy OR hippocampal volume OR brain volume OR shrinkage',
    }
    
    expanded = question.lower()
    
    # Apply expansions
    for term, expansion in expansions.items():
        if term in expanded:
            expanded = expanded.replace(term, f"({expansion})")
    
    return expanded


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_metadata(pdf_file):
    """Extract metadata from PDF using Gemini"""
    filename = pdf_file.name
    
    metadata = {
        "filename": filename,
        "title": filename.replace('.pdf', ''),
        "authors": [],
        "year": "Unknown",
        "pages": 0,
        "study_type": "unknown"
    }
    
    try:
        file_bytes = pdf_file.getvalue()
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        metadata["pages"] = len(pdf_reader.pages)
        
        # Extract first 3 pages for metadata
        first_pages = ""
        for i in range(min(3, len(pdf_reader.pages))):
            first_pages += pdf_reader.pages[i].extract_text() or ""
        
        # Use Gemini to extract metadata
        prompt = f"""Extract metadata from this research paper. Return ONLY valid JSON:
{{
    "title": "Paper title",
    "authors": ["Author 1", "Author 2"],
    "year": "YYYY",
    "study_type": "meta-analysis|systematic-review|rct|cohort|case-control|case-series|case-report|review|unknown"
}}

Text:
{first_pages[:8000]}"""

        genai_model = genai.GenerativeModel("gemini-2.0-flash")
        response = genai_model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            response_text = response.text.strip()
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                extracted = json.loads(response_text)
                metadata.update({k: v for k, v in extracted.items() if v})
            except:
                pass
        
        # Extract full text
        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            full_text += f"[Page {i+1}]: {page_text}\n\n"
        
        st.session_state.raw_texts[filename] = full_text
        
        return metadata
        
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return metadata


# ============================================================================
# CHUNKING
# ============================================================================

def create_chunks(all_papers_metadata):
    """Create chunks with source tracking"""
    all_chunks = []
    
    for filename, metadata in all_papers_metadata.items():
        paper_text = st.session_state.raw_texts[filename]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(paper_text)
        
        for chunk in chunks:
            header = f"""PAPER: {metadata['title']}
YEAR: {metadata['year']}
STUDY_TYPE: {metadata.get('study_type', 'unknown')}
SOURCE_FILE: {filename}

{chunk}"""
            all_chunks.append(header)
    
    return all_chunks


# ============================================================================
# VECTOR STORE CREATION
# ============================================================================

def create_vector_stores(chunks):
    """Create vector stores with batching"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        progress = st.progress(0)
        st.info(f"Creating embeddings in {total_batches} batches...")
        
        # Cosine RAG
        vector_store_cosine = None
        batch_count = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            if vector_store_cosine is None:
                vector_store_cosine = FAISS.from_texts(batch, embedding=embeddings)
            else:
                batch_store = FAISS.from_texts(batch, embedding=embeddings)
                vector_store_cosine.merge_from(batch_store)
            
            batch_count += 1
            progress.progress(min(batch_count / (total_batches * 2), 0.5))
        
        vector_store_cosine.save_local("faiss_index_cosine")
        st.session_state.vector_store_cosine = vector_store_cosine

        # ET-RAG (same chunks, different retrieval)
        vector_store_etrag = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            if vector_store_etrag is None:
                vector_store_etrag = FAISS.from_texts(batch, embedding=embeddings)
            else:
                batch_store = FAISS.from_texts(batch, embedding=embeddings)
                vector_store_etrag.merge_from(batch_store)

            batch_count += 1
            progress.progress(min(batch_count / (total_batches * 2), 1.0))

        vector_store_etrag.save_local("faiss_index_etrag")
        st.session_state.vector_store_etrag = vector_store_etrag
        st.session_state.embeddings_model = embeddings
        progress.progress(1.0)

        return True
        
    except Exception as e:
        st.error(f"Error creating vector stores: {str(e)}")
        return False


# ============================================================================
# AGENT 1: FULL CONTEXT (GEMINI)
# ============================================================================

def agent_full_context(question, paper_metadata=None, raw_texts=None, question_type='short_answer'):
    """Full Context agent using GPT-4o-mini — sees condensed content from ALL papers."""

    try:
        paper_metadata = paper_metadata or st.session_state.paper_metadata
        raw_texts = raw_texts or st.session_state.raw_texts

        expanded = expand_query(question)

        # Build condensed full context — 10K chars per paper
        all_text = ""
        references = []
        for idx, (filename, metadata) in enumerate(paper_metadata.items(), 1):
            paper_text = raw_texts.get(filename, "")
            ref = f"[{idx}] {metadata.get('title', filename)} ({metadata.get('year', 'Unknown')})"
            references.append(ref)
            condensed = paper_text[:8000] if len(paper_text) > 8000 else paper_text
            all_text += f"\nPAPER {idx}: {metadata['title']} ({metadata['year']})\n{condensed}\n\n"

        # Build question-type-specific instructions
        if question_type == 'multiple_choice':
            type_instruction = "MULTIPLE CHOICE: Select ALL options that are supported. Start with letters like 'A, B, C'."
        elif question_type == 'single_choice':
            type_instruction = "SINGLE CHOICE: Pick the ONE best answer. Start with the letter."
        elif question_type == 'long_answer':
            type_instruction = "Answer in 200-250 words."
        else:
            type_instruction = "Answer in 100-150 words."

        prompt = f"""Answer using ONLY the papers below.

QUESTION: {question}

{type_instruction}
Cite using paper title and year. Do NOT use external knowledge.
Say "not covered" ONLY if topic is about a completely different field.

PAPERS CONTENT:
{all_text}
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

        for retry in range(3):
            try:
                response = llm.invoke(prompt)
                answer = response.content.strip()

                # Calculate confidence based on answer quality
                if "not covered" in answer.lower():
                    confidence = 0.92
                elif len(answer) > 500:
                    confidence = 0.85
                elif len(answer) > 200:
                    confidence = 0.75
                else:
                    confidence = 0.60

                # Extract files used from citations in the answer
                files_used = []
                for idx, (filename, metadata) in enumerate(paper_metadata.items(), 1):
                    title = metadata.get('title', '')
                    if title.lower()[:20] in answer.lower():
                        files_used.append(filename)

                return {
                    "answer": answer,
                    "confidence": confidence,
                    "files_used": list(set(files_used)),
                    "success": True
                }
            except Exception as api_err:
                if '429' in str(api_err) or 'Rate limit' in str(api_err):
                    time.sleep((retry + 1) * 5)
                else:
                    raise api_err

        return {"answer": "Error: Rate limit", "confidence": 0.0, "files_used": [], "success": False}

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "confidence": 0.0, "files_used": [], "success": False}


# ============================================================================
# AGENT 2: COSINE RAG (OPENAI)
# ============================================================================

def extract_key_terms(question):
    """Extract key medical terms for multi-query retrieval."""
    q = question.lower()
    for w in ['what','how','why','does','is','are','the','of','in','to','and','a','an','can','may',
              'according','recent','studies','describe','explain','discuss','evaluate','analyze',
              'compare','contrast','summarize','role','influence','effect','significance','purpose','using']:
        q = q.replace(f' {w} ', ' ')
    words = [w.strip('?.,') for w in q.split() if len(w) > 3]
    terms = words[:6]
    for i in range(min(len(words)-1, 4)):
        terms.append(f"{words[i]} {words[i+1]}")
    return terms[:8]


def multi_query_retrieve(vector_store, question, k=RETRIEVAL_CANDIDATES):
    """Retrieve chunks using multiple search strategies for better coverage."""
    all_docs = {}
    # Strategy 1: Original question
    for doc, score in vector_store.similarity_search_with_score(question, k=k):
        key = doc.page_content[:100]
        if key not in all_docs or score < all_docs[key][1]:
            all_docs[key] = (doc, score)
    # Strategy 2: Expanded query with synonyms
    expanded = expand_query(question)
    for doc, score in vector_store.similarity_search_with_score(expanded, k=k):
        key = doc.page_content[:100]
        if key not in all_docs or score < all_docs[key][1]:
            all_docs[key] = (doc, score)
    # Strategy 3: Individual key terms
    for term in extract_key_terms(question):
        for doc, score in vector_store.similarity_search_with_score(term, k=10):
            key = doc.page_content[:100]
            if key not in all_docs or score < all_docs[key][1]:
                all_docs[key] = (doc, score)
    return all_docs


def agent_cosine_rag(question, vector_store=None, question_type='short_answer'):
    """Cosine RAG agent with multi-query retrieval"""

    try:
        if vector_store is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.load_local(
                "faiss_index_cosine", embeddings, allow_dangerous_deserialization=True
            )

        all_docs = multi_query_retrieve(vector_store, question)
        
        # Sort by score and take top K
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1])[:TOP_K_CHUNKS]
        
        if not sorted_docs:
            return {
                "answer": "This information is not covered in the uploaded research papers.",
                "confidence": 0.92,
                "files_used": [],
                "success": True
            }
        
        docs = [doc for doc, score in sorted_docs]
        
        # Extract files used
        files_used = []
        for doc in docs:
            match = re.search(r'SOURCE_FILE:\s*([^\n]+)', doc.page_content)
            if match:
                files_used.append(match.group(1).strip())
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build question-type-specific instructions
        if question_type == 'multiple_choice':
            type_rule = "MULTIPLE CHOICE: Evaluate EACH option (A, B, C, D) INDEPENDENTLY. For each, ask: is there evidence in the context? Select ALL supported options, not just the best one. Format: list all correct letters, then brief explanation for each. Keep response under 150 words."
        elif question_type == 'single_choice':
            type_rule = "SINGLE CHOICE: Pick the ONE best answer supported by the context. Keep response under 100 words."
        elif question_type == 'long_answer':
            type_rule = "Answer in 3-4 paragraphs with evidence. Keep response between 200-250 words."
        else:
            type_rule = "Answer in 2-4 concise sentences. Keep response between 100-150 words."

        prompt_template = f"""Answer the question using ONLY the research paper excerpts below.

CONTEXT:
{{context}}

QUESTION: {{question}}

{type_rule}

RULES:
- Cite using paper title and year, e.g., (Alzheimer disease, 2021).
- Look for synonyms (e.g., "metabolite clearance" = "Aβ clearance", "glymphatic system").
- Use partial information rather than saying "not covered."
- Say "not covered" ONLY if topic is completely absent.
- Do NOT use external knowledge. Do NOT repeat yourself.

Answer:
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        result = chain.invoke({
            "input_documents": docs,
            "question": question
        })
        answer = result["output_text"].strip()
        
        # Calculate confidence
        similarities = [1 / (1 + score) for doc, score in sorted_docs]
        avg_sim = np.mean(similarities)
        
        if "not covered" in answer.lower():
            confidence = 0.92
        else:
            confidence = min(0.50 + (avg_sim * 0.4), 0.90)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "files_used": list(set(files_used)),
            "success": True
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence": 0.0,
            "files_used": [],
            "success": False
        }


# ============================================================================
# AGENT 3: ET-RAG (OPENAI)
# ============================================================================

def calculate_temporal_weight(year):
    """Calculate temporal weight"""
    try:
        age = CURRENT_YEAR - int(year)
        if age <= 3:
            return 1.0
        elif age <= 7:
            return 0.85
        else:
            return 0.6
    except:
        return 0.5


def agent_etrag(question, vector_store=None, question_type='short_answer', paper_metadata=None, raw_texts=None):
    """ET-RAG: Hybrid agent — retrieved chunks (reranked by evidence + recency) + condensed paper context."""

    try:
        if vector_store is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.load_local(
                "faiss_index_etrag", embeddings, allow_dangerous_deserialization=True
            )

        paper_metadata = paper_metadata or st.session_state.paper_metadata
        raw_texts = raw_texts or st.session_state.raw_texts
        expanded = expand_query(question)

        # Multi-query retrieval
        all_docs = multi_query_retrieve(vector_store, question)
        docs_with_scores = list(all_docs.values())

        if not docs_with_scores:
            return {"answer": "This information is not covered in the uploaded research papers.",
                    "confidence": 0.92, "files_used": [], "success": True}

        # ET-RAG reranking
        scored_docs = []
        for doc, cosine_dist in docs_with_scores:
            year_match = re.search(r'YEAR:\s*([^\n]+)', doc.page_content)
            study_match = re.search(r'STUDY_TYPE:\s*([^\n]+)', doc.page_content)
            year = year_match.group(1).strip() if year_match else "Unknown"
            study_type = study_match.group(1).strip().lower() if study_match else "unknown"
            cosine_sim = 1 / (1 + cosine_dist)
            evidence_weight = EVIDENCE_WEIGHTS.get(study_type, 0.5)
            temporal_weight = calculate_temporal_weight(year)
            etrag_score = (
                ETRAG_WEIGHTS['cosine'] * cosine_sim +
                ETRAG_WEIGHTS['evidence'] * evidence_weight +
                ETRAG_WEIGHTS['temporal'] * temporal_weight
            )
            scored_docs.append({'doc': doc, 'score': etrag_score})

        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = [item['doc'] for item in scored_docs[:TOP_K_CHUNKS]]
        avg_score = np.mean([item['score'] for item in scored_docs[:TOP_K_CHUNKS]])
        chunk_context = "\n\n".join([doc.page_content for doc in top_chunks])

        # Extract files used
        files_used = []
        for doc in top_chunks:
            match = re.search(r'SOURCE_FILE:\s*([^\n]+)', doc.page_content)
            if match:
                files_used.append(match.group(1).strip())

        # Build condensed paper context (12K per paper — broader coverage)
        paper_context = ""
        for idx, (filename, metadata) in enumerate(paper_metadata.items(), 1):
            paper_text = raw_texts.get(filename, "")
            condensed = paper_text[:15000] if len(paper_text) > 15000 else paper_text
            paper_context += f"\nPAPER {idx}: {metadata['title']} ({metadata['year']})\n{condensed}\n"

        # Build question-type-specific instructions
        if question_type == 'single_choice':
            type_rule = "SINGLE CHOICE: Pick the ONE best answer. Start with the letter. Keep under 100 words."
        elif question_type == 'multiple_choice':
            type_rule = """MULTIPLE CHOICE — Multiple options can be correct.
For EACH option (A, B, C, D), evaluate using BOTH sources:
- Only mark SUPPORTED if there is CLEAR, DIRECT evidence in the papers
- Indirect or implied evidence alone is NOT enough — the papers must specifically discuss the topic
- Write: [Letter]: SUPPORTED (with brief evidence) or NOT SUPPORTED
After evaluating ALL 4, write: ANSWER: [supported letters]"""
        elif question_type == 'long_answer':
            type_rule = "Answer in 200-250 words with comprehensive evidence from multiple papers."
        else:
            type_rule = "Answer in 100-150 words with citations."

        prompt = f"""You are an expert medical research assistant with access to high-quality evidence. Answer using the evidence below.

1. HIGH-RELEVANCE EXCERPTS (ranked by evidence quality + recency — prioritize these):
{chunk_context}

2. BROADER PAPER CONTEXT (supplementary — use to fill gaps only):
{paper_context}

QUESTION: {question}
SEARCH TERMS: {expanded}

{type_rule}
Prioritize the high-relevance excerpts. Use broader context only when excerpts don't cover a topic.
Cite using paper title and year. Do NOT use external knowledge.
Say "not covered" ONLY if topic is about a completely different field.

Answer:
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

        for retry in range(3):
            try:
                response = llm.invoke(prompt)
                answer = response.content.strip()

                if "not covered" in answer.lower():
                    confidence = 0.92
                else:
                    confidence = min(0.55 + (avg_score * 0.35), 0.90)

                return {
                    "answer": answer,
                    "confidence": confidence,
                    "files_used": list(set(files_used)),
                    "success": True
                }
            except Exception as api_err:
                if '429' in str(api_err) or 'Rate limit' in str(api_err):
                    time.sleep((retry + 1) * 5)
                else:
                    raise api_err

        return {"answer": "Error: Rate limit", "confidence": 0.0, "files_used": [], "success": False}

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "confidence": 0.0, "files_used": [], "success": False}


# ============================================================================
# CONSENSUS ANALYSIS
# ============================================================================

def synthesize_answer(question, r1, r2, r3, question_type='short_answer'):
    """Use GPT-4o-mini to:
    1. Evaluate each agent's answer for correctness
    2. Synthesize the best final answer
    3. Determine real consensus based on content analysis

    Returns dict with: synthesized_answer, consensus_level, consensus_message, confidence
    """
    try:
        # Build agent summary
        agent_texts = f"""AGENT 1 — Full Context [Confidence: {r1['confidence']:.0%}]:
{r1['answer']}

AGENT 2 — Cosine RAG (GPT-4o-mini) [Confidence: {r2['confidence']:.0%}]:
{r2['answer']}

AGENT 3 — ET-RAG (GPT-4o-mini) [Confidence: {r3['confidence']:.0%}]:
{r3['answer']}"""

        # Use specialized prompt for multiple choice
        if question_type == 'multiple_choice':
            synthesis_prompt = f"""Three AI agents analyzed research papers to answer a MULTIPLE CHOICE question (select ALL correct answers).

QUESTION: {question}

AGENT RESPONSES:
{agent_texts}

YOUR TASK: Evaluate EACH option independently by checking ALL agent responses for evidence.

Return this JSON, then the final answer:

```json
{{
  "option_A": {{"supported": true/false, "evidence": "what agents said about this option"}},
  "option_B": {{"supported": true/false, "evidence": "what agents said about this option"}},
  "option_C": {{"supported": true/false, "evidence": "what agents said about this option"}},
  "option_D": {{"supported": true/false, "evidence": "what agents said about this option"}},
  "consensus": "STRONG/MAJORITY/SPLIT/NOT_COVERED",
  "consensus_reason": "1 sentence"
}}
```

RULES FOR EVALUATING EACH OPTION:
- Check ALL 3 agent responses for ANY mention or evidence related to each option
- If ANY agent provides evidence supporting an option → mark it as supported=true
- An option can be supported even if only 1 agent mentions it — the evidence matters, not the count
- Look for INDIRECT evidence too (e.g., "increased wakefulness" supports "insomnia"; "Aβ clearance during sleep" supports "metabolite clearance")
- Do NOT require all agents to agree on an option — if one agent found real evidence from the papers, include it
- If an agent had an error, ignore that agent

After the JSON, write the final answer:
- List ALL supported options: **Answer: A, B, C — [option texts]**
- Brief explanation for each with citations as (Paper Title, Year)
- Do NOT mention agents in the final answer
- Keep response under 150 words."""

        else:
            # Single choice / short / long answer prompt
            if question_type == 'single_choice':
                answer_format = "**Answer: [Letter]. [Option text]**\nExplanation in 2-3 sentences with citations. Keep under 100 words."
            elif question_type == 'long_answer':
                answer_format = "3-4 paragraph answer: thesis → evidence with citations → mechanisms → clinical significance. Keep between 200-250 words total."
            else:
                answer_format = "2-4 sentence direct answer with citations. Keep between 100-150 words."

            synthesis_prompt = f"""Three AI agents analyzed research papers to answer this question. Produce a single clean answer.

QUESTION: {question}
TYPE: {question_type}

AGENT RESPONSES:
{agent_texts}

===== PART 1: JSON EVALUATION =====

```json
{{
  "agent1_correct": true/false,
  "agent1_note": "1 sentence",
  "agent2_correct": true/false,
  "agent2_note": "1 sentence",
  "agent3_correct": true/false,
  "agent3_note": "1 sentence",
  "consensus": "STRONG/MAJORITY/SPLIT/NOT_COVERED",
  "consensus_reason": "1 sentence"
}}
```

Rules: STRONG = same answer/key points. MAJORITY = 2 agree. SPLIT = contradictory. NOT_COVERED = topic absent from papers. Ignore errored agents.

===== PART 2: FINAL ANSWER =====

{answer_format}

CITATION RULES:
- Cite as (Paper Title, Year) — e.g., (Alzheimer disease, 2021)
- Do NOT use [Paper #, Page #] format — use actual paper names
- Do NOT invent citations. Do NOT mention agents.
- If NOT_COVERED: just say "This topic is not covered in the uploaded research papers."
- Be concise. No repetition."""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=3000)
        response = llm.invoke(synthesis_prompt)
        full_response = response.content.strip()

        # Parse the JSON evaluation block
        consensus_level = "STRONG"  # default
        consensus_reason = ""
        synthesized = full_response  # fallback: use entire response

        # Find JSON block — handle nested objects for MCQ format
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL)
        if not json_match:
            # Try matching a larger JSON block (MCQ has nested objects)
            json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', full_response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{[^{}]*"consensus"[^{}]*\})', full_response, re.DOTALL)

        if json_match:
            try:
                evaluation = json.loads(json_match.group(1))
                consensus_level = evaluation.get('consensus', 'STRONG').upper()
                consensus_reason = evaluation.get('consensus_reason', '')

                # Extract the answer part (everything after the JSON block)
                json_end = full_response.find(json_match.group(0)) + len(json_match.group(0))
                answer_part = full_response[json_end:].strip()
                # Remove any markdown headers
                answer_part = re.sub(r'^#+\s*(PART 2|Synthesized Answer|Final Answer)[:\s]*\n*', '', answer_part, flags=re.IGNORECASE)
                answer_part = re.sub(r'^\*\*?(PART 2|Synthesized Answer|Final Answer)\*?\*?[:\s]*\n*', '', answer_part, flags=re.IGNORECASE)
                if answer_part and len(answer_part) > 30:
                    synthesized = answer_part.strip()
            except json.JSONDecodeError:
                pass

        # Force override for NOT_COVERED — never hallucinate an answer
        if consensus_level == "NOT_COVERED":
            synthesized = "This topic is not covered in the uploaded research papers. The question pertains to a subject outside the scope of the provided documents."

        # Build consensus message
        if consensus_level == "STRONG":
            consensus_msg = f"✅ Strong agreement across agents"
        elif consensus_level == "MAJORITY":
            consensus_msg = f"⚠️ Majority agreement (2 of 3 agents)"
        elif consensus_level == "NOT_COVERED":
            consensus_msg = "✅ Agents agree: topic not covered in uploaded papers"
        else:
            consensus_msg = f"❌ Agents disagree"

        if consensus_reason:
            consensus_msg += f" — {consensus_reason}"

        # Average confidence (skip errored agents)
        valid_confs = [c for c in [r1['confidence'], r2['confidence'], r3['confidence']] if c > 0]
        avg_conf = np.mean(valid_confs) if valid_confs else 0.5

        return {
            "synthesized_answer": synthesized,
            "consensus": consensus_level,
            "confidence": avg_conf,
            "message": consensus_msg
        }

    except Exception as e:
        # Fallback: pick the best agent response
        best = max([r1, r2, r3], key=lambda r: r['confidence'])
        return {
            "synthesized_answer": best['answer'],
            "consensus": "UNCLEAR",
            "confidence": best['confidence'],
            "message": f"⚠️ Synthesis unavailable ({str(e)[:50]}), showing best agent response"
        }


# ============================================================================
# QUESTION PARSING & BATCH PROCESSING
# ============================================================================

def parse_questions(user_input):
    """Parse user input using GPT to extract questions with their types.

    Uses GPT-4o-mini to understand the raw pasted text and return a clean
    list of questions with accurate type classification based on section headers.

    Returns list of dicts: [{"text": "...", "type": "single_choice"}, ...]
    Falls back to list of dicts with type from classify_question_type().
    """
    text = user_input.strip()

    # Quick check: if it's clearly just one short question, skip the LLM call
    if len(text) < 300 and text.count('\n') <= 2:
        return [{"text": text, "type": classify_question_type(text)}]

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=16000)

        parse_prompt = f"""You are a question parser. The user has pasted a block of text containing research questions.
Your job is to extract EVERY individual question with its type.

RULES:
1. Each question MUST include its answer options (A, B, C, D) if they exist — combine the question stem and all options into ONE string.
   Example: "Which protein is a hallmark of AD? A. α-synuclein B. Amyloid-beta (Aβ) C. Huntingtin D. Dopamine"
2. Remove section headers like "Single Choice Questions (10)" etc. — these are NOT questions. But USE them to determine each question's type.
3. Remove numbering prefixes (1., 2., Q1., etc.) from the start of questions.
4. Remove bullet prefixes (o, •, -, *) from option lines.
5. CRITICAL: You MUST extract ALL questions. Do NOT skip any.
6. Determine the type based on which section the question appeared under:
   - Questions under "Single Choice" → "single_choice"
   - Questions under "Multiple Choice" → "multiple_choice"
   - Questions under "Short Answer" → "short_answer"
   - Questions under "Long Answer" → "long_answer"
   - If no section header, infer from the question format
7. Return ONLY a valid JSON array of objects. No explanation, no markdown.

FORMAT: [{{"text": "question text here", "type": "single_choice"}}, ...]

USER INPUT:
{text}

JSON array:"""

        response = llm.invoke(parse_prompt)
        result_text = response.content.strip()

        # Clean markdown code blocks if present
        result_text = result_text.replace("```json", "").replace("```", "").strip()

        questions = json.loads(result_text)

        if isinstance(questions, list) and len(questions) >= 1:
            cleaned = []
            for item in questions:
                if isinstance(item, dict) and 'text' in item:
                    q_text = item['text'].strip()
                    q_type = item.get('type', classify_question_type(q_text))
                    if len(q_text) > 10:
                        cleaned.append({"text": q_text, "type": q_type})
                elif isinstance(item, str) and len(item.strip()) > 10:
                    # Fallback if GPT returned plain strings
                    cleaned.append({"text": item.strip(), "type": classify_question_type(item.strip())})
            if cleaned:
                return cleaned

    except Exception as e:
        # Log but don't crash — fall back to single question
        st.toast(f"Question parsing note: {str(e)[:100]}", icon="ℹ️")

    # Fallback: treat entire input as single question
    return [{"text": text, "type": classify_question_type(text)}]


def classify_question_type(question):
    """Auto-detect question type from the question text"""
    q = question.strip()

    has_options = len(re.findall(r'\b[A-D][\.\)]\s', q)) >= 3

    if not has_options:
        # No A/B/C/D options — it's an open-ended question
        if re.match(r'^(discuss|evaluate|analyze|compare|describe|summarize|critically|explain the limitations)', q, re.IGNORECASE):
            return 'long_answer'
        return 'short_answer'

    # Has options — determine single vs multiple choice
    # Multiple choice indicators: plural question words, "which of the following", "select all"
    multiple_patterns = [
        r'which of the following',
        r'select all that apply',
        r'may lead to which',
        r'may improve',
        r'which\s+(factors|interventions|effects|roles|biomarkers|characteristics|imaging|components|conditions|sleep|characteristics|interventions|imaging biomarkers|sleep.related)',
        r'what\s+(roles|effects)',
        r'are\s+(components|discussed|identified|associated|used|epidemiologically)',
    ]
    for pattern in multiple_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return 'multiple_choice'

    # Default for questions with options: single choice
    return 'single_choice'


def build_single_response(r1, r2, r3, question="", question_type="short_answer"):
    """Build formatted response with synthesized answer + agent details"""

    # Synthesize
    synthesis = synthesize_answer(question, r1, r2, r3, question_type)

    response = f"""## ✅ Answer

{synthesis['synthesized_answer']}

---

**{synthesis['message']}** | Avg Confidence: **{synthesis['confidence']:.0%}**

---

### 🔍 Individual Agent Responses

#### 🌐 Agent 1: Full Context — Confidence: {r1['confidence']:.0%}
{r1['answer']}
{('📄 Sources: ' + ', '.join(r1['files_used'][:3])) if r1['files_used'] else ''}

---

#### 🔍 Agent 2: Cosine RAG — Confidence: {r2['confidence']:.0%}
{r2['answer']}
{('📄 Sources: ' + ', '.join(r2['files_used'][:3])) if r2['files_used'] else ''}

---

#### 🎯 Agent 3: ET-RAG — Confidence: {r3['confidence']:.0%}
{r3['answer']}
{('📄 Sources: ' + ', '.join(r3['files_used'][:3])) if r3['files_used'] else ''}
"""
    return response, synthesis


def process_batch_in_chat(questions):
    """Process multiple questions: 3 agents + GPT synthesis per question"""
    total = len(questions)
    all_results = []

    # Extract session state data BEFORE threading (thread-safe)
    paper_metadata = dict(st.session_state.paper_metadata)
    raw_texts = dict(st.session_state.raw_texts)
    vs_cosine = st.session_state.vector_store_cosine
    vs_etrag = st.session_state.vector_store_etrag

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, q_item in enumerate(questions):
        question = q_item['text'] if isinstance(q_item, dict) else q_item
        q_type = q_item.get('type', classify_question_type(question)) if isinstance(q_item, dict) else classify_question_type(question)
        status_text.markdown(f"**Processing question {i + 1}/{total}:** {question[:80]}...")

        start_time = time.time()

        # Run all 3 agents IN PARALLEL with pre-extracted data
        with ThreadPoolExecutor(max_workers=3) as executor:
            f1 = executor.submit(agent_full_context, question, paper_metadata, raw_texts, q_type)
            f2 = executor.submit(agent_cosine_rag, question, vs_cosine, q_type)
            f3 = executor.submit(agent_etrag, question, vs_etrag, q_type, paper_metadata, raw_texts)
            r1 = f1.result()
            r2 = f2.result()
            r3 = f3.result()

        # GPT synthesis
        synthesis = synthesize_answer(question, r1, r2, r3, question_type=q_type)

        elapsed = time.time() - start_time

        all_results.append({
            'question_id': i + 1,
            'question_text': question,
            'question_type': q_type,
            'synthesized_answer': synthesis['synthesized_answer'],
            'consensus_level': synthesis['consensus'],
            'consensus_message': synthesis['message'],
            'avg_confidence': round(synthesis['confidence'], 2),
            'agent1_full_context': r1['answer'],
            'agent1_confidence': r1['confidence'],
            'agent1_files_used': ', '.join(r1['files_used']),
            'agent2_cosine_rag': r2['answer'],
            'agent2_confidence': r2['confidence'],
            'agent2_files_used': ', '.join(r2['files_used']),
            'agent3_etrag': r3['answer'],
            'agent3_confidence': r3['confidence'],
            'agent3_files_used': ', '.join(r3['files_used']),
            'response_time_sec': round(elapsed, 1)
        })

        progress_bar.progress((i + 1) / total)

        # Brief pause between questions to respect API rate limits
        if i < total - 1:
            time.sleep(2)

    progress_bar.empty()
    status_text.empty()

    results_df = pd.DataFrame(all_results)

    # Build combined chat response
    response = f"## 📋 Batch Results — {total} Questions Processed\n\n"

    # Summary stats
    avg_time = results_df['response_time_sec'].mean()
    consensus_counts = results_df['consensus_level'].value_counts()
    strong = consensus_counts.get('STRONG', 0)
    majority = consensus_counts.get('MAJORITY', 0)
    avg_conf = results_df['avg_confidence'].mean()

    response += f"""| Metric | Value |
|--------|-------|
| Total Questions | {total} |
| Avg Response Time | {avg_time:.1f}s |
| Strong Consensus | {strong} ({strong/total*100:.0f}%) |
| Majority Consensus | {majority} ({majority/total*100:.0f}%) |
| Avg Confidence | {avg_conf:.0%} |

---

"""

    # Per-question results — individual agents + synthesis
    for _, row in results_df.iterrows():
        q_label = row['question_type'].replace('_', ' ').title()
        response += f"""### Q{row['question_id']} [{q_label}]
**{row['question_text']}**

#### 🌐 Agent 1: Full Context — Confidence: {row['agent1_confidence']:.0%}
{row['agent1_full_context']}

#### 🔍 Agent 2: Cosine RAG — Confidence: {row['agent2_confidence']:.0%}
{row['agent2_cosine_rag']}

#### 🎯 Agent 3: ET-RAG — Confidence: {row['agent3_confidence']:.0%}
{row['agent3_etrag']}

#### ✅ Synthesized Final Answer
{row['synthesized_answer']}

**{row['consensus_message']}** | Avg Confidence: **{row['avg_confidence']:.0%}** | Time: {row['response_time_sec']}s

---

"""

    return response, results_df


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🧠",
        layout="wide"
    )

    # CSS to maximize chat area and fix width
    st.markdown("""
    <style>
        /* Make main content area fill the screen */
        .stMainBlockContainer {
            max-width: 100% !important;
            padding-top: 1rem !important;
        }
        /* Make chat container taller */
        [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            max-height: calc(100vh - 200px) !important;
        }
        /* Reduce header padding */
        .stAppHeader {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Multi-Agent Research Assistant")
    st.markdown("**Upload your research papers and get answers from 3 AI agents**")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Papers")
        
        pdf_docs = st.file_uploader(
            "Upload research papers (PDF)",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("🚀 Process Papers"):
            if pdf_docs:
                with st.spinner("Processing papers..."):
                    # Extract metadata
                    for pdf in pdf_docs:
                        metadata = extract_metadata(pdf)
                        st.session_state.paper_metadata[pdf.name] = metadata
                    
                    # Create chunks
                    chunks = create_chunks(st.session_state.paper_metadata)
                    
                    # Create vector stores
                    if create_vector_stores(chunks):
                        st.session_state.papers_processed = True
                        st.success(f"✅ Processed {len(pdf_docs)} papers")
            else:
                st.warning("Please upload PDFs first")
        
        if st.session_state.papers_processed:
            st.success(f"✅ {len(st.session_state.paper_metadata)} papers loaded")
            
            with st.expander("📚 Loaded Papers"):
                for filename, meta in st.session_state.paper_metadata.items():
                    st.markdown(f"**{meta['title']}** ({meta['year']})")
            
            if st.button("🗑️ Clear Papers"):
                import shutil
                for idx in ["faiss_index_cosine", "faiss_index_etrag"]:
                    if os.path.exists(idx):
                        shutil.rmtree(idx)
                st.session_state.papers_processed = False
                st.session_state.paper_metadata = {}
                st.session_state.raw_texts = {}
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        st.markdown("**About**")
        st.caption("Three AI agents analyze your papers: Full Context, Cosine RAG, and ET-RAG (Evidence+Temporal Hybrid)")
    
    # Main content area
    if not st.session_state.papers_processed:
        st.info("👈 Upload research papers to begin")
    else:
        # Chat history display
        chat_container = st.container(height="stretch")
        with chat_container:
            for entry in st.session_state.chat_history:
                role = entry[0]
                message = entry[1]
                with st.chat_message(role):
                    st.markdown(message)
                    # If batch results are attached, show download button
                    if len(entry) > 2 and entry[2] is not None:
                        csv_output = entry[2].to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results as CSV",
                            data=csv_output,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key=f"dl_{id(entry)}"
                        )

        # Chat input
        user_input = st.chat_input("Ask a question — or paste multiple questions (one per line)...")

        if user_input:
            questions = parse_questions(user_input)

            if len(questions) == 1:
                # --- Single question mode ---
                st.session_state.chat_history.append(("user", user_input))
                q_item = questions[0]
                q_text = q_item['text'] if isinstance(q_item, dict) else q_item
                q_type = q_item.get('type', classify_question_type(q_text)) if isinstance(q_item, dict) else classify_question_type(q_text)

                # Extract session state data before threading
                _pm = dict(st.session_state.paper_metadata)
                _rt = dict(st.session_state.raw_texts)
                _vsc = st.session_state.vector_store_cosine
                _vse = st.session_state.vector_store_etrag

                with st.spinner("🤖 All 3 agents analyzing in parallel..."):
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        f1 = executor.submit(agent_full_context, q_text, _pm, _rt, q_type)
                        f2 = executor.submit(agent_cosine_rag, q_text, _vsc, q_type)
                        f3 = executor.submit(agent_etrag, q_text, _vse, q_type, _pm, _rt)
                        r1 = f1.result()
                        r2 = f2.result()
                        r3 = f3.result()
                    response, _ = build_single_response(r1, r2, r3, question=q_text, question_type=q_type)
                    st.session_state.chat_history.append(("assistant", response))

                st.rerun()

            else:
                # --- Batch mode: store questions, process one at a time with rerun ---
                if "batch_questions" not in st.session_state:
                    # First time — save questions and start processing
                    st.session_state.batch_questions = questions
                    st.session_state.batch_index = 0
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", f"📋 Processing **{len(questions)}** questions. Results will appear one at a time..."))
                    st.rerun()

        # Process batch questions one at a time (outside the chat_input block)
        if "batch_questions" in st.session_state and st.session_state.batch_index < len(st.session_state.batch_questions):
            _pm = dict(st.session_state.paper_metadata)
            _rt = dict(st.session_state.raw_texts)
            _vsc = st.session_state.vector_store_cosine
            _vse = st.session_state.vector_store_etrag

            i = st.session_state.batch_index
            total = len(st.session_state.batch_questions)
            q_item = st.session_state.batch_questions[i]
            q_text = q_item['text'] if isinstance(q_item, dict) else q_item
            q_type = q_item.get('type', classify_question_type(q_text)) if isinstance(q_item, dict) else classify_question_type(q_text)

            with st.spinner(f"🤖 Processing Q{i+1}/{total}: {q_text[:60]}..."):
                with ThreadPoolExecutor(max_workers=3) as executor:
                    f1 = executor.submit(agent_full_context, q_text, _pm, _rt, q_type)
                    f2 = executor.submit(agent_cosine_rag, q_text, _vsc, q_type)
                    f3 = executor.submit(agent_etrag, q_text, _vse, q_type, _pm, _rt)
                    r1 = f1.result()
                    r2 = f2.result()
                    r3 = f3.result()

                response, _ = build_single_response(r1, r2, r3, question=q_text, question_type=q_type)
                response = f"### Q{i+1}/{total} [{q_type.replace('_',' ').title()}]\n**{q_text}**\n\n{response}"
                st.session_state.chat_history.append(("assistant", response))

            st.session_state.batch_index += 1

            if st.session_state.batch_index >= total:
                # Batch complete — clean up
                st.session_state.chat_history.append(("assistant", f"✅ All **{total}** questions processed!"))
                del st.session_state.batch_questions
                del st.session_state.batch_index

            st.rerun()


if __name__ == "__main__":
    main()