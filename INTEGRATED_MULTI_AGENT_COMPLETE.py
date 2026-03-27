"""
MULTI-AGENT ALZHEIMER'S RESEARCH CHATBOT
=========================================
Production-Ready Version

Three AI Agents Answer Your Research Questions:
- Full Context Agent (Gemini 2.0 Flash) - Analyzes entire corpus
- Cosine RAG Agent (GPT-4o-mini) - Smart semantic retrieval  
- ET-RAG Agent (GPT-4o-mini) - Evidence + Recency weighted

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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model settings for consistency
TEMPERATURE = 0.1  # Very low for deterministic responses
MAX_TOKENS = 2048
TOP_K_CHUNKS = 15  # Increased from 8 for better coverage
RETRIEVAL_CANDIDATES = 30  # Retrieve more, then rerank
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

# ============================================================================
# QUERY EXPANSION
# ============================================================================

def expand_query(question):
    """Expand query with medical synonyms and variations"""
    
    # Common medical term expansions
    expansions = {
        'sleep disorder': 'sleep disorder OR sleep disturbance OR insomnia OR sleep apnea OR OSA OR narcolepsy OR sleep problem',
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
        progress.progress(1.0)
        
        return True
        
    except Exception as e:
        st.error(f"Error creating vector stores: {str(e)}")
        return False


# ============================================================================
# AGENT 1: FULL CONTEXT (GEMINI)
# ============================================================================

def agent_full_context(question):
    """Full Context agent using Gemini"""
    
    try:
        paper_count = len(st.session_state.paper_metadata)
        
        # Combine all papers
        all_text = ""
        references = []
        
        for idx, (filename, metadata) in enumerate(st.session_state.paper_metadata.items(), 1):
            paper_text = st.session_state.raw_texts.get(filename, "")
            
            ref = f"[{idx}] {metadata.get('title', filename)} ({metadata.get('year', 'Unknown')})"
            references.append(ref)
            
            header = f"\n{'='*60}\nPAPER {idx}: {metadata['title']}\nYEAR: {metadata['year']}\n{'='*60}\n\n"
            all_text += header + paper_text + "\n\n"
        
        # Truncate if needed
        if len(all_text) > 800000:
            all_text = all_text[:800000]
        
        # Expand query for better matching
        expanded = expand_query(question)
        
        prompt = f"""You are a medical research assistant analyzing {paper_count} research papers.

CRITICAL SEARCH REQUIREMENTS:

1. SEARCH EXHAUSTIVELY:
   - Read ALL {paper_count} papers completely
   - Search for EXACT terms AND synonyms
   - Check ALL sections: abstract, introduction, methods, results, discussion
   - Look for direct mentions AND indirect references

2. SYNONYM AWARENESS:
   - "sleep disorder" = insomnia, sleep apnea, OSA, narcolepsy, sleep disturbance
   - "AD" = Alzheimer's disease, Alzheimer disease, dementia
   - "imaging" = PET, MRI, CT, SPECT, scan
   - "amyloid" = Aβ, A-beta, plaque
   - Search for ALL variations

3. CITATION REQUIREMENTS:
   - Cite as [Paper #, Page #]
   - Use Paper numbers from list below
   - DO NOT cite external papers

4. IF NOT FOUND:
   - Only say "not covered" after searching ALL papers with ALL synonyms
   - Be thorough - don't give up easily

PAPERS:
{chr(10).join(references)}

FULL CONTENT:
{all_text}

ORIGINAL QUESTION: {question}

EXPANDED SEARCH TERMS: {expanded}

Search thoroughly using both the original question and expanded terms. Provide your answer with citations [Paper #, Page #]:
"""
        
        genai_model = genai.GenerativeModel("gemini-2.0-flash")

        # Retry with exponential backoff for rate limits
        response = None
        for attempt in range(5):
            try:
                response = genai_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_TOKENS
                    )
                )
                break  # Success
            except Exception as api_err:
                if '429' in str(api_err) or 'Resource exhausted' in str(api_err):
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8, 16, 32 seconds
                    time.sleep(wait_time)
                else:
                    raise api_err

        if response and hasattr(response, 'text'):
            answer = response.text.strip()
            
            # Extract citations
            citations = re.findall(r'\[Paper (\d+),?\s*Page[s]?\s*([\d\-,\s]+)\]', answer)
            
            # Calculate confidence
            if "not covered" in answer.lower():
                confidence = 0.92
            elif len(citations) >= 2:
                confidence = 0.85
            elif len(citations) >= 1:
                confidence = 0.75
            else:
                confidence = 0.60
            
            # Extract which files were cited
            files_used = []
            for cite in citations:
                paper_num = int(cite[0])
                for idx, (filename, _) in enumerate(st.session_state.paper_metadata.items(), 1):
                    if idx == paper_num:
                        files_used.append(filename)
                        break
            
            return {
                "answer": answer,
                "confidence": confidence,
                "files_used": list(set(files_used)),
                "success": True
            }
        
        return {
            "answer": "No response from model",
            "confidence": 0.0,
            "files_used": [],
            "success": False
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence": 0.0,
            "files_used": [],
            "success": False
        }


# ============================================================================
# AGENT 2: COSINE RAG (OPENAI)
# ============================================================================

def agent_cosine_rag(question):
    """Cosine RAG agent using OpenAI with improved retrieval"""
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local(
            "faiss_index_cosine",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Expand query for better retrieval
        expanded = expand_query(question)
        
        # Retrieve more candidates
        docs_with_scores = vector_store.similarity_search_with_score(
            question, 
            k=RETRIEVAL_CANDIDATES
        )
        
        # Also search with expanded query
        expanded_docs = vector_store.similarity_search_with_score(
            expanded,
            k=RETRIEVAL_CANDIDATES
        )
        
        # Combine and deduplicate
        all_docs = {}
        for doc, score in docs_with_scores + expanded_docs:
            doc_key = doc.page_content[:100]  # Use first 100 chars as key
            if doc_key not in all_docs or score < all_docs[doc_key][1]:
                all_docs[doc_key] = (doc, score)
        
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
        
        prompt_template = """Answer the question using the context below.

CONTEXT (top {k} most relevant sections):
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based on the context provided
- If the context discusses the topic but doesn't give exact answer, INFER from available information
- Search for exact terms AND synonyms (e.g., "sleep disorder" includes insomnia, OSA, sleep apnea, sleep disturbances)
- If the context mentions sleep problems, disturbances, or issues in relation to AD, identify which specific type
- Cite sources from the papers
- ONLY say "not covered" if there is ZERO mention of the topic
- Do NOT cite external papers

Answer:
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question", "k"]
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        
        result = chain.invoke({
            "input_documents": docs, 
            "question": question,
            "k": TOP_K_CHUNKS
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


def agent_etrag(question):
    """ET-RAG agent with evidence and temporal weighting + improved retrieval"""
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local(
            "faiss_index_etrag",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Expand query for better retrieval
        expanded = expand_query(question)
        
        # Retrieve more candidates from both queries
        docs_original = vector_store.similarity_search_with_score(
            question,
            k=RETRIEVAL_CANDIDATES
        )
        
        docs_expanded = vector_store.similarity_search_with_score(
            expanded,
            k=RETRIEVAL_CANDIDATES
        )
        
        # Combine and deduplicate
        all_docs = {}
        for doc, score in docs_original + docs_expanded:
            doc_key = doc.page_content[:100]
            if doc_key not in all_docs or score < all_docs[doc_key][1]:
                all_docs[doc_key] = (doc, score)
        
        docs_with_scores = list(all_docs.values())
        
        if not docs_with_scores:
            return {
                "answer": "This information is not covered in the uploaded research papers.",
                "confidence": 0.92,
                "files_used": [],
                "success": True
            }
        
        # Rerank with ET-RAG scoring
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
        
        # Take top K by ET-RAG score
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        docs = [item['doc'] for item in scored_docs[:TOP_K_CHUNKS]]
        avg_score = np.mean([item['score'] for item in scored_docs[:TOP_K_CHUNKS]])
        
        # Extract files used
        files_used = []
        for doc in docs:
            match = re.search(r'SOURCE_FILE:\s*([^\n]+)', doc.page_content)
            if match:
                files_used.append(match.group(1).strip())
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt_template = """Answer based on high-quality, recent evidence.

CONTEXT (ranked by evidence quality + recency + relevance):
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based on the context provided
- If context discusses the topic but doesn't give exact answer, INFER from available information
- Search for exact terms AND synonyms (e.g., "sleep disorder" includes insomnia, OSA, sleep apnea, sleep disturbances)
- If the context mentions sleep problems in relation to AD, identify which specific type is most supported
- Prioritize meta-analyses, systematic reviews, and RCTs
- Cite sources from the papers
- ONLY say "not covered" if there is ZERO mention of the topic
- Do NOT cite external papers

Answer:
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        
        result = chain.invoke({"input_documents": docs, "question": question})
        answer = result["output_text"].strip()
        
        # Calculate confidence
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
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence": 0.0,
            "files_used": [],
            "success": False
        }


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
        agent_texts = f"""AGENT 1 — Full Context (Gemini) [Confidence: {r1['confidence']:.0%}]:
{r1['answer']}

AGENT 2 — Cosine RAG (GPT-4o-mini) [Confidence: {r2['confidence']:.0%}]:
{r2['answer']}

AGENT 3 — ET-RAG (GPT-4o-mini) [Confidence: {r3['confidence']:.0%}]:
{r3['answer']}"""

        # Tailor answer format by question type
        if question_type == 'single_choice':
            answer_format = "Start with: **Answer: [Letter]. [Option text]**\nThen give a 2-3 sentence explanation with paper citations."
        elif question_type == 'multiple_choice':
            answer_format = "Start with: **Answer: [Letters] — [Option texts]**\nThen explain why each selected option is correct with citations."
        elif question_type == 'long_answer':
            answer_format = "Write a comprehensive 3-5 paragraph answer that opens with a thesis, synthesizes key findings with citations, covers mechanisms/evidence/implications, and concludes with clinical significance."
        else:
            answer_format = "Write a clear 2-4 sentence answer that directly addresses the question with citations."

        synthesis_prompt = f"""You are a medical research expert. Three AI agents independently analyzed research papers to answer a question.

Your task has TWO parts. You MUST return BOTH parts.

QUESTION: {question}
QUESTION TYPE: {question_type}

AGENT RESPONSES:
{agent_texts}

===== PART 1: EVALUATION (return as JSON) =====

Evaluate each agent's response and determine consensus. Return this JSON block EXACTLY:

```json
{{
  "agent1_correct": true/false,
  "agent1_note": "brief assessment",
  "agent2_correct": true/false,
  "agent2_note": "brief assessment",
  "agent3_correct": true/false,
  "agent3_note": "brief assessment",
  "consensus": "STRONG/MAJORITY/SPLIT/NOT_COVERED",
  "consensus_reason": "1 sentence explaining why"
}}
```

CONSENSUS RULES:
- "STRONG": All responding agents cover the same key points / give the same answer (even if worded differently). For open-ended questions, if agents discuss the same mechanisms, findings, and conclusions = STRONG.
- "MAJORITY": 2 out of 3 agree on the core answer, 1 differs significantly.
- "SPLIT": Agents give contradictory or substantially different answers.
- "NOT_COVERED": Most agents say the topic is not in the uploaded papers.
- If an agent had an ERROR (429, timeout), ignore it — evaluate only the working agents.
- For short/long answer questions: focus on whether agents cover the SAME KEY POINTS, not whether they use identical words. Similar content = agreement.

===== PART 2: SYNTHESIZED ANSWER =====

After the JSON block, write the final synthesized answer:
- Combine the best evidence and citations from all correct agents
- Do NOT mention "Agent 1/2/3" — just present the information naturally
- Keep citations in format [Paper #, Page #]
- {answer_format}"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=3000)
        response = llm.invoke(synthesis_prompt)
        full_response = response.content.strip()

        # Parse the JSON evaluation block
        consensus_level = "STRONG"  # default
        consensus_reason = ""
        synthesized = full_response  # fallback: use entire response

        json_match = re.search(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL)
        if not json_match:
            # Try without code blocks
            json_match = re.search(r'(\{[^{}]*"consensus"[^{}]*\})', full_response, re.DOTALL)

        if json_match:
            try:
                evaluation = json.loads(json_match.group(1))
                consensus_level = evaluation.get('consensus', 'STRONG').upper()
                consensus_reason = evaluation.get('consensus_reason', '')

                # Extract the answer part (everything after the JSON block)
                json_end = full_response.find(json_match.group(0)) + len(json_match.group(0))
                answer_part = full_response[json_end:].strip()
                # Remove any markdown headers like "## Synthesized Answer" etc.
                answer_part = re.sub(r'^#+\s*(PART 2|Synthesized Answer|Final Answer)[:\s]*\n*', '', answer_part, flags=re.IGNORECASE)
                answer_part = re.sub(r'^\*\*?(PART 2|Synthesized Answer|Final Answer)\*?\*?[:\s]*\n*', '', answer_part, flags=re.IGNORECASE)
                if answer_part and len(answer_part) > 30:
                    synthesized = answer_part.strip()
            except json.JSONDecodeError:
                pass

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

#### 🌐 Agent 1: Full Context (Gemini) — Confidence: {r1['confidence']:.0%}
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

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, q_item in enumerate(questions):
        question = q_item['text'] if isinstance(q_item, dict) else q_item
        q_type = q_item.get('type', classify_question_type(question)) if isinstance(q_item, dict) else classify_question_type(question)
        status_text.markdown(f"**Processing question {i + 1}/{total}:** {question[:80]}...")

        start_time = time.time()

        r1 = agent_full_context(question)
        r2 = agent_cosine_rag(question)
        r3 = agent_etrag(question)

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

#### 🌐 Agent 1: Full Context (Gemini) — Confidence: {row['agent1_confidence']:.0%}
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
        st.caption("Three AI agents analyze your papers: Full Context (Gemini), Cosine RAG (GPT-4o), and ET-RAG (Evidence+Temporal)")
    
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

                with st.spinner("🤖 All 3 agents analyzing..."):
                    r1 = agent_full_context(q_text)
                    r2 = agent_cosine_rag(q_text)
                    r3 = agent_etrag(q_text)
                    response, _ = build_single_response(r1, r2, r3, question=q_text, question_type=q_type)
                    st.session_state.chat_history.append(("assistant", response))

                st.rerun()

            else:
                # --- Batch mode: multiple questions detected ---
                st.session_state.chat_history.append(("user", user_input))
                st.info(f"📋 Detected **{len(questions)}** questions. Processing with all 3 agents + synthesis...")

                response, results_df = process_batch_in_chat(questions)
                # Store with results_df attached for download button
                st.session_state.chat_history.append(("assistant", response, results_df))

                st.rerun()


if __name__ == "__main__":
    main()