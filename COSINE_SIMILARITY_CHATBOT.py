import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import traceback
import io

# Load environment variables
try:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in environment variables.")
    else:
        genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error loading environment: {e}")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "papers_processed" not in st.session_state:
    st.session_state.papers_processed = False
if "paper_metadata" not in st.session_state:
    st.session_state.paper_metadata = {}
if "raw_texts" not in st.session_state:
    st.session_state.raw_texts = {}

def format_filename_as_title(filename):
    """Convert filename to a readable title by removing extensions and replacing special chars"""
    # Remove file extension
    title = filename.rsplit('.', 1)[0]
    # Replace hyphens and underscores with spaces
    title = title.replace('-', ' ').replace('_', ' ')
    # Extract year if present at beginning
    year_match = re.match(r'^(19|20)\d{2}\s+(.+)$', title)
    if year_match:
        year, rest = year_match.groups()
        title = f"{rest} ({year})"
    return title

def extract_metadata_from_first_page(text):
    """Try to extract title and authors from the first page text"""
    metadata = {
        "extracted_title": None,
        "extracted_authors": [],
        "extracted_abstract": None,
    }
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Try to find title - usually the first prominent text on the page
    # Look for the first sentence that ends with a period or line break
    title_candidates = re.findall(r'^(.*?[A-Z][^.!?]*(?:[.!?]|$))', text)
    if title_candidates and len(title_candidates[0]) > 15:  # Avoid short headers
        metadata["extracted_title"] = title_candidates[0].strip()
    
    # Try to find authors - often follows the title
    # Look for patterns like "by Author Name, Author Name" or just "Author Name, Author Name"
    author_patterns = [
        r'(?:by|authors?)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
        r'([A-Z][a-z]+ [A-Z][a-z]+(?:,[A-Z][a-z]+ [A-Z][a-z]+)*)'
    ]
    
    for pattern in author_patterns:
        author_matches = re.search(pattern, text)
        if author_matches:
            authors_text = author_matches.group(1)
            authors = [a.strip() for a in authors_text.split(',')]
            if authors:
                metadata["extracted_authors"] = authors
                break
    
    # Try to find abstract
    abstract_pattern = r'abstract[:\s]+(.*?)(?:introduction|keywords|background|methods|results)'
    abstract_match = re.search(abstract_pattern, text.lower(), re.IGNORECASE | re.DOTALL)
    if abstract_match:
        metadata["extracted_abstract"] = abstract_match.group(1).strip()
    
    return metadata

def extract_year_from_text(text):
    """Extract publication year from text content"""
    # Look for patterns like "published in 2023" or "© 2023" or just the year with context
    year_patterns = [
        r'(?:published|copyright|©|year)\s+(?:in\s+)?(?:the\s+)?(?:year\s+)?(19|20)\d{2}',
        r'(?:19|20)\d{2}\s+(?:vol|volume)',
        r'received:?\s+\w+\s+(19|20)\d{2}',
        r'accepted:?\s+\w+\s+(19|20)\d{2}'
    ]
    
    for pattern in year_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Extract just the year from the matched text
            year_match = re.search(r'(19|20)\d{2}', match.group(0))
            if year_match:
                return year_match.group(0)
    
    # Fallback: look for any 4-digit year between 1900-2030
    years = re.findall(r'((?:19|20)[0-9]{2})', text)
    valid_years = [y for y in years if 1900 <= int(y) <= 2030]
    if valid_years:
        # Prefer more recent years as they're more likely to be publication year
        return sorted(valid_years, reverse=True)[0]
    
    return "Unknown"

def extract_paper_metadata_with_gemini(pdf_file):
    """Extract metadata from a research paper PDF using Google Gemini"""
    filename = pdf_file.name
    
    # Initialize with basic metadata
    metadata = {
        "filename": filename,
        "title": format_filename_as_title(filename),
        "authors": [],
        "year": "Unknown",
        "abstract": "",
        "pages": 0,
        "source": filename
    }
    
    try:
        # Create a copy of the file in memory to avoid stream position issues
        file_bytes = pdf_file.getvalue()
        file_stream = io.BytesIO(file_bytes)
        pdf_reader = PdfReader(file_stream)
        
        metadata["pages"] = len(pdf_reader.pages)
        
        # Extract text from first few pages for analysis
        first_pages_text = ""
        for i in range(min(5, len(pdf_reader.pages))):
            page_text = pdf_reader.pages[i].extract_text() or ""
            first_pages_text += page_text + "\n\n"
        
        # Prepare prompt for Gemini
        prompt = f"""
        Extract metadata from this research paper. The text below is from the first few pages of a PDF research paper.
        
        Please extract and return ONLY the following information in this exact JSON format:
        
        {{
            "title": "The full title of the paper", 
            "authors": ["Author 1", "Author 2", ...],
            "year": "The publication year (just the year, like 2023)",
            "journal": "Journal name if available, otherwise 'Unknown'",
            "abstract": "The paper's abstract. If not clearly identifiable, return a brief summary of the paper's focus based on the introduction",
            "keywords": ["keyword1", "keyword2", ...] (if available, otherwise empty array)
        }}
        
        Only return valid JSON without any additional text, explanation, or formatting.
        
        PAPER TEXT:
        {first_pages_text[:10000]}  # Limit to 4000 chars to stay within token limits
        """
        
        # Call Google Gemini to extract metadata
        try:
            genai_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = genai_model.generate_content(prompt)
            
            # Parse the response
            if response and hasattr(response, 'text'):
                import json
                
                # Clean the response to ensure it's valid JSON
                response_text = response.text.strip()
                # Remove markdown code block formatting if present
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "", 1)
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                try:
                    extracted_data = json.loads(response_text)
                    
                    # Update metadata with extracted information
                    if extracted_data.get("title"):
                        metadata["title"] = extracted_data["title"]
                    
                    if extracted_data.get("authors") and isinstance(extracted_data["authors"], list):
                        metadata["authors"] = extracted_data["authors"]
                    
                    if extracted_data.get("year"):
                        metadata["year"] = extracted_data["year"]
                    
                    if extracted_data.get("abstract"):
                        metadata["abstract"] = extracted_data["abstract"]
                    
                    # Add additional metadata fields
                    if extracted_data.get("journal"):
                        metadata["journal"] = extracted_data["journal"]
                    
                    if extracted_data.get("keywords") and isinstance(extracted_data["keywords"], list):
                        metadata["keywords"] = extracted_data["keywords"]
                        
                except json.JSONDecodeError as e:
                    st.warning(f"Could not parse Gemini response for {filename}. Using fallback extraction.")
                    # If JSON parsing fails, use the fallback method
                    fallback_meta, _ = extract_paper_metadata_fallback(pdf_file)
                    for key, value in fallback_meta.items():
                        if key not in metadata or not metadata[key]:
                            metadata[key] = value
        except Exception as e:
            st.warning(f"Error using Gemini for {filename}: {str(e)}. Using fallback extraction.")
            # If Gemini fails, use the fallback method
            fallback_meta, _ = extract_paper_metadata_fallback(pdf_file)
            for key, value in fallback_meta.items():
                if key not in metadata or not metadata[key]:
                    metadata[key] = value
        
        # Get full text for later processing
        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            full_text += f"[Page {i+1}]: {page_text}\n\n"
        
        # Store full text in session state for later access
        st.session_state.raw_texts[filename] = full_text
        
        return metadata, pdf_reader.pages
        
    except Exception as e:
        st.error(f"Error extracting metadata for {filename}: {str(e)}")
        return metadata, []

# Rename the original function for fallback use
def extract_paper_metadata_fallback(pdf_file):
    """Extract metadata from a research paper PDF using regex methods (fallback)"""
    filename = pdf_file.name
    
    # Initialize with basic metadata
    metadata = {
        "filename": filename,
        "title": format_filename_as_title(filename),
        "authors": [],
        "year": "Unknown",
        "abstract": "",
        "pages": 0,
        "source": filename
    }
    
    try:
        # Create a copy of the file in memory to avoid stream position issues
        file_bytes = pdf_file.getvalue()
        file_stream = io.BytesIO(file_bytes)
        pdf_reader = PdfReader(file_stream)
        
        metadata["pages"] = len(pdf_reader.pages)
        
        # Try to get PDF metadata
        if pdf_reader.metadata:
            meta = pdf_reader.metadata
            if hasattr(meta, 'title') and meta.title:
                metadata["title"] = meta.title
            if hasattr(meta, 'author') and meta.author:
                # Try to split authors
                authors = meta.author.split(',') if ',' in meta.author else [meta.author]
                metadata["authors"] = [author.strip() for author in authors]
        
        # Extract text from first page for analysis
        first_page_text = ""
        if len(pdf_reader.pages) > 0:
            first_page_text = pdf_reader.pages[0].extract_text() or ""
        
        # Get text from first few pages to improve metadata extraction
        first_pages_text = first_page_text
        for i in range(1, min(3, len(pdf_reader.pages))):
            page_text = pdf_reader.pages[i].extract_text() or ""
            first_pages_text += " " + page_text
        
        # Try to extract metadata from text content
        extracted_meta = extract_metadata_from_first_page(first_pages_text)
        
        # Update metadata if better information was found
        if extracted_meta["extracted_title"] and len(extracted_meta["extracted_title"]) > 10:
            metadata["title"] = extracted_meta["extracted_title"]
        
        if extracted_meta["extracted_authors"]:
            metadata["authors"] = extracted_meta["extracted_authors"]
        
        if extracted_meta["extracted_abstract"]:
            metadata["abstract"] = extracted_meta["extracted_abstract"]
        
        # Try to extract year from filename first
        year_from_filename = re.search(r'(19|20)\d{2}', filename)
        if year_from_filename:
            metadata["year"] = year_from_filename.group(0)
        else:
            # Try to find year in text content
            year = extract_year_from_text(first_pages_text)
            if year != "Unknown":
                metadata["year"] = year
            
        # Extract citations
        citations = re.findall(r'\[\d+\]', first_pages_text)
        if citations:
            metadata["has_citations"] = True
        
        return metadata, pdf_reader.pages
        
    except Exception as e:
        st.error(f"Error in fallback metadata extraction for {filename}: {str(e)}")
        return metadata, []

# Update the get_paper_text function to use the Gemini-based extraction
def get_paper_text(pdf_docs):
    """Extract text from research paper PDFs with metadata using Gemini"""
    all_text = ""
    all_pages_count = 0
    processed_papers = 0
    
    try:
        for pdf in pdf_docs:
            try:
                filename = pdf.name
                
                # Extract metadata and pages using Gemini-based method
                metadata, pages = extract_paper_metadata_with_gemini(pdf)
                st.session_state.paper_metadata[filename] = metadata
                
                all_pages_count += len(pages)
                processed_papers += 1
                
                # Get the full text from session state
                paper_text = st.session_state.raw_texts.get(filename, "")
                
                if not paper_text:
                    st.warning(f"No text could be extracted from {filename}")
                    continue
                
                # Prepare paper header with metadata
                paper_header = f"PAPER: {metadata['title']}\n"
                paper_header += f"AUTHORS: {', '.join(metadata['authors']) if metadata['authors'] else 'Unknown'}\n"
                paper_header += f"YEAR: {metadata['year']}\n"
                
                # Add journal if available
                if metadata.get('journal'):
                    paper_header += f"JOURNAL: {metadata['journal']}\n"
                
                # Add keywords if available
                if metadata.get('keywords') and metadata['keywords']:
                    paper_header += f"KEYWORDS: {', '.join(metadata['keywords'])}\n"
                
                paper_header += f"SOURCE: {metadata['source']}\n"
                
                # Add abstract if available
                if metadata.get('abstract'):
                    paper_header += f"\nABSTRACT: {metadata['abstract']}\n"
                
                # Add full text with paper header
                paper_text = paper_header + "\n" + paper_text
                
                # Add to the combined text with clear paper boundaries
                all_text += f"\n\n{'='*50}\n{paper_text}\n{'='*50}\n\n"
                
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {str(e)}")
                continue
        
        return all_text, processed_papers, all_pages_count
    except Exception as e:
        st.error(f"Error in PDF extraction: {str(e)}")
        return "", 0, 0

def clean_research_text(text):
    """Clean extracted text with special handling for research papers"""
    try:
        # Replace multiple newlines with a single one
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix hyphenated words that got split across lines
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        return text
    except Exception as e:
        st.warning(f"Warning in text cleaning: {str(e)}")
        return text  # Return original text if cleaning fails

def get_research_chunks(text):
    """Split text into optimized chunks for research paper understanding with improved paper tracking"""
    try:
        # Clean the text first
        clean_content = clean_research_text(text)
        
        # Create a splitter optimized for research papers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks to capture more context in research papers
            chunk_overlap=300,  # More overlap to maintain context across complex concepts
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on paragraph boundaries
        )
        
        chunks = text_splitter.split_text(clean_content)
        
        # Add chunk index information to help track sources
        indexed_chunks = []
        for i, chunk in enumerate(chunks):
            # Try to identify which paper this belongs to
            paper_match = re.search(r'PAPER: ([^\n]+)', chunk)
            paper_title = paper_match.group(1) if paper_match else "Unknown Paper"
            
            # Try to extract year
            year_match = re.search(r'YEAR: ([^\n]+)', chunk)
            paper_year = year_match.group(1) if year_match else "Unknown Year"
            
            # Try to extract authors
            authors_match = re.search(r'AUTHORS: ([^\n]+)', chunk)
            paper_authors = authors_match.group(1) if authors_match else "Unknown Authors"
            
            # Try to extract journal
            journal_match = re.search(r'JOURNAL: ([^\n]+)', chunk)
            paper_journal = journal_match.group(1) if journal_match else ""
            
            # Create a clear paper identifier header
            paper_id = f"Paper: \"{paper_title}\" ({paper_year}) by {paper_authors}"
            if paper_journal:
                paper_id += f" - {paper_journal}"
            
            # Add metadata to the chunk
            chunk_header = f"[Chunk {i+1}] {paper_id}\n"
            chunk_header += f"SOURCE_PAPER_TITLE: {paper_title}\n"
            chunk_header += f"SOURCE_PAPER_YEAR: {paper_year}\n"
            chunk_header += f"SOURCE_PAPER_AUTHORS: {paper_authors}\n"
            if paper_journal:
                chunk_header += f"SOURCE_PAPER_JOURNAL: {paper_journal}\n"
            chunk_header += f"CHUNK_INDEX: {i+1}\n"
            chunk_header += "--------------------\n"
            
            indexed_chunk = f"{chunk_header}{chunk}"
            indexed_chunks.append(indexed_chunk)
        
        return indexed_chunks
    except Exception as e:
        st.error(f"Error in text chunking: {str(e)}")
        # Fallback to basic chunking
        try:
            basic_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            return basic_splitter.split_text(text)
        except:
            # Last resort: manual chunking
            chunks = []
            for i in range(0, len(text), 1500):
                chunks.append(text[i:i+1500])
            return chunks

def get_vector_store(text_chunks):
    """Create and save a vector store from text chunks"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create the vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save the vector store locally
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.error(f"Trace: {traceback.format_exc()}")
        return None

def get_research_qa_chain():
    """Create a specialized chain for research paper analysis"""
    try:
        prompt_template = """
        You are a research assistant specialized in analyzing scientific papers about Alzheimer's disease. You have been provided with content from several research papers.
        
        Answer the question based ONLY on the information in the provided context. The context contains excerpts from research papers.
        
        PAY SPECIAL ATTENTION to the source paper information provided at the beginning of each chunk. Each chunk has metadata in this format:
        SOURCE_PAPER_TITLE: [title of the paper]
        SOURCE_PAPER_YEAR: [year of publication]
        SOURCE_PAPER_AUTHORS: [authors of the paper]
        SOURCE_PAPER_JOURNAL: [journal name if available]
        
        When appropriate:
        - Identify conflicting findings or statements between papers
        - Compare methodologies and results across papers
        - Synthesize information from multiple papers
        - Highlight limitations and gaps in the research
        - Cite specific papers by their titles and years when referring to their findings
        
        When citing a paper, use this format: "According to [Title] ([Year]) by [Authors], ..."
        
        If information is missing or unclear, state this explicitly rather than making assumptions.
        If papers disagree on a topic, explain the different perspectives and possible reasons for disagreement.
        
        If asked for a paper summary, provide a structured summary including: research question, methodology, key findings, and limitations.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Detailed, research-focused answer:
        """

        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Use the appropriate Gemini model
            temperature=0.2,  # Lower temperature for more factual responses
            top_p=0.95,       # Slightly limit token selection for more accurate replies
            top_k=40          # Keep a good variety of token options
        )

        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        chain = load_qa_chain(
            model, 
            chain_type="stuff",  # 'stuff' method puts all documents into the prompt
            prompt=prompt
        )

        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def get_relevant_documents(question, top_k=5):
    """Get the most relevant document chunks using advanced retrieval techniques,
    including 4 chunks before and after each match for better context"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the vector store
        try:
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading document index: {e}")
            return []
        
        # Determine if this is a specific query type
        is_summary_request = any(term in question.lower() for term in ["summarize", "summary", "summarization"])
        
        # If asking for a summary, adjust the retrieval strategy
        if is_summary_request:
            # Try to identify which paper to summarize
            paper_titles = [meta.get("title", "").lower() for meta in st.session_state.paper_metadata.values()]
            
            matched_papers = []
            for title in paper_titles:
                if title.lower() != "unknown title" and title.lower() in question.lower():
                    matched_papers.append(title)
            
            if matched_papers:
                # Found a specific paper to summarize - adjust the query
                specific_paper = matched_papers[0]
                enhanced_query = f"full content of paper titled {specific_paper}"
                top_k = 8  # Get more chunks to ensure we capture the full paper
            else:
                # General summary request
                enhanced_query = question
        else:
            enhanced_query = question
        
        # Try MMR search first
        try:
            # Use Maximum Marginal Relevance to get diverse but relevant results
            main_docs = vector_store.max_marginal_relevance_search(
                enhanced_query,
                k=top_k,
                fetch_k=top_k*3,  # Fetch more candidates, then select top_k most diverse
                lambda_mult=0.7  # Higher lambda puts more emphasis on relevance vs. diversity
            )
        except Exception as e:
            st.warning(f"MMR search failed, falling back to standard search: {e}")
            # Fall back to similarity search if MMR fails
            main_docs = vector_store.similarity_search(
                enhanced_query,
                k=top_k
            )
        
        # If no documents found, return empty list
        if not main_docs:
            return []
        
        # Load all chunks to identify neighbors
        # We'll need this to find chunks before and after matches
        all_chunks = []
        try:
            # Get all document chunks from the vector store
            # This is a simplified approach - in a production environment with many documents,
            # you'd want a more efficient way to retrieve neighboring chunks
            all_docs = vector_store.similarity_search(
                "",  # Empty query to get all documents
                k=1000  # Adjust based on your expected maximum document count
            )
            all_chunks = all_docs
        except Exception as e:
            st.warning(f"Could not retrieve all chunks for context expansion: {e}")
            return main_docs  # Fall back to just the main matches if we can't get all chunks
        
        # Extract chunk numbers for sorting
        def extract_chunk_number(doc):
            try:
                chunk_text = doc.page_content
                chunk_match = re.search(r'\[Chunk (\d+)\]', chunk_text)
                if chunk_match:
                    return int(chunk_match.group(1))
                return 0  # Default if no chunk number found
            except:
                return 0
        
        # Sort all chunks by their chunk number
        all_chunks.sort(key=extract_chunk_number)
        
        # Dictionary to track which chunks we've already included
        included_chunks = {}
        
        # Find the chunks before and after each main match
        expanded_docs = []
        for doc in main_docs:
            # Skip if we've already included this chunk
            doc_chunk_num = extract_chunk_number(doc)
            if doc_chunk_num in included_chunks:
                continue
                
            # Mark this chunk as included
            included_chunks[doc_chunk_num] = True
            expanded_docs.append(doc)
            
            # Find chunks before and after
            for i, chunk in enumerate(all_chunks):
                chunk_num = extract_chunk_number(chunk)
                
                # If this is our target chunk, get 4 before and 4 after
                if chunk_num == doc_chunk_num:
                    # Add 4 chunks before
                    for j in range(max(0, i-4), i):
                        before_chunk = all_chunks[j]
                        before_chunk_num = extract_chunk_number(before_chunk)
                        if before_chunk_num not in included_chunks:
                            expanded_docs.append(before_chunk)
                            included_chunks[before_chunk_num] = True
                    
                    # Add 4 chunks after
                    for j in range(i+1, min(len(all_chunks), i+5)):
                        after_chunk = all_chunks[j]
                        after_chunk_num = extract_chunk_number(after_chunk)
                        if after_chunk_num not in included_chunks:
                            expanded_docs.append(after_chunk)
                            included_chunks[after_chunk_num] = True
                    
                    break
        
        # Sort expanded docs by chunk number for a coherent context
        expanded_docs.sort(key=extract_chunk_number)
        
        return expanded_docs
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def process_user_question(user_question):
    """Process user question with specialized handling for research paper queries"""
    if not user_question.strip():
        return
    
    # Show the user's question in chat
    st.session_state.chat_history.append(("user", user_question))
    
    # Create a placeholder for the assistant's message
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.text("Analyzing research papers...")
    
    try:
        # Check if this is a meta question about the papers themselves
        is_paper_list_question = any(term in user_question.lower() for term in 
                                    ["what papers", "which papers", "list papers", "papers uploaded", 
                                     "paper titles", "titles of papers", "names of papers", "loaded papers"])
        
        if is_paper_list_question:
            # Just list the papers without using the QA chain
            paper_list = []
            for filename, metadata in st.session_state.paper_metadata.items():
                title = metadata.get("title", filename)
                authors = ", ".join(metadata.get("authors", [])) or "Unknown"
                year = metadata.get("year", "Unknown")
                paper_list.append(f"- \"{title}\" ({year}) by {authors}")
            
            if paper_list:
                response_text = "Here are the papers I have information about:\n\n" + "\n".join(paper_list)
            else:
                response_text = "I don't have information about any specific papers yet. Please upload some research papers."
        else:
            # Is this a request to summarize a specific paper?
            is_summary_request = any(term in user_question.lower() for term in ["summarize", "summary", "summarization"])
            
            # Get relevant documents
            docs = get_relevant_documents(user_question, top_k=5 if not is_summary_request else 8)
            
            if not docs:
                response_text = "I couldn't find relevant information in the research papers. Please try rephrasing your question or upload relevant papers."
            else:
                # Create conversation chain
                chain = get_research_qa_chain()
                
                if not chain:
                    response_text = "Sorry, I'm having trouble connecting to the language model. Please check your API key configuration."
                else:
                    # Get response
                    response = chain(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                    
                    # Extract the text from the response
                    response_text = response["output_text"]
        
    except Exception as e:
        response_text = f"Sorry, I encountered an error while analyzing the research papers: {str(e)}"
    
    # Format response text to force proper wrapping
    # Split long lines and ensure proper line breaks
    formatted_lines = []
    for line in response_text.split('\n'):
        # Wrap very long lines (ensure lines are not too long)
        if len(line) > 80:
            import textwrap
            wrapped_lines = textwrap.wrap(line, width=80, break_long_words=False, replace_whitespace=False)
            formatted_lines.extend(wrapped_lines)
        else:
            formatted_lines.append(line)
    
    formatted_response = '\n'.join(formatted_lines)
    
    # Update the placeholder with the actual response using HTML wrapping
    thinking_placeholder.markdown(f'<div style="max-width: 100%; word-wrap: break-word;">{formatted_response}</div>', unsafe_allow_html=True)
    
    # Add response to chat history
    st.session_state.chat_history.append(("assistant", formatted_response))

def display_chat_history():
    """Display the chat history with no scrollbars in user messages but preserving response formatting"""
    if not st.session_state.chat_history:
        st.info("No conversation history yet. Ask a question to begin!")
        return
    
    # Display messages in chronological order
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            if role == "user":
                # User messages: ensure no scrollbars
                st.markdown(
                    f'<div style="width:100%; overflow:visible; white-space:normal; word-wrap:break-word;">{message}</div>',
                    unsafe_allow_html=True
                )
            else:
                # Assistant messages: use standard markdown with normal formatting
                st.markdown(message)

def generate_paper_insights():
    """Generate initial insights about uploaded papers"""
    papers = list(st.session_state.paper_metadata.values())
    if not papers:
        return "No papers have been processed yet."
    
    paper_count = len(papers)
    years = [int(p.get("year", "0")) for p in papers if p.get("year", "").isdigit()]
    
    if years:
        earliest_year = min(years)
        latest_year = max(years)
        year_range = f"Publication years range from {earliest_year} to {latest_year}"
    else:
        year_range = "Publication years unknown"
    
    insights = [
        f"📚 {paper_count} research papers processed",
        f"📅 {year_range}"
    ]
    
    return "\n".join(insights)

def main():
    """Main application function"""
    try:
        st.set_page_config(
            page_title="Alzheimer's Research Assistant",
            page_icon="🧠",
            layout="wide"
        )
        
        st.header("🧠 Alzheimer's Disease Research Assistant")
        
        # Create a two-column layout
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("📁 Upload Research Papers")
            pdf_docs = st.file_uploader(
                "Upload Alzheimer's research papers (PDF)",
                accept_multiple_files=True,
                type=['pdf']
            )
            
            process_button = st.button("Process Papers")
            
            if process_button and pdf_docs:
                with st.spinner("Processing research papers..."):
                    try:
                        # Process PDFs in multiple steps with progress indicators
                        progress_bar = st.progress(0)
                        st.info("Step 1/3: Extracting text and metadata from PDFs...")
                        raw_text, processed_papers, total_pages = get_paper_text(pdf_docs)
                        progress_bar.progress(33)
                        
                        if not raw_text:
                            st.error("No text was extracted from the PDFs. Please upload text-based research papers.")
                        else:
                            st.info(f"Step 2/3: Creating research-optimized text chunks from {total_pages} pages...")
                            text_chunks = get_research_chunks(raw_text)
                            progress_bar.progress(66)
                            
                            if not text_chunks:
                                st.error("Error creating text chunks.")
                            else:
                                st.info(f"Step 3/3: Building knowledge base with {len(text_chunks)} chunks...")
                                vector_store = get_vector_store(text_chunks)
                                progress_bar.progress(100)
                                
                                if vector_store:
                                    # Show stats
                                    st.success(f"✅ Processing complete! Created {len(text_chunks)} chunks from {processed_papers} research papers ({total_pages} pages total).")
                                    st.session_state.papers_processed = True
                                else:
                                    st.error("Error building the knowledge base.")
                    except Exception as e:
                        st.error(f"Error during document processing: {str(e)}")
                        st.error(f"Trace: {traceback.format_exc()}")
            
            # Display paper insights
            if st.session_state.get("papers_processed"):
                st.info(generate_paper_insights())
                
                # Paper management
                if st.button("Clear Papers"):
                    # Remove the stored index
                    if os.path.exists("faiss_index"):
                        import shutil
                        try:
                            shutil.rmtree("faiss_index")
                            st.session_state.papers_processed = False
                            st.session_state.paper_metadata = {}
                            st.session_state.raw_texts = {}
                            st.success("Papers cleared. Please upload new research papers.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing papers: {str(e)}")
            
            # Paper metadata explorer
            if st.session_state.get("papers_processed") and st.session_state.paper_metadata:
                with st.expander("📋 Processed Papers Details"):
                    for filename, metadata in st.session_state.paper_metadata.items():
                        title = metadata.get('title', filename)
                        
                        st.markdown(f"**{title}**")
                        st.markdown(f"Authors: {', '.join(metadata.get('authors', [])) or 'Unknown'}")
                        st.markdown(f"Year: {metadata.get('year', 'Unknown')}")
                        
                        # Display journal if available
                        if metadata.get('journal') and metadata['journal'] != 'Unknown':
                            st.markdown(f"Journal: {metadata['journal']}")
                        
                        st.markdown(f"Pages: {metadata.get('pages', 0)}")
                        
                        # Show abstract if available (without using nested expander)
                        if metadata.get('abstract') and len(metadata['abstract']) > 0:
                            st.markdown("**Abstract:**")
                            st.markdown(f"<div style='background-color:#f0f2f6; padding:8px; border-radius:5px; margin-bottom:10px; font-size:0.9em; color:black;'>{metadata['abstract']}</div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
            
            # Tips
            with st.expander("💡 Research Assistant Capabilities"):
                st.markdown("""
                You can ask me to:
                - **Summarize** individual papers on Alzheimer's disease
                - **Compare** findings across multiple papers
                - Identify **conflicting results or methodologies**
                - Extract specific information about **pathogenesis, treatments, or biomarkers**
                - Find **gaps or limitations** in the Alzheimer's research
                - Synthesize information about specific aspects like **amyloid, tau, neuroinflammation**
                - List all uploaded papers
                """)
        
        with col1:
            # Display chat interface
            st.subheader("💬 Alzheimer's Research Assistant")
            
            # Show a warning if no papers processed
            if not st.session_state.get("papers_processed", False):
                st.warning("⚠️ Please upload and process Alzheimer's research papers before asking questions.")
            
            # Create a container for the chat interface with custom CSS
            chat_container = st.container()
            
            # Add custom CSS for chat layout to ensure no scrolling in user questions only
            st.markdown("""
            <style>
            /* Remove scrollbars from user messages only */
            div[data-testid="stChatMessage"][data-testid*="user"] {
                overflow: visible !important;
                overflow-x: visible !important;
                overflow-y: visible !important;
                max-height: none !important;
                height: auto !important;
            }
            
            /* Ensure user message text wrapping */
            div[data-testid="stChatMessage"][data-testid*="user"] p,
            div[data-testid="stChatMessage"][data-testid*="user"] div {
                white-space: normal !important;
                overflow-wrap: break-word !important;
                word-wrap: break-word !important;
                word-break: break-word !important;
            }
            
            /* Maintain width constraints */
            div[data-testid="stChatMessage"] {
                max-width: 100% !important;
                width: 100% !important;
            }
            
            /* Make pre and code blocks wrap properly */
            pre, code {
                white-space: pre-wrap !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Use two columns with specific heights to create the fixed input at bottom
            messages_container = st.container(height=500)
            
            # Display existing chat history in the scrollable area
            with messages_container:
                display_chat_history()
            
            # This empty space pushes content to the bottom
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
            
            # Chat input at the bottom
            user_question = st.chat_input("Ask a question about Alzheimer's disease research...")
            
            if user_question and st.session_state.get("papers_processed", False):
                process_user_question(user_question)
                
                # Auto-scroll to bottom after new message
                st.rerun()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.error(f"Trace: {traceback.format_exc()}")

if __name__ == "__main__":
    print("Starting Alzheimer's Research Analysis Assistant...")
    main()
    print("Application initialized")