import streamlit as st
from PyPDF2 import PdfReader
import os
import google.generativeai as genai
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

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "papers_processed" not in st.session_state:
    st.session_state.papers_processed = False
if "paper_metadata" not in st.session_state:
    st.session_state.paper_metadata = {}
if "raw_texts" not in st.session_state:
    st.session_state.raw_texts = {}

def process_pdfs_simple(pdf_docs):
    """Simply extract text from PDFs without complex metadata extraction"""
    try:
        # Dictionary to store extracted text
        paper_texts = {}
        paper_metadata = {}
        
        for pdf in pdf_docs:
            filename = pdf.name
            st.info(f"Extracting text from {filename}...")
            
            # Create basic metadata entry
            paper_metadata[filename] = {
                "filename": filename,
                "title": filename.replace('.pdf', ''),
                "source": filename
            }
            
            # Create a copy of the file in memory
            file_bytes = pdf.getvalue()
            file_stream = io.BytesIO(file_bytes)
            
            # Extract text
            try:
                pdf_reader = PdfReader(file_stream)
                full_text = ""
                
                # Add page numbers for reference
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    full_text += f"[Page {i+1}]: {page_text}\n\n"
                
                # Store the extracted text
                paper_texts[filename] = full_text
                
                # Update metadata with page count
                paper_metadata[filename]["pages"] = len(pdf_reader.pages)
            except Exception as e:
                st.error(f"Error extracting text from {filename}: {str(e)}")
        
        # Store in session state
        st.session_state.raw_texts = paper_texts
        st.session_state.paper_metadata = paper_metadata
        st.session_state.papers_processed = True
        
        return len(paper_texts), sum(meta.get("pages", 0) for meta in paper_metadata.values())
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return 0, 0

def display_chat_history():
    """Display the chat history with proper formatting"""
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

def process_user_question_simple(user_question):
    """Directly send the question and all paper text to Google Gemini without any processing"""
    if not user_question.strip():
        return
    
    # Show the user's question in chat
    st.session_state.chat_history.append(("user", user_question))
    
    # Create a placeholder for the assistant's message
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.text("Sending all paper text to Gemini...")
    
    try:
        # Check if this is a meta question about the papers
        is_paper_list_question = any(term in user_question.lower() for term in 
                                   ["what papers", "which papers", "list papers", "papers uploaded", 
                                    "paper titles", "titles of papers", "names of papers", "loaded papers"])
        
        if is_paper_list_question:
            # Just list the papers
            paper_list = []
            for filename in st.session_state.raw_texts.keys():
                paper_list.append(f"- {filename}")
            
            if paper_list:
                response_text = "Here are the papers I have information about:\n\n" + "\n".join(paper_list)
            else:
                response_text = "I don't have information about any papers yet. Please upload some research papers."
        else:
            # Combine all paper text
            all_paper_text = ""
            paper_count = 0
            
            for filename, text in st.session_state.raw_texts.items():
                paper_count += 1
                
                # Simple header for each paper
                paper_header = f"\n\n{'='*30} PAPER {paper_count}: {filename} {'='*30}\n\n"
                
                # Add to the combined text
                all_paper_text += paper_header + text + "\n\n"
            
            if not all_paper_text:
                response_text = "No paper content available. Please upload some research papers."
            else:
                # Simple prompt for Gemini
                prompt = f"""
                I will give you the complete text extracted from PDF  papers\n
                Understand the entire information properly, and this will be the only knowledge base\n 
                HERE IS THE COMPLETE TEXT FROM ALL PAPERS:
                {all_paper_text}
                Now from the above information asnwer the following question, only answer from the information provided.\n
                MY QUESTION IS:
                {user_question}
                Try to give the best answer from your understanding of the information, mention from which paper the infromation is taken from when required after the sentences.
                Only use the information to answer the question
                """
                
                thinking_placeholder.text(f"Analyzing text from {paper_count} papers...")
                
                # Call Google Gemini directly - using gemini-2.0-flash
                try:
                    genai_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                    response = genai_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2,
                            top_p=0.95,
                            top_k=40,
                            max_output_tokens=4096
                        )
                    )
                    
                    if response and hasattr(response, 'text'):
                        response_text = response.text.strip()
                    else:
                        response_text = "No valid response received from Gemini. Please try again."
                except Exception as e:
                    # Try with gemini-pro if the first attempt fails
                    try:
                        thinking_placeholder.text(f"First attempt failed. Trying with gemini-pro model...")
                        genai_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                        response = genai_model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.2,
                                top_p=0.95,
                                top_k=40,
                                max_output_tokens=4096
                            )
                        )
                        
                        if response and hasattr(response, 'text'):
                            response_text = response.text.strip()
                        else:
                            response_text = "No valid response received from Gemini. Please try again."
                    except Exception as e2:
                        response_text = f"Error calling Gemini models: {str(e2)}"
        
    except Exception as e:
        response_text = f"Sorry, I encountered an error while analyzing the papers: {str(e)}"
    
    # Format response text
    formatted_lines = []
    for line in response_text.split('\n'):
        if len(line) > 80:
            import textwrap
            wrapped_lines = textwrap.wrap(line, width=80, break_long_words=False, replace_whitespace=False)
            formatted_lines.extend(wrapped_lines)
        else:
            formatted_lines.append(line)
    
    formatted_response = '\n'.join(formatted_lines)
    
    # Update the placeholder with the response
    thinking_placeholder.markdown(f'<div style="max-width: 100%; word-wrap: break-word;">{formatted_response}</div>', unsafe_allow_html=True)
    
    # Add response to chat history
    st.session_state.chat_history.append(("assistant", formatted_response))

def main():
    """Simplified main function that directly feeds PDF text to Gemini without processing"""
    try:
        st.set_page_config(
            page_title="Alzheimer's Research Assistant",
            page_icon="🧠",
            layout="wide"
        )
        
        st.header("🧠 Alzheimer's Disease Research Assistant")
        st.markdown("### Simple PDF-to-Gemini Analysis")
        
        # Create a two-column layout
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("📁 Upload Research Papers")
            pdf_docs = st.file_uploader(
                "Upload research papers (PDF)",
                accept_multiple_files=True,
                type=['pdf']
            )
            
            process_button = st.button("Extract Text from PDFs")
            
            if process_button and pdf_docs:
                with st.spinner("Extracting text from PDFs..."):
                    # Use simplified PDF processing
                    paper_count, page_count = process_pdfs_simple(pdf_docs)
                    
                    if paper_count > 0:
                        st.success(f"✅ Successfully extracted text from {paper_count} PDFs ({page_count} pages total).")
                    else:
                        st.error("No text was extracted from the PDFs. Please check the files and try again.")
            
            # Paper management
            if st.session_state.get("papers_processed"):
                # Show simple paper list
                papers = list(st.session_state.raw_texts.keys())
                if papers:
                    st.info(f"📚 {len(papers)} PDFs processed")
                    
                    # Show file list in expander
                    with st.expander("📋 Processed PDF Files"):
                        for filename in papers:
                            st.markdown(f"- {filename}")
                
                # Clear papers button
                if st.button("Clear Papers"):
                    st.session_state.papers_processed = False
                    st.session_state.paper_metadata = {}
                    st.session_state.raw_texts = {}
                    st.success("Papers cleared. Please upload new PDFs.")
                    st.rerun()
            
            # Help tips
            with st.expander("💡 How This Works"):
                st.markdown("""
                This is a simplified PDF analysis tool that:
                
                1. Extracts raw text from your uploaded PDFs
                2. When you ask a question, sends your question + the complete text from all PDFs directly to Google Gemini
                3. Returns Gemini's answer with no intermediate processing or chunking
                
                This direct approach ensures that all paper content is considered for every question.
                """)
        
        with col1:
            # Display chat interface
            st.subheader("💬 Ask Questions About Your Papers")
            
            # Show a warning if no papers processed
            if not st.session_state.get("papers_processed", False):
                st.warning("⚠️ Please upload and process PDF files before asking questions.")
            
            # Create container for chat history
            messages_container = st.container(height=500)
            
            # Display chat history
            with messages_container:
                display_chat_history()
            
            # Space at bottom
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
            
            # Chat input
            user_question = st.chat_input("Ask a question about the content in your PDFs...")
            
            if user_question and st.session_state.get("papers_processed", False):
                # Use the simple direct approach
                process_user_question_simple(user_question)
                
                # Refresh display
                st.rerun()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.error(f"Trace: {traceback.format_exc()}")

if __name__ == "__main__":
    print("Starting Simple PDF-to-Gemini Analysis Tool...")
    main()
    print("Application initialized")