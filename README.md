Alzheimer's Research Assistant
A specialized chatbot for analyzing, retrieving, and synthesizing information from Alzheimer's disease research papers.
📋 Project Overview
This project implements a disease-specific chatbot designed to help researchers, healthcare professionals, and students navigate the complex landscape of Alzheimer's disease research. The system can analyze multiple research papers simultaneously, extract relevant information, identify contradictions or agreements between different studies, and provide comprehensive answers to specific questions.
Two distinct implementation approaches were developed and compared:

RAG (Retrieval-Augmented Generation) Approach: Uses cosine similarity to retrieve relevant chunks of text from papers
Full-Context Approach: Provides the entire document content to the language model

🌟 Key Features

PDF Research Paper Processing: Extract text and metadata from uploaded PDF research papers
Natural Language Question Answering: Ask questions in natural language about the content of papers
Cross-Paper Analysis: Compare findings across multiple papers
Source Attribution: Responses include citations to specific papers
Contradiction Identification: System can identify conflicting findings or statements
Summarization: Generate summaries of individual papers or topics
User-Friendly Interface: Clean Streamlit web interface for easy interaction

🔧 Technology Stack

Python: Core programming language
Streamlit: Web application framework
Google Gemini API: Foundation language model
PyPDF2: PDF text extraction
FAISS: Vector database for similarity search (RAG version)
LangChain: Framework for LLM application development (RAG version)
Regular Expressions: For text processing and metadata extraction

📂 Repository Structure
This repository contains two main implementation files:

ver2.py: The RAG (Retrieval-Augmented Generation) implementation
ver3.py: The Full-Context implementation

RAG Implementation (ver2.py)
The RAG implementation follows these steps:

Extract text and metadata from research papers
Split text into optimized chunks
Generate embeddings for each chunk
Store embeddings in a FAISS vector database
When a question is asked, retrieve the most relevant chunks using cosine similarity
Use retrieved chunks as context for the LLM to generate a response

Full-Context Implementation (ver3.py)
The Full-Context implementation follows these steps:

Extract text from research papers with minimal preprocessing
When a question is asked, send the entire content of all papers to the LLM
Generate a response based on the complete document content

🚀 Installation and Setup
Prerequisites

Python 3.8 or higher
Google Gemini API key

Installation Steps

Clone this repository:
git clone https://github.com/yourusername/alzheimers-research-assistant.git
cd alzheimers-research-assistant

Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages:
pip install -r requirements.txt

Create a .env file in the project root directory with your Google API key:
GOOGLE_API_KEY=your_google_api_key_here


Running the Application
To run the RAG implementation:
streamlit run ver2.py
To run the Full-Context implementation:
streamlit run ver3.py
Access the application at http://localhost:8501 in your web browser.
📊 Performance Comparison
Our evaluation of both implementations yielded the following results:
MetricRAG ApproachFull-Context ApproachOverall Accuracy95%98%Response TimeFaster (1-3 seconds)Slower (5-10+ seconds)Token UsageLower (selective context)Higher (entire document)Complex QuestionsSome limitationsBetter handlingSource AttributionExplicit and preciseMore generalContradictionsSometimes missedBetter identificationSummarizationGood performanceExcellent performance
💡 Usage Examples
The application allows you to:

Upload Research Papers: Upload multiple PDF research papers about Alzheimer's disease
Ask Questions: Input natural language questions about the content of the papers
Get Detailed Answers: Receive responses that synthesize information across papers

Example questions:

"What are the primary pathological hallmarks of Alzheimer's disease?"
"Summarize the findings of Paper X regarding amyloid-beta"
"Compare the methodologies used in these papers for measuring tau protein"
"Are there any contradictions between these papers about the effectiveness of drug Y?"
"What are the limitations mentioned across these studies?"

🔮 Future Improvements

Hybrid Implementation: Develop a system that dynamically switches between RAG and Full-Context approaches based on question type
Enhanced Retrieval Methods: Implement more sophisticated similarity search algorithms
Improved PDF Processing: Better handling of tables, figures, and multi-column layouts
Extension to Other Diseases: Adapt the system to other medical domains

📚 References

Wu, E., et al. (2023). Generalist large language models in medical artificial intelligence. Nature Medicine, 29, 2448-2458. https://www.nature.com/articles/s41591-023-02448-8
Nori, H., et al. (2023). Capabilities of large language models in the clinical and biomedical domain. Nature, 586, 291-302. https://www.nature.com/articles/s41586-023-06291-2

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👥 Contributors
Ruchith
Revanth


📧 Contact
For questions or feedback, please open an issue on this repository.
