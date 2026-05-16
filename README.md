# RAG_Sanskrit_Hariom

## Sanskrit Document RAG Chatbot

### Requirements
- Python 3.11
- Ollama with mistral model

### Setup

1. Create and activate virtual environment:
   source venv/bin/activate

2. Install dependencies:
   pip install langchain langchain-community langchain-ollama langchain-core langchain-text-splitters faiss-cpu sentence-transformers streamlit python-docx pypdf ollama docx2txt

3. Download Mistral model:
   ollama pull mistral

4. Place Sanskrit documents in data/ folder

### Run

Terminal 1 — start Ollama:
ollama serve

Terminal 2 — start app:
cd code
streamlit run rag_app.py

Open browser at: http://localhost:8501

### Architecture
- Loader: LangChain Docx2txt / PyPDF
- Chunking: RecursiveCharacterTextSplitter (500 chars, 100 overlap)
- Embeddings: all-MiniLM-L6-v2 (CPU)
- Vector Store: FAISS
- LLM: Mistral 7B via Ollama (temperature=0)
- UI: Streamlit chat interface

### Anti-Hallucination
- temperature=0.0
- top_k=10
- repeat_penalty=1.3
- Strict prompt with 7 rules
- English summaries injected into vector DB