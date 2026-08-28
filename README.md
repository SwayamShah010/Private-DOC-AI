# 🔐 PrivateDocs AI

> A local-first AI document search, question-answering, and summarization system built with RAG.

PrivateDocs AI lets users upload PDF documents, index them locally, ask natural-language questions, receive source-grounded answers with page citations, and generate document summaries. The project is designed with privacy in mind: documents, embeddings, vector data, and LLM inference can remain on the user's own machine.

## ✨ Features

* 📄 PDF upload and text extraction
* 🧠 Local embeddings using Sentence Transformers
* 🔎 Semantic vector search
* 🔤 BM25 keyword search
* 🔀 Hybrid semantic + keyword retrieval
* 🎯 Optional cross-encoder reranking
* 🤖 Retrieval-Augmented Generation (RAG)
* 📌 Source and page citations
* 📝 Document summarization
* 🖼️ OCR support for scanned PDFs
* 🗂️ Local ChromaDB vector storage
* 🔒 Local LLM inference using Ollama
* 🖥️ Streamlit web interface
* 🧪 Unit and integration testing
* 📊 RAG evaluation workflow
* 🐳 Docker support

## 🏗️ Architecture

```text
PDF Upload
    ↓
Text Extraction / OCR
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
ChromaDB Vector Store
    ↓
Semantic Search + BM25
    ↓
Hybrid Retrieval
    ↓
Optional Reranking
    ↓
RAG Pipeline
    ↓
Ollama / Llama 3.1
    ↓
Answer + Page Citations
```

## 🛠️ Tech Stack

| Component       | Technology              |
| --------------- | ----------------------- |
| Language        | Python                  |
| UI              | Streamlit               |
| PDF Processing  | PyMuPDF                 |
| OCR             | Tesseract + pytesseract |
| Embeddings      | Sentence Transformers   |
| Embedding Model | `all-MiniLM-L6-v2`      |
| Vector Database | ChromaDB                |
| Keyword Search  | BM25                    |
| RAG Framework   | LangChain               |
| Local LLM       | Ollama                  |
| Default Model   | `llama3.1:8b`           |
| Testing         | Pytest                  |
| Packaging       | Docker                  |

## 📁 Project Structure

```text
PrivateDocs-AI/
│
├── app/
│   ├── citations/
│   ├── config/
│   ├── embeddings/
│   ├── generation/
│   ├── ingestion/
│   ├── retrieval/
│   ├── storage/
│   └── ui/
│
├── data/
├── evaluation/
├── tests/
├── docs/
│
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── run.py
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd PrivateDocs-AI
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🤖 Ollama Setup

Install Ollama from:

```text
https://ollama.com/
```

Check Ollama:

```powershell
ollama list
```

Download the model:

```powershell
ollama pull llama3.1:8b
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

Test the model:

```powershell
ollama run llama3.1:8b
```

Exit using:

```text
/bye
```

## ⚙️ Environment Setup

Copy:

```text
.env.example
```

to:

```text
.env
```

On Windows:

```powershell
Copy-Item .env.example .env
```

Default configuration:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

EMBEDDING_MODEL=all-MiniLM-L6-v2

CHUNK_SIZE=800
CHUNK_OVERLAP=150

TOP_K=5

ENABLE_HYBRID_SEARCH=true
HYBRID_CANDIDATE_K=20
RRF_K=60

ENABLE_RERANKING=false
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_CANDIDATE_K=20

DOCUMENTS_DIR=data/documents
VECTOR_STORE_DIR=data/vector_store

ENABLE_OCR=true
OCR_LANGUAGE=eng
OCR_DPI=300

PRIVACY_MODE=local_only
```

## ▶️ Run the Project

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Run:

```powershell
python run.py
```

Or:

```powershell
python -m streamlit run app/ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## 📖 How to Use

1. Start Ollama.
2. Run `python run.py`.
3. Upload one or more PDFs.
4. Let PrivateDocs AI extract and index the documents.
5. Select the document collection.
6. Ask a question.
7. Review the generated answer with page citations.
8. Use the summarization tab to generate document summaries.

## 🔍 Hybrid Retrieval

PrivateDocs AI combines semantic and keyword retrieval.

### Semantic Search

Semantic search understands the meaning of a query using embeddings.

For example:

```text
Query:
"What are the rules for returning an item?"

Relevant text:
"Customers can request a refund within 30 days..."
```

### BM25 Keyword Search

BM25 is useful for exact terms such as:

```text
Invoice IDs
Employee IDs
Product codes
Names
Acronyms
Technical terms
```

### Reciprocal Rank Fusion

Results from semantic search and BM25 are merged using Reciprocal Rank Fusion before being passed to the RAG pipeline.

## 🎯 Optional Reranking

Enable reranking in `.env`:

```env
ENABLE_RERANKING=true
```

This can improve retrieval accuracy but may increase query processing time.

## 🔒 Privacy

PrivateDocs AI uses a local-first architecture.

By default:

```env
PRIVACY_MODE=local_only
```

This means:

* PDFs remain on the local machine.
* Embeddings are generated locally.
* ChromaDB data stays local.
* Ollama runs the LLM locally.
* No external LLM API key is required.

The embedding and reranking models may require internet access during their first download.

## 🖼️ OCR

PrivateDocs AI supports OCR for scanned PDFs using Tesseract.

After installing Tesseract, verify it with:

```powershell
tesseract --version
```

Normal text-based PDFs can still be processed without OCR.

## 🧪 Testing

Run:

```powershell
pytest tests/ -v
```

The project includes tests for:

* PDF ingestion
* Chunking
* OCR
* Indexing
* Semantic retrieval
* Keyword search
* Hybrid retrieval
* Reranking
* Citations
* Prompt generation
* RAG pipeline
* Summarization
* UI integration

## 📊 Evaluation

Run the evaluation workflow with:

```powershell
python evaluation/run_eval.py
```

The evaluation system can measure areas such as:

* Retrieval accuracy
* Citation correctness
* Groundedness
* No-answer handling
* Response latency

## 🐳 Docker

Docker support is included.

Run:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

## 🛠️ Common Errors

### Streamlit not installed

```powershell
python -m pip install streamlit
```

### Sentence Transformers not installed

```powershell
python -m pip install sentence-transformers==3.1.1
```

### ChromaDB not installed

```powershell
python -m pip install chromadb==0.5.5
```

### Ollama connection error

Check:

```powershell
ollama list
```

Then:

```powershell
ollama pull llama3.1:8b
```

Test:

```powershell
ollama run llama3.1:8b
```

### Port `11434` already in use

If `ollama serve` returns:

```text
bind: Only one usage of each socket address...
```

Ollama is already running.

Do not start another Ollama server.

Continue with:

```powershell
ollama list
python run.py
```

## 🌐 Deployment

The default version depends on a local Ollama server, so it cannot be deployed directly like a static Netlify application.

Possible deployment options include:

* Streamlit Community Cloud
* Render
* Railway
* Hugging Face Spaces
* Docker VPS/cloud server

For public cloud deployment, the local Ollama component may need to be replaced with or connected to a hosted LLM service.

## 🔮 Future Improvements

* DOCX support
* TXT support
* PPTX support
* CSV support
* Conversational memory
* Chat history
* Document comparison
* Table-aware PDF extraction
* Improved OCR
* Metadata filtering
* Advanced reranking
* Hosted LLM fallback
* Authentication
* Export summaries
* Export answers
* Analytics dashboard

## 🎯 Real-World Applications

PrivateDocs AI can be useful for:

* 📚 Students searching textbooks and notes
* 🔬 Researchers analyzing papers
* 👨‍💻 Developers searching documentation
* 🏢 Organizations working with internal documents
* 📑 Professionals analyzing reports
* 🔐 Privacy-sensitive document analysis

## 📌 Project Goal

PrivateDocs AI demonstrates how **Retrieval-Augmented Generation, local embeddings, vector databases, hybrid retrieval, OCR, and local LLMs** can be combined into a practical privacy-focused document intelligence system.

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit the changes.
5. Open a pull request.

## ⭐ Support

If you find this project useful, consider starring the repository.

---

**Built with Python, Streamlit, Sentence Transformers, ChromaDB, LangChain, BM25, and Ollama.**
