# Funding Navigator AI

Funding Navigator AI is a Retrieval-Augmented Generation (RAG) project that helps users retrieve information about Startup India funding schemes, government policies, grants, and startup support programs using semantic search.

The project collects information from Startup India resources, preprocesses the data, generates embeddings, stores them in Milvus, and retrieves relevant information through a FastAPI backend.

---

## Features

- Scrapes Startup India webpages using Selenium and BeautifulSoup
- Cleans and preprocesses collected documents
- Semantic document chunking using LangChain
- SentenceTransformer embeddings
- Milvus vector database for semantic retrieval
- FastAPI REST API for question answering
- DeepEval-based retrieval evaluation
- Ollama (Llama 3 8B) used for answer generation and evaluation experiments

---

## Tech Stack

- Python
- FastAPI
- LangChain
- SentenceTransformers
- Milvus
- Ollama (Llama 3 8B)
- DeepEval
- Selenium
- BeautifulSoup

---

## Project Workflow

```text
Startup India Website
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
LangChain Chunking
        │
        ▼
SentenceTransformer Embeddings
        │
        ▼
Milvus Vector Database
        │
        ▼
FastAPI Backend
        │
        ▼
Relevant Chunks Retrieved
        │
        ▼
Llama 3 (Ollama)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/search` | Retrieve relevant chunks for a question |
| POST | `/batch_search` | Retrieve answers for multiple questions |

---

## Project Structure

```text
genai-project/
│
├── app.py
├── qa.py
├── query_milvus.py
├── chunking.py
├── clean.py
├── html_to_markdown.py
├── retrieve_chunks.py
├── requirements.txt
└── README.md
```

---

## Running the Project

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Documentation

Detailed API execution screenshots and sample test cases are available in the **Test Cases** document included in this repository.

---

## Future Improvements

- Hybrid search
- Cloud deployment
- Conversational memory

---

## Author

**Prakrithi Jain**

B.E. Computer Science and Engineering  
BMS College of Engineering
