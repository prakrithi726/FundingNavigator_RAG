# Funding Navigator AI

Funding Navigator AI is an end-to-end Retrieval-Augmented Generation (RAG) application that helps users explore Startup India schemes, funding opportunities, government policies, tax benefits, and startup support initiatives through natural language queries.

The application retrieves relevant information using semantic search with Milvus and generates grounded responses using a locally hosted Meta Llama 3 8B Instruct model through Ollama.

---

## Features

- Semantic search over Startup India documents
- Retrieval-Augmented Generation (RAG)
- Local LLM inference using Meta Llama 3 8B Instruct (Ollama)
- Milvus Vector Database for semantic retrieval
- FastAPI backend
- Streamlit frontend
- HTML to Markdown document preprocessing
- Recursive text chunking using LangChain
- SentenceTransformer embeddings (all-MiniLM-L6-v2)

---

## System Architecture

```
User Question
      │
      ▼
Streamlit Frontend
      │
      ▼
FastAPI Backend
      │
      ▼
SentenceTransformer Embedding
      │
      ▼
Milvus Vector Database
      │
      ▼
Top-5 Relevant Chunks
      │
      ▼
Meta Llama 3 8B Instruct (Ollama)
      │
      ▼
Generated Response
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Meta Llama 3 8B Instruct (Ollama) |
| Vector Database | Milvus |
| Embedding Model | SentenceTransformer (all-MiniLM-L6-v2) |
| Framework | LangChain |
| Web Scraping | Selenium |
| Document Processing | Markdownify |
| Containerization | Docker |

---

## Project Workflow

1. Scrape Startup India webpages using Selenium.
2. Convert HTML into Markdown documents.
3. Clean and preprocess the extracted text.
4. Split documents into chunks using LangChain's RecursiveCharacterTextSplitter.
5. Generate embeddings using SentenceTransformer.
6. Store embeddings in Milvus Vector Database.
7. Retrieve the most relevant chunks for a user query.
8. Pass the retrieved context to Meta Llama 3 through Ollama.
9. Generate a grounded response.
10. Display the answer using the Streamlit interface.

---

## Sample Questions

- What is Karnataka's Startup Policy?
- What is tax exemption under Section 80IAC?
- Can a Partnership Firm avail SISFS benefits?
- What are different funding support schemes for Indian startups?
- What is the eligibility criteria for Mahila Coir Yojana?

---

## Future Improvements

- Semantic chunking for improved retrieval quality
- Hybrid retrieval (Dense + BM25)
- Source citation support
- Conversation history
- Cloud deployment
- Support for multiple LLMs

---

## Author

**Prakrithi Jain**

B.Tech in Computer Science and Engineering  
BMS College of Engineering
