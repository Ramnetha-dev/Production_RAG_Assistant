# Production-Grade GenAI Assistant (RAG)

Flask-based RAG chat assistant with:
- document chunking
- embedding generation (OpenAI)
- SQLite storage for chunks + vectors
- cosine similarity retrieval
- grounded answer generation with citations
- browser chat UI

## 1) Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set:

```bash
OPENAI_API_KEY=...

# Optional: app can run without OPENAI_API_KEY using local fallback retrieval.
# Set the key to enable OpenAI embeddings + LLM answer generation.
```

## 2) Run

```bash
python app.py
```

Open: `http://localhost:5000`

## 3) How It Works

1. `POST /reindex`:
   - reads `docs.json`
   - chunks each document
   - generates embeddings
   - stores chunks + vectors in SQLite (`rag_store.db`)
2. `POST /chat`:
   - embeds user query
   - retrieves top similar chunks
   - sends retrieved context + history to LLM
   - returns grounded answer + citations

## 4) API Endpoints

- `GET /health` - service and index status
- `POST /reindex` - rebuild index from `docs.json`
- `POST /chat` - ask a grounded question
- `POST /reset-memory` - clear session memory

## 5) Submission Checklist

- GitHub repository link
- deployed app URL
- short screen recording link
