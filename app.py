import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI


BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "rag_store.db"
DOCS_PATH = BASE_DIR / "docs.json"

load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
TOP_K = _env_int("TOP_K", 3)
CHUNK_SIZE = _env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 80)
SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.20)
MAX_TURNS = _env_int("MAX_TURNS", 5)


api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key) if api_key else None
LOCAL_EMBED_DIM = 512

app = Flask(__name__)

# Conversation memory per session id for simple demo use-case.
memory: Dict[str, List[Dict[str, str]]] = {}


@dataclass
class Chunk:
    doc_id: str
    title: str
    idx: int
    content: str
    source: str


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY,
                vector_json TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    words = text.split(" ")
    if len(words) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        slice_words = words[start : start + chunk_size]
        if not slice_words:
            continue
        chunks.append(" ".join(slice_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def read_documents() -> List[dict]:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Missing docs file at: {DOCS_PATH}")
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("docs.json must contain a list of objects.")
    return data


def build_chunks(docs: List[dict]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, doc in enumerate(docs):
        title = str(doc.get("title", f"Document {i+1}")).strip() or f"Document {i+1}"
        source = str(doc.get("source", title)).strip() or title
        content = str(doc.get("content", "")).strip()
        doc_id = str(doc.get("id", f"doc_{i+1}"))
        if not content:
            continue
        parts = chunk_text(content)
        for idx, part in enumerate(parts):
            chunks.append(Chunk(doc_id=doc_id, title=title, idx=idx, content=part, source=source))
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    def _local_embed(batch: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in batch:
            vec = np.zeros(LOCAL_EMBED_DIM, dtype=np.float32)
            tokens = re.findall(r"\w+", text.lower())
            for token in tokens:
                idx = hash(token) % LOCAL_EMBED_DIM
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors

    if client is not None:
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            return _local_embed(texts)

    # Offline fallback: deterministic token-hash embeddings for local retrieval.
    return _local_embed(texts)


def clear_index() -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM chunks")
        conn.commit()


def index_documents() -> Dict[str, int]:
    docs = read_documents()
    chunks = build_chunks(docs)
    if not chunks:
        clear_index()
        return {"docs": len(docs), "chunks": 0}

    vectors = embed_texts([c.content for c in chunks])
    with get_db() as conn:
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM chunks")
        for chunk, vector in zip(chunks, vectors):
            cur = conn.execute(
                """
                INSERT INTO chunks (doc_id, title, chunk_index, content, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk.doc_id, chunk.title, chunk.idx, chunk.content, chunk.source),
            )
            chunk_id = cur.lastrowid
            conn.execute(
                "INSERT INTO embeddings (chunk_id, vector_json) VALUES (?, ?)",
                (chunk_id, json.dumps(vector)),
            )
        conn.commit()
    return {"docs": len(docs), "chunks": len(chunks)}


def load_index_rows() -> List[sqlite3.Row]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.doc_id, c.title, c.chunk_index, c.content, c.source, e.vector_json
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id
            """
        ).fetchall()
    return rows


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, object]]:
    rows = load_index_rows()
    if not rows:
        return []
    if client is None:
        qtokens = set(tokenize(query))
        scored_local: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            ctokens = set(tokenize(row["content"]))
            if not qtokens or not ctokens:
                score = 0.0
            else:
                overlap = len(qtokens & ctokens)
                score = overlap / max(1, len(qtokens))
            scored_local.append((score, row))
        scored_local.sort(key=lambda x: x[0], reverse=True)
        top_local = scored_local[: max(1, top_k)]
        result_local: List[Dict[str, object]] = []
        for score, row in top_local:
            result_local.append(
                {
                    "chunk_id": row["id"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "source": row["source"],
                    "score": round(score, 4),
                }
            )
        return result_local

    qvec = np.array(embed_texts([query])[0], dtype=np.float32)
    scored: List[Tuple[float, sqlite3.Row]] = []
    for row in rows:
        vec = np.array(json.loads(row["vector_json"]), dtype=np.float32)
        if vec.shape != qvec.shape:
            continue
        score = cosine_similarity(qvec, vec)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [(score, row) for score, row in scored if score >= SIMILARITY_THRESHOLD]
    top = (filtered if filtered else scored)[: max(1, top_k)]
    result = []
    for score, row in top:
        result.append(
            {
                "chunk_id": row["id"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "source": row["source"],
                "score": round(score, 4),
            }
        )
    return result


def build_messages(session_id: str, user_message: str, contexts: List[Dict[str, object]]) -> List[Dict[str, str]]:
    context_text = "\n\n".join(
        [
            f"[{i+1}] title={c['title']}; source={c['source']}; score={c['score']}\n{c['content']}"
            for i, c in enumerate(contexts)
        ]
    )
    system_prompt = (
        "You are a precise support assistant. Answer using ONLY the retrieved context. "
        "If context is insufficient, say you do not know. Keep answers concise and factual. "
        "Always add citation tags like [1], [2] that map to provided chunks."
    )
    history = memory.get(session_id, [])
    history_pairs = history[-(MAX_TURNS * 2) :]

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history_pairs)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n{context_text}\n\n"
                f"User question:\n{user_message}\n\n"
                "Respond with grounded answer and citations."
            ),
        }
    )
    return messages


def generate_answer(session_id: str, user_message: str, contexts: List[Dict[str, object]]) -> str:
    if client is None:
        if not contexts:
            titles = [doc.get("title", "") for doc in read_documents()]
            title_hint = ", ".join([t for t in titles if t][:5])
            answer = (
                "I do not know based on the indexed documents. "
                f"Try topics like: {title_hint}."
            )
        else:
            snippets = [f"[{i+1}] {c['content']}" for i, c in enumerate(contexts[:2])]
            answer = "OpenAI key not configured, returning top retrieved context:\n\n" + "\n\n".join(snippets)
        memory.setdefault(session_id, []).extend(
            [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
        )
        return answer

    try:
        messages = build_messages(session_id, user_message, contexts)
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.1,
        )
        answer = completion.choices[0].message.content or "I do not know based on current documents."
    except Exception:
        if not contexts:
            titles = [doc.get("title", "") for doc in read_documents()]
            title_hint = ", ".join([t for t in titles if t][:5])
            answer = (
                "OpenAI is unavailable right now; using document-only fallback. "
                f"Try topics like: {title_hint}."
            )
        else:
            snippets = [f"[{i+1}] {c['content']}" for i, c in enumerate(contexts[:2])]
            answer = "OpenAI is unavailable right now, returning top retrieved context:\n\n" + "\n\n".join(snippets)
    memory.setdefault(session_id, []).extend(
        [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
    )
    return answer


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    with get_db() as conn:
        chunk_count = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    return jsonify(
        {
            "status": "ok",
            "indexed_chunks": chunk_count,
            "has_openai_key": bool(api_key),
            "embed_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
        }
    )


@app.post("/reindex")
def reindex():
    try:
        start = time.time()
        stats = index_documents()
        elapsed = round((time.time() - start) * 1000, 2)
        return jsonify({"ok": True, "stats": stats, "elapsed_ms": elapsed})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()
    session_id = str(payload.get("session_id", "default"))
    if not user_message:
        return jsonify({"ok": False, "error": "Message is required."}), 400
    try:
        start = time.time()
        contexts = retrieve(user_message, top_k=TOP_K)
        answer = generate_answer(session_id, user_message, contexts)
        elapsed = round((time.time() - start) * 1000, 2)
        return jsonify(
            {
                "ok": True,
                "answer": answer,
                "citations": [
                    {
                        "id": i + 1,
                        "title": c["title"],
                        "source": c["source"],
                        "score": c["score"],
                        "chunk_index": c["chunk_index"],
                    }
                    for i, c in enumerate(contexts)
                ],
                "latency_ms": elapsed,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/reset-memory")
def reset_memory():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "default"))
    memory[session_id] = []
    return jsonify({"ok": True, "session_id": session_id})


def ensure_index_ready() -> None:
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    if count == 0:
        index_documents()


def bootstrap_app() -> None:
    # Needed for WSGI servers (e.g., gunicorn on Render), where __main__ is not executed.
    init_db()


bootstrap_app()


if __name__ == "__main__":
    init_db()
    ensure_index_ready()
    port = _env_int("PORT", 5000)
    app.run(host="0.0.0.0", port=port, debug=True)
