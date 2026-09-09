"""
rag_pipeline.py — SHIELD RAG Chatbot Pipeline
===============================================
Features:
  • Ingest trip reports (ETL output) into ChromaDB
  • OpenAI embeddings + GPT-4o-mini for grounded answers
  • Rolling conversation memory
  • ask_sync() for Streamlit (blocking)
  • Falls back gracefully when OpenAI key is missing
  • Session-scoped — each TripSession gets its own collection
  • clear_history() / clear_documents() / chunk_count
"""

from __future__ import annotations

import hashlib
import textwrap
import time
import uuid
from typing import Optional

from config import settings
from db import get_session
from models import ChatMessage
from openai import OpenAI


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 20]


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:

    def __init__(self):
        self._session_id   = str(uuid.uuid4())
        self._history:     list[dict] = []
        self._doc_count:   int        = 0
        self._seen_hashes: set[str]   = set()

        self._client  = None   # OpenAI lazy init
        self._chroma  = None   # ChromaDB lazy init
        self._col     = None   # collection lazy init
        self._emb_fn  = None

    # ── Lazy inits ────────────────────────────────────────────────────────────

    def _get_openai(self):
        if self._client is None:
         if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set.")


        self._client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        return self._client

    def _get_collection(self):
     if self._col is None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        # Load free embedding model
        self._embed_model = SentenceTransformer(settings.EMBED_MODEL)

        # Custom embedding function
        def embed_fn(texts):
            return self._embed_model.encode(texts).tolist()

        if settings.chroma_in_memory:
            self._chroma = chromadb.Client()
        else:
            self._chroma = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

        self._col = self._chroma.get_or_create_collection(
            name=f"shield_{self._session_id}",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        return self._col

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest_report(self, report_text: str, source: str = "trip_report") -> int:
        """
        Chunk a trip report string and upsert into ChromaDB.
        Returns number of NEW chunks added.
        """
        if not report_text.strip():
            return 0

        chunks = _chunk_text(
            report_text,
            settings.RAG_CHUNK_SIZE,
            settings.RAG_CHUNK_OVERLAP,
        )

        new_docs  = []
        new_ids   = []
        new_metas = []

        for i, chunk in enumerate(chunks):
            h = _content_hash(chunk)
            if h in self._seen_hashes:
                continue
            self._seen_hashes.add(h)
            cid = f"{source}_{self._doc_count + len(new_docs)}_{h}"
            new_docs.append(chunk)
            new_ids.append(cid)
            new_metas.append({"source": source, "chunk": i, "hash": h})

        if not new_docs:
            return 0

        try:
            col = self._get_collection()
            col.upsert(documents=new_docs, ids=new_ids, metadatas=new_metas)
            self._doc_count += len(new_docs)
        except Exception as exc:
            return 0

        return len(new_docs)

    # ── Ask (sync — for Streamlit) ────────────────────────────────────────────

    def ask(
        self,
        query:      str,
        session_id: Optional[int] = None,
    ) -> str:
        """
        Full RAG ask. Blocking. Use directly in Streamlit.
        Persists messages to DB when session_id is provided.
        """
        if not query.strip():
            return "Please type a question."

        # Fallback when no OpenAI key
        # Fallback when no Groq key
        if not settings.GROQ_API_KEY:
            return (
        "⚠️ Chatbot requires a Groq API key.\n"
        "Add `GROQ_API_KEY=gsk-...` to your `.env` file and restart."
    )

        start = time.perf_counter()

        context_block, n_chunks = self._retrieve(query)
        system_prompt           = self._build_system(context_block, n_chunks > 0)

        messages = [{"role": "system", "content": system_prompt}]
        messages += self._history[-(settings.RAG_HISTORY_TURNS * 2):]
        messages.append({"role": "user", "content": query})

        try:
            client = self._get_openai()
            resp   = client.chat.completions.create(
                model       = settings.OPENAI_MODEL,
                messages    = messages,
                temperature = settings.OPENAI_TEMPERATURE,
                max_tokens  = settings.OPENAI_MAX_TOKENS,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as exc:
            answer = f"⚠️ LLM error: {exc}"

        latency_ms = int((time.perf_counter() - start) * 1000)

        self._history.append({"role": "user",      "content": query})
        self._history.append({"role": "assistant", "content": answer})

        # Persist to DB
        if session_id:
            try:
                with get_session() as db:
                    db.add(ChatMessage(
                        session_id  = session_id,
                        role        = "user",
                        content     = query,
                    ))
                    db.add(ChatMessage(
                        session_id  = session_id,
                        role        = "assistant",
                        content     = answer,
                        chunks_used = n_chunks,
                        latency_ms  = latency_ms,
                    ))
                    db.commit()
            except Exception:
                pass

        return answer

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> tuple[str, int]:
        if self._doc_count == 0:
            return "", 0
        try:
            col     = self._get_collection()
            results = col.query(
                query_texts = [query],
                n_results   = min(settings.RAG_TOP_K, self._doc_count),
                include     = ["documents", "distances", "metadatas"],
            )
        except Exception:
            return "", 0

        docs      = results["documents"][0]
        distances = results["distances"][0]
        metas     = results["metadatas"][0]

        good = [
            (doc, meta)
            for doc, dist, meta in zip(docs, distances, metas)
            if dist <= settings.RAG_SIM_THRESHOLD
        ]
        if not good:
            good = list(zip(docs[:2], metas[:2]))

        parts = []
        for i, (doc, meta) in enumerate(good):
            parts.append(f"[{meta.get('source','trip')} | chunk {i+1}]\n{doc}")

        return "\n\n---\n\n".join(parts), len(good)

    # ── System prompt ─────────────────────────────────────────────────────────

    def _build_system(self, context: str, has_context: bool) -> str:
        if has_context:
            return textwrap.dedent(f"""
                You are SHIELD, an intelligent road safety assistant.

                Use ONLY the trip report data below to answer questions.
                If information is not in the report, say so clearly.
                Be concise, safety-focused, and friendly.
                Always include actionable advice when relevant.

                TRIP DATA:
                ──────────────────────────────
                {context}
                ──────────────────────────────
            """).strip()
        else:
            return textwrap.dedent("""
                You are SHIELD, an intelligent road safety assistant.
                No trip data has been loaded yet.
                Tell the user to upload a dashcam image or video first,
                then they can ask questions about their trip.
                You can still answer general road safety questions.
            """).strip()

    # ── Utility ───────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self._history = []

    def clear_documents(self) -> None:
        if self._chroma and self._col:
            try:
                self._chroma.delete_collection(f"shield_{self._session_id}")
            except Exception:
                pass
        self._session_id   = str(uuid.uuid4())
        self._col          = None
        self._doc_count    = 0
        self._seen_hashes  = set()

    @property
    def chunk_count(self) -> int:
        return self._doc_count

    @property
    def history_length(self) -> int:
        return len(self._history) // 2
