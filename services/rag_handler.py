"""
services/rag_handler.py — Local ChromaDB RAG with BGE embeddings for document Q&A.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from typing import Dict, List

import chromadb
import nltk
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from FlagEmbedding import FlagModel
from loguru import logger

from config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_DB_DIR,
    RAG_EMBED_MODEL,
    RAG_RELEVANCE_THRESHOLD,
    RAG_TOP_K,
)

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass


class _BGEEmbeddingFunction(EmbeddingFunction):
    """Offline ChromaDB embedding function backed by FlagEmbedding (BAAI/bge-*)."""

    def __init__(self, model_name: str) -> None:
        self._model = FlagModel(model_name, use_fp16=False, devices=["cpu"])

    def __call__(self, input: Documents) -> Embeddings:
        # Symmetric encoding for both passages and queries: the asymmetric
        # encode_queries() path pushed borderline matches over the threshold.
        embeddings = self._model.encode(list(input))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.where(norms == 0, 1, norms)).tolist()


class RAGHandler:
    """A local ChromaDB collection of document chunks, queried per turn."""

    def __init__(self) -> None:
        os.makedirs(RAG_DB_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self.client = chromadb.PersistentClient(path=RAG_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name="documents_bge",
            embedding_function=_BGEEmbeddingFunction(RAG_EMBED_MODEL),
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[rag] Ready — {self.collection.count()} chunks stored")

    def has_documents(self) -> bool:
        return self.collection.count() > 0

    def add_document(self, text: str, source: str) -> Dict:
        """Chunk and index one document. Re-uploading the same source is a no-op."""
        existing = self.collection.get(where={"source": source}, limit=1)
        if existing.get("ids"):
            metas = existing.get("metadatas") or [{}]
            return {
                "doc_id": metas[0].get("doc_id", "existing"),
                "chunks": 0, "source": source, "skipped": True,
            }

        doc_id = uuid.uuid4().hex[:8]
        chunks = self._chunk(text)
        if not chunks:
            return {"doc_id": doc_id, "chunks": 0, "source": source}

        with self._lock:
            self.collection.add(
                documents=chunks,
                ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
                metadatas=[
                    {"doc_id": doc_id, "source": source, "chunk": i}
                    for i in range(len(chunks))
                ],
            )
        logger.info(f"[rag] Added '{source}' as {doc_id}: {len(chunks)} chunks")
        return {"doc_id": doc_id, "chunks": len(chunks), "source": source}

    def query_texts_only(self, query_text: str, n_results: int = RAG_TOP_K) -> List[str]:
        """Top-K chunk texts within the relevance threshold; empty if none qualify."""
        if not self.has_documents():
            return []
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "distances"],
        )
        docs = (results.get("documents") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        if not docs:
            return []

        hits = [d for d, dist in zip(docs, dists) if dist <= RAG_RELEVANCE_THRESHOLD]
        logger.debug(
            f"[rag] distances={[round(d, 3) for d in dists]} "
            f"threshold={RAG_RELEVANCE_THRESHOLD} kept={len(hits)}"
        )
        return hits

    def list_documents(self) -> List[Dict]:
        items = self.collection.get(include=["metadatas"])
        seen: Dict[str, str] = {}
        for meta in (items.get("metadatas") or []):
            did = meta.get("doc_id", "")
            if did and did not in seen:
                seen[did] = meta.get("source", did)
        return [{"id": k, "name": v} for k, v in seen.items()]

    def delete_document(self, doc_id: str) -> int:
        ids = self.collection.get(where={"doc_id": doc_id}).get("ids", [])
        if ids:
            with self._lock:
                self.collection.delete(ids=ids)
        logger.info(f"[rag] Deleted doc {doc_id}: {len(ids)} chunks")
        return len(ids)

    def _chunk(self, text: str) -> List[str]:
        """Split into ~RAG_CHUNK_SIZE-word chunks on sentence boundaries, with overlap."""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = None

        if not sentences:
            words = text.split()
            step = max(1, RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP)
            return [
                " ".join(words[i:i + RAG_CHUNK_SIZE])
                for i in range(0, len(words), step)
                if words[i:i + RAG_CHUNK_SIZE]
            ]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > RAG_CHUNK_SIZE and current:
                chunks.append(" ".join(current))
                overlap = " ".join(current).split()[-RAG_CHUNK_OVERLAP:]
                current = [" ".join(overlap)] if overlap else []
                current_len = len(overlap)
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(" ".join(current))
        return chunks
