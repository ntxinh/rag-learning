"""Retrieval: embed question → top-k → threshold → ranked chunks.

Standalone — knows nothing about generation. Output preserves source
attribution; "no usable evidence" is a first-class (empty) result, not an error.
"""

from __future__ import annotations

import logging

from rag_learning.embeddings.service import EmbeddingService
from rag_learning.types import Chunk, RetrievedChunk
from rag_learning.vector_store.chroma_store import ChromaStore

log = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore,
        *,
        top_k: int,
        similarity_threshold: float | None = None,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self, question: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        k = top_k or self.top_k
        vector = self._embeddings.embed(question.strip())
        hits = self._store.query(vector, k)

        out: list[RetrievedChunk] = []
        for hit in hits:
            similarity = 1.0 - hit["distance"]  # cosine distance → similarity
            if (
                self.similarity_threshold is not None
                and similarity < self.similarity_threshold
            ):
                continue
            meta = hit["metadata"]
            chunk = Chunk(
                chunk_id=hit["id"],
                document_id=str(meta.get("document_id", "")),
                text=hit["text"],
                metadata=meta,
            )
            out.append(
                RetrievedChunk(
                    rank=len(out) + 1,
                    chunk=chunk,
                    distance=hit["distance"],
                    similarity=similarity,
                )
            )

        log.info("retrieve(%.40r) -> %d chunks (k=%d)", question, len(out), k)
        return out
