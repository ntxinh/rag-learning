"""ChromaDB persistent store adapter.

Collection uses Chroma's default cosine-distance space: distance 0 = identical,
2 = opposite. `query` returns raw distances — callers (retrieval) convert to
similarity; never present a distance as a score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rag_learning.types import Chunk

log = logging.getLogger(__name__)


class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str):
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        log.info("vector store ready: %s @ %s", collection_name, persist_dir)

    # -- writes ---------------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace; identical chunk_ids are no-ops."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=np.asarray(vectors, dtype=np.float32),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def delete_document(self, source: str) -> int:
        """Remove all chunks for a source path; returns deleted count."""
        found = self._collection.get(where={"source": source}, include=[])
        if not found["ids"]:
            return 0
        self._collection.delete(ids=found["ids"])
        return len(found["ids"])

    def reset(self) -> None:
        """Wipe every entry. Guarded by callers (CLI --yes, API admin)."""
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)

    # -- reads ----------------------------------------------------------------

    def query(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Raw ranked hits: id, text, metadata, distance — unfiltered."""
        res = self._collection.query(
            query_embeddings=np.asarray([vector], dtype=np.float32),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        docs = res["documents"]
        metas = res["metadatas"]
        dists = res["distances"]
        for row_id, doc, meta, dist in zip(
            res["ids"][0],
            (docs or [[]])[0],
            (metas or [[]])[0],
            (dists or [[]])[0],
            strict=True,
        ):
            out.append(
                {
                    "id": row_id,
                    "text": doc,
                    "metadata": dict(meta),
                    "distance": float(dist),
                }
            )
        return out

    def document_id_for_source(self, source: str) -> str | None:
        """document_id of chunks currently indexed for this source, if any."""
        res = self._collection.get(
            where={"source": source}, include=["metadatas"], limit=1
        )
        metas = res["metadatas"]
        if not res["ids"] or not metas:
            return None
        return str(metas[0].get("document_id") or "")

    def list_documents(self) -> list[dict[str, Any]]:
        """One row per document_id present in the collection."""
        res = self._collection.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for meta in res["metadatas"] or []:
            doc_id = str(meta.get("document_id", ""))
            entry = docs.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "file_name": meta.get("file_name", ""),
                    "source": meta.get("source", ""),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["file_name"])

    def stats(self) -> dict[str, Any]:
        return {
            "collection": self._collection.name,
            "chunks": self._collection.count(),
            "documents": len(self.list_documents()),
        }
