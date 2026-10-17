"""Real-Chroma round-trip: upsert → query → delete → stats.

Uses a tmp dir and hand-made vectors — no embedding model needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_learning.types import Chunk
from rag_learning.vector_store.chroma_store import ChromaStore

pytestmark = pytest.mark.integration


def _chunk(i: int, doc_id: str = "d1", source: str = "data/f.md") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{i}",
        document_id=doc_id,
        text=f"text {i}",
        metadata={
            "document_id": doc_id,
            "source": source,
            "file_name": "f.md",
            "file_type": "md",
            "chunk_index": i,
            "content_hash": doc_id,
        },
    )


def test_round_trip(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma", "test")
    chunks = [_chunk(0), _chunk(1), _chunk(2, doc_id="d2", source="data/g.md")]
    vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
    store.upsert(chunks, vectors)

    # query closest to first vector
    hits = store.query([1.0, 0.0, 0.0], 2)
    assert len(hits) == 2
    assert hits[0]["id"] == "d1:0"
    assert hits[0]["distance"] < hits[1]["distance"]

    # idempotent re-upsert
    store.upsert(chunks, vectors)
    assert store.stats()["chunks"] == 3

    # delete one document only
    assert store.delete_document("data/g.md") == 1
    assert store.document_id_for_source("data/g.md") is None
    assert store.document_id_for_source("data/f.md") == "d1"

    docs = store.list_documents()
    assert [d["document_id"] for d in docs] == ["d1"]
    assert docs[0]["chunks"] == 2

    store.reset()
    assert store.stats()["chunks"] == 0


def test_persistence_across_instances(tmp_path: Path) -> None:
    d = tmp_path / "chroma"
    ChromaStore(d, "test").upsert([_chunk(0)], [[1.0, 0.0, 0.0]])
    assert ChromaStore(d, "test").stats()["chunks"] == 1
