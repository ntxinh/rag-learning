"""hit-rate / MRR / precision on synthetic ranked results."""

from __future__ import annotations

from rag_learning.evaluation.metrics import first_hit_rank, mrr, precision_at_k
from rag_learning.types import Chunk, RetrievedChunk


def _rc(rank: int, file_name: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk=Chunk(
            chunk_id=f"d:{rank}",
            document_id="d",
            text="t",
            metadata={"file_name": file_name},
        ),
        distance=0.0,
        similarity=1.0,
    )


def test_first_hit_rank():
    chunks = [_rc(1, "a.md"), _rc(2, "b.md"), _rc(3, "c.md")]
    assert first_hit_rank(chunks, ["c.md"]) == 3
    assert first_hit_rank(chunks, ["zzz.md"]) is None


def test_mrr():
    chunks = [_rc(1, "a.md"), _rc(2, "b.md")]
    assert mrr(chunks, ["a.md"]) == 1.0
    assert mrr(chunks, ["b.md"]) == 0.5
    assert mrr(chunks, ["none"]) == 0.0


def test_precision_at_k():
    chunks = [_rc(1, "a.md"), _rc(2, "x.md"), _rc(3, "b.md"), _rc(4, "y.md")]
    # expected {a,b}; top-2 = a,x → 1/2 hits
    assert precision_at_k(chunks, ["a.md", "b.md"], 2) == 0.5
    # top-4 = a,x,b,y → 2/4
    assert precision_at_k(chunks, ["a.md", "b.md"], 4) == 0.5
    assert precision_at_k(chunks, ["a.md"], 2) == 0.5


def test_empty_results():
    assert first_hit_rank([], ["a.md"]) is None
    assert mrr([], ["a.md"]) == 0.0
    assert precision_at_k([], ["a.md"], 5) == 0.0
