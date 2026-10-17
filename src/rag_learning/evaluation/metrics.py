"""Retrieval metrics: hit-rate@k, MRR, precision@k. Pure functions on results."""

from __future__ import annotations

from rag_learning.types import RetrievedChunk


def first_hit_rank(
    chunks: list[RetrievedChunk], expected_sources: list[str]
) -> int | None:
    """Rank of the first result whose file_name is expected; None if absent."""
    expected = set(expected_sources)
    for rc in chunks:
        if rc.chunk.metadata.get("file_name") in expected:
            return rc.rank
    return None


def mrr(chunks: list[RetrievedChunk], expected_sources: list[str]) -> float:
    rank = first_hit_rank(chunks, expected_sources)
    return 1.0 / rank if rank else 0.0


def precision_at_k(
    chunks: list[RetrievedChunk], expected_sources: list[str], k: int
) -> float:
    if k <= 0:
        return 0.0
    top = [rc for rc in chunks if rc.rank <= k]
    if not top:
        return 0.0
    expected = set(expected_sources)
    hits = sum(1 for rc in top if rc.chunk.metadata.get("file_name") in expected)
    return hits / len(top)


def hit_at_k(chunks: list[RetrievedChunk], expected_sources: list[str], k: int) -> bool:
    return any(rc.rank <= k for rc in chunks if first_hit_rank([rc], expected_sources))
