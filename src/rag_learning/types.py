"""Shared value types. One home so no layer imports another layer's internals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """One indexed text fragment."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its rank and scores."""

    rank: int
    chunk: Chunk
    distance: float  # cosine distance: 0 identical .. 2 opposite
    similarity: float  # 1 - distance


@dataclass(frozen=True)
class Citation:
    """Validated pointer from an answer to a retrieved chunk."""

    citation_id: str  # "S1", "S2", ...
    file_name: str
    source: str
    page: int | None
    chunk_id: str
    excerpt: str


@dataclass
class Answer:
    """Generation output: grounded text plus validated citations."""

    text: str
    citations: list[Citation]
    insufficient_evidence: bool = False
    invalid_citations_removed: int = 0


@dataclass
class IngestionReport:
    """Per-run stats; a file lands in exactly one bucket."""

    indexed: list[str] = field(default_factory=list)  # document_ids
    skipped: list[str] = field(default_factory=list)  # empty/unsupported files
    failed: dict[str, str] = field(default_factory=dict)  # source -> error
