"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)


class CitationOut(BaseModel):
    citation_id: str
    file_name: str
    source: str
    page: int | None
    chunk_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    insufficient_evidence: bool
    invalid_citations_removed: int


class ChunkOut(BaseModel):
    rank: int
    chunk_id: str
    file_name: str
    source: str
    page: int | None
    similarity: float
    distance: float
    text: str


class RetrieveResponse(BaseModel):
    question: str
    results: list[ChunkOut]


class IngestRequest(BaseModel):
    directory: str | None = None  # default: configured data dir


class IngestResponse(BaseModel):
    indexed: list[str]
    skipped: list[str]
    failed: dict[str, str]


class StatsResponse(BaseModel):
    collection: str
    chunks: int
    documents: int


class DocumentOut(BaseModel):
    document_id: str
    file_name: str
    source: str
    chunks: int


class DeleteResponse(BaseModel):
    deleted: int
