"""HTTP endpoints over the application services."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from rag_learning.generation.service import GenerationError
from rag_learning.types import RetrievedChunk

from .schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    DeleteResponse,
    DocumentOut,
    IngestRequest,
    IngestResponse,
    RetrieveResponse,
    StatsResponse,
)

router = APIRouter()


def _chunk_out(rc: RetrievedChunk) -> ChunkOut:
    meta = rc.chunk.metadata
    return ChunkOut(
        rank=rc.rank,
        chunk_id=rc.chunk.chunk_id,
        file_name=str(meta.get("file_name", "")),
        source=str(meta.get("source", "")),
        page=meta.get("page"),
        similarity=round(rc.similarity, 4),
        distance=round(rc.distance, 4),
        text=rc.chunk.text,
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(body: AskRequest, request: Request) -> RetrieveResponse:
    qa = request.app.state.qa
    try:
        results = qa.retrieve(body.question, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RetrieveResponse(
        question=body.question, results=[_chunk_out(r) for r in results]
    )


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    qa = request.app.state.qa
    try:
        result = qa.ask(body.question, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    a = result.answer
    return AskResponse(
        answer=a.text,
        citations=[c.__dict__ for c in a.citations],
        insufficient_evidence=a.insufficient_evidence,
        invalid_citations_removed=a.invalid_citations_removed,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    ing = request.app.state.ingestion
    report = ing.ingest(Path(body.directory) if body.directory else None)
    return IngestResponse(
        indexed=report.indexed, skipped=report.skipped, failed=report.failed
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(request: Request) -> list[DocumentOut]:
    return [
        DocumentOut(**d) for d in request.app.state.ingestion.store.list_documents()
    ]


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, request: Request) -> DeleteResponse:
    deleted = request.app.state.ingestion.remove_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="document not found")
    return DeleteResponse(deleted=deleted)


@router.get("/stats", response_model=StatsResponse)
def stats(request: Request) -> StatsResponse:
    return StatsResponse(**request.app.state.ingestion.stats())


@router.post("/reset", response_model=StatsResponse)
def reset(request: Request) -> StatsResponse:
    request.app.state.ingestion.reset()
    return stats(request)
