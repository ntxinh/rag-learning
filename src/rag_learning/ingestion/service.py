"""Ingestion pipeline: discover → load → normalize → chunk → embed → upsert.

Idempotent via content-addressed ids:
- document_id = sha256(full normalized text)[:16] — content follows the doc,
  not the filename (rename an unchanged file → same document).
- chunk_id = f"{document_id}:{chunk_index}"
Changed file → new document_id; stale chunks for that source are deleted
first, so old and new representations never coexist.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from rag_learning.embeddings.service import EmbeddingService
from rag_learning.types import Chunk, IngestionReport
from rag_learning.vector_store.chroma_store import ChromaStore

from .chunking import chunk_text
from .loaders import LoadedDocument, discover, load

log = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaStore,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._embeddings = embedding_service
        self._store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest_directory(self, directory: Path) -> IngestionReport:
        report = IngestionReport()
        files = discover(directory)
        log.info("discovered %d supported files under %s", len(files), directory)
        for path in files:
            try:
                self.ingest_file(path)
                report.indexed.append(path.name)
            except ValueError as exc:
                log.warning("skipped %s: %s", path.name, exc)
                report.skipped.append(path.name)
            except Exception as exc:  # per-file isolation: report, keep going
                log.error("failed to ingest %s: %s", path.name, exc)
                report.failed[path.name] = str(exc)
        return report

    def ingest_file(self, path: Path) -> str:
        """Index one file; returns its document_id. Raises on failure."""
        doc = load(path)
        document_id = _document_id(doc)
        source = str(doc.source)

        # Replaced-content cutover: drop any prior chunks for this source.
        existing = self._store.document_id_for_source(source)
        if existing == document_id:
            log.info(
                "%s unchanged (document %s), skipping re-index",
                doc.file_name,
                document_id,
            )
            return document_id
        if existing:
            self._store.delete_document(source)

        chunks = self._make_chunks(doc, document_id)
        if not chunks:
            raise ValueError(f"{doc.file_name} produced no chunks")

        vectors = self._embeddings.embed_batch([c.text for c in chunks])
        self._store.upsert(chunks, vectors)
        log.info(
            "indexed %s as document %s (%d chunks)",
            doc.file_name,
            document_id,
            len(chunks),
        )
        return document_id

    def _make_chunks(self, doc: LoadedDocument, document_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for page_no, page_text in enumerate(doc.pages, start=1):
            for tc in chunk_text(page_text, self._chunk_size, self._chunk_overlap):
                is_pdf = doc.file_type == "pdf"
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}:{index}",
                        document_id=document_id,
                        text=tc.text,
                        metadata={
                            "document_id": document_id,
                            "source": str(doc.source),
                            "file_name": doc.file_name,
                            "file_type": doc.file_type,
                            "chunk_index": index,
                            "content_hash": document_id,
                            # page numbers only meaningful for PDFs
                            **({"page": page_no} if is_pdf else {}),
                        },
                    )
                )
                index += 1
        return chunks


def _document_id(doc: LoadedDocument) -> str:
    """Content-addressed id: identical content → identical id."""
    h = hashlib.sha256()
    for page in doc.pages:
        h.update(page.encode("utf-8"))
    return h.hexdigest()[:16]
