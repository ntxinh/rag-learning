"""Application seam: one place that wires Settings → services for callers.

CLI and API both build this; tests can inject fakes at the same constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_learning.config import Settings
from rag_learning.embeddings.service import EmbeddingService
from rag_learning.ingestion.service import IngestionService as _Pipeline
from rag_learning.types import IngestionReport
from rag_learning.vector_store.chroma_store import ChromaStore


class IngestionApp:
    """Thin orchestrator over the ingestion pipeline."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService | None = None,
        store: ChromaStore | None = None,
    ):
        self._settings = settings
        self.embeddings = embeddings or EmbeddingService(
            settings.embedding_model, settings.embedding_device
        )
        self.store = store or ChromaStore(settings.vector_dir, settings.collection_name)
        self._pipeline = _Pipeline(
            self.embeddings,
            self.store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest(self, directory: Path | None = None) -> IngestionReport:
        return self._pipeline.ingest_directory(directory or self._settings.data_dir)

    def remove_document(self, source: str) -> int:
        """Delete chunks for a source path or document_id."""
        # Try source first, then document_id.
        deleted = self.store.delete_document(source)
        if deleted:
            return deleted
        for doc in self.store.list_documents():
            if doc["document_id"] == source:
                return self.store.delete_document(str(doc["source"]))
        return 0

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def reset(self) -> None:
        self.store.reset()
