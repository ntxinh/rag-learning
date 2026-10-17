"""Application seam: retrieve → generate, shared by CLI and API."""

from __future__ import annotations

from dataclasses import dataclass

from rag_learning.config import Settings
from rag_learning.embeddings.service import EmbeddingService
from rag_learning.generation.service import GenerationService
from rag_learning.retrieval.service import RetrievalService
from rag_learning.types import Answer, RetrievedChunk
from rag_learning.vector_store.chroma_store import ChromaStore


@dataclass
class QAResult:
    answer: Answer
    evidence: list[RetrievedChunk]


class QuestionAnsweringApp:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService | None = None,
        store: ChromaStore | None = None,
        generator: GenerationService | None = None,
    ):
        self._settings = settings
        self.embeddings = embeddings or EmbeddingService(
            settings.embedding_model, settings.embedding_device
        )
        self.store = store or ChromaStore(settings.vector_dir, settings.collection_name)
        self.retriever = RetrievalService(
            self.embeddings,
            self.store,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )
        self.generator = generator or GenerationService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            max_context_chars=settings.max_context_chars,
        )

    def retrieve(
        self, question: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, top_k=top_k)

    def ask(self, question: str, *, top_k: int | None = None) -> QAResult:
        evidence = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, evidence)
        return QAResult(answer=answer, evidence=evidence)
