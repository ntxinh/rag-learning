"""FastAPI app factory. Services are built once and shared via app.state."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_learning.application.ingestion_service import IngestionApp
from rag_learning.application.question_answering import QuestionAnsweringApp
from rag_learning.config import get_settings
from rag_learning.embeddings.service import EmbeddingService
from rag_learning.logging_config import configure_logging
from rag_learning.vector_store.chroma_store import ChromaStore

from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    # Share one embedding model + store across both app services.
    embeddings = EmbeddingService(settings.embedding_model, settings.embedding_device)
    store = ChromaStore(settings.vector_dir, settings.collection_name)
    app.state.ingestion = IngestionApp(settings, embeddings=embeddings, store=store)
    app.state.qa = QuestionAnsweringApp(settings, embeddings=embeddings, store=store)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="rag-learning", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
