"""Ollama generation path — auto-skipped when the server or model is absent."""

from __future__ import annotations

import httpx
import pytest

from rag_learning.config import get_settings
from rag_learning.generation.service import GenerationService
from rag_learning.types import Chunk, RetrievedChunk

pytestmark = pytest.mark.integration


def _ollama_up() -> bool:
    s = get_settings()
    try:
        r = httpx.get(f"{s.ollama_base_url}/api/tags", timeout=2)
        return s.ollama_model in r.text
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="Ollama/model unavailable")


def test_generate_with_citation():
    s = get_settings()
    svc = GenerationService(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        temperature=0.0,
        timeout=s.llm_timeout,
        max_context_chars=s.max_context_chars,
    )
    chunk = Chunk(
        chunk_id="d:0",
        document_id="d",
        text="The sky appears blue because of Rayleigh scattering.",
        metadata={"file_name": "sky.md", "source": "data/sky.md"},
    )
    rc = RetrievedChunk(rank=1, chunk=chunk, distance=0.1, similarity=0.9)
    answer = svc.generate("Why is the sky blue?", [rc])
    assert answer.text
    assert not answer.insufficient_evidence
    # citations may be empty if the model doesn't cite — but none may be invalid
    assert answer.invalid_citations_removed == 0
