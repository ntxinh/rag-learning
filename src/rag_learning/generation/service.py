"""Ollama chat generation + citation validation.

No vector-store/embedding imports — evidence chunks arrive as arguments.
No evidence → insufficient-evidence answer WITHOUT calling the LLM.
"""

from __future__ import annotations

import logging
import re

import httpx

from rag_learning.types import Answer, Citation, RetrievedChunk

from .prompt import build_messages, pack_context

log = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = (
    "I don't have enough relevant information in the indexed documents to answer that."
)

_CITATION_RE = re.compile(r"\[(S\d+)\]")


class GenerationError(RuntimeError):
    """Ollama unreachable, timed out, or missing the model."""


class GenerationService:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        timeout: float,
        max_context_chars: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_context_chars = max_context_chars

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        packed = pack_context(chunks, self.max_context_chars)
        if not packed:
            log.info("no evidence under context budget; skipping LLM call")
            return Answer(
                text=INSUFFICIENT_EVIDENCE, citations=[], insufficient_evidence=True
            )

        messages = build_messages(question, packed)
        raw = self._chat(messages)
        return self._attach_citations(raw, packed)

    # -- internals ------------------------------------------------------------

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
        except httpx.ConnectError as exc:
            raise GenerationError(
                f"cannot reach Ollama at {self.base_url} — is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise GenerationError(
                f"Ollama timed out after {self.timeout}s — reduce context or raise "
                "RAG_LLM_TIMEOUT"
            ) from exc

        if resp.status_code == 404:
            raise GenerationError(
                f"model '{self.model}' not found in Ollama — "
                f"run `ollama pull {self.model}`"
            )
        resp.raise_for_status()
        return str(resp.json()["message"]["content"])

    def _attach_citations(
        self, text: str, packed: list[tuple[str, RetrievedChunk]]
    ) -> Answer:
        """Map every emitted [Sx] to its real chunk; strip and count invented ids."""
        valid = {cid: rc for cid, rc in packed}

        def strip_unknown(m: re.Match[str]) -> str:
            return m.group(0) if m.group(1) in valid else ""

        cleaned, removed = _strip_invalid(text, valid)
        if removed:
            log.warning("stripped %d invented citation(s) from answer", removed)

        used_ids = [m.group(1) for m in _CITATION_RE.finditer(cleaned)]
        citations: list[Citation] = []
        seen: set[str] = set()
        for cid in used_ids:
            if cid in seen:
                continue
            seen.add(cid)
            rc = valid[cid]
            meta = rc.chunk.metadata
            citations.append(
                Citation(
                    citation_id=cid,
                    file_name=str(meta.get("file_name", "")),
                    source=str(meta.get("source", "")),
                    page=meta.get("page"),
                    chunk_id=rc.chunk.chunk_id,
                    excerpt=rc.chunk.text[:200],
                )
            )
        return Answer(
            text=cleaned.strip(),
            citations=citations,
            insufficient_evidence=False,
            invalid_citations_removed=removed,
        )


def _strip_invalid(text: str, valid: dict[str, RetrievedChunk]) -> tuple[str, int]:
    removed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal removed
        if m.group(1) in valid:
            return m.group(0)
        removed += 1
        return ""

    return _CITATION_RE.sub(repl, text), removed
