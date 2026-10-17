"""Deterministic character-based chunker.

Split preference: paragraph boundary → sentence boundary → hard cut.
`chunk_size`/`chunk_overlap` are CHARACTERS, not tokens.
Invariant: every emitted chunk is non-empty and <= chunk_size chars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")  # noqa: RUF001 — CJK punctuation intentional


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def _hard_cut(text: str, size: int, overlap: int) -> list[str]:
    """Plain windows with overlap; the guaranteed-termination fallback."""
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Split `text` into deterministic chunks.

    Raises ValueError on invalid config. Pure function — same input, same output.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    if not text.strip():
        return []

    # Elementary segments: paragraphs that fit, else sentences, else hard cuts.
    segments: list[str] = []
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if len(para) <= chunk_size:
            segments.append(para)
        else:
            for sent in _split_sentences(para):
                if len(sent) <= chunk_size:
                    segments.append(sent.strip())
                else:
                    segments.extend(_hard_cut(sent, chunk_size, chunk_overlap))

    chunks: list[str] = []
    current = ""
    for seg in segments:
        candidate = f"{current}\n\n{seg}" if current else seg
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = seg

    if current:
        chunks.append(current)

    # Carry `chunk_overlap` chars of tail context into the next chunk,
    # trimmed to a word boundary so we don't start mid-word.
    out: list[TextChunk] = []
    for i, text_piece in enumerate(chunks):
        if i > 0 and chunk_overlap > 0:
            tail = chunks[i - 1][-chunk_overlap:]
            word_start = tail.find(" ")
            if word_start != -1:
                tail = tail[word_start + 1 :]
            merged = f"{tail}\n{text_piece}"
            if len(merged) <= chunk_size:
                text_piece = merged
            # else: piece already at budget; skip overlap for this chunk
        text_piece = text_piece.strip()
        if text_piece:
            out.append(TextChunk(text=text_piece, chunk_index=len(out)))
    return out
