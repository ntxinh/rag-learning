"""Prompt packing, budget, citation-id mapping, and validation."""

from __future__ import annotations

from rag_learning.generation.prompt import build_messages, pack_context
from rag_learning.generation.service import _strip_invalid
from rag_learning.types import Chunk, RetrievedChunk


def _rc(rank: int, text: str, file_name: str = "f.md") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"doc:{rank}",
        document_id="doc",
        text=text,
        metadata={"file_name": file_name, "source": f"data/{file_name}"},
    )
    return RetrievedChunk(rank=rank, chunk=chunk, distance=0.1, similarity=0.9)


def test_pack_context_assigns_ids_in_rank_order():
    packed = pack_context([_rc(1, "aaa"), _rc(2, "bbb")], 10_000)
    assert [cid for cid, _ in packed] == ["S1", "S2"]


def test_pack_context_respects_budget_drops_wholes():
    # each block ~250 chars; budget 300 → only first fits
    chunks = [_rc(i, "x" * 200) for i in range(1, 5)]
    packed = pack_context(chunks, 300)
    assert len(packed) == 1


def test_pack_context_tries_shorter_later_chunks():
    big = _rc(1, "x" * 500)
    small = _rc(2, "tiny")
    packed = pack_context([big, small], 400)
    assert len(packed) == 1
    assert packed[0][1].chunk.text == "tiny"


def test_build_messages_contains_question_and_sources():
    packed = pack_context([_rc(1, "evidence text")], 10_000)
    msgs = build_messages("q?", packed)
    assert msgs[0]["role"] == "system"
    assert "q?" in msgs[1]["content"]
    assert "[S1]" in msgs[1]["content"]
    assert "evidence text" in msgs[1]["content"]


def test_strip_invalid_citations():
    valid = {"S1": _rc(1, "x"), "S2": _rc(2, "y")}
    text = "fact [S1] other [S9] more [S2]"
    cleaned, removed = _strip_invalid(text, valid)
    assert "[S1]" in cleaned and "[S2]" in cleaned
    assert "[S9]" not in cleaned
    assert removed == 1
