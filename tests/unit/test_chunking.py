"""Chunker edge cases mandated by the spec: empty, boundary, oversize,
paragraphs, overlap, unicode/Vietnamese, invalid config."""

from __future__ import annotations

import itertools

import pytest

from rag_learning.ingestion.chunking import chunk_text


def test_empty_input():
    assert chunk_text("", 100, 20) == []
    assert chunk_text("   \n\n  ", 100, 20) == []


def test_short_text_single_chunk():
    out = chunk_text("hello world", 100, 20)
    assert len(out) == 1
    assert out[0].text == "hello world"
    assert out[0].chunk_index == 0


def test_exact_boundary():
    text = "x" * 100
    out = chunk_text(text, 100, 20)
    assert len(out) == 1
    assert out[0].text == text


def test_long_text_splits_within_size():
    text = ("word " * 400).strip()  # ~2000 chars, no paragraph breaks
    out = chunk_text(text, 500, 50)
    assert len(out) > 1
    assert all(len(c.text) <= 500 for c in out)
    assert [c.chunk_index for c in out] == list(range(len(out)))


def test_paragraph_boundary_preferred():
    p1 = "a" * 400
    p2 = "b" * 400
    p3 = "c" * 400
    text = f"{p1}\n\n{p2}\n\n{p3}"
    out = chunk_text(text, 900, 100)
    # 900 fits two paragraphs, not three → expect a paragraph-aligned split
    assert len(out) == 2
    assert out[0].text.startswith("a" * 100)


def test_overlap_carries_context():
    paragraphs = [f"{'abcdefghij' * 20}_p{i}" for i in range(4)]
    text = "\n\n".join(paragraphs)
    out = chunk_text(text, 300, 50)
    assert len(out) > 1
    # every chunk after the first shares tail text with its predecessor
    for prev, cur in itertools.pairwise(out):
        tail = prev.text[-50:]
        shared = any(tail[i:] in cur.text for i in range(len(tail)))
        assert shared, "expected overlap between consecutive chunks"


def test_unicode_and_vietnamese():
    text = "Tiếng Việt có dấu. " * 80 + "\n\n" + "日本語のテキスト。" * 30
    out = chunk_text(text, 300, 30)
    assert len(out) > 1
    assert all(len(c.text) <= 300 for c in out)
    assert "Tiếng Việt" in out[0].text


def test_invalid_config():
    with pytest.raises(ValueError):
        chunk_text("abc", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 100)  # overlap == size
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 150)
    with pytest.raises(ValueError):
        chunk_text("abc", 100, -1)


def test_no_empty_chunks_and_terminates():
    text = "\n\n".join(f"para {i} " + "x" * 250 for i in range(20))
    out = chunk_text(text, 200, 60)
    assert out
    assert all(c.text.strip() for c in out)
