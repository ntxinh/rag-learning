"""Settings validation: positive sizes, overlap < size."""

from __future__ import annotations

import pytest

from rag_learning.config import Settings


def test_defaults_valid():
    s = Settings()
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k > 0


def test_overlap_must_be_below_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=200)


def test_positive_constraints():
    with pytest.raises(ValueError):
        Settings(chunk_size=0)
    with pytest.raises(ValueError):
        Settings(top_k=0)
    with pytest.raises(ValueError):
        Settings(max_context_chars=-1)


def test_threshold_range():
    assert Settings(similarity_threshold=0.5).similarity_threshold == 0.5
    with pytest.raises(ValueError):
        Settings(similarity_threshold=1.5)
