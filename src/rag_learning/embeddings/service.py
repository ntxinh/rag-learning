"""Sentence Transformers embedding service.

One instance per process; the model loads lazily on first use and is reused
for indexing AND querying (they must always be the same model/config).
Vectors are L2-normalized so Chroma cosine distance is well-defined.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                log.info(
                    "loading embedding model %s on %s", self.model_name, self.device
                )
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load embedding model '{self.model_name}': {exc}. "
                    "Check the model id and network (first run downloads weights)."
                ) from exc
        return self._model

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("cannot embed empty text")
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("cannot embed an empty batch")
        if any(not t.strip() for t in texts):
            raise ValueError("batch contains empty text")
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
