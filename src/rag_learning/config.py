"""Central configuration. Everything reads Settings; nothing reads os.environ."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rag_learning/config.py -> repo root
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    data_dir: Path = ROOT / "data" / "documents"
    vector_dir: Path = ROOT / "storage" / "chroma"
    collection_name: str = "rag_documents"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    top_k: int = Field(default=5, gt=0)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=8000, gt=0)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_temperature: float = Field(default=0.1, ge=0.0)
    llm_timeout: float = Field(default=120.0, gt=0)

    log_level: str = "INFO"

    @field_validator("data_dir", "vector_dir", mode="before")
    @classmethod
    def _resolve_path(cls, v: object) -> Path:
        return Path(str(v)) if Path(str(v)).is_absolute() else ROOT / str(v)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"< chunk_size ({self.chunk_size})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
