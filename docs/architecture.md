# Architecture

How the pieces fit together. For *what* each stage does see
[rag-pipeline.md](rag-pipeline.md); for *why*, see [DESIGN.md](../DESIGN.md).

## Layers

```
┌────────────────────── entry points ──────────────────────┐
│  cli.py (Typer commands)      api/ (FastAPI + schemas)   │
└──────────────────────────┬───────────────────────────────┘
┌──────────────────────────▼───────────────────────────────┐
│  application/                                            │
│  IngestionService        QuestionAnsweringService        │
└───┬──────────────┬──────────────┬──────────────┬─────────┘
┌───▼────┐   ┌─────▼─────┐   ┌───▼────────┐   ┌──▼─────────┐
│ingestion│   │retrieval/ │   │generation/ │   │evaluation/ │
│ loaders │   │ service   │   │ prompt     │   │ dataset    │
│ chunking│   │           │   │ service    │   │ metrics    │
│ service │   │           │   │ (Ollama)   │   │            │
└───┬────┘   └─────┬─────┘   └────────────┘   └────────────┘
    │              │
┌───▼──────────┐ ┌─▼────────────────┐
│ embeddings/  │ │ vector_store/    │
│ service      │ │ chroma_store     │
│ (SentenceTr.)│ │ (Chroma persistent)│
└──────────────┘ └──────────────────┘
```

## Dependency rules

- Down-pointing arrows only. `application/` composes; entry points call
  `application/` and nothing deeper.
- `ingestion/` and `retrieval/` never import `generation/` (or each other).
  `generation/` never imports `vector_store/` — it receives chunks.
- `evaluation/` drives the application services; it is a consumer, not a
  dependency.
- `config.py` (`Settings`) and `logging_config.py` are the only modules every
  layer may touch. `Settings` is injected via constructors.

## Why this shape

- **Each stage is replaceable.** Swap Chroma for another store, Sentence
  Transformers for an API embedder, Ollama for a different LLM — the seam is
  one constructor argument.
- **Each stage is testable alone.** Chunker is a pure function; retrieval can
  run against a temp Chroma dir; generation is faked at the HTTP boundary.
- **No magic.** There is no framework orchestrator hiding the flow; reading
  `application/question_answering.py` tells you the whole ask-path.

## Process model

Single process, single user. CLI and API share in-process service instances.
The embedding model loads once and is reused for the process lifetime;
Chroma persists under `storage/`; Ollama runs out-of-process on localhost.

## Extension seams (deferred work slots in cleanly)

| Want later          | Where it plugs in                                  |
|---------------------|----------------------------------------------------|
| Reranker            | `retrieval/service.py`, after top-k fetch          |
| Lexical/hybrid      | second candidate source merged in retrieval        |
| New file format     | one loader in `ingestion/loaders.py`               |
| Token chunking      | `ingestion/chunking.py` (same `Chunk` output)      |
| Different vector DB | reimplement `chroma_store` behind the same methods |
