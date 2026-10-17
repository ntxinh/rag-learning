# rag-learning

A small, hand-rolled Retrieval-Augmented Generation (RAG) application built for
learning. Every pipeline stage — ingestion, chunking, embedding, vector search,
retrieval, prompt construction, generation, citation validation, evaluation —
is implemented directly on top of the underlying libraries, with no RAG
framework in the way.

## Stack

| Layer          | Choice                                             |
|----------------|----------------------------------------------------|
| Language       | Python 3.12 (pinned via `mise`)                    |
| Deps           | `uv` + `pyproject.toml` + committed `uv.lock`      |
| Embeddings     | Sentence Transformers `paraphrase-multilingual-MiniLM-L12-v2` (EN + VI, ~470 MB) |
| Vector store   | ChromaDB, persistent local, cosine distance        |
| LLM            | Ollama, default `llama3.1:8b`                      |
| API            | FastAPI + Pydantic                                 |
| Docs parsing   | PyMuPDF (PDF), stdlib (`.txt`, `.md`)              |
| Quality        | Ruff, mypy, pytest, HTTPX                          |

No LangChain / LlamaIndex / Haystack. No Docker, Redis, or external DB.

## Prerequisites

- [mise](https://mise.jdx.dev/) — provisions Python and `uv`
- [Ollama](https://ollama.com/) — running locally with a chat model pulled:
  `ollama pull llama3.1:8b`

## Quickstart

```bash
mise install            # Python 3.12 + uv
mise run sync           # or: uv sync --all-groups

# drop .txt/.md/.pdf files into data/documents/
mise run ingest         # index them into ChromaDB (storage/chroma/)

mise run ask -- "your question here"   # answer + citations
mise run api                            # FastAPI on http://127.0.0.1:8000
```

Everything also works through `make` (`make help`) and `uv run`
(`uv run rag --help`, `uv run pytest`, …).

## Commands

| Command                    | What it does                                       |
|----------------------------|----------------------------------------------------|
| `rag ingest [path]`        | Index supported files (idempotent per content hash)|
| `rag documents list`       | List indexed documents                             |
| `rag documents remove <id>`| Delete one document's chunks from the index        |
| `rag retrieve "question"`  | Show ranked chunks — no LLM call                   |
| `rag ask "question"`       | Retrieve + generate grounded answer + citations    |
| `rag evaluate [dataset]`   | Score retrieval/generation against a test set      |
| `rag reset`                | Wipe the collection (guarded; `--yes` to skip)     |
| `rag stats`                | Collection/document/chunk counts                   |

## Configuration

All settings come from `rag_learning.config` (Pydantic Settings) and can be
overridden with environment variables — see `.env.example`. Key knobs:
`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_EMBEDDING_MODEL`,
`RAG_OLLAMA_MODEL`, `RAG_LLM_TEMPERATURE`, `RAG_MAX_CONTEXT_CHARS`,
`RAG_SIMILARITY_THRESHOLD`, `RAG_LOG_LEVEL`.

Changing `RAG_EMBEDDING_MODEL` **requires rebuilding the index**
(`rag reset && rag ingest`) — vectors from different models are not comparable.

## Project layout

```
src/rag_learning/     package (config, cli, api/, ingestion/, embeddings/,
                      vector_store/, retrieval/, generation/, evaluation/,
                      application/)
tests/                unit/, integration/, fixtures/, evaluation/
data/documents/       files to index (gitignored contents)
data/evaluation/      question/answer test sets
storage/              ChromaDB persistence (gitignored)
docs/                 architecture, pipeline, evaluation, troubleshooting
DESIGN.md             the design doc for the whole system
```

## Documentation

- [DESIGN.md](DESIGN.md) — system design, data model, decisions
- [docs/architecture.md](docs/architecture.md) — layers and module boundaries
- [docs/rag-pipeline.md](docs/rag-pipeline.md) — stage-by-stage walkthrough
- [docs/evaluation.md](docs/evaluation.md) — eval dataset format and metrics
- [docs/troubleshooting.md](docs/troubleshooting.md) — common failures

## Status

Scaffolding and documentation stage: project metadata and docs exist; the
`src/` package is implemented per DESIGN.md as the next step.
