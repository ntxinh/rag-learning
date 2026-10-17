# AGENTS.md — working guide for this repository

Educational RAG app. The point is *visible pipeline mechanics*; optimize for
clarity over cleverness.

## Non-negotiables

- **No RAG frameworks.** No LangChain, LlamaIndex, Haystack, or similar —
  not even "just for one helper." The pipeline is hand-rolled on purpose.
- `uv` manages everything Python: `uv sync --all-groups`, `uv run ...`,
  `uv add <pkg>`. Never `pip install`, never touch a venv directly.
- `uv.lock` is committed. `storage/chroma/`, `data/documents/` contents,
  model weights, and `.env` are not.
- `src/` layout; package name `rag_learning`. New code goes in the module
  named for its layer (see DESIGN.md) — don't invent parallel structures.

## Commands

| Task        | Command                                    |
|-------------|--------------------------------------------|
| Sync deps   | `mise run sync` / `uv sync --all-groups`   |
| Test        | `uv run pytest` (or `mise run test`)       |
| Lint        | `uv run ruff check .`                      |
| Format      | `uv run ruff format .`                     |
| Typecheck   | `uv run mypy src tests`                    |
| All checks  | `mise run check` / `make check`            |
| Run API     | `uv run uvicorn rag_learning.api.app:app`  |
| CLI         | `uv run rag --help`                        |

## Conventions

- Type-annotate everything; mypy must stay clean. Ruff is formatter + linter.
- Constructor injection for services (embeddings, store, LLM client). No
  module-level singletons; no `Protocol` unless two implementations or a
  test seam actually need it.
- Chunk size is **characters**, not tokens. Keep it that way.
- Chroma distance metric is cosine distance — never present it as a
  similarity score without the documented conversion.
- Tests: unit tests live under `tests/unit/`, mirroring `src/` modules.
  LLM and embedding model calls are faked at service boundaries; only
  `tests/integration/` may touch real Ollama/Chroma (marked and skippable).
- Configuration only through `rag_learning.config.Settings` — no
  `os.environ` reads scattered through modules.

## Definition of done for a change

`mise run check` green; docs updated if behavior/config/CLI surface changed;
DESIGN.md stays truthful about module responsibilities.
