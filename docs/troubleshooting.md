# Troubleshooting

Ordered by where the failure shows up.

## Setup

**`mise: command not found` / wrong Python**
`mise install` reads `mise.toml`. Verify: `mise ls`, `uv run python -V`
should print 3.12.x. If `uv` is missing: `mise install uv` or
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

**`uv sync` fails on a dependency**
Python version mismatch with a wheel (PyMuPDF, torch). Confirm
`uv run python -V` is 3.12. `uv sync --all-groups --reinstall` fixes a
half-built env.

**First `rag ingest` is slow / downloads something**
The embedding model downloads once (~470 MB) to the HF cache. Subsequent
runs are local. Pre-warm: `uv run python -c "from rag_learning.embeddings.service import EmbeddingService; EmbeddingService()"`.

## Ingestion

**Files skipped**
`rag ingest` reports `{indexed, skipped, failed}` — skipped means empty or
unsupported extension. `failed` rows carry the exception; run with
`RAG_LOG_LEVEL=DEBUG` for the loader traceback.

**PDF produces no text**
Scanned/image PDFs have no text layer — PyMuPDF returns empty. This project
does not OCR (deferred). Verify with `pdftotext file.pdf -` or open the PDF
and try selecting text.

**Re-ingesting created duplicates**
It shouldn't — chunks upsert by deterministic id. If it did, `rag reset`
then `rag ingest`, and report it: that's a bug in `document_id`/`chunk_id`
derivation.

**Changed file still answers from the old content**
Old chunks are deleted per-`source` before re-index. Check `rag documents
list` — two entries for one file name means the delete-before-upsert path
failed; file an issue, `rag reset` to recover.

## Retrieval

**`rag retrieve` returns nothing**
1. Anything indexed? `rag stats` → 0 chunks means ingestion, not retrieval.
2. Threshold too strict: unset `RAG_SIMILARITY_THRESHOLD` and retry.
3. Question too short/empty → validation rejects it.

**Results feel irrelevant**
- Confirm similarity semantics: scores near 1.0 are good; a "0.2" is *distance-like* only if you're looking at raw Chroma output.
- Vietnamese/English mixed corpus needs the multilingual model — check
  `RAG_EMBEDDING_MODEL`; if it was changed after ingest, the index is
  stale: `rag reset && rag ingest`.
- Chunk size too large dilutes passages; too small loses context. Iterate
  with `RAG_CHUNK_SIZE` + `rag retrieve`.

## Generation

**`Could not connect to Ollama`**
`ollama serve` isn't running (usually a background service). Check
`curl localhost:11434/api/tags`. Wrong host? `RAG_OLLAMA_BASE_URL`.

**`model 'llama3.1:8b' not found`**
`ollama pull llama3.1:8b`, or point `RAG_OLLAMA_MODEL` at one you have
(`ollama list`).

**Timeout**
Long context on CPU is slow. Raise `RAG_LLM_TIMEOUT`, lower
`RAG_MAX_CONTEXT_CHARS`, or lower `RAG_TOP_K`.

**Answer ignores the evidence / hallucinates sources**
Invented `[Sx]` ids are stripped and logged — check logs for
`invalid citation`. If the answer is ungrounded with *valid* ids, that's a
model-limit issue: try a different Ollama model or tighten the prompt in
`generation/prompt.py` (and its unit test).

**`insufficient evidence` when you know the answer is in there**
This is a retrieval miss, not a generation failure — run
`rag retrieve "<question>"` and debug retrieval (above).

## Index / storage

**Want a clean slate**
`rag reset` (guarded) or `make clean`. Chroma lives in `storage/chroma/`;
deleting the directory is equivalent.

**Chroma errors about tenant/schema after an upgrade**
Persistent DBs can outlive code versions. `rag reset && rag ingest` is the
documented recovery — there is no migration path by design.
