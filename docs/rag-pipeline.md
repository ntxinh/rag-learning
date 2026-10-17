# The RAG Pipeline, Stage by Stage

Follow one file in, then one question through. Code references point at
`src/rag_learning/`.

## Ingestion: a file becomes vectors

```
data/documents/report.pdf
        │
        ▼  ingestion/service.py: discover
        walk dir → [*.txt, *.md, *.pdf]
        ▼  ingestion/loaders.py: load
        report.pdf → PyMuPDF → [(page 1, "…"), (page 2, "…")]
        ▼  normalize
        \r\n→\n · collapse 3+ blank lines → 1 · trim line ends
        ▼  ingestion/chunking.py: chunk
        paragraphs joined ≤ chunk_size chars,
        overlap tail carried into next chunk
        ▼
        [Chunk(text, chunk_index, metadata{source, page, …})]
        ▼  embeddings/service.py
        SentenceTransformer.encode(batch) → float32 vectors, L2-normalized
        ▼  vector_store/chroma_store.py
        upsert(chunk_id, vector, text, metadata)
        ▼
        storage/chroma/  (persistent)
```

**Inspect it:** `rag documents list`, `rag stats`, or open the collection —
each chunk's metadata tells you exactly where it came from.

**Idempotency:** `document_id = sha256(text)`, `chunk_id = "{doc}:{index}"`.
Re-run `rag ingest` on unchanged files and nothing changes. Edit a file and
its old chunks are replaced wholesale.

## Answering: a question becomes a grounded answer

```
"what does section 3 say about X?"
        │
        ▼  retrieval/service.py
        embed question (SAME model as indexing)
        ▼
        Chroma query(top_k=5) → ranked (distance, chunk) pairs
        ▼  optional threshold on similarity = 1 − cosine_distance
        ▼
        Evidence: [(rank, Sx, text, source, page, score)]
        │
        ├── empty → "insufficient evidence" response, NO LLM call
        │
        ▼  generation/prompt.py
        system rules + question + [S1]..[Sn] context blocks
        (packed under max_context_chars)
        ▼  generation/service.py
        Ollama /api/chat (temp 0.1, timeout-bounded)
        ▼
        raw answer text
        ▼  citation validation
        keep only [Sx] that map to real retrieved chunks;
        strip + log invented ones
        ▼
        answer + citations[{id, file, page, chunk_id, excerpt}]
```

**Inspect it:** `rag retrieve "question"` runs everything up to (not
including) the prompt stage — the exact context the LLM would see.

## Key semantics to not get wrong

- **Distance vs similarity.** Chroma returns cosine *distance*
  (`0` identical → `2` opposite). The API reports `similarity = 1 − d`.
  A "score of 0.9" means near-identical.
- **Chars, not tokens.** `chunk_size = 1000` means 1000 characters (~200–300
  English tokens; Vietnamese differs more). Tuning retrieval? Adjust chars,
  verify with `rag retrieve`.
- **Same embedding both directions.** Indexing and querying share one model,
  one normalization, one metric. Change the model → rebuild the index.
- **Overlap carries context, not correctness.** Overlap keeps sentences that
  straddle boundaries intact in at least one chunk; too much wastes the
  context budget, too little loses boundary sentences.
