# RAG Pipeline Overview

This project implements a retrieval-augmented generation pipeline in pure
Python, without high-level RAG frameworks.

## Ingestion

Documents in `.txt`, `.md`, and `.pdf` formats are loaded, normalized, and
split into deterministic chunks. Chunk size is measured in **characters**,
not tokens — the default is 1000 characters with a 200-character overlap.

Each chunk carries metadata: `document_id`, `chunk_id`, `source`,
`file_name`, `file_type`, `chunk_index`, `content_hash`, and `page` for PDFs.
Document ids are content-addressed: `document_id` is a truncated SHA-256 of
the normalized text, and `chunk_id` is the document id plus the chunk index.
Re-ingesting an unchanged file is a no-op.

## Embeddings and storage

Embeddings come from Sentence Transformers, specifically
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — a ~470 MB
multilingual model that handles English and Vietnamese and runs fast on CPU.
Vectors are L2-normalized and stored in a persistent ChromaDB collection
using cosine distance.

## Retrieval and generation

A question is embedded with the same model, matched against Chroma's top-k
(cosine distance; reported similarity = 1 − distance), optionally filtered
by a similarity threshold, then packed into a prompt under a character
budget. Ollama generates the answer with citations `[S1]`, `[S2]`, … that
map to the retrieved chunks. Citation ids the model invents are stripped
and logged.
