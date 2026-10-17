# Evaluation

A small harness for a small system. It answers two questions: *did retrieval
surface the right chunks?* and *did the answer use them?* — reproducibly.

## Dataset format

JSONL, one object per line, in `data/evaluation/`:

```jsonl
{"id": "q1", "question": "What is the chunk overlap default?", "expected_sources": ["README.md"], "reference_answer": "…"}
{"id": "q2", "question": "…", "expected_sources": ["report.pdf"], "expected_pages": [3]}
```

| Field               | Required | Meaning                                        |
|---------------------|----------|------------------------------------------------|
| `id`                | yes      | stable case id (for diffs between runs)        |
| `question`          | yes      | the query                                      |
| `expected_sources`  | yes      | file names retrieval should surface            |
| `expected_pages`    | no       | expected PDF pages (checked when present)      |
| `reference_answer`  | no       | human-written answer, printed for side-by-side |
| `tags`              | no       | e.g. `"vi"`, `"negative"` for slicing results  |

Write cases that exercise the corpus you actually ingest — including
**negative cases** (`expected_sources: []`) where the right answer is
"insufficient evidence".

## Run it

```bash
rag evaluate                              # data/evaluation/questions.jsonl
rag evaluate path/to/other.jsonl
rag evaluate --json report.json
```

## Metrics

Retrieval (computed per question, then aggregated):

| Metric       | Definition                                                        |
|--------------|-------------------------------------------------------------------|
| hit-rate@k   | ≥1 expected source appears in the top-k results                   |
| MRR          | 1/rank of the first expected source (0 if absent) — position matters |
| precision@k  | fraction of top-k results that are expected sources                |

Answer (reported, partially manual):

| Check                      | Automatic? | What it means                                   |
|----------------------------|------------|-------------------------------------------------|
| citations-valid            | yes        | every `[Sx]` in the answer maps to a real chunk |
| evidence-used              | yes        | ≥1 citation when evidence existed (or correct refusal) |
| claim support              | no         | does each citation actually support the claim?   |
|                            |            | valid id ≠ supported claim — review flagged rows |

## Reading output

Per-question rows show question, hit/miss, first-hit rank, citation validity,
and the answer next to `reference_answer`. The aggregate block prints
hit-rate@k, MRR, mean precision@k, and citation-valid fraction.

Low hit-rate → tune `top_k`, chunk size/overlap, or check ingestion coverage
(`rag stats`). High hit-rate with invalid citations → generation/prompt
problem, not retrieval.

## Failure analysis ritual

1. Reproduce the miss: `rag retrieve "<question>"` — is the right chunk even
   in the index? (`rag stats`, `grep` the source file.)
2. In index but not retrieved → embedding/threshold/top-k issue.
3. Retrieved but unused → prompt or context-budget issue.
4. Cited but unsupported → model grounding issue; check the excerpt.

Record findings next to the dataset (`data/evaluation/NOTES.md`) — the point
of the project is seeing *why* RAG fails.
