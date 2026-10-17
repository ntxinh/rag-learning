"""`rag` command-line interface. Services built once in the callback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from rag_learning.application.ingestion_service import IngestionApp
from rag_learning.application.question_answering import QuestionAnsweringApp
from rag_learning.config import get_settings
from rag_learning.embeddings.service import EmbeddingService
from rag_learning.evaluation.dataset import load_dataset
from rag_learning.evaluation.metrics import first_hit_rank, mrr, precision_at_k
from rag_learning.generation.service import GenerationError
from rag_learning.logging_config import configure_logging
from rag_learning.vector_store.chroma_store import ChromaStore

app = typer.Typer(help="rag-learning: hand-rolled local RAG.", no_args_is_help=True)
documents_app = typer.Typer(help="Manage indexed documents.", no_args_is_help=True)
app.add_typer(documents_app, name="documents")

_ingestion: IngestionApp | None = None
_qa: QuestionAnsweringApp | None = None


def _build() -> None:
    global _ingestion, _qa
    if _ingestion is not None:
        return
    settings = get_settings()
    configure_logging(settings.log_level)
    embeddings = EmbeddingService(settings.embedding_model, settings.embedding_device)
    store = ChromaStore(settings.vector_dir, settings.collection_name)
    _ingestion = IngestionApp(settings, embeddings=embeddings, store=store)
    _qa = QuestionAnsweringApp(settings, embeddings=embeddings, store=store)


@app.callback()
def _main() -> None:
    _build()


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Directory of documents")] = Path(
        "data/documents"
    ),
) -> None:
    """Index supported files (.txt/.md/.pdf) into the vector store."""
    assert _ingestion is not None
    report = _ingestion.ingest(path)
    typer.echo(
        f"indexed={len(report.indexed)} skipped={len(report.skipped)} "
        f"failed={len(report.failed)}"
    )
    for name, err in report.failed.items():
        typer.echo(f"  FAILED {name}: {err}", err=True)
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def retrieve(
    question: Annotated[str, typer.Argument()],
    top_k: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Show ranked chunks for a question — no LLM call."""
    assert _qa is not None
    results = _qa.retrieve(question, top_k=top_k)
    if not results:
        typer.echo("no results")
        return
    for r in results:
        meta = r.chunk.metadata
        page = f" p.{meta['page']}" if meta.get("page") is not None else ""
        typer.echo(
            f"#{r.rank}  sim={r.similarity:.3f}  {meta.get('file_name')}{page}  "
            f"[{r.chunk.chunk_id}]"
        )
        typer.echo(f"    {r.chunk.text[:160].replace(chr(10), ' ')}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    top_k: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Retrieve + generate a grounded answer with citations."""
    assert _qa is not None
    try:
        result = _qa.ask(question, top_k=top_k)
    except GenerationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(result.answer.text)
    if result.answer.citations:
        typer.echo("\nSources:")
        for c in result.answer.citations:
            page = f" p.{c.page}" if c.page is not None else ""
            typer.echo(f"  [{c.citation_id}] {c.file_name}{page} ({c.chunk_id})")


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Argument()] = Path(
        "data/evaluation/questions.jsonl"
    ),
    report_json: Annotated[Path | None, typer.Option("--json")] = None,
) -> None:
    """Score retrieval/generation against a JSONL dataset."""
    assert _qa is not None
    cases = load_dataset(dataset)
    if not cases:
        typer.echo("empty dataset", err=True)
        raise typer.Exit(code=1)

    rows: list[dict[str, Any]] = []
    hits = mrrs = 0.0
    for case in cases:
        evidence = _qa.retrieve(case.question)
        rank = first_hit_rank(evidence, case.expected_sources)
        hit = rank is not None
        hits += hit
        mrrs += mrr(evidence, case.expected_sources)
        try:
            result = _qa.ask(case.question)
            a = result.answer
            row = {
                "id": case.id,
                "hit": hit,
                "rank": rank,
                "precision@k": round(
                    precision_at_k(
                        evidence, case.expected_sources, _qa.retriever.top_k
                    ),
                    3,
                ),
                "citations_valid": a.invalid_citations_removed == 0,
                "insufficient_evidence": a.insufficient_evidence,
                "answer": a.text,
                "reference_answer": case.reference_answer,
            }
        except GenerationError as exc:
            row = {"id": case.id, "hit": hit, "rank": rank, "error": str(exc)}
        rows.append(row)
        status = "HIT" if hit else "MISS"
        typer.echo(f"[{status}] {case.id} rank={rank} :: {case.question[:60]}")

    n = len(cases)
    typer.echo(f"\nhit-rate@k={hits / n:.2f}  MRR={mrrs / n:.3f}  (n={n})")
    if report_json:
        report_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


@documents_app.command("list")
def documents_list() -> None:
    """List indexed documents."""
    assert _ingestion is not None
    docs = _ingestion.store.list_documents()
    if not docs:
        typer.echo("no documents indexed")
        return
    for d in docs:
        typer.echo(f"{d['document_id']}  {d['file_name']}  ({d['chunks']} chunks)")


@documents_app.command("remove")
def documents_remove(target: Annotated[str, typer.Argument()]) -> None:
    """Remove a document by source path or document_id."""
    assert _ingestion is not None
    deleted = _ingestion.remove_document(target)
    if not deleted:
        typer.echo("not found", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"deleted {deleted} chunk(s)")


@app.command()
def stats() -> None:
    """Show collection stats."""
    assert _ingestion is not None
    s = _ingestion.stats()
    typer.echo(
        f"collection={s['collection']} documents={s['documents']} chunks={s['chunks']}"
    )


@app.command()
def reset(
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
) -> None:
    """Wipe the vector collection. Destructive."""
    assert _ingestion is not None
    if not yes:
        typer.confirm("Delete ALL indexed chunks?", abort=True)
    _ingestion.reset()
    typer.echo("index cleared")
