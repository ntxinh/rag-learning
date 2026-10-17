"""File → text loaders. One per format; page-aware for PDF.

Text normalization (shared): \r\n/\r → \n, trim trailing whitespace per line,
collapse runs of 3+ newlines to 2. Never touches whitespace inside lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class LoadedDocument:
    """Extracted text; `pages` is per-page text for PDFs, else one entry."""

    source: Path
    file_name: str
    file_type: str  # extension without dot
    pages: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load(path: Path) -> LoadedDocument:
    """Dispatch on extension; raises ValueError for unsupported or empty files."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type: {path.suffix}")
    pages = _load_pdf(path) if suffix == ".pdf" else [_read_text(path)]
    pages = [normalize(p) for p in pages]
    if not any(pages):
        raise ValueError(f"no extractable text in {path.name}")
    return LoadedDocument(
        source=path, file_name=path.name, file_type=suffix.lstrip("."), pages=pages
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> list[str]:
    """One entry per page so chunks can carry their page number."""
    pages: list[str] = []
    doc = pymupdf.open(path)
    try:
        for page in doc.pages():
            pages.append(page.get_text("text"))
    finally:
        doc.close()
    return pages


def discover(directory: Path) -> list[Path]:
    """All supported files under directory, recursively, in stable order."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
