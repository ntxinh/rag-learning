"""JSONL evaluation dataset: one question per line.

{"id": "q1", "question": "…", "expected_sources": ["file.md"],
 "expected_pages": [3], "reference_answer": "…", "tags": ["vi"]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_sources: list[str]
    expected_pages: list[int] = field(default_factory=list)
    reference_answer: str | None = None
    tags: list[str] = field(default_factory=list)


def load_dataset(path: Path) -> list[EvalCase]:
    """Parse a JSONL dataset; skips blank lines, rejects malformed rows."""
    cases: list[EvalCase] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    id=str(obj["id"]),
                    question=str(obj["question"]),
                    expected_sources=[str(s) for s in obj["expected_sources"]],
                    expected_pages=[int(p) for p in obj.get("expected_pages", [])],
                    reference_answer=obj.get("reference_answer"),
                    tags=[str(t) for t in obj.get("tags", [])],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed eval case: {exc}") from exc
    return cases
