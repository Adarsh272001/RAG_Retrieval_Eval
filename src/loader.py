"""Load documents (.md, .txt, .pdf) from a folder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str  # file name, e.g. "attendance_policy.md"
    text: str


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pypdf to load PDF files: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_corpus(folder: str | Path) -> list[Document]:
    """Read every .md/.txt/.pdf file in ``folder`` (non-recursive, sorted by name)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {folder}")
    docs: list[Document] = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        if text.strip():
            docs.append(Document(doc_id=path.name, text=text))
    if not docs:
        raise ValueError(f"No .md/.txt/.pdf documents found in {folder}")
    return docs
