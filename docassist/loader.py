"""Stage 1 — document ingestion.

PDFs are read page by page with `PyPDFLoader` so that every piece of text keeps
the filename and page number it came from. That metadata is what makes the
citations at the other end of the pipeline verifiable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from docassist.config import Settings

logger = logging.getLogger(__name__)

# PDF extractors leave words split across line breaks ("confiden-\ntial") and
# ragged single newlines mid-sentence. Both confuse the chunker and the
# embedding model, so they get normalised away before anything else happens.
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_REPEATED_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Undo the usual PDF-extraction damage without destroying paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SINGLE_NEWLINE.sub(" ", text)
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    text = _REPEATED_SPACES.sub(" ", text)
    return text.strip()


def discover_files(settings: Settings) -> list[Path]:
    """Every supported document under the configured folder, sorted."""
    if not settings.pdf_dir.exists():
        raise FileNotFoundError(
            f"Document folder not found: {settings.pdf_dir}\n"
            f"Create it and drop PDFs in, or run: python -m scripts.generate_test_pdfs"
        )

    wanted = {".pdf", *settings.text_extensions}
    files = [
        path
        for path in sorted(settings.pdf_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in wanted and not path.name.startswith(".")
    ]
    return files


def _load_one(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        # PyPDFLoader emits one Document per page and sets metadata["page"]
        # (0-indexed). We re-index to 1 below so citations match what a human
        # sees in a PDF reader.
        return PyPDFLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8").load()


def load_documents(settings: Settings) -> list[Document]:
    """Load every document in `settings.pdf_dir` into page-level Documents.

    Metadata on each returned Document:
        source     basename, e.g. "Acme_Employee_Handbook_2026.pdf"
        path       absolute path on disk
        page       1-indexed page number (1 for non-paginated text files)
        doc_type   "pdf" | "text"
    """
    files = discover_files(settings)
    if not files:
        raise FileNotFoundError(
            f"No PDFs or text files in {settings.pdf_dir}\n"
            f"Run: python -m scripts.generate_test_pdfs   (creates sample documents)"
        )

    documents: list[Document] = []
    skipped_pages = 0

    for path in files:
        try:
            pages = _load_one(path)
        except Exception as exc:  # a single unreadable file shouldn't kill the run
            logger.warning("Skipping %s — could not be parsed (%s)", path.name, exc)
            continue

        kept = 0
        for page in pages:
            body = clean_text(page.page_content)
            if len(body) < 20:
                # Cover pages, dividers and scanned images extract to nothing.
                # Indexing them adds noise to every future search.
                skipped_pages += 1
                continue

            raw_page = page.metadata.get("page")
            page_number = int(raw_page) + 1 if isinstance(raw_page, int) else 1

            documents.append(
                Document(
                    page_content=body,
                    metadata={
                        "source": path.name,
                        "path": str(path),
                        "page": page_number,
                        "doc_type": "pdf" if path.suffix.lower() == ".pdf" else "text",
                    },
                )
            )
            kept += 1

        logger.info("Loaded %-45s %3d page(s) with text", path.name, kept)

    logger.info(
        "Ingestion complete: %d file(s), %d page(s) kept, %d empty page(s) skipped",
        len(files),
        len(documents),
        skipped_pages,
    )
    return documents
