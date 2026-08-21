from langchain_core.documents import Document

from docassist.chunker import chunk_documents, chunk_id
from docassist.config import load_settings
from docassist.loader import clean_text


def _settings():
    return load_settings()


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert clean_text("confiden-\ntial data") == "confidential data"


def test_clean_text_keeps_paragraphs_but_unwraps_lines():
    cleaned = clean_text("line one\nline two\n\nnew paragraph")
    assert cleaned == "line one line two\n\nnew paragraph"


def test_chunks_respect_size_and_overlap():
    settings = _settings()
    long_text = " ".join(f"Sentence number {i} about company policy." for i in range(400))
    doc = Document(page_content=long_text, metadata={"source": "a.pdf", "page": 1})

    chunks = chunk_documents([doc], settings)

    assert len(chunks) > 1
    # RecursiveCharacterTextSplitter may overshoot slightly to avoid splitting a
    # word, but should stay in the same ballpark as the configured size.
    assert all(len(c.page_content) <= settings.chunk_size + 200 for c in chunks)
    # Overlap means consecutive chunks share text.
    tail = chunks[0].page_content[-100:]
    assert any(word in chunks[1].page_content for word in tail.split()[:5])


def test_chunks_carry_source_metadata_for_citations():
    settings = _settings()
    doc = Document(
        page_content="Employees receive 30 days of annual leave per year. " * 40,
        metadata={"source": "handbook.pdf", "page": 7},
    )

    chunks = chunk_documents([doc], settings)

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.metadata["source"] == "handbook.pdf"
        assert chunk.metadata["page"] == 7
        assert "chunk_id" in chunk.metadata
        assert "start_index" in chunk.metadata


def test_chunk_id_is_stable_across_runs():
    doc = Document(page_content="same text", metadata={"source": "a.pdf", "page": 2})
    other = Document(page_content="same text", metadata={"source": "b.pdf", "page": 2})

    # Re-ingestion must upsert rather than duplicate, which requires the id to
    # depend only on content plus origin.
    assert chunk_id(doc) == chunk_id(
        Document(page_content="same text", metadata={"source": "a.pdf", "page": 2})
    )
    assert chunk_id(doc) != chunk_id(other)


def test_tiny_fragments_are_dropped():
    settings = _settings()
    doc = Document(page_content="Heading", metadata={"source": "a.pdf", "page": 1})
    assert chunk_documents([doc], settings) == []
