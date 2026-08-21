"""Enterprise Document Assistant — Streamlit front end.

    streamlit run app.py

Upload PDFs, index them, ask questions, and see exactly which passage of which
file each sentence of the answer came from.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from docassist.config import load_settings
from docassist.embeddings import build_embeddings
from docassist.ingest import build_index
from docassist.logging_utils import setup_logging
from docassist.pipeline import RAGPipeline, RAGResponse
from docassist.vectorstore import open_store

st.set_page_config(
    page_title="Enterprise Document Assistant",
    page_icon="📄",
    layout="wide",
)

setup_logging()

SAMPLE_QUESTIONS = [
    "How many days of paid time off do employees get?",
    "What must I do if my work laptop is stolen?",
    "How long are customer shipment records kept?",
    "Can I work from outside Germany?",
    "Summarise the Q3 operations results.",
]


# ---------------------------------------------------------------------------
# Cached resources
#
# Streamlit re-runs this whole file on every interaction. Without caching, the
# embedding model would be reloaded from disk on every keystroke.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline(cache_key: int = 0) -> RAGPipeline:
    settings = load_settings()
    store = open_store(settings, build_embeddings(settings))
    return RAGPipeline(settings=settings, store=store)


def refresh_pipeline() -> None:
    """Drop the cached pipeline so it picks up a rebuilt index."""
    get_pipeline.clear()
    st.session_state.cache_key = st.session_state.get("cache_key", 0) + 1


# ---------------------------------------------------------------------------
# Sidebar — corpus management and configuration
# ---------------------------------------------------------------------------


def render_sidebar(pipeline: RAGPipeline) -> None:
    settings = pipeline.settings

    with st.sidebar:
        st.header("📚 Indexed documents")

        stats = pipeline.stats()
        if stats.chunk_count == 0:
            st.warning("The index is empty. Add documents below, then build the index.")
        else:
            col_a, col_b = st.columns(2)
            col_a.metric("Documents", stats.source_count)
            col_b.metric("Chunks", stats.chunk_count)
            for source, count in stats.sources.items():
                st.caption(f"• {source} — {count} chunks")

        st.divider()
        st.subheader("Add documents")

        uploads = st.file_uploader(
            "PDF, TXT or MD",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads and st.button("Save & index", use_container_width=True, type="primary"):
            settings.pdf_dir.mkdir(parents=True, exist_ok=True)
            for upload in uploads:
                # Path() strips any directory component a crafted filename
                # might carry, so an upload can't escape the corpus folder.
                target = settings.pdf_dir / Path(upload.name).name
                target.write_bytes(upload.getbuffer())
            with st.spinner(f"Indexing {len(uploads)} file(s)…"):
                report = build_index(settings)
            refresh_pipeline()
            st.success(report.summary())
            st.rerun()

        st.divider()
        st.subheader("Index")

        if st.button("🔄 Re-index folder", use_container_width=True):
            with st.spinner("Re-indexing…"):
                report = build_index(settings)
            refresh_pipeline()
            st.success(report.summary())
            st.rerun()

        if st.button("🧨 Rebuild from scratch", use_container_width=True):
            with st.spinner("Rebuilding…"):
                report = build_index(settings, rebuild=True)
            refresh_pipeline()
            st.success(report.summary())
            st.rerun()

        if stats.chunk_count == 0 and not any(settings.pdf_dir.glob("*.pdf")):
            if st.button("📄 Generate sample documents", use_container_width=True):
                from scripts.generate_test_pdfs import generate

                with st.spinner("Writing sample PDFs and indexing…"):
                    generate(settings.pdf_dir)
                    report = build_index(settings, rebuild=True)
                refresh_pipeline()
                st.success(report.summary())
                st.rerun()

        st.divider()
        st.subheader("Configuration")
        llm_label = (
            "no model — answers quoted from the retrieved text"
            if settings.llm_provider == "extractive"
            else f"`{settings.llm_model}`"
        )
        st.caption(
            f"**Embeddings** `{settings.embedding_provider}` · "
            f"`{settings.embedding_model.split('/')[-1]}`\n\n"
            f"**LLM** `{settings.llm_provider}` · {llm_label}\n\n"
            f"**Chunking** {settings.chunk_size} chars / {settings.chunk_overlap} overlap\n\n"
            f"**Retrieval** top-{settings.top_k}, {settings.search_type}, "
            f"threshold {settings.score_threshold}"
        )
        st.caption("Change these in `.env`, then restart the app.")

        if shutil.which("ollama") is None and settings.llm_provider == "ollama":
            st.error("LLM_PROVIDER=ollama but the `ollama` binary was not found.")


# ---------------------------------------------------------------------------
# Answer rendering
# ---------------------------------------------------------------------------


def render_response(response: RAGResponse) -> None:
    if response.failed:
        st.error(response.answer, icon="⚠️")
        st.caption(
            "This is a configuration problem, not a limitation of your documents — "
            "the question was never answered either way."
        )
        return

    if response.abstained:
        st.info(response.answer, icon="🚫")
        st.caption(
            "The assistant refused rather than guessing — this is the intended "
            "behaviour when the documents do not cover the question."
        )
    else:
        st.markdown(response.answer)

    if response.invalid_citations:
        st.error(
            f"The answer cited passages that were never retrieved "
            f"({response.invalid_citations}). Treat it as unreliable.",
            icon="⚠️",
        )

    if response.sources:
        st.markdown("##### Sources")
        for source in response.sources:
            with st.expander(f"**{source.label}** — relevance {source.score:.2f}"):
                st.write(source.snippet)

    footer = [f"`{response.model}`", f"{response.latency_ms} ms"]
    if response.usage:
        footer.append(
            f"{response.usage.get('input_tokens', 0)} in / "
            f"{response.usage.get('output_tokens', 0)} out tokens"
        )
    if not response.abstained:
        footer.append(f"{response.citation_coverage:.0%} of claims cited")
    footer.append(f"{len(response.retrieved)} chunks retrieved")
    st.caption(" · ".join(footer))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.session_state.setdefault("cache_key", 0)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)

    pipeline = get_pipeline(st.session_state.cache_key)
    render_sidebar(pipeline)

    st.title("📄 Enterprise Document Assistant")
    st.caption(
        "Ask questions about your internal documents. Every answer is generated "
        "only from retrieved passages, and cites the file and page it came from."
    )

    stats = pipeline.stats()
    if stats.chunk_count == 0:
        st.warning(
            "No documents are indexed yet. Use the sidebar to upload PDFs — or "
            "generate a sample corpus to try the system immediately.",
            icon="👈",
        )
        return

    if not st.session_state.history:
        st.markdown("**Try one of these:**")
        columns = st.columns(len(SAMPLE_QUESTIONS))
        for column, question in zip(columns, SAMPLE_QUESTIONS):
            if column.button(question, use_container_width=True):
                st.session_state.pending = question
                st.rerun()

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            render_response(entry["response"])

    question = st.chat_input("Ask a question about the indexed documents…")
    if st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                response = pipeline.safe_answer(question)
            render_response(response)
        st.session_state.history.append({"question": question, "response": response})


if __name__ == "__main__":
    main()
