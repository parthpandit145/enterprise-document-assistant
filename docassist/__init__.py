"""Enterprise Document Assistant — a source-grounded RAG pipeline over PDFs.

Pipeline shape:

    PDFs -> text extraction -> chunking -> embeddings -> Chroma vector store
         -> semantic retrieval -> LLM with retrieved context
         -> grounded answer + citations
"""

from docassist.config import Settings, load_settings

__all__ = ["Settings", "load_settings", "__version__"]
__version__ = "0.1.0"
