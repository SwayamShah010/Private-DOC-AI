"""
Indexing service: the orchestration layer that ties ingestion, chunking,
embedding, and storage together into one "index this PDF" operation.

Kept separate from the UI (app/ui/) so the Streamlit layer only ever
calls one function and displays its result - all the real logic stays
testable and swappable without touching UI code, per PRD Section 18's
maintainability requirement.
"""
from dataclasses import dataclass

from app.config.logging_config import get_logger
from app.embeddings.embedder import Embedder
from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_extractor import extract_pdf
from app.storage.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class IndexingResult:
    filename: str
    success: bool
    document_id: str | None = None
    chunk_count: int = 0
    likely_scanned: bool = False
    ocr_pages_used: int = 0
    error: str | None = None


def index_pdf_file(
    file_path: str,
    filename: str,
    collection_id: str = "default",
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    ocr_enabled: bool | None = None,
) -> IndexingResult:
    """Run one PDF through the full pipeline: extract -> chunk -> embed -> store.

    Never raises for expected failure modes (corrupted PDF, scanned PDF
    with no extractable text even after OCR) - those come back as a
    non-success IndexingResult so the UI can show a clear, specific
    message instead of a stack trace (PRD Section 13: "Clear error
    messages for unsupported, corrupted, or scanned PDFs").
    """
    doc_result = extract_pdf(file_path, filename=filename, ocr_enabled=ocr_enabled)

    if doc_result.error:
        return IndexingResult(filename=filename, success=False, error=doc_result.error)

    if doc_result.likely_scanned:
        if doc_result.ocr_unavailable:
            error = (
                "This looks like a scanned PDF, and OCR couldn't run because the "
                "Tesseract engine isn't installed on this machine. Install it "
                "(see README) and try again."
            )
        else:
            error = (
                "This looks like a scanned PDF, and OCR could not extract "
                "readable text from it (very low image quality, or genuinely "
                "blank pages)."
            )
        return IndexingResult(
            filename=filename,
            success=False,
            document_id=doc_result.document_id,
            likely_scanned=True,
            error=error,
        )

    chunks = chunk_document(doc_result, collection_id=collection_id)
    if not chunks:
        return IndexingResult(
            filename=filename,
            success=False,
            document_id=doc_result.document_id,
            error="No text content could be extracted from this document.",
        )

    embedder = embedder or Embedder()
    vector_store = vector_store or VectorStore()

    texts = [c.chunk_text for c in chunks]
    embeddings = embedder.embed_texts(texts)
    for c in chunks:
        c.embedding_model = embedder.model_name

    added = vector_store.add_chunks(chunks, embeddings)

    logger.info("Indexed '%s': %d chunks added to collection '%s'.", filename, added, collection_id)
    return IndexingResult(
        filename=filename,
        success=True,
        document_id=doc_result.document_id,
        chunk_count=added,
        ocr_pages_used=doc_result.ocr_pages_used,
    )
