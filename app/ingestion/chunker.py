"""
Chunking: splits extracted page text into overlapping chunks.

Key design decision: chunking happens PER PAGE, not across the whole
document. A chunk never spans two pages. This costs a little retrieval
quality at page boundaries (a sentence split across pages 4/5 becomes
two chunks instead of one continuous one), but in exchange every chunk
has one unambiguous, always-correct page number - which is what the
PRD's citation requirements (Section 11) depend on: "Citation generation
should use retrieval metadata rather than asking the LLM to invent page
numbers." Cross-page chunking would need each chunk to store a page
*range*, which complicates citations for a small quality gain. This is
a documented tradeoff, not an oversight.

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph breaks first, then sentences, then words - avoiding mid-word
cuts wherever possible.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.ingestion.chunk_models import Chunk
from app.ingestion.models import DocumentResult

logger = get_logger(__name__)


def chunk_document(
    doc_result: DocumentResult,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection_id: str = "default",
) -> list[Chunk]:
    """Turn a DocumentResult (page-level text) into a flat list of Chunks.

    chunk_size / chunk_overlap default to the values in app/config/settings.py
    but can be overridden per-call - this is what powers the "configurable
    chunking" requirement (PRD Section 7.1) and the future Settings screen
    (Section 13).
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in doc_result.pages:
        if page.is_empty or not page.text.strip():
            continue  # nothing to chunk - e.g. a page OCR (Phase 9) couldn't
                      # recover text from, or OCR is disabled/unavailable

        page_pieces = splitter.split_text(page.text)
        for i, piece in enumerate(page_pieces):
            chunk_id = f"{doc_result.document_id}_p{page.page_number}_c{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_result.document_id,
                    filename=doc_result.filename,
                    page_number=page.page_number,
                    chunk_text=piece,
                    chunk_index=i,
                    collection_id=collection_id,
                )
            )

    logger.info(
        "Chunked '%s': %d pages -> %d chunks (size=%d, overlap=%d)",
        doc_result.filename, len(doc_result.pages), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks
