"""
Shared data models for ingestion output.

Keeping this as a plain dataclass (not tied to any specific vector DB or
framework) means ingestion stays swappable - if we ever replace PyMuPDF
or ChromaDB, nothing downstream needs to change its interface.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PageContent:
    """Extracted text from a single PDF page."""
    document_id: str
    filename: str
    page_number: int          # 1-indexed - matches what a human sees in a PDF viewer
    text: str
    is_empty: bool = False    # True if extraction (incl. OCR fallback) found no usable text
    extraction_error: str | None = None
    ocr_used: bool = False    # True if this page's text came from OCR (Phase 9), not the text layer


@dataclass
class DocumentResult:
    """Full result of ingesting one PDF file."""
    document_id: str
    filename: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    likely_scanned: bool = False   # True if most pages have no usable text even after OCR
    ocr_pages_used: int = 0        # how many pages needed and got an OCR fallback
    # True if OCR would have helped (some pages were empty) but the Tesseract
    # engine isn't installed on this machine - lets callers give a specific,
    # actionable error instead of a generic "looks scanned" message.
    ocr_unavailable: bool = False
    error: str | None = None       # set if the whole file failed to open
