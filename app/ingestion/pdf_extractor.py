"""
PDF text extraction.

Uses PyMuPDF (imported as `fitz`) - chosen per the PRD (Section 9) for
being fast and reliable on a wide range of real-world PDFs, including
ones with unusual encodings that trip up simpler parsers.

Design notes:
- One corrupted/unreadable page must NOT crash the whole document
  (PRD Section 18, Reliability NFR) - each page is extracted in its
  own try/except.
- Page numbers are preserved 1-indexed so they match what a user sees
  in their PDF viewer - this is what citations will point to later.
- Phase 9: if a page's normal text layer comes back empty (a scanned/
  image page), we fall back to OCR (app/ingestion/ocr.py) before giving
  up on it. `likely_scanned` reflects the *post-OCR* state, so a
  document only gets flagged as unreadable if OCR genuinely couldn't
  extract anything either (or isn't installed).
"""
import hashlib
import uuid

import pymupdf  # PyMuPDF (the `fitz` alias is deprecated as of 1.24+)

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.ingestion.models import DocumentResult, PageContent
from app.ingestion.ocr import is_tesseract_available, ocr_page

logger = get_logger(__name__)

# If more than this fraction of pages extract as empty, we suspect a scanned PDF.
SCANNED_HEURISTIC_THRESHOLD = 0.6


def _make_document_id(file_path: str) -> str:
    """Stable ID derived from file path + content hash prefix, so re-ingesting
    the same file is idempotent-ish and traceable (useful for duplicate
    detection later in Phase 8)."""
    with open(file_path, "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()[:10]
    return f"doc_{digest}_{uuid.uuid4().hex[:6]}"


def extract_pdf(
    file_path: str,
    filename: str | None = None,
    ocr_enabled: bool | None = None,
) -> DocumentResult:
    """Extract page-level text from a single PDF file.

    Never raises for a malformed *page* - only for a file that cannot be
    opened at all (e.g. not a valid PDF), in which case the returned
    DocumentResult.error is set instead of raising, so callers can keep
    processing the rest of a batch upload.

    `ocr_enabled` defaults to settings.enable_ocr; pages whose normal text
    layer is empty get a Tesseract OCR pass (Phase 9) before being
    counted as genuinely empty.
    """
    filename = filename or file_path.split("/")[-1]
    ocr_enabled = settings.enable_ocr if ocr_enabled is None else ocr_enabled

    try:
        doc = pymupdf.open(file_path)
    except Exception as exc:  # noqa: BLE001 - any failure to open must be caught
        logger.error("Failed to open PDF '%s': %s", filename, exc)
        return DocumentResult(
            document_id=_make_document_id(file_path) if _safe_exists(file_path) else f"doc_error_{uuid.uuid4().hex[:6]}",
            filename=filename,
            total_pages=0,
            error=f"Could not open file as a PDF: {exc}",
        )

    document_id = _make_document_id(file_path)
    pages: list[PageContent] = []
    empty_count = 0
    ocr_pages_used = 0
    ocr_would_help = False  # any page stayed empty after normal extraction

    for page_index in range(len(doc)):
        page_number = page_index + 1  # 1-indexed for humans
        try:
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
            is_empty = len(text) == 0
            ocr_used = False

            if is_empty:
                ocr_would_help = True
                if ocr_enabled:
                    ocr_text = ocr_page(page, dpi=settings.ocr_dpi, language=settings.ocr_language)
                    if ocr_text:
                        text = ocr_text
                        is_empty = False
                        ocr_used = True
                        ocr_pages_used += 1
                        logger.info("Page %d of '%s' recovered via OCR (%d chars).", page_number, filename, len(text))

            if is_empty:
                empty_count += 1

            pages.append(
                PageContent(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    text=text,
                    is_empty=is_empty,
                    ocr_used=ocr_used,
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-page failures
            logger.warning("Page %d of '%s' failed to extract: %s", page_number, filename, exc)
            pages.append(
                PageContent(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    text="",
                    is_empty=True,
                    extraction_error=str(exc),
                )
            )
            empty_count += 1

    total_pages = len(doc)
    doc.close()

    likely_scanned = (
        total_pages > 0 and (empty_count / total_pages) >= SCANNED_HEURISTIC_THRESHOLD
    )
    # OCR "would have helped" (some page needed it) but never got a real
    # chance: either it's disabled, or Tesseract isn't installed. Distinct
    # from "OCR ran and still found nothing" (e.g. a truly blank page, or
    # very low image quality) - callers use this to give a specific,
    # actionable error rather than a generic "looks scanned" message.
    ocr_unavailable = ocr_would_help and (not ocr_enabled or not is_tesseract_available())

    if likely_scanned:
        logger.info(
            "'%s' looks like a scanned PDF (%d/%d pages empty after%s OCR).",
            filename, empty_count, total_pages, "" if ocr_enabled else " skipping",
        )

    return DocumentResult(
        document_id=document_id,
        filename=filename,
        total_pages=total_pages,
        pages=pages,
        likely_scanned=likely_scanned,
        ocr_pages_used=ocr_pages_used,
        ocr_unavailable=ocr_unavailable,
    )


def _safe_exists(file_path: str) -> bool:
    try:
        with open(file_path, "rb"):
            return True
    except Exception:  # noqa: BLE001
        return False
