"""
OCR fallback for scanned/image-only PDF pages (Phase 9).

Uses PyMuPDF's own page rasterizer (page.get_pixmap) to render a page to
an image - no extra system dependency like poppler needed, since
PyMuPDF already links against its own rendering engine. That image is
then handed to pytesseract, a thin wrapper around the Tesseract OCR
engine (a separate system binary - see README for install instructions).

Deliberately isolated in its own module so the rest of ingestion doesn't
need to know or care whether Tesseract is installed: _tesseract_available()
is checked once and cached, and every public function here degrades to
"no OCR text" rather than raising, so a machine without Tesseract falls
back to exactly the pre-Phase-9 behavior (scanned pages come back empty)
instead of crashing.
"""
import pymupdf
import pytesseract
from PIL import Image

from app.config.logging_config import get_logger

logger = get_logger(__name__)

_tesseract_available_cache: bool | None = None
_warned_unavailable = False


def is_tesseract_available() -> bool:
    """Cached check for whether the Tesseract binary is reachable. Checked
    once per process since it can't change mid-run and get_tesseract_version()
    shells out, which we don't want to repeat for every empty page."""
    global _tesseract_available_cache, _warned_unavailable
    if _tesseract_available_cache is None:
        try:
            pytesseract.get_tesseract_version()
            _tesseract_available_cache = True
        except Exception as exc:  # noqa: BLE001 - covers missing binary, bad PATH, etc.
            _tesseract_available_cache = False
            if not _warned_unavailable:
                logger.warning(
                    "Tesseract OCR engine not found (%s) - scanned PDF pages will "
                    "be skipped instead of OCR'd. See README for install instructions.",
                    exc,
                )
                _warned_unavailable = True
    return _tesseract_available_cache


def ocr_page(page: "pymupdf.Page", dpi: int = 300, language: str = "eng") -> str:
    """Rasterize one PDF page and run OCR on it. Returns extracted text,
    or "" if Tesseract isn't available or OCR finds nothing / fails.

    Never raises - a single unreadable/corrupt page must not take down
    the whole document (same reliability principle as pdf_extractor's
    per-page try/except, PRD Section 18).
    """
    if not is_tesseract_available():
        return ""

    try:
        pixmap = page.get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except Exception as exc:  # noqa: BLE001 - isolate per-page OCR failures
        logger.warning("OCR failed on page %d: %s", getattr(page, "number", -1) + 1, exc)
        return ""
