"""
Unit tests for app/ingestion/ocr.py

Builds a real "scanned" PDF by rendering text into a PIL image and
embedding that image (no text layer) as the page content - this is a
faithful stand-in for an actual scanned document, and exercises the real
Tesseract binary rather than mocking it, since Tesseract is a normal
system dependency in this environment (see README).
"""
import pymupdf
import pytest
from PIL import Image, ImageDraw

from app.ingestion import ocr as ocr_module
from app.ingestion.ocr import is_tesseract_available, ocr_page


def _make_scanned_pdf(tmp_path, text: str, image_size=(800, 200)) -> str:
    """A single-page PDF whose only content is a rasterized image of
    `text` - no embedded text layer, i.e. what a real scanner produces."""
    img_path = tmp_path / "page_image.png"
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, image_size[1] // 2 - 10), text, fill="black")
    img.save(img_path)

    pdf_path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=image_size[0], height=image_size[1])
    page.insert_image(page.rect, filename=str(img_path))
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture(autouse=True)
def reset_tesseract_cache():
    """is_tesseract_available() caches its result at module level - reset
    it before/after each test so patches in one test don't leak into the
    next."""
    ocr_module._tesseract_available_cache = None
    ocr_module._warned_unavailable = False
    yield
    ocr_module._tesseract_available_cache = None
    ocr_module._warned_unavailable = False


def test_is_tesseract_available_returns_true_when_installed():
    # Tesseract is installed in this environment (see README setup step).
    assert is_tesseract_available() is True


def test_ocr_page_recovers_text_from_a_rendered_image(tmp_path):
    pdf_path = _make_scanned_pdf(tmp_path, "Warranty period is twenty four months.")
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)

    text = ocr_page(page, dpi=300, language="eng")
    doc.close()

    assert "warranty" in text.lower()
    assert "twenty four months" in text.lower()


def test_ocr_page_on_truly_blank_page_returns_empty_string(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    pdf_path = tmp_path / "blank.pdf"
    doc.save(str(pdf_path))
    doc.close()

    doc2 = pymupdf.open(str(pdf_path))
    text = ocr_page(doc2.load_page(0), dpi=300, language="eng")
    doc2.close()

    assert text == ""


def test_ocr_page_returns_empty_string_when_tesseract_unavailable(tmp_path, monkeypatch):
    pdf_path = _make_scanned_pdf(tmp_path, "Some scanned content.")
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)

    def _raise(*args, **kwargs):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(ocr_module.pytesseract, "get_tesseract_version", _raise)

    text = ocr_page(page, dpi=300, language="eng")
    doc.close()

    assert text == ""


def test_is_tesseract_available_result_is_cached(monkeypatch):
    call_count = 0
    real_check = ocr_module.pytesseract.get_tesseract_version

    def _counting_check():
        nonlocal call_count
        call_count += 1
        return real_check()

    monkeypatch.setattr(ocr_module.pytesseract, "get_tesseract_version", _counting_check)

    is_tesseract_available()
    is_tesseract_available()
    is_tesseract_available()

    assert call_count == 1
