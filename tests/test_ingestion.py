"""
Unit tests for app/ingestion/pdf_extractor.py

Uses PyMuPDF itself to generate throwaway test PDFs in a temp dir,
so these tests need no fixture files committed to the repo.
"""
import pymupdf
import pytest

from app.ingestion.pdf_extractor import extract_pdf


@pytest.fixture
def sample_pdf(tmp_path):
    """A well-formed 2-page PDF with real text on both pages."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Page one content about privacy and RAG.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Page two content about citations.")
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def blank_pdf(tmp_path):
    """A PDF with pages but no text at all - simulates a scanned document."""
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return str(path)


def test_extracts_correct_page_count(sample_pdf):
    result = extract_pdf(sample_pdf, filename="sample.pdf")
    assert result.total_pages == 2
    assert len(result.pages) == 2
    assert result.error is None


def test_page_numbers_are_1_indexed(sample_pdf):
    result = extract_pdf(sample_pdf, filename="sample.pdf")
    assert [p.page_number for p in result.pages] == [1, 2]


def test_extracted_text_is_correct(sample_pdf):
    result = extract_pdf(sample_pdf, filename="sample.pdf")
    assert "privacy" in result.pages[0].text.lower()
    assert "citations" in result.pages[1].text.lower()


def test_document_id_is_generated_and_stable_format(sample_pdf):
    result = extract_pdf(sample_pdf, filename="sample.pdf")
    assert result.document_id.startswith("doc_")


def test_nonexistent_or_corrupted_file_does_not_raise(tmp_path):
    bad_file = tmp_path / "not_a_pdf.pdf"
    bad_file.write_text("this is definitely not a PDF")

    # Must not raise - must return a DocumentResult with .error set instead
    result = extract_pdf(str(bad_file), filename="not_a_pdf.pdf")
    assert result.error is not None
    assert result.total_pages == 0
    assert result.pages == []


def test_blank_pages_flagged_as_likely_scanned(blank_pdf):
    result = extract_pdf(blank_pdf, filename="blank.pdf")
    assert result.total_pages == 3
    assert all(p.is_empty for p in result.pages)
    assert result.likely_scanned is True


def test_filename_defaults_to_path_basename_when_not_given(sample_pdf):
    result = extract_pdf(sample_pdf)  # no filename arg
    assert result.filename == "sample.pdf"


# --- Phase 9: OCR fallback ---

def _make_scanned_pdf(tmp_path, text: str, image_size=(800, 200)):
    from PIL import Image, ImageDraw
    img_path = tmp_path / "scan_source.png"
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


def test_ocr_recovers_text_from_a_scanned_page(tmp_path):
    pdf_path = _make_scanned_pdf(tmp_path, "Refunds are issued within thirty days.")

    result = extract_pdf(pdf_path, filename="scanned.pdf", ocr_enabled=True)

    assert result.likely_scanned is False  # OCR recovered it, so it's no longer "unreadable"
    assert result.pages[0].is_empty is False
    assert result.pages[0].ocr_used is True
    assert "thirty days" in result.pages[0].text.lower()
    assert result.ocr_pages_used == 1


def test_ocr_disabled_leaves_scanned_page_empty(tmp_path):
    pdf_path = _make_scanned_pdf(tmp_path, "Refunds are issued within thirty days.")

    result = extract_pdf(pdf_path, filename="scanned.pdf", ocr_enabled=False)

    assert result.likely_scanned is True
    assert result.pages[0].is_empty is True
    assert result.pages[0].ocr_used is False
    assert result.ocr_pages_used == 0


def test_ocr_unavailable_flag_set_when_tesseract_missing(tmp_path, monkeypatch):
    from app.ingestion import ocr as ocr_module

    pdf_path = _make_scanned_pdf(tmp_path, "Refunds are issued within thirty days.")

    def _raise(*args, **kwargs):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(ocr_module.pytesseract, "get_tesseract_version", _raise)
    ocr_module._tesseract_available_cache = None
    ocr_module._warned_unavailable = False

    result = extract_pdf(pdf_path, filename="scanned.pdf", ocr_enabled=True)

    ocr_module._tesseract_available_cache = None  # don't leak into other tests
    ocr_module._warned_unavailable = False

    assert result.likely_scanned is True
    assert result.ocr_unavailable is True
    assert result.ocr_pages_used == 0


def test_mixed_document_only_flags_pages_that_stay_empty(tmp_path):
    # One real-text page + one scanned page: OCR should recover the scanned
    # one, and the document overall should NOT be flagged as likely_scanned.
    scanned_pdf_path = _make_scanned_pdf(tmp_path, "Warranty is twelve months.")
    scanned_doc = pymupdf.open(scanned_pdf_path)
    scanned_page = scanned_doc.load_page(0)

    combined_path = tmp_path / "mixed.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "This page has a normal, searchable text layer.")
    doc.insert_pdf(scanned_doc, start_at=-1)
    doc.save(str(combined_path))
    doc.close()
    scanned_doc.close()

    result = extract_pdf(str(combined_path), filename="mixed.pdf", ocr_enabled=True)

    assert result.total_pages == 2
    assert result.likely_scanned is False
    assert result.pages[0].ocr_used is False
    assert result.pages[1].ocr_used is True
    assert "twelve months" in result.pages[1].text.lower()
