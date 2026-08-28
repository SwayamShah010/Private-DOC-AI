"""
Unit tests for app/storage/indexing_service.py

Uses a real PDF (generated on the fly with PyMuPDF, no network needed)
and a real VectorStore (ChromaDB, no network needed), but a fake
Embedder that returns deterministic vectors instead of downloading a
real Hugging Face model - consistent with how Phase 3/4 were tested.
"""
import pymupdf
import pytest

from app.storage.indexing_service import index_pdf_file
from app.storage.vector_store import VectorStore


class FakeEmbedder:
    model_name = "fake-embedder-for-tests"

    def embed_texts(self, texts):
        return [[float(len(t) % 7), 0.1, 0.2] for t in texts]


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "vs"))


@pytest.fixture
def good_pdf(tmp_path):
    path = tmp_path / "good.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "This document discusses privacy-first RAG systems in detail.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Citations and grounding are essential for trustworthy AI answers.")
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def blank_pdf(tmp_path):
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def corrupted_pdf(tmp_path):
    path = tmp_path / "corrupted.pdf"
    path.write_text("not a real pdf")
    return str(path)


def test_indexes_a_valid_pdf_end_to_end(good_pdf, store):
    result = index_pdf_file(good_pdf, "good.pdf", embedder=FakeEmbedder(), vector_store=store)

    assert result.success is True
    assert result.chunk_count > 0
    assert result.error is None
    assert store.collection_chunk_count("default") == result.chunk_count


def test_indexed_chunks_are_actually_searchable_afterward(good_pdf, store):
    index_pdf_file(good_pdf, "good.pdf", embedder=FakeEmbedder(), vector_store=store)

    chunks = store.get_document_chunks(
        [d["document_id"] for d in store.list_documents()][0]
    )
    assert len(chunks) > 0
    assert any("privacy" in c["chunk_text"].lower() for c in chunks)


def test_scanned_pdf_returns_clear_error_not_a_crash(blank_pdf, store):
    result = index_pdf_file(blank_pdf, "blank.pdf", embedder=FakeEmbedder(), vector_store=store)

    assert result.success is False
    assert result.likely_scanned is True
    assert "scanned" in result.error.lower()
    assert store.collection_chunk_count("default") == 0  # nothing should have been stored


def test_corrupted_pdf_returns_clear_error_not_a_crash(corrupted_pdf, store):
    result = index_pdf_file(corrupted_pdf, "corrupted.pdf", embedder=FakeEmbedder(), vector_store=store)

    assert result.success is False
    assert result.error is not None
    assert store.collection_chunk_count("default") == 0


def test_respects_custom_collection_id(good_pdf, store):
    result = index_pdf_file(good_pdf, "good.pdf", collection_id="research", embedder=FakeEmbedder(), vector_store=store)

    assert result.success is True
    assert store.collection_chunk_count("research") == result.chunk_count
    assert store.collection_chunk_count("default") == 0


# --- Phase 9: OCR fallback integration ---

@pytest.fixture
def scanned_pdf_with_text(tmp_path):
    """A 'scanned' PDF (image only, no text layer) that OCR can recover."""
    from PIL import Image, ImageDraw
    img_path = tmp_path / "scan_source.png"
    img = Image.new("RGB", (800, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 90), "Refunds are issued within thirty days.", fill="black")
    img.save(img_path)

    path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=200)
    page.insert_image(page.rect, filename=str(img_path))
    doc.save(str(path))
    doc.close()
    return str(path)


def test_scanned_pdf_indexes_successfully_via_ocr(scanned_pdf_with_text, store):
    result = index_pdf_file(scanned_pdf_with_text, "scanned.pdf", embedder=FakeEmbedder(), vector_store=store)

    assert result.success is True
    assert result.likely_scanned is False
    assert result.ocr_pages_used == 1
    assert result.chunk_count > 0

    chunks = store.get_document_chunks(result.document_id)
    assert any("thirty days" in c["chunk_text"].lower() for c in chunks)


def test_scanned_pdf_with_ocr_disabled_gives_generic_scanned_error(scanned_pdf_with_text, store):
    result = index_pdf_file(scanned_pdf_with_text, "scanned.pdf", embedder=FakeEmbedder(), vector_store=store, ocr_enabled=False)

    assert result.success is False
    assert result.likely_scanned is True
    assert "scanned" in result.error.lower()


def test_scanned_pdf_with_tesseract_missing_gives_actionable_error(scanned_pdf_with_text, store, monkeypatch):
    from app.ingestion import ocr as ocr_module

    def _raise(*args, **kwargs):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(ocr_module.pytesseract, "get_tesseract_version", _raise)
    ocr_module._tesseract_available_cache = None
    ocr_module._warned_unavailable = False

    result = index_pdf_file(scanned_pdf_with_text, "scanned.pdf", embedder=FakeEmbedder(), vector_store=store)

    ocr_module._tesseract_available_cache = None
    ocr_module._warned_unavailable = False

    assert result.success is False
    assert result.likely_scanned is True
    assert "tesseract" in result.error.lower()
    assert "install" in result.error.lower()
