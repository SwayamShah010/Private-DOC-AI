"""
Unit tests for app/ingestion/chunker.py
"""
import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.models import DocumentResult, PageContent


def _make_doc(pages_text: dict[int, str], document_id="doc_x", filename="x.pdf") -> DocumentResult:
    pages = [
        PageContent(document_id=document_id, filename=filename, page_number=n, text=t)
        for n, t in pages_text.items()
    ]
    return DocumentResult(document_id=document_id, filename=filename, total_pages=len(pages), pages=pages)


def test_chunk_size_and_overlap_are_respected():
    text = "word " * 500  # long enough to force multiple chunks
    doc = _make_doc({1: text})
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.chunk_text) <= 200 + 20  # splitter respects size, small slack for separators


def test_chunks_never_cross_page_boundary():
    doc = _make_doc({1: "Page one text. " * 30, 2: "Page two text. " * 30})
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)

    for c in chunks:
        assert c.page_number in (1, 2)
        if c.page_number == 1:
            assert "Page two" not in c.chunk_text
        if c.page_number == 2:
            assert "Page one" not in c.chunk_text


def test_empty_pages_produce_no_chunks():
    doc = _make_doc({1: "Real content here that is long enough.", 2: ""})
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)

    assert all(c.page_number != 2 for c in chunks)
    assert len(chunks) >= 1


def test_chunk_ids_are_unique():
    doc = _make_doc({1: "word " * 300})
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_metadata_matches_source_document():
    doc = _make_doc({1: "Some content about privacy and RAG systems."}, document_id="doc_abc", filename="report.pdf")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc_abc"
    assert chunks[0].filename == "report.pdf"
    assert chunks[0].page_number == 1


def test_overlap_greater_than_or_equal_to_size_raises():
    doc = _make_doc({1: "short text"})
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, chunk_overlap=150)


def test_custom_collection_id_is_applied():
    doc = _make_doc({1: "Content for a specific collection."})
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50, collection_id="research_papers")

    assert all(c.collection_id == "research_papers" for c in chunks)
