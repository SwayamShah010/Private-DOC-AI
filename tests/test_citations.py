"""
Unit tests for app/citations/citation_mapper.py
"""
from app.citations.citation_mapper import build_citations, format_citation_label


def _hit(chunk_id, filename, page, text, distance):
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "distance": distance,
        "metadata": {"filename": filename, "page_number": page, "document_id": "d1"},
    }


def test_builds_one_citation_per_hit():
    hits = [
        _hit("c1", "a.pdf", 1, "text one", 0.1),
        _hit("c2", "a.pdf", 2, "text two", 0.2),
    ]
    citations = build_citations(hits)
    assert len(citations) == 2
    assert citations[0].filename == "a.pdf"
    assert citations[0].page_number == 1


def test_deduplicates_same_document_and_page():
    hits = [
        _hit("c1", "a.pdf", 1, "first chunk from page 1", 0.1),
        _hit("c2", "a.pdf", 1, "second chunk from page 1", 0.15),
        _hit("c3", "a.pdf", 2, "different page", 0.2),
    ]
    citations = build_citations(hits)
    assert len(citations) == 2  # (a.pdf, page 1) collapsed, (a.pdf, page 2) kept


def test_dedup_keeps_first_most_relevant_occurrence():
    hits = [
        _hit("c1", "a.pdf", 1, "most relevant text", 0.1),
        _hit("c2", "a.pdf", 1, "less relevant duplicate page", 0.4),
    ]
    citations = build_citations(hits)
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"


def test_different_filenames_same_page_are_not_deduped():
    hits = [
        _hit("c1", "a.pdf", 1, "from doc a", 0.1),
        _hit("c2", "b.pdf", 1, "from doc b", 0.2),
    ]
    citations = build_citations(hits)
    assert len(citations) == 2


def test_empty_hits_returns_empty_citations():
    assert build_citations([]) == []


def test_format_citation_label():
    hits = [_hit("c1", "report.pdf", 4, "text", 0.1)]
    citation = build_citations(hits)[0]
    assert format_citation_label(citation) == "report.pdf, p. 4"
