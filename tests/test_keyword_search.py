"""
Unit tests for app/retrieval/keyword_search.py

Uses a real VectorStore (ChromaDB, tmp_path) purely as chunk storage -
KeywordSearch never touches embeddings, so this needs no embedding model
and no network access.
"""
import pytest

from app.ingestion.chunk_models import Chunk
from app.retrieval.keyword_search import KeywordSearch
from app.storage.vector_store import VectorStore


def _chunk(chunk_id, document_id, page, text, collection_id="default"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=page,
        chunk_text=text,
        collection_id=collection_id,
    )


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "vector_store"))


@pytest.fixture
def keyword_search(store):
    return KeywordSearch(vector_store=store)


def _seed(store):
    # Fake embeddings - KeywordSearch never reads them, only chunk_text.
    chunks = [
        _chunk("c1", "d1", 1, "The invoice number is INV-4471 and is due in 30 days."),
        _chunk("c2", "d1", 2, "Employees may request paid time off with two weeks notice."),
        _chunk("c3", "d1", 3, "Refunds are processed within 30 business days of the request."),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    store.add_chunks(chunks, embeddings)


def test_returns_chunk_matching_exact_keyword(store, keyword_search):
    _seed(store)

    hits = keyword_search.search("INV-4471", top_k=3)

    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == "c1"


def test_ranks_more_relevant_chunk_first(store, keyword_search):
    _seed(store)

    hits = keyword_search.search("paid time off notice", top_k=3)

    assert hits[0]["chunk_id"] == "c2"


def test_query_with_no_matching_terms_returns_empty(store, keyword_search):
    _seed(store)

    hits = keyword_search.search("quantum entanglement spacecraft", top_k=3)

    assert hits == []


def test_empty_collection_returns_empty_list(keyword_search):
    hits = keyword_search.search("anything", top_k=3, collection_id="nonexistent")
    assert hits == []


def test_respects_top_k(store, keyword_search):
    _seed(store)

    hits = keyword_search.search("days", top_k=1)

    assert len(hits) == 1


def test_hits_include_bm25_score_and_metadata(store, keyword_search):
    _seed(store)

    hits = keyword_search.search("invoice", top_k=1)

    assert hits[0]["bm25_score"] > 0
    assert hits[0]["metadata"]["document_id"] == "d1"


def test_index_rebuilds_when_chunks_added(store, keyword_search):
    _seed(store)
    assert keyword_search.search("skydiving", top_k=3) == []

    store.add_chunks(
        [_chunk("c4", "d1", 4, "Skydiving is not covered under this policy.")],
        [[0.7, 0.8]],
    )

    hits = keyword_search.search("skydiving", top_k=3)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c4"


def test_collections_are_isolated(store, keyword_search):
    store.add_chunks(
        [_chunk("c1", "d1", 1, "unique term zephyr", collection_id="alpha")],
        [[0.1, 0.2]],
    )
    store.add_chunks(
        [_chunk("c2", "d2", 1, "different content entirely", collection_id="beta")],
        [[0.3, 0.4]],
    )

    hits = keyword_search.search("zephyr", top_k=3, collection_id="beta")
    assert hits == []

    hits = keyword_search.search("zephyr", top_k=3, collection_id="alpha")
    assert len(hits) == 1
