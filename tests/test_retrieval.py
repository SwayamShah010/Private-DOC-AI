"""
Unit tests for app/storage/vector_store.py

Uses synthetic (deterministic, seeded) embedding vectors instead of a
real embedding model - this tests the storage/retrieval logic in
complete isolation from Phase 3's Embedder, which needs a one-time
Hugging Face model download unavailable in restricted environments.
Each test gets its own tmp_path so ChromaDB collections never leak
between tests.
"""
import random

import pytest

from app.ingestion.chunk_models import Chunk
from app.storage.vector_store import VectorStore


def _fake_vector(seed: int, dim: int = 8) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _make_chunk(chunk_id, document_id, page_number, text, collection_id="default"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=page_number,
        chunk_text=text,
        collection_id=collection_id,
    )


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "vector_store"))


def test_add_and_count_chunks(store):
    chunks = [_make_chunk("c1", "d1", 1, "hello"), _make_chunk("c2", "d1", 2, "world")]
    embeddings = [_fake_vector(1), _fake_vector(2)]

    added = store.add_chunks(chunks, embeddings)

    assert added == 2
    assert store.collection_chunk_count("default") == 2


def test_add_chunks_mismatched_lengths_raises(store):
    chunks = [_make_chunk("c1", "d1", 1, "hello")]
    with pytest.raises(ValueError):
        store.add_chunks(chunks, embeddings=[])


def test_query_returns_nearest_neighbor_first(store):
    chunks = [
        _make_chunk("c1", "d1", 1, "target chunk"),
        _make_chunk("c2", "d1", 2, "far chunk"),
        _make_chunk("c3", "d1", 3, "farther chunk"),
    ]
    embeddings = [_fake_vector(1), _fake_vector(50), _fake_vector(999)]
    store.add_chunks(chunks, embeddings)

    hits = store.query(query_embedding=_fake_vector(1), top_k=3)

    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_query_respects_top_k(store):
    chunks = [_make_chunk(f"c{i}", "d1", i, f"chunk {i}") for i in range(5)]
    embeddings = [_fake_vector(i) for i in range(5)]
    store.add_chunks(chunks, embeddings)

    hits = store.query(query_embedding=_fake_vector(0), top_k=2)

    assert len(hits) == 2


def test_query_on_empty_collection_returns_empty_list(store):
    hits = store.query(query_embedding=_fake_vector(1), top_k=5, collection_id="nonexistent")
    assert hits == []


def test_metadata_round_trips_correctly(store):
    chunk = _make_chunk("c1", "doc_abc", 7, "some text", collection_id="research")
    store.add_chunks([chunk], [_fake_vector(1)])

    hits = store.query(query_embedding=_fake_vector(1), top_k=1, collection_id="research")

    assert hits[0]["metadata"]["document_id"] == "doc_abc"
    assert hits[0]["metadata"]["page_number"] == 7
    assert hits[0]["chunk_text"] == "some text"


def test_collections_are_isolated_from_each_other(store):
    store.add_chunks(
        [_make_chunk("c1", "d1", 1, "in default", collection_id="default")],
        [_fake_vector(1)],
    )
    store.add_chunks(
        [_make_chunk("c2", "d2", 1, "in research", collection_id="research")],
        [_fake_vector(2)],
    )

    assert store.collection_chunk_count("default") == 1
    assert store.collection_chunk_count("research") == 1

    default_hits = store.query(query_embedding=_fake_vector(1), collection_id="default")
    assert all(h["chunk_id"] != "c2" for h in default_hits)


def test_delete_document_removes_only_its_chunks(store):
    store.add_chunks(
        [
            _make_chunk("c1", "d1", 1, "keep me"),
            _make_chunk("c2", "d1", 2, "keep me too"),
            _make_chunk("c3", "d2", 1, "delete me"),
        ],
        [_fake_vector(1), _fake_vector(2), _fake_vector(3)],
    )

    store.delete_document("d2")

    assert store.collection_chunk_count("default") == 2
    remaining_ids = {h["chunk_id"] for h in store.query(query_embedding=_fake_vector(1), top_k=10)}
    assert "c3" not in remaining_ids


def test_list_collections_reflects_created_collections(store):
    store.add_chunks([_make_chunk("c1", "d1", 1, "x", collection_id="alpha")], [_fake_vector(1)])
    store.add_chunks([_make_chunk("c2", "d2", 1, "y", collection_id="beta")], [_fake_vector(2)])

    names = store.list_collections()

    assert "alpha" in names
    assert "beta" in names


def test_get_document_chunks_returns_only_that_document(store):
    store.add_chunks(
        [
            _make_chunk("c1", "d1", 1, "doc1 page1"),
            _make_chunk("c2", "d1", 2, "doc1 page2"),
            _make_chunk("c3", "d2", 1, "doc2 page1"),
        ],
        [_fake_vector(1), _fake_vector(2), _fake_vector(3)],
    )

    chunks = store.get_document_chunks("d1")

    assert len(chunks) == 2
    assert all(c["metadata"]["document_id"] == "d1" for c in chunks)


def test_get_document_chunks_returns_in_reading_order(store):
    # Deliberately add out of order and with chunk_index that would sort
    # wrong lexically (c10 vs c2) if chunk_id string were used for ordering.
    c_page2 = Chunk(chunk_id="d1_p2_c0", document_id="d1", filename="d1.pdf",
                     page_number=2, chunk_text="page2 text", chunk_index=0)
    c_page1_late = Chunk(chunk_id="d1_p1_c10", document_id="d1", filename="d1.pdf",
                          page_number=1, chunk_text="page1 chunk10", chunk_index=10)
    c_page1_early = Chunk(chunk_id="d1_p1_c2", document_id="d1", filename="d1.pdf",
                           page_number=1, chunk_text="page1 chunk2", chunk_index=2)

    store.add_chunks(
        [c_page2, c_page1_late, c_page1_early],
        [_fake_vector(1), _fake_vector(2), _fake_vector(3)],
    )

    chunks = store.get_document_chunks("d1")
    texts_in_order = [c["chunk_text"] for c in chunks]

    assert texts_in_order == ["page1 chunk2", "page1 chunk10", "page2 text"]


def test_list_documents_aggregates_page_counts(store):
    store.add_chunks(
        [
            _make_chunk("c1", "d1", 1, "a"),
            _make_chunk("c2", "d1", 2, "b"),
            _make_chunk("c3", "d1", 2, "c"),  # same page, different chunk
            _make_chunk("c4", "d2", 1, "d"),
        ],
        [_fake_vector(i) for i in range(4)],
    )

    docs = store.list_documents()
    by_id = {d["document_id"]: d for d in docs}

    assert by_id["d1"]["page_count"] == 2  # pages 1 and 2, deduplicated
    assert by_id["d2"]["page_count"] == 1


def test_list_documents_empty_collection_returns_empty_list(store):
    assert store.list_documents("empty_collection") == []
