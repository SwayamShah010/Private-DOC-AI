"""
Unit tests for app/retrieval/hybrid_search.py

Uses fake semantic and keyword retrievers (pre-canned hit lists) so the
fusion logic (Reciprocal Rank Fusion) is tested in isolation, without a
real embedding model, vector DB, or BM25 index.
"""
from app.retrieval.hybrid_search import HybridSearch


def _hit(chunk_id, text="text", distance=None, bm25_score=None, doc="d1", page=1):
    hit = {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "metadata": {"filename": f"{doc}.pdf", "page_number": page, "document_id": doc},
    }
    if distance is not None:
        hit["distance"] = distance
    if bm25_score is not None:
        hit["bm25_score"] = bm25_score
    return hit


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=None, collection_id="default"):
        return self._hits[:top_k] if top_k else list(self._hits)


def test_chunk_found_by_both_retrievers_ranks_above_single_source():
    # c1 is #1 in both lists -> highest fused score.
    # c2 is only found by semantic, c3 only by keyword.
    semantic = FakeRetriever([_hit("c1", distance=0.1), _hit("c2", distance=0.3)])
    keyword = FakeRetriever([_hit("c1", bm25_score=5.0), _hit("c3", bm25_score=4.0)])
    hybrid = HybridSearch(semantic_search=semantic, keyword_search=keyword)

    hits = hybrid.search("query", top_k=3)

    assert hits[0]["chunk_id"] == "c1"


def test_semantic_only_hit_keeps_its_distance():
    semantic = FakeRetriever([_hit("c1", distance=0.2)])
    keyword = FakeRetriever([])
    hybrid = HybridSearch(semantic_search=semantic, keyword_search=keyword)

    hits = hybrid.search("query", top_k=3)

    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["distance"] == 0.2


def test_keyword_only_hit_has_none_distance():
    semantic = FakeRetriever([])
    keyword = FakeRetriever([_hit("c1", bm25_score=3.0)])
    hybrid = HybridSearch(semantic_search=semantic, keyword_search=keyword)

    hits = hybrid.search("query", top_k=3)

    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["distance"] is None
    assert hits[0]["bm25_score"] == 3.0


def test_results_are_sorted_by_fused_score_descending():
    semantic = FakeRetriever([_hit("c1", distance=0.1), _hit("c2", distance=0.9)])
    keyword = FakeRetriever([_hit("c3", bm25_score=10.0)])
    hybrid = HybridSearch(semantic_search=semantic, keyword_search=keyword)

    hits = hybrid.search("query", top_k=3)

    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_respects_top_k_after_fusion():
    semantic = FakeRetriever([_hit(f"c{i}", distance=i * 0.1) for i in range(10)])
    keyword = FakeRetriever([])
    hybrid = HybridSearch(semantic_search=semantic, keyword_search=keyword)

    hits = hybrid.search("query", top_k=2)

    assert len(hits) == 2


def test_empty_results_from_both_retrievers_returns_empty_list():
    hybrid = HybridSearch(semantic_search=FakeRetriever([]), keyword_search=FakeRetriever([]))

    hits = hybrid.search("query", top_k=5)

    assert hits == []


def test_rrf_k_changes_relative_weighting_but_not_crash():
    semantic = FakeRetriever([_hit("c1", distance=0.1), _hit("c2", distance=0.2)])
    keyword = FakeRetriever([_hit("c2", bm25_score=8.0), _hit("c1", bm25_score=1.0)])

    hybrid_default = HybridSearch(semantic_search=semantic, keyword_search=keyword, rrf_k=60)
    hybrid_small_k = HybridSearch(semantic_search=semantic, keyword_search=keyword, rrf_k=1)

    # Both should run without error and return both chunks.
    assert {h["chunk_id"] for h in hybrid_default.search("q", top_k=2)} == {"c1", "c2"}
    assert {h["chunk_id"] for h in hybrid_small_k.search("q", top_k=2)} == {"c1", "c2"}
