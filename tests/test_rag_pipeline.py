"""
Unit tests for app/generation/rag_pipeline.py

Uses a fake SemanticSearch (no real embedding model or vector DB needed)
and a mocked OllamaClient (no live Ollama server needed) so the
orchestration logic - the relevance gate, decline handling, citation
attachment - is fully testable without any local services running.
"""
from unittest.mock import MagicMock

from app.generation.rag_pipeline import RAGPipeline
from app.generation.prompt_builder import NO_ANSWER_PHRASE


class FakeSearch:
    """Stands in for SemanticSearch - returns pre-canned hits instead of
    doing real embedding + vector lookup."""
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=None, collection_id="default"):
        return self._hits


def _hit(chunk_id, filename, page, text, distance):
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "distance": distance,
        "metadata": {"filename": filename, "page_number": page, "document_id": "d1"},
    }


def test_declines_without_calling_llm_when_no_hits():
    search = FakeSearch(hits=[])
    llm = MagicMock()
    pipeline = RAGPipeline(search=search, llm=llm)

    result = pipeline.answer("what is X?")

    assert result.declined is True
    assert result.answer_text == NO_ANSWER_PHRASE
    assert result.citations == []
    llm.generate.assert_not_called()  # relevance gate should skip the LLM entirely


def test_declines_without_calling_llm_when_all_hits_below_relevance_threshold():
    weak_hits = [_hit("c1", "a.pdf", 1, "barely related", distance=5.0)]
    search = FakeSearch(hits=weak_hits)
    llm = MagicMock()
    pipeline = RAGPipeline(search=search, llm=llm, relevance_threshold=1.5)

    result = pipeline.answer("what is X?")

    assert result.declined is True
    llm.generate.assert_not_called()


def test_calls_llm_and_attaches_citations_when_relevant_hits_found():
    good_hits = [_hit("c1", "a.pdf", 3, "X is defined as Y.", distance=0.2)]
    search = FakeSearch(hits=good_hits)
    llm = MagicMock()
    llm.generate.return_value = "X is Y."
    pipeline = RAGPipeline(search=search, llm=llm, relevance_threshold=1.5)

    result = pipeline.answer("what is X?")

    assert result.declined is False
    assert result.answer_text == "X is Y."
    assert len(result.citations) == 1
    assert result.citations[0].filename == "a.pdf"
    assert result.citations[0].page_number == 3
    llm.generate.assert_called_once()


def test_llm_declining_clears_citations_even_with_relevant_hits():
    good_hits = [_hit("c1", "a.pdf", 3, "unrelated to the actual question", distance=0.2)]
    search = FakeSearch(hits=good_hits)
    llm = MagicMock()
    llm.generate.return_value = NO_ANSWER_PHRASE
    pipeline = RAGPipeline(search=search, llm=llm, relevance_threshold=1.5)

    result = pipeline.answer("what is X?")

    assert result.declined is True
    assert result.citations == []  # nothing to attribute if the model declined


def test_relevance_threshold_filters_individual_hits_not_just_whole_query():
    mixed_hits = [
        _hit("c1", "a.pdf", 1, "relevant", distance=0.3),
        _hit("c2", "b.pdf", 1, "irrelevant", distance=5.0),
    ]
    search = FakeSearch(hits=mixed_hits)
    llm = MagicMock()
    llm.generate.return_value = "Answer based on relevant chunk."
    pipeline = RAGPipeline(search=search, llm=llm, relevance_threshold=1.5)

    result = pipeline.answer("what is X?")

    assert result.declined is False
    assert len(result.citations) == 1
    assert result.citations[0].filename == "a.pdf"  # the irrelevant one was filtered out


# --- Phase 8: hybrid search / reranking integration with the gate ---

def _hybrid_hit(chunk_id, filename, page, text, distance=None, bm25_score=None):
    hit = {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "metadata": {"filename": filename, "page_number": page, "document_id": "d1"},
        "distance": distance,
    }
    if bm25_score is not None:
        hit["bm25_score"] = bm25_score
    return hit


def test_keyword_only_hybrid_hit_with_no_distance_passes_the_gate():
    # HybridSearch sets distance=None for chunks only BM25 found - the gate
    # should trust the retriever's fusion rather than declining for lack of
    # a semantic distance to compare against the threshold.
    hits = [_hybrid_hit("c1", "a.pdf", 1, "INV-4471 details", distance=None, bm25_score=6.0)]
    search = FakeSearch(hits=hits)
    llm = MagicMock()
    llm.generate.return_value = "Here are the invoice details."
    pipeline = RAGPipeline(search=search, llm=llm, relevance_threshold=1.5)

    result = pipeline.answer("what is invoice INV-4471?")

    assert result.declined is False
    assert len(result.citations) == 1


def test_reranker_score_supersedes_distance_in_gate():
    # Even a hit with a "bad" raw distance should be judged by its
    # rerank_score once reranking has run, not the original distance.
    hits = [_hit("c1", "a.pdf", 1, "X is defined as Y.", distance=5.0)]
    search = FakeSearch(hits=hits)
    llm = MagicMock()
    llm.generate.return_value = "X is Y."

    class FakeReranker:
        def rerank(self, query, candidates, top_k=None):
            reranked = [{**c, "rerank_score": 3.0} for c in candidates]
            return reranked[:top_k] if top_k else reranked

    pipeline = RAGPipeline(
        search=search, llm=llm, reranker=FakeReranker(),
        relevance_threshold=1.5, rerank_relevance_threshold=0.0,
    )

    result = pipeline.answer("what is X?")

    assert result.declined is False
    llm.generate.assert_called_once()


def test_reranker_declining_hit_below_threshold_skips_llm():
    hits = [_hit("c1", "a.pdf", 1, "barely related", distance=0.1)]
    search = FakeSearch(hits=hits)
    llm = MagicMock()

    class FakeReranker:
        def rerank(self, query, candidates, top_k=None):
            reranked = [{**c, "rerank_score": -2.0} for c in candidates]
            return reranked[:top_k] if top_k else reranked

    pipeline = RAGPipeline(
        search=search, llm=llm, reranker=FakeReranker(),
        rerank_relevance_threshold=0.0,
    )

    result = pipeline.answer("what is X?")

    assert result.declined is True
    llm.generate.assert_not_called()
