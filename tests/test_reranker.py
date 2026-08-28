"""
Unit tests for app/retrieval/reranker.py

The real cross-encoder model requires a one-time Hugging Face download,
unavailable in restricted environments - so these tests patch
sentence_transformers.CrossEncoder with a deterministic fake that scores
pairs by a simple, predictable rule instead of loading real weights.
"""
from unittest.mock import patch

import app.retrieval.reranker as reranker_module
from app.retrieval.reranker import Reranker


class FakeCrossEncoder:
    """Scores each (query, chunk_text) pair by how many words they share -
    deterministic and good enough to prove reranking reorders correctly."""

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


def _hit(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "metadata": {"filename": "a.pdf", "page_number": 1, "document_id": "d1"},
        "distance": 0.5,
    }


def setup_function(_):
    # Fresh model cache per test so patches don't leak across tests.
    reranker_module._model_cache.clear()


def test_reorders_hits_by_relevance():
    hits = [
        _hit("c1", "totally unrelated content about gardening"),
        _hit("c2", "the notice period for termination is 30 days"),
    ]
    with patch.object(reranker_module, "CrossEncoder", FakeCrossEncoder):
        reranker = Reranker(model_name="fake-model")
        result = reranker.rerank("what is the notice period for termination", hits)

    assert result[0]["chunk_id"] == "c2"


def test_adds_rerank_score_field():
    hits = [_hit("c1", "some content")]
    with patch.object(reranker_module, "CrossEncoder", FakeCrossEncoder):
        reranker = Reranker(model_name="fake-model")
        result = reranker.rerank("some query", hits)

    assert "rerank_score" in result[0]


def test_respects_top_k():
    hits = [_hit(f"c{i}", f"content {i}") for i in range(5)]
    with patch.object(reranker_module, "CrossEncoder", FakeCrossEncoder):
        reranker = Reranker(model_name="fake-model")
        result = reranker.rerank("content", hits, top_k=2)

    assert len(result) == 2


def test_empty_hits_returns_empty_list():
    with patch.object(reranker_module, "CrossEncoder", FakeCrossEncoder):
        reranker = Reranker(model_name="fake-model")
        result = reranker.rerank("query", [])

    assert result == []


def test_model_is_cached_across_instances():
    with patch.object(reranker_module, "CrossEncoder", FakeCrossEncoder):
        Reranker(model_name="fake-model")
        Reranker(model_name="fake-model")

    assert len(reranker_module._model_cache) == 1
