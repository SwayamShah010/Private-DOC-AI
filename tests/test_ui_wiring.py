"""
Regression test for a real bug caught during Phase 11 review: the
Streamlit UI (app/ui/streamlit_app.py) was still hardcoding a plain
SemanticSearch for its RAGPipeline, even after Phase 8 added hybrid
search and reranking - meaning the live app never actually benefited
from either feature, despite them being fully implemented and tested
elsewhere. This test locks in the fix: get_search()/get_reranker()
must respect settings.enable_hybrid_search / settings.enable_reranking.

Uses fakes for Embedder and Reranker (both would otherwise need a
Hugging Face model download) to keep this fast and network-free - the
point of this test is the *wiring*, not embedding/reranking quality,
which is already covered elsewhere (test_hybrid_search.py, etc.).
"""
from dataclasses import replace
from unittest.mock import MagicMock

import app.ui.streamlit_app as ui
from app.storage.vector_store import VectorStore


class FakeEmbedder:
    model_name = "fake"
    def embed_texts(self, texts):
        return [[0.0, 0.0] for _ in texts]
    def embed_query(self, q):
        return [0.0, 0.0]


def _reset_cached_resources():
    # st.cache_resource-decorated functions cache across test runs unless
    # explicitly cleared - each test needs a clean slate to see the effect
    # of monkeypatched settings.
    for fn in (ui.get_vector_store, ui.get_embedder, ui.get_search, ui.get_reranker):
        fn.clear()


def test_get_search_returns_hybrid_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "settings", replace(ui.settings, enable_hybrid_search=True, enable_reranking=False))
    monkeypatch.setattr(ui, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ui, "VectorStore", lambda: VectorStore(persist_dir=str(tmp_path)))
    _reset_cached_resources()

    from app.retrieval.hybrid_search import HybridSearch
    search = ui.get_search()

    assert isinstance(search, HybridSearch)
    _reset_cached_resources()


def test_get_search_returns_semantic_only_when_hybrid_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "settings", replace(ui.settings, enable_hybrid_search=False))
    monkeypatch.setattr(ui, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ui, "VectorStore", lambda: VectorStore(persist_dir=str(tmp_path)))
    _reset_cached_resources()

    from app.retrieval.hybrid_search import HybridSearch
    search = ui.get_search()

    assert not isinstance(search, HybridSearch)
    assert type(search).__name__ == "SemanticSearch"
    _reset_cached_resources()


def test_get_reranker_is_none_when_disabled(monkeypatch):
    monkeypatch.setattr(ui, "settings", replace(ui.settings, enable_reranking=False))
    _reset_cached_resources()

    assert ui.get_reranker() is None
    _reset_cached_resources()


def test_get_reranker_builds_one_when_enabled(monkeypatch):
    monkeypatch.setattr(ui, "settings", replace(ui.settings, enable_reranking=True))
    monkeypatch.setattr(ui, "Reranker", MagicMock(return_value="a-reranker-instance"))
    _reset_cached_resources()

    assert ui.get_reranker() == "a-reranker-instance"
    _reset_cached_resources()


def test_get_rag_pipeline_passes_search_and_reranker_through(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "settings", replace(ui.settings, enable_hybrid_search=False, enable_reranking=False))
    monkeypatch.setattr(ui, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ui, "VectorStore", lambda: VectorStore(persist_dir=str(tmp_path)))
    monkeypatch.setattr(ui, "get_ollama_client", MagicMock(return_value=MagicMock()))
    _reset_cached_resources()

    pipeline = ui.get_rag_pipeline()

    assert pipeline.search is ui.get_search()
    assert pipeline.reranker is None
    _reset_cached_resources()
