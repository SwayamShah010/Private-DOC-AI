"""
Retrieval factory: builds "the retriever" for the rest of the app to use,
based on settings.enable_hybrid_search / settings.enable_reranking.

This is the one place that knows how Phase 4's SemanticSearch, Phase 8's
HybridSearch, and Phase 8's Reranker fit together. Everything downstream
(RAGPipeline, the summarizer, the UI) just calls .search(query, top_k,
collection_id) on whatever this returns - see PRD Section 15 Phase 8:
"adding hybrid/BM25 search... means adding a sibling module, not
rewriting" the callers.
"""
from app.config.settings import settings
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.semantic_search import SemanticSearch


def build_default_retriever():
    """Semantic-only, or hybrid (semantic + BM25 keyword) per
    settings.enable_hybrid_search. This is what most of the app should
    call - it does NOT include reranking, which needs the extra
    `rerank_candidate_k` -> rerank -> `top_k` shape handled explicitly by
    build_default_reranker() below (RAGPipeline wires the two together)."""
    if settings.enable_hybrid_search:
        return HybridSearch()
    return SemanticSearch()


def build_default_reranker() -> Reranker | None:
    """None if reranking is disabled (the default) - callers should treat
    a None reranker as "skip this step", not fail."""
    if not settings.enable_reranking:
        return None
    return Reranker()
