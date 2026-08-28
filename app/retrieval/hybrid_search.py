"""
Hybrid search: fuses SemanticSearch (Phase 4) and KeywordSearch (Phase 8)
results using Reciprocal Rank Fusion (RRF).

Why RRF specifically: semantic distance (L2) and BM25 score live on
completely different, uncalibrated scales, so averaging or normalizing
them directly is fragile and dataset-dependent. RRF sidesteps that by
only looking at each chunk's *rank* within each retriever's list, not
its raw score:

    rrf_score(chunk) = sum over retrievers where chunk appears of
                        1 / (rrf_k + rank_in_that_retriever)

A chunk ranked highly by either retriever (or both) rises to the top; a
chunk only one retriever found still gets credit instead of being
dropped. This is the standard approach (Cormack et al., 2009) and is
what most production hybrid-search systems use for exactly this reason.

Same .search() signature as SemanticSearch, so RAGPipeline and the
summarizer can treat "a retriever" as an interchangeable dependency
(see app/retrieval/factory.py) without caring which one is plugged in.
"""
from app.config.logging_config import get_logger
from app.config.settings import settings
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.semantic_search import SemanticSearch
from app.storage.vector_store import VectorStore

logger = get_logger(__name__)


class HybridSearch:
    def __init__(
        self,
        semantic_search: SemanticSearch | None = None,
        keyword_search: KeywordSearch | None = None,
        rrf_k: int | None = None,
    ):
        vector_store = VectorStore()
        self.semantic_search = semantic_search or SemanticSearch(vector_store=vector_store)
        self.keyword_search = keyword_search or KeywordSearch(vector_store=vector_store)
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        collection_id: str = "default",
        candidate_k: int | None = None,
    ) -> list[dict]:
        """Return the top-k chunks by fused rank as plain dicts:
        {chunk_id, chunk_text, metadata, distance, bm25_score, score}.

        `distance` is carried over from the semantic retriever when a
        chunk was found there (needed by RAGPipeline's relevance gate);
        it's None for chunks that only the keyword retriever surfaced.
        `score` is the fused RRF score all results are ranked by - higher
        is better, regardless of source.
        """
        top_k = top_k or settings.top_k
        candidate_k = candidate_k or settings.hybrid_candidate_k
        # Pull a wider candidate pool from each retriever than we'll
        # ultimately return - fusion needs room to promote a chunk that
        # ranked, say, 8th semantically but 1st by keyword.
        candidate_k = max(candidate_k, top_k)

        semantic_hits = self.semantic_search.search(query, top_k=candidate_k, collection_id=collection_id)
        keyword_hits = self.keyword_search.search(query, top_k=candidate_k, collection_id=collection_id)

        by_chunk_id: dict[str, dict] = {}
        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(semantic_hits):
            cid = hit["chunk_id"]
            by_chunk_id[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, hit in enumerate(keyword_hits):
            cid = hit["chunk_id"]
            if cid in by_chunk_id:
                # Merge: keep the semantic hit's fields (has `distance`)
                # but add the keyword score for visibility/debugging.
                by_chunk_id[cid]["bm25_score"] = hit["bm25_score"]
            else:
                by_chunk_id[cid] = {**hit, "distance": None}
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for cid in ranked_ids[:top_k]:
            hit = dict(by_chunk_id[cid])
            hit["score"] = rrf_scores[cid]
            results.append(hit)

        logger.info(
            "Hybrid query %r: %d semantic + %d keyword candidates -> %d fused hits (collection=%s)",
            query, len(semantic_hits), len(keyword_hits), len(results), collection_id,
        )
        return results
