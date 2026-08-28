"""
Cross-encoder reranking: a second, more accurate but more expensive
relevance pass over a small set of candidates.

Bi-encoders (Embedder, Phase 3) embed the query and each chunk
*separately* and compare vectors - fast enough to search thousands of
chunks, but the model never actually looks at query and chunk together.
A cross-encoder feeds the (query, chunk) pair into the model jointly, so
it can weigh how well they actually match - notably better relevance
judgments, but far too slow to run over a whole corpus. The standard
pattern (used here) is: retrieve a wider candidate set cheaply (hybrid
search), then rerank only that small set with the cross-encoder.

Lazy-loaded and cached the same way Phase 3's Embedder is, since loading
a cross-encoder model has the same one-time cost (and, on first use, the
same one-time Hugging Face download) - see settings.enable_reranking for
why this is opt-in rather than on by default.
"""
from sentence_transformers import CrossEncoder

from app.config.logging_config import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

_model_cache: dict[str, CrossEncoder] = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _model_cache:
        logger.info("Loading cross-encoder reranker '%s' (first load may download it)...", model_name)
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model
        self.model = _get_model(self.model_name)

    def rerank(self, query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
        """Re-score `hits` for how well each actually answers `query`, and
        return them sorted best-first (each hit gains a `rerank_score`
        field). Truncates to top_k if given, otherwise returns all of them
        re-ordered."""
        if not hits:
            return []

        pairs = [(query, h["chunk_text"]) for h in hits]
        scores = self.model.predict(pairs)

        rescored = []
        for hit, score in zip(hits, scores):
            new_hit = dict(hit)
            new_hit["rerank_score"] = float(score)
            rescored.append(new_hit)

        rescored.sort(key=lambda h: h["rerank_score"], reverse=True)
        return rescored[:top_k] if top_k else rescored
