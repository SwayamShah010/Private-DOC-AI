"""
Semantic search: turns a natural-language query into an embedding and
retrieves the most similar chunks from the vector store.

This is intentionally a thin orchestration layer over Embedder (Phase 3)
and VectorStore (this phase) - it has almost no logic of its own, which
is the point: it's easy to test each half in isolation and swap either
one out later (e.g. adding hybrid/BM25 search in Phase 8 means adding
a sibling to this module, not rewriting it).
"""
from app.config.logging_config import get_logger
from app.config.settings import settings
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore

logger = get_logger(__name__)


class SemanticSearch:
    def __init__(self, embedder: Embedder | None = None, vector_store: VectorStore | None = None):
        # Lazily created if not supplied, so callers who already have an
        # Embedder loaded (expensive to load) can reuse it instead of
        # reloading the model.
        self._embedder = embedder
        self.vector_store = vector_store or VectorStore()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int | None = None,
        collection_id: str = "default",
    ) -> list[dict]:
        top_k = top_k or settings.top_k
        query_embedding = self.embedder.embed_query(query)
        hits = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_id=collection_id,
        )
        logger.info("Query %r returned %d hits from collection=%s", query, len(hits), collection_id)
        return hits
