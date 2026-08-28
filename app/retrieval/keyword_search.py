"""
Keyword search: BM25 ranking over the full chunk corpus of a collection.

This is the other half of Phase 8's hybrid retrieval. Semantic search
(Phase 4) is good at "meaning" but can blur past exact terms - part
numbers, product names, acronyms, dollar figures - that a user's
question quotes verbatim. BM25 is the classic fix: pure lexical overlap,
no embedding model, no network call, deterministic.

Deliberately NOT persisted like the vector store is: rank_bm25's index
is cheap to build (milliseconds to low seconds for the corpus sizes a
local single-user tool like this deals with) and chunks change whenever
a document is added or deleted, so rebuilding on demand is simpler and
more correct than tracking invalidation. We do cache the built index
per collection for the lifetime of one KeywordSearch instance, keyed by
chunk count, so repeated queries in the same session (e.g. multiple
questions in one chat) don't rebuild it every time.
"""
import re

from rank_bm25 import BM25Okapi

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.storage.vector_store import VectorStore

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenizer. Deliberately simple (no
    stemming/stopwords) - BM25's term-frequency weighting already
    down-weights common words, and a simple tokenizer is easier to reason
    about and keeps this dependency-free beyond rank_bm25 itself."""
    return _TOKEN_RE.findall(text.lower())


class KeywordSearch:
    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()
        # collection_id -> (chunk_count_when_built, BM25Okapi, chunk_list, tokenized_corpus)
        self._index_cache: dict[str, tuple[int, BM25Okapi, list[dict], list[list[str]]]] = {}

    def _get_index(self, collection_id: str) -> tuple[BM25Okapi, list[dict], list[list[str]]]:
        chunks = self.vector_store.get_all_chunks(collection_id)
        cached = self._index_cache.get(collection_id)
        if cached is not None and cached[0] == len(chunks):
            return cached[1], cached[2], cached[3]

        tokenized_corpus = [_tokenize(c["chunk_text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._index_cache[collection_id] = (len(chunks), bm25, chunks, tokenized_corpus)
        logger.info(
            "Built BM25 index for collection=%s (%d chunks).", collection_id, len(chunks)
        )
        return bm25, chunks, tokenized_corpus

    def search(
        self,
        query: str,
        top_k: int | None = None,
        collection_id: str = "default",
    ) -> list[dict]:
        """Return the top-k chunks by BM25 score, as plain dicts:
        {chunk_id, chunk_text, metadata, bm25_score}. Empty list if the
        collection has no chunks yet."""
        top_k = top_k or settings.top_k
        bm25, chunks, tokenized_corpus = self._get_index(collection_id)

        if bm25 is None or not chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)

        scores = bm25.get_scores(query_tokens)
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

        hits = []
        for i in ranked[:top_k]:
            # Gate on actual lexical overlap rather than the raw BM25 score
            # sign: rank_bm25's IDF formula can go negative for terms that
            # appear in most/all documents of a small corpus (e.g. a single-
            # document collection, or a term that's genuinely common across
            # the corpus) - relying on "score > 0" would wrongly discard
            # real matches in exactly those small-collection cases this
            # local, single-user tool is likely to hit often.
            if not (query_token_set & set(tokenized_corpus[i])):
                continue
            hits.append({
                "chunk_id": chunks[i]["chunk_id"],
                "chunk_text": chunks[i]["chunk_text"],
                "metadata": chunks[i]["metadata"],
                "bm25_score": float(scores[i]),
            })

        logger.info(
            "Keyword query %r returned %d hits from collection=%s", query, len(hits), collection_id
        )
        return hits
