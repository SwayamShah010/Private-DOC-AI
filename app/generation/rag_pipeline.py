"""
The full RAG Q&A pipeline: retrieve -> build grounded prompt -> generate
-> attach citations. This is what the UI (Phase 7) will call directly.

Three-layer defense against unsupported answers (PRD Section 11 + 21):
  1. A relevance gate BEFORE calling the LLM at all - if nothing
     relevant was retrieved, we skip generation entirely and return the
     decline message immediately. Cheaper, faster, and more reliable
     than hoping the LLM notices weak context on its own.
  2. (Phase 8) If reranking is enabled, a cross-encoder pass narrows the
     candidates down to the ones that actually answer the question
     before the gate even runs, so the gate is judging better evidence.
  3. The grounded prompt itself (Phase 5's prompt_builder), which
     instructs the model to decline when the provided sources are
     insufficient even if the relevance gate let borderline cases through.

Retrieval itself (semantic-only vs. hybrid semantic+keyword) is decided
by app.retrieval.factory based on settings - this module doesn't care
which one it got, only that it implements .search(query, top_k,
collection_id).
"""
from dataclasses import dataclass, field

from app.citations.citation_mapper import Citation, build_citations
from app.config.logging_config import get_logger
from app.config.settings import settings
from app.generation.ollama_client import OllamaClient
from app.generation.prompt_builder import NO_ANSWER_PHRASE, build_grounded_prompt, is_decline_response
from app.retrieval.factory import build_default_reranker, build_default_retriever
from app.retrieval.reranker import Reranker

logger = get_logger(__name__)

# ChromaDB default distance is L2 (squared Euclidean) over normalized-ish
# embedding space; empirically, all-MiniLM-L6-v2 gives well-matched query/
# chunk pairs a distance well under this. Loosely-related content is well
# above it. MUST be re-tuned by hand once you're testing against real
# documents and a real embedding model - treat this as a starting point,
# not a calibrated constant.
DEFAULT_RELEVANCE_THRESHOLD = 1.5

# cross-encoder/ms-marco-MiniLM-L-6-v2 outputs a raw (unbounded) relevance
# logit, not a probability - roughly, positive means "relevant", negative
# means "not". 0.0 is a reasonable starting cutoff. Like the distance
# threshold above, this is a starting point that benefits from tuning
# against real queries once reranking is turned on (PRD Section 15 Phase 8:
# "tune top-k and thresholds").
DEFAULT_RERANK_RELEVANCE_THRESHOLD = 0.0


@dataclass
class AnswerResult:
    query: str
    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    declined: bool = False
    retrieved_chunk_count: int = 0


class RAGPipeline:
    def __init__(
        self,
        search=None,
        llm: OllamaClient | None = None,
        reranker: Reranker | None = None,
        relevance_threshold: float | None = None,
        rerank_relevance_threshold: float | None = None,
    ):
        # `search` is duck-typed: SemanticSearch (Phase 4) and HybridSearch
        # (Phase 8) both implement .search(query, top_k, collection_id), so
        # either works here. Defaults to whichever settings.enable_hybrid_search
        # says to use.
        self.search = search if search is not None else build_default_retriever()
        self.llm = llm or OllamaClient()
        # `reranker=None` (the default) means "don't rerank", not "use the
        # default reranker" - building one loads/downloads a model, so we
        # only do that if settings.enable_reranking is actually on.
        self.reranker = reranker if reranker is not None else build_default_reranker()
        self.relevance_threshold = (
            relevance_threshold if relevance_threshold is not None else DEFAULT_RELEVANCE_THRESHOLD
        )
        self.rerank_relevance_threshold = (
            rerank_relevance_threshold
            if rerank_relevance_threshold is not None
            else DEFAULT_RERANK_RELEVANCE_THRESHOLD
        )

    def _is_relevant(self, hit: dict) -> bool:
        """A hit counts as relevant if:
        - it was reranked: its cross-encoder score clears the rerank threshold: this
          supersedes distance/BM25 since it's a direct relevance judgment.
        - it has a semantic `distance`: that distance clears the (lower-is-
          better) threshold, same as pre-Phase-8 behavior.
        - it's a keyword-only hybrid hit (`distance` is None): there's no
          semantic distance to gate on, so we trust that BM25 found genuine
          lexical overlap and let it through - hybrid's RRF fusion has
          already ranked it against the alternatives.
        """
        if "rerank_score" in hit:
            return hit["rerank_score"] >= self.rerank_relevance_threshold
        distance = hit.get("distance")
        if distance is None:
            return True
        return distance <= self.relevance_threshold

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        collection_id: str = "default",
    ) -> AnswerResult:
        top_k = top_k or settings.top_k

        if self.reranker is not None:
            # Phase 8: retrieve a wider candidate pool, then let the (more
            # accurate, more expensive) cross-encoder pick the real top_k
            # out of it, rather than trusting the cheaper retriever's order.
            candidate_k = max(settings.rerank_candidate_k, top_k)
            candidates = self.search.search(query, top_k=candidate_k, collection_id=collection_id)
            hits = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            hits = self.search.search(query, top_k=top_k, collection_id=collection_id)

        # Layer 1 (+ Layer 2 if reranking): relevance gate - filter out weak
        # matches before they ever reach the LLM.
        relevant_hits = [h for h in hits if self._is_relevant(h)]

        if not relevant_hits:
            logger.info("No sufficiently relevant chunks for query %r - declining without calling LLM.", query)
            return AnswerResult(
                query=query,
                answer_text=NO_ANSWER_PHRASE,
                citations=[],
                declined=True,
                retrieved_chunk_count=len(hits),
            )

        citations = build_citations(relevant_hits)
        prompt = build_grounded_prompt(query, citations)

        # Layer 3: even with relevant chunks, the model itself may still
        # decline (e.g. chunks are topically close but don't actually
        # answer the specific question asked).
        answer_text = self.llm.generate(prompt)
        declined = is_decline_response(answer_text)

        return AnswerResult(
            query=query,
            answer_text=answer_text,
            citations=[] if declined else citations,
            declined=declined,
            retrieved_chunk_count=len(relevant_hits),
        )
