"""
Summarization: single-document (PRD Section 7.1) and multi-document /
collection-level (PRD Section 7.2) summaries.

Design note on long documents: this MVP uses a single-pass approach -
concatenate a document's chunks in reading order and send them to the
LLM in one prompt. This is simple and works well for typical documents
(a few pages to a few dozen pages) that fit within the local model's
context window. It will NOT work well for very long documents (hundreds
of pages) that exceed context length - a proper fix is map-reduce
summarization (summarize chunks in batches, then summarize the
summaries), which is a reasonable Phase 8+ enhancement but deliberately
out of scope for the MVP per the PRD's own guidance (Section 21 risk:
"Project becomes too large" -> "Finish MVP before advanced features").
This limitation should be stated plainly in the README, not hidden.
"""
from dataclasses import dataclass

from app.config.logging_config import get_logger
from app.generation.ollama_client import OllamaClient
from app.storage.vector_store import VectorStore

logger = get_logger(__name__)

# Rough safety cap on how much raw text we pack into one summarization
# prompt. This is a character count, not a token count, so it's a
# conservative approximation - tune based on the actual model's context
# window once you're testing with real documents.
MAX_SUMMARY_INPUT_CHARS = 12000

_SINGLE_DOC_INSTRUCTIONS = """Summarize the following document clearly and concisely.
Cover the main points, key facts, and overall purpose of the document.
Do not add information that isn't in the text. Do not speculate about content that may be missing.
"""

_MULTI_DOC_INSTRUCTIONS = """You are given summaries of several documents from the same collection.
Write a single combined summary that captures the shared themes and key points across all of them.
Note any significant differences or contradictions between documents if present.
Do not add information that isn't in the provided summaries.
"""


@dataclass
class SummaryResult:
    summary_text: str
    filename: str | None = None       # set for single-document summaries
    document_ids: list[str] | None = None  # set for multi-document summaries
    truncated: bool = False           # True if input text had to be cut down to fit


class Summarizer:
    def __init__(self, vector_store: VectorStore | None = None, llm: OllamaClient | None = None):
        self.vector_store = vector_store or VectorStore()
        self.llm = llm or OllamaClient()

    def summarize_document(self, document_id: str, collection_id: str = "default") -> SummaryResult:
        chunks = self.vector_store.get_document_chunks(document_id, collection_id=collection_id)
        if not chunks:
            return SummaryResult(
                summary_text="No content found for this document. It may not have been indexed yet.",
                filename=None,
            )

        filename = chunks[0]["metadata"]["filename"]
        full_text, truncated = self._assemble_text(chunks)

        prompt = f"{_SINGLE_DOC_INSTRUCTIONS}\n--- DOCUMENT: {filename} ---\n{full_text}\n--- END DOCUMENT ---\n\nSummary:"
        summary_text = self.llm.generate(prompt)

        logger.info("Summarized document '%s' (%d chunks, truncated=%s)", filename, len(chunks), truncated)
        return SummaryResult(summary_text=summary_text, filename=filename, truncated=truncated)

    def summarize_collection(self, collection_id: str = "default") -> SummaryResult:
        """Multi-document summary: summarize each document individually,
        then combine those summaries into one collection-level summary.
        This scales better than concatenating every document's raw text,
        since summaries are much shorter than source text."""
        documents = self.vector_store.list_documents(collection_id=collection_id)
        if not documents:
            return SummaryResult(summary_text="This collection has no indexed documents yet.", document_ids=[])

        per_doc_summaries = []
        document_ids = []
        for doc in documents:
            result = self.summarize_document(doc["document_id"], collection_id=collection_id)
            per_doc_summaries.append(f"[{doc['filename']}]\n{result.summary_text}")
            document_ids.append(doc["document_id"])

        combined_input = "\n\n".join(per_doc_summaries)
        prompt = f"{_MULTI_DOC_INSTRUCTIONS}\n--- PER-DOCUMENT SUMMARIES ---\n{combined_input}\n--- END ---\n\nCombined summary:"
        summary_text = self.llm.generate(prompt)

        logger.info("Summarized collection '%s' (%d documents)", collection_id, len(documents))
        return SummaryResult(summary_text=summary_text, document_ids=document_ids)

    @staticmethod
    def _assemble_text(chunks: list[dict]) -> tuple[str, bool]:
        """Join chunks in reading order, respecting the character cap."""
        pieces = []
        total_len = 0
        truncated = False
        for c in chunks:
            piece = c["chunk_text"]
            if total_len + len(piece) > MAX_SUMMARY_INPUT_CHARS:
                truncated = True
                break
            pieces.append(piece)
            total_len += len(piece)
        return "\n".join(pieces), truncated
