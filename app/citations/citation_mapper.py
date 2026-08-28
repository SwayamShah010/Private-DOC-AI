"""
Citations: mapping retrieved chunks to displayable, verifiable sources.

Critical design point (PRD Section 11): "Citation generation should use
retrieval metadata rather than asking the LLM to invent page numbers."
This module NEVER asks the LLM to produce a citation - citations are
built entirely from the metadata already attached to retrieved chunks
in Phase 4. The LLM only ever sees labeled context blocks; it cannot
fabricate a page number because it never generates one - we build the
citation list ourselves after retrieval, before the LLM is even called.
"""
from dataclasses import dataclass


@dataclass
class Citation:
    filename: str
    page_number: int
    chunk_id: str
    chunk_text: str
    # Lower = more similar (raw L2 score from the vector store). None for
    # hits that a Phase 8 hybrid search found only via BM25 keyword match,
    # which has no semantic distance to report.
    relevance_distance: float | None


def build_citations(hits: list[dict]) -> list[Citation]:
    """Convert raw vector-store hits into deduplicated Citation objects,
    ordered by relevance. Two chunks from the same document+page collapse
    into one citation (a user doesn't need "page 4" listed three times)."""
    seen: dict[tuple[str, int], Citation] = {}

    for hit in hits:
        meta = hit["metadata"]
        key = (meta["filename"], meta["page_number"])
        if key not in seen:
            seen[key] = Citation(
                filename=meta["filename"],
                page_number=meta["page_number"],
                chunk_id=hit["chunk_id"],
                chunk_text=hit["chunk_text"],
                relevance_distance=hit["distance"],
            )
        # if already seen, keep the first (most relevant) occurrence - hits
        # arrive pre-sorted by distance from VectorStore.query()

    return list(seen.values())


def format_citation_label(citation: Citation) -> str:
    """Human-readable label, e.g. 'report.pdf, p. 4' - used in the UI and
    in the prompt's source labels."""
    return f"{citation.filename}, p. {citation.page_number}"
