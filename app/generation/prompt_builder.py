"""
Builds the prompt sent to the local LLM.

Implements PRD Section 11's grounding requirements directly:
- "Every factual answer should be grounded in retrieved document content."
- "If sufficient evidence is not retrieved, the system should say it
   cannot find enough information instead of inventing an answer."
- "Prompts should explicitly instruct the LLM not to fabricate sources."

Note what this prompt does NOT ask the model to do: produce page
numbers or source names. It only asks for the answer text. Citations
are attached afterward from retrieval metadata (see app/citations/),
which removes the LLM's ability to hallucinate a citation entirely,
rather than just asking it nicely not to.
"""
from app.citations.citation_mapper import Citation, format_citation_label

NO_ANSWER_PHRASE = "I don't have enough information in the provided documents to answer that."

_SYSTEM_INSTRUCTIONS = f"""You are a document assistant. Answer the user's question using ONLY the numbered source excerpts below.

Rules:
- Base your answer strictly on the provided sources. Do not use outside knowledge.
- Do not invent, guess, or assume any information not present in the sources.
- If the sources do not contain enough information to answer, respond with exactly: "{NO_ANSWER_PHRASE}"
- Do not mention source numbers, filenames, or page numbers in your answer - citations are handled separately.
- Be concise and direct.
"""


def build_grounded_prompt(query: str, citations: list[Citation]) -> str:
    """Assemble the full prompt: system instructions + labeled source
    excerpts + the user's question."""
    if not citations:
        # No retrieved context at all - PRD requires declining rather than
        # letting the LLM improvise. We still route this through the LLM
        # with zero sources so it produces the standard decline phrase
        # consistently, rather than hand-writing every "no answer" case.
        sources_block = "(no relevant sources were found)"
    else:
        blocks = []
        for i, c in enumerate(citations, start=1):
            label = format_citation_label(c)
            blocks.append(f"[Source {i} - {label}]\n{c.chunk_text}")
        sources_block = "\n\n".join(blocks)

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n"
        f"--- SOURCES ---\n{sources_block}\n--- END SOURCES ---\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def is_decline_response(answer_text: str) -> bool:
    """Detect whether the model declined to answer (used to decide whether
    to still show citations - if it declined, we show none, since there's
    nothing to attribute)."""
    return NO_ANSWER_PHRASE.lower() in answer_text.lower()
