"""
Unit tests for app/generation/prompt_builder.py
"""
from app.citations.citation_mapper import Citation
from app.generation.prompt_builder import NO_ANSWER_PHRASE, build_grounded_prompt, is_decline_response


def _citation(filename="a.pdf", page=1, text="Some source text.", chunk_id="c1"):
    return Citation(filename=filename, page_number=page, chunk_id=chunk_id, chunk_text=text, relevance_distance=0.1)


def test_prompt_includes_source_text():
    prompt = build_grounded_prompt("What is X?", [_citation(text="X is defined here.")])
    assert "X is defined here." in prompt


def test_prompt_includes_the_question():
    prompt = build_grounded_prompt("What is the refund policy?", [_citation()])
    assert "What is the refund policy?" in prompt


def test_prompt_instructs_no_fabrication():
    prompt = build_grounded_prompt("anything", [_citation()])
    assert "do not invent" in prompt.lower() or "do not use outside knowledge" in prompt.lower()


def test_prompt_includes_decline_instruction():
    prompt = build_grounded_prompt("anything", [_citation()])
    assert NO_ANSWER_PHRASE in prompt


def test_prompt_with_no_citations_still_well_formed():
    prompt = build_grounded_prompt("anything", [])
    assert "no relevant sources were found" in prompt
    assert "Question: anything" in prompt


def test_prompt_labels_multiple_sources_distinctly():
    citations = [
        _citation(filename="a.pdf", page=1, text="First excerpt.", chunk_id="c1"),
        _citation(filename="b.pdf", page=3, text="Second excerpt.", chunk_id="c2"),
    ]
    prompt = build_grounded_prompt("q", citations)
    assert "[Source 1 - a.pdf, p. 1]" in prompt
    assert "[Source 2 - b.pdf, p. 3]" in prompt
    assert "First excerpt." in prompt
    assert "Second excerpt." in prompt


def test_prompt_never_asks_model_to_produce_page_numbers():
    prompt = build_grounded_prompt("q", [_citation()])
    assert "mention source numbers" in prompt.lower() or "citations are handled separately" in prompt.lower()


def test_is_decline_response_detects_exact_phrase():
    assert is_decline_response(NO_ANSWER_PHRASE) is True


def test_is_decline_response_case_insensitive_and_substring():
    assert is_decline_response(f"Well, {NO_ANSWER_PHRASE.lower()}") is True


def test_is_decline_response_false_for_normal_answer():
    assert is_decline_response("The refund window is 30 days.") is False
