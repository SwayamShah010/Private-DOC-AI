"""
Unit tests for app/generation/summarizer.py

Uses a fake VectorStore (no real ChromaDB/embeddings needed) and a
mocked OllamaClient, so summarization orchestration logic - text
assembly, ordering, truncation, multi-document combination - is fully
testable without any local services running.
"""
from unittest.mock import MagicMock

import pytest

from app.generation.summarizer import MAX_SUMMARY_INPUT_CHARS, Summarizer


class FakeVectorStore:
    def __init__(self, chunks_by_doc: dict[str, list[dict]], documents: list[dict]):
        self._chunks_by_doc = chunks_by_doc
        self._documents = documents

    def get_document_chunks(self, document_id, collection_id="default"):
        return self._chunks_by_doc.get(document_id, [])

    def list_documents(self, collection_id="default"):
        return self._documents


def _chunk(text, filename="a.pdf", page=1):
    return {"chunk_text": text, "metadata": {"filename": filename, "page_number": page}}


def test_summarize_document_calls_llm_with_assembled_text():
    store = FakeVectorStore(
        chunks_by_doc={"d1": [_chunk("First part."), _chunk("Second part.")]},
        documents=[],
    )
    llm = MagicMock()
    llm.generate.return_value = "A short summary."
    summarizer = Summarizer(vector_store=store, llm=llm)

    result = summarizer.summarize_document("d1")

    assert result.summary_text == "A short summary."
    assert result.filename == "a.pdf"
    assert result.truncated is False
    prompt_sent = llm.generate.call_args[0][0]
    assert "First part." in prompt_sent
    assert "Second part." in prompt_sent


def test_summarize_document_with_no_chunks_does_not_call_llm():
    store = FakeVectorStore(chunks_by_doc={}, documents=[])
    llm = MagicMock()
    summarizer = Summarizer(vector_store=store, llm=llm)

    result = summarizer.summarize_document("missing_doc")

    assert "no content found" in result.summary_text.lower()
    llm.generate.assert_not_called()


def test_summarize_document_truncates_when_over_char_limit():
    huge_chunk_text = "x" * (MAX_SUMMARY_INPUT_CHARS + 5000)
    store = FakeVectorStore(
        chunks_by_doc={"d1": [_chunk(huge_chunk_text), _chunk("this should be cut off")]},
        documents=[],
    )
    llm = MagicMock()
    llm.generate.return_value = "summary"
    summarizer = Summarizer(vector_store=store, llm=llm)

    result = summarizer.summarize_document("d1")

    assert result.truncated is True
    prompt_sent = llm.generate.call_args[0][0]
    assert "this should be cut off" not in prompt_sent


def test_summarize_collection_summarizes_each_document_then_combines():
    store = FakeVectorStore(
        chunks_by_doc={
            "d1": [_chunk("Doc one content.", filename="a.pdf")],
            "d2": [_chunk("Doc two content.", filename="b.pdf")],
        },
        documents=[
            {"document_id": "d1", "filename": "a.pdf", "page_count": 1},
            {"document_id": "d2", "filename": "b.pdf", "page_count": 1},
        ],
    )
    llm = MagicMock()
    # First two calls summarize each doc, third call combines them
    llm.generate.side_effect = ["Summary of A.", "Summary of B.", "Combined summary of A and B."]
    summarizer = Summarizer(vector_store=store, llm=llm)

    result = summarizer.summarize_collection()

    assert result.summary_text == "Combined summary of A and B."
    assert set(result.document_ids) == {"d1", "d2"}
    assert llm.generate.call_count == 3

    combine_prompt = llm.generate.call_args_list[2][0][0]
    assert "Summary of A." in combine_prompt
    assert "Summary of B." in combine_prompt


def test_summarize_collection_with_no_documents_does_not_call_llm():
    store = FakeVectorStore(chunks_by_doc={}, documents=[])
    llm = MagicMock()
    summarizer = Summarizer(vector_store=store, llm=llm)

    result = summarizer.summarize_collection()

    assert "no indexed documents" in result.summary_text.lower()
    llm.generate.assert_not_called()
