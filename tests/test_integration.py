"""
End-to-end integration tests, matching PRD Section 20's four bullets
exactly:
  1. Upload -> extraction -> indexing
  2. Question -> retrieval -> generation -> citations
  3. Summary generation
  4. Multiple-document collections

Unlike the per-module unit tests elsewhere (each of which fakes out its
neighbors to test one module in isolation), these wire real components
together across module boundaries: real PDF -> real extract_pdf -> real
chunker -> real VectorStore (ChromaDB) -> real HybridSearch -> real
RAGPipeline -> real citation_mapper. The only fakes are for the two
things that need external network access in a normal run - the
embedding model (Hugging Face download) and the LLM (a running Ollama
server) - exactly the same pattern used to validate the pipeline
end-to-end during Phase 8/9/10 development.
"""
import pymupdf
import pytest

from app.generation.rag_pipeline import RAGPipeline
from app.generation.summarizer import Summarizer
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.semantic_search import SemanticSearch
from app.storage.indexing_service import index_pdf_file
from app.storage.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words embedder - no Hugging Face download
    needed, but still meaningfully discriminative between topics (unlike
    a purely random vector), so retrieval results are realistic enough
    to assert on."""
    model_name = "fake-integration-test"
    VOCAB = [
        "vacation", "days", "15", "sick", "leave", "10", "remote", "work",
        "warranty", "24", "months", "refund", "invoice", "revenue",
        "million", "employees", "password", "security", "capital", "france",
    ]

    def _vec(self, text):
        tokens = text.lower().split()
        v = [float(sum(1 for t in tokens if w in t)) for w in self.VOCAB]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_texts(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, q):
        return self._vec(q)


class FakeLLM:
    """Stands in for OllamaClient - returns a canned, clearly-grounded
    answer instead of actually calling a model."""
    def __init__(self, response="Based on the provided context, here is the answer."):
        self.response = response
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return self.response


def _make_pdf(tmp_path, filename, pages_text):
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    path = tmp_path / filename
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "vector_store"))


# --- 1. Upload -> extraction -> indexing ---

def test_upload_extraction_indexing_end_to_end(tmp_path, store):
    pdf_path = _make_pdf(tmp_path, "hr.pdf", [
        "Vacation Policy\nFull-time employees accrue 15 days of paid vacation per year.",
        "Sick Leave\nEmployees receive 10 paid sick days per calendar year.",
    ])

    result = index_pdf_file(pdf_path, "hr.pdf", embedder=FakeEmbedder(), vector_store=store)

    assert result.success is True
    assert result.chunk_count > 0

    chunks = store.get_document_chunks(result.document_id)
    assert len(chunks) == result.chunk_count
    assert any("15 days" in c["chunk_text"] for c in chunks)
    assert all(c["metadata"]["filename"] == "hr.pdf" for c in chunks)
    # Page numbers preserved for citation integrity.
    assert {c["metadata"]["page_number"] for c in chunks} == {1, 2}


# --- 2. Question -> retrieval -> generation -> citations ---

def test_question_retrieval_generation_citations_end_to_end(tmp_path, store):
    pdf_path = _make_pdf(tmp_path, "hr.pdf", [
        "Vacation Policy\nFull-time employees accrue 15 days of paid vacation per year.",
        "Sick Leave\nEmployees receive 10 paid sick days per calendar year.",
    ])
    index_pdf_file(pdf_path, "hr.pdf", embedder=FakeEmbedder(), vector_store=store)

    search = HybridSearch(
        semantic_search=SemanticSearch(embedder=FakeEmbedder(), vector_store=store),
        keyword_search=KeywordSearch(vector_store=store),
    )
    llm = FakeLLM("Employees accrue 15 days of paid vacation per year.")
    pipeline = RAGPipeline(search=search, llm=llm, relevance_threshold=1.5)

    answer = pipeline.answer("How many vacation days do employees get?")

    assert answer.declined is False
    assert len(answer.citations) >= 1
    assert answer.citations[0].filename == "hr.pdf"
    assert answer.citations[0].page_number == 1
    assert len(llm.calls) == 1
    # The grounded context actually made it into the prompt sent to the LLM.
    assert "15 days" in llm.calls[0]


def test_unanswerable_question_declines_without_calling_llm(tmp_path, store):
    pdf_path = _make_pdf(tmp_path, "hr.pdf", [
        "Vacation Policy\nFull-time employees accrue 15 days of paid vacation per year.",
    ])
    index_pdf_file(pdf_path, "hr.pdf", embedder=FakeEmbedder(), vector_store=store)

    search = SemanticSearch(embedder=FakeEmbedder(), vector_store=store)
    llm = FakeLLM()
    pipeline = RAGPipeline(search=search, llm=llm, relevance_threshold=1.5)

    answer = pipeline.answer("What is the capital of France?")

    assert answer.declined is True
    assert answer.citations == []
    assert len(llm.calls) == 0  # relevance gate should skip generation entirely


# --- 3. Summary generation ---

def test_summary_generation_end_to_end(tmp_path, store):
    pdf_path = _make_pdf(tmp_path, "warranty.pdf", [
        "Warranty Overview\nThe Acme Widget Pro is covered for 24 months from purchase.",
        "Exclusions\nThis warranty does not cover water damage or misuse.",
    ])
    result = index_pdf_file(pdf_path, "warranty.pdf", embedder=FakeEmbedder(), vector_store=store)

    llm = FakeLLM("The Acme Widget Pro has a 24-month warranty, excluding water damage and misuse.")
    summarizer = Summarizer(llm=llm, vector_store=store)

    summary = summarizer.summarize_document(result.document_id)

    assert summary.summary_text.strip() != ""
    assert summary.filename == "warranty.pdf"
    assert len(llm.calls) >= 1


# --- 4. Multiple-document collections ---

def test_multiple_documents_in_one_collection_are_all_retrievable(tmp_path, store):
    hr_path = _make_pdf(tmp_path, "hr.pdf", ["Vacation Policy\nEmployees accrue 15 days of paid vacation per year."])
    warranty_path = _make_pdf(tmp_path, "warranty.pdf", ["Warranty\nThe Widget Pro is covered for 24 months."])

    r1 = index_pdf_file(hr_path, "hr.pdf", embedder=FakeEmbedder(), vector_store=store)
    r2 = index_pdf_file(warranty_path, "warranty.pdf", embedder=FakeEmbedder(), vector_store=store)

    assert r1.success and r2.success
    assert store.collection_chunk_count("default") == r1.chunk_count + r2.chunk_count

    search = SemanticSearch(embedder=FakeEmbedder(), vector_store=store)
    vacation_hits = search.search("How many vacation days?", top_k=1)
    warranty_hits = search.search("How long is the warranty?", top_k=1)

    assert vacation_hits[0]["metadata"]["filename"] == "hr.pdf"
    assert warranty_hits[0]["metadata"]["filename"] == "warranty.pdf"


def test_documents_in_different_collections_do_not_leak_into_each_other(tmp_path, store):
    hr_path = _make_pdf(tmp_path, "hr.pdf", ["Vacation Policy\nEmployees accrue 15 days of paid vacation per year."])
    warranty_path = _make_pdf(tmp_path, "warranty.pdf", ["Warranty\nThe Widget Pro is covered for 24 months."])

    index_pdf_file(hr_path, "hr.pdf", collection_id="project_a", embedder=FakeEmbedder(), vector_store=store)
    index_pdf_file(warranty_path, "warranty.pdf", collection_id="project_b", embedder=FakeEmbedder(), vector_store=store)

    search = SemanticSearch(embedder=FakeEmbedder(), vector_store=store)
    hits_in_a = search.search("warranty", top_k=5, collection_id="project_a")
    hits_in_b = search.search("warranty", top_k=5, collection_id="project_b")

    assert all(h["metadata"]["filename"] == "hr.pdf" for h in hits_in_a)
    assert all(h["metadata"]["filename"] == "warranty.pdf" for h in hits_in_b)
