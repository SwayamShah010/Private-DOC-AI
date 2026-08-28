"""
Integration test for evaluation/run_eval.py's orchestration - proves the
harness itself (indexing the corpus, running retrieval, aggregating
metrics) works end to end, independent of whether a real embedding
model or Ollama is available.

Uses a small deterministic bag-of-words "embedder" instead of the real
Embedder (which needs a Hugging Face download) - good enough to validate
retrieval finds the right pages for at least the exact-keyword questions,
which is all this test needs to prove the harness's plumbing works.
"""
import re

from evaluation.dataset.generate_dataset import DOCUMENTS
from evaluation.run_eval import _load_questions, run_evaluation


class _FakeBagOfWordsEmbedder:
    """Deterministic, network-free stand-in for the real Embedder. Builds
    its vocabulary from the eval corpus + questions so it's meaningfully
    discriminative for this specific test, unlike a purely random fake."""
    model_name = "fake-bow-test"

    def __init__(self):
        words = set()
        for spec in DOCUMENTS.values():
            for title, body in spec["pages"]:
                words.update(re.findall(r"[a-z0-9]+", f"{title} {body}".lower()))
        for q in _load_questions():
            words.update(re.findall(r"[a-z0-9]+", q["question"].lower()))
        self.vocab = sorted(words)

    def _vec(self, text):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        v = [float(tokens.count(w)) for w in self.vocab]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_texts(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, q):
        return self._vec(q)


def test_harness_runs_end_to_end_retrieval_only():
    results, agg = run_evaluation(
        top_k=5, use_hybrid=True, use_rerank=False,
        retrieval_only=True, embedder=_FakeBagOfWordsEmbedder(),
    )

    assert agg.total_questions == len(results) == 60
    assert agg.known_answer_count == 35
    assert agg.no_answer_count == 15
    assert agg.cross_document_count == 10
    # Generation was skipped for every question (retrieval_only=True).
    assert agg.generation_skipped_count == 60


def test_harness_retrieval_finds_the_right_page_for_easy_questions():
    # A reasonably faithful embedder should retrieve the exact page for
    # at least some direct, single-fact questions - proves the full
    # index -> embed -> search chain is wired correctly, not just that
    # it runs without crashing.
    results, _ = run_evaluation(
        top_k=5, use_hybrid=True, use_rerank=False,
        retrieval_only=True, embedder=_FakeBagOfWordsEmbedder(),
    )
    known_answer_results = [r for r in results if r["category"] == "known_answer"]
    hit_rate = sum(1 for r in known_answer_results if r["recall"] == 1.0) / len(known_answer_results)
    assert hit_rate > 0.5


def test_harness_semantic_only_mode_also_runs():
    results, agg = run_evaluation(
        top_k=5, use_hybrid=False, use_rerank=False,
        retrieval_only=True, embedder=_FakeBagOfWordsEmbedder(),
    )
    assert agg.total_questions == len(results) == 60


def test_harness_results_have_expected_shape():
    results, _ = run_evaluation(
        top_k=3, use_hybrid=True, use_rerank=False,
        retrieval_only=True, embedder=_FakeBagOfWordsEmbedder(),
    )
    r = results[0]
    for key in ("id", "category", "question", "expected_sources", "retrieved_sources",
                "precision", "recall", "retrieval_latency_ms", "generation_skipped"):
        assert key in r
    assert r["generation_skipped"] is True
    assert r["retrieval_latency_ms"] >= 0
