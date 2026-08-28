"""
Unit tests for evaluation/metrics.py - all pure functions, no vector
store, embeddings, or Ollama needed.
"""
from evaluation.metrics import (
    AggregateMetrics,
    aggregate,
    citation_correctness,
    groundedness_proxy,
    precision_recall,
)


# --- precision_recall ---

def test_precision_recall_perfect_match():
    p, r = precision_recall([("a.pdf", 1)], [("a.pdf", 1)])
    assert p == 1.0 and r == 1.0


def test_precision_recall_partial_overlap():
    retrieved = [("a.pdf", 1), ("b.pdf", 2)]
    expected = [("a.pdf", 1), ("c.pdf", 3)]
    p, r = precision_recall(retrieved, expected)
    assert p == 0.5  # 1 of 2 retrieved was relevant
    assert r == 0.5  # 1 of 2 relevant was retrieved


def test_precision_recall_no_overlap():
    p, r = precision_recall([("a.pdf", 1)], [("b.pdf", 2)])
    assert p == 0.0 and r == 0.0


def test_precision_recall_nothing_retrieved_and_nothing_expected():
    # Correctly matches a no-answer question with an empty result set.
    p, r = precision_recall([], [])
    assert p == 1.0 and r == 1.0


def test_precision_recall_nothing_retrieved_but_something_expected():
    p, r = precision_recall([], [("a.pdf", 1)])
    assert p == 0.0 and r == 0.0


def test_precision_recall_something_retrieved_but_nothing_expected():
    p, r = precision_recall([("a.pdf", 1)], [])
    assert p == 0.0 and r == 0.0


def test_precision_recall_recall_at_k_with_extra_irrelevant_hits():
    retrieved = [("a.pdf", 1), ("x.pdf", 9), ("y.pdf", 9)]
    expected = [("a.pdf", 1)]
    p, r = precision_recall(retrieved, expected)
    assert p == 1 / 3
    assert r == 1.0


# --- citation_correctness ---

def test_citation_correctness_all_correct():
    assert citation_correctness([("a.pdf", 1)], [("a.pdf", 1)]) == 1.0


def test_citation_correctness_all_wrong():
    assert citation_correctness([("a.pdf", 1)], [("b.pdf", 2)]) == 0.0


def test_citation_correctness_partial():
    citations = [("a.pdf", 1), ("b.pdf", 2)]
    expected = [("a.pdf", 1)]
    assert citation_correctness(citations, expected) == 0.5


def test_citation_correctness_no_citations_returns_1():
    # Nothing to be wrong about - see docstring for why this is 1.0, not 0.0.
    assert citation_correctness([], [("a.pdf", 1)]) == 1.0


# --- groundedness_proxy ---

def test_groundedness_true_when_all_keywords_present():
    assert groundedness_proxy("The vacation policy allows 15 days per year.", ["15"], declined=False) is True


def test_groundedness_false_when_a_keyword_is_missing():
    assert groundedness_proxy("The vacation policy allows some days per year.", ["15"], declined=False) is False


def test_groundedness_is_case_insensitive_and_whitespace_normalized():
    # Simulates a keyword phrase that got line-wrapped in the source text
    # extraction, and an answer using different casing.
    assert groundedness_proxy("This is covered for WATER   DAMAGE too.", ["water damage"], declined=False) is True


def test_groundedness_no_answer_question_correctly_declined_is_grounded():
    assert groundedness_proxy("I don't have enough information to answer that.", [], declined=True) is True


def test_groundedness_no_answer_question_that_answered_anyway_is_not_grounded():
    # No expected_keywords means this was a no-answer question - if the
    # pipeline didn't decline, it fabricated something.
    assert groundedness_proxy("Sure, the answer is 42.", [], declined=False) is False


def test_groundedness_known_answer_question_that_declined_is_not_grounded():
    assert groundedness_proxy("I don't know.", ["15"], declined=True) is False


# --- aggregate ---

def _result(
    category, precision=1.0, recall=1.0, declined=False, generation_skipped=False,
    citation_correctness=1.0, grounded=True, retrieval_latency_ms=10.0, answer_latency_ms=100.0,
):
    return {
        "category": category,
        "precision": precision,
        "recall": recall,
        "declined": declined,
        "generation_skipped": generation_skipped,
        "citation_correctness": citation_correctness,
        "grounded": grounded,
        "retrieval_latency_ms": retrieval_latency_ms,
        "answer_latency_ms": answer_latency_ms,
    }


def test_aggregate_counts_categories_correctly():
    results = [
        _result("known_answer"), _result("known_answer"),
        _result("no_answer", precision=None, recall=None, declined=True),
        _result("cross_document"),
    ]
    agg = aggregate(results)
    assert agg.total_questions == 4
    assert agg.known_answer_count == 2
    assert agg.no_answer_count == 1
    assert agg.cross_document_count == 1


def test_aggregate_mean_precision_recall_excludes_no_answer_questions():
    results = [
        _result("known_answer", precision=1.0, recall=1.0),
        _result("known_answer", precision=0.0, recall=0.0),
        _result("no_answer", precision=None, recall=None, declined=True),
    ]
    agg = aggregate(results)
    assert agg.mean_precision == 0.5
    assert agg.mean_recall == 0.5


def test_aggregate_decline_accuracy_only_from_no_answer_questions():
    results = [
        _result("no_answer", precision=None, recall=None, declined=True),   # correct
        _result("no_answer", precision=None, recall=None, declined=False),  # incorrect - fabricated
    ]
    agg = aggregate(results)
    assert agg.decline_accuracy == 0.5


def test_aggregate_false_decline_rate_from_known_and_cross_document_only():
    results = [
        _result("known_answer", declined=False),  # correct
        _result("known_answer", declined=True),   # false decline
        _result("cross_document", declined=True), # false decline
    ]
    agg = aggregate(results)
    assert agg.false_decline_rate == 2 / 3


def test_aggregate_generation_skipped_questions_excluded_from_generation_metrics():
    results = [
        _result("known_answer", generation_skipped=True),
        _result("known_answer", generation_skipped=False, grounded=True),
    ]
    agg = aggregate(results)
    assert agg.generation_skipped_count == 1
    assert agg.groundedness_rate == 1.0  # only the non-skipped one counted


def test_aggregate_latency_percentiles():
    results = [_result("known_answer", retrieval_latency_ms=v, answer_latency_ms=v * 10) for v in [10, 20, 30, 40, 50]]
    agg = aggregate(results)
    assert agg.retrieval_latency_p50_ms == 30
    assert agg.retrieval_latency_p95_ms in (40, 50)  # depends on percentile interpolation choice


def test_aggregate_empty_results_does_not_crash():
    agg = aggregate([])
    assert agg.total_questions == 0
    assert agg.mean_precision == 0.0
    assert isinstance(agg, AggregateMetrics)
