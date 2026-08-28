"""
Metric computation for the Phase 10 evaluation harness. Kept separate
from run_eval.py's orchestration (indexing, retrieval, generation) so
these functions are pure, fast, and unit-testable without needing a
vector store, embeddings, or Ollama - see tests/test_eval_metrics.py.

Metrics implemented, mapped to PRD Section 15 Phase 10 / Section 20:
- retrieval precision/recall  -> precision_recall()
- citation correctness        -> citation_correctness()
- answer groundedness         -> groundedness_proxy() (see its docstring
                                  for an honest statement of what this
                                  proxy does and doesn't verify)
- latency                     -> computed inline in run_eval.py (just
                                  wall-clock timing, nothing to unit test)
"""
import re
from dataclasses import dataclass, field


def _normalize(text: str) -> str:
    """Lowercase + collapse all whitespace runs to a single space. PDF
    text extraction can wrap a phrase across a line break (e.g. "water\\n
    damage"), which would silently break a naive substring check -
    normalizing whitespace first avoids false negatives from that."""
    return re.sub(r"\s+", " ", text.lower()).strip()


Source = tuple[str, int]  # (filename, page_number)


def precision_recall(retrieved: list[Source], expected: list[Source]) -> tuple[float, float]:
    """Standard set-based precision/recall over (filename, page) pairs.

    - precision: of what was retrieved, how much was actually relevant?
    - recall: of what's actually relevant, how much did we retrieve?

    For no-answer questions (expected == []), precision/recall aren't
    meaningful in the usual sense - callers should use
    `retrieval_correctly_empty` semantics instead (see run_eval.py) and
    skip these two metrics for that category.
    """
    retrieved_set = set(retrieved)
    expected_set = set(expected)

    if not retrieved_set and not expected_set:
        return 1.0, 1.0
    if not retrieved_set:
        return 0.0, 0.0
    if not expected_set:
        # Nothing was supposed to be relevant - precision is undefined by
        # the usual formula (0/len(retrieved) = 0.0 is the conventional
        # choice, treating any hit as "not relevant" by definition).
        return 0.0, 0.0

    overlap = retrieved_set & expected_set
    precision = len(overlap) / len(retrieved_set)
    recall = len(overlap) / len(expected_set)
    return precision, recall


def citation_correctness(citation_sources: list[Source], expected: list[Source]) -> float:
    """Fraction of the answer's citations that point to an actually-correct
    (filename, page) - PRD Section 11: "citations point to the correct
    document/page whenever metadata permits." 1.0 if there are no
    citations to check (nothing to be wrong about); this is intentionally
    generous - a *missing* citation is a recall problem, tracked
    separately, not a correctness problem."""
    if not citation_sources:
        return 1.0
    correct = sum(1 for c in citation_sources if c in expected)
    return correct / len(citation_sources)


def groundedness_proxy(answer_text: str, expected_keywords: list[str], declined: bool) -> bool:
    """A lightweight, deterministic stand-in for "is this answer actually
    grounded in the source content," WITHOUT another LLM call to judge it.

    Honest limitation (documented here rather than glossed over, per the
    PRD's own "Honest Project Assessment" ethos, Section 25): this checks
    whether the expected factual keywords appear in the answer text. That
    catches the common failure modes - hallucinated numbers, wrong facts,
    an answer that dodges the question - but it is NOT a substitute for a
    real fact-check or an LLM-as-judge groundedness score. A model could
    in principle include the right keywords while still being unfaithful
    to the source in some other way this proxy wouldn't catch. Treat this
    metric as "did the answer contain the facts it needed to," not "is
    this answer verified correct."

    For no-answer questions (expected_keywords == []), groundedness means
    the pipeline correctly declined rather than fabricating something.
    """
    if not expected_keywords:
        return declined

    if declined:
        return False  # should have answered but didn't

    normalized_answer = _normalize(answer_text)
    return all(_normalize(kw) in normalized_answer for kw in expected_keywords)


@dataclass
class AggregateMetrics:
    """Summary stats across every question in one evaluation run."""
    total_questions: int = 0
    known_answer_count: int = 0
    no_answer_count: int = 0
    cross_document_count: int = 0

    mean_precision: float = 0.0   # known_answer + cross_document only
    mean_recall: float = 0.0      # known_answer + cross_document only

    decline_accuracy: float = 0.0     # no_answer questions that were correctly declined
    false_decline_rate: float = 0.0   # known_answer/cross_document questions wrongly declined

    citation_correctness_rate: float = 0.0   # mean across answered questions with citations
    groundedness_rate: float = 0.0           # mean across all questions (see groundedness_proxy)

    retrieval_latency_p50_ms: float = 0.0
    retrieval_latency_p95_ms: float = 0.0
    answer_latency_p50_ms: float = 0.0
    answer_latency_p95_ms: float = 0.0

    generation_skipped_count: int = 0  # questions where Ollama wasn't reachable


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(round(pct / 100 * (len(s) - 1))), len(s) - 1)
    return s[idx]


def aggregate(results: list[dict]) -> AggregateMetrics:
    """`results` is a list of per-question result dicts as built by
    run_eval.py - see that module for the exact shape."""
    agg = AggregateMetrics(total_questions=len(results))

    retrieval_pr = []  # (precision, recall) for known_answer/cross_document
    decline_checks = []  # (category, declined) for no_answer questions
    false_declines = []  # bool, for known_answer/cross_document questions
    citation_rates = []
    grounded_flags = []
    retrieval_latencies = []
    answer_latencies = []

    for r in results:
        category = r["category"]
        if category == "known_answer":
            agg.known_answer_count += 1
        elif category == "no_answer":
            agg.no_answer_count += 1
        elif category == "cross_document":
            agg.cross_document_count += 1

        if category in ("known_answer", "cross_document"):
            retrieval_pr.append((r["precision"], r["recall"]))
            if not r["generation_skipped"]:
                false_declines.append(r["declined"])
                if not r["declined"]:
                    citation_rates.append(r["citation_correctness"])
        elif category == "no_answer":
            if not r["generation_skipped"]:
                decline_checks.append(r["declined"])

        if not r["generation_skipped"]:
            grounded_flags.append(r["grounded"])
            answer_latencies.append(r["answer_latency_ms"])
        else:
            agg.generation_skipped_count += 1

        retrieval_latencies.append(r["retrieval_latency_ms"])

    if retrieval_pr:
        agg.mean_precision = sum(p for p, _ in retrieval_pr) / len(retrieval_pr)
        agg.mean_recall = sum(rec for _, rec in retrieval_pr) / len(retrieval_pr)
    if decline_checks:
        agg.decline_accuracy = sum(decline_checks) / len(decline_checks)
    if false_declines:
        agg.false_decline_rate = sum(false_declines) / len(false_declines)
    if citation_rates:
        agg.citation_correctness_rate = sum(citation_rates) / len(citation_rates)
    if grounded_flags:
        agg.groundedness_rate = sum(grounded_flags) / len(grounded_flags)

    agg.retrieval_latency_p50_ms = _percentile(retrieval_latencies, 50)
    agg.retrieval_latency_p95_ms = _percentile(retrieval_latencies, 95)
    agg.answer_latency_p50_ms = _percentile(answer_latencies, 50)
    agg.answer_latency_p95_ms = _percentile(answer_latencies, 95)

    return agg
