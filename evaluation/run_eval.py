"""
Phase 10 evaluation harness.

Runs every question in evaluation/dataset/questions.json through the
real ingestion + retrieval (+ generation, if Ollama is reachable)
pipeline, and reports the metrics the PRD asks for (Section 15 Phase 10 /
Section 20 "Evaluation Tests"): retrieval precision/recall, citation
correctness, answer groundedness, and latency.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --retrieval-only   # skip Ollama even if it's running
    python evaluation/run_eval.py --top-k 3 --no-hybrid --no-rerank

What this does NOT require to run: Ollama. If Ollama isn't reachable,
generation-based metrics (groundedness, citation correctness, decline
accuracy, answer latency) are skipped and clearly marked as such in the
report - retrieval metrics (precision/recall, retrieval latency) still
run fully, since they only need the embedding model and vector store.
This means you can sanity-check retrieval quality without a running LLM,
and get the complete picture once Ollama is up.

Each run is fully isolated: it builds a fresh, temporary ChromaDB
collection, indexes the eval corpus into it, runs the eval, and cleans
up - it never touches your real `data/vector_store`.
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.citations.citation_mapper import build_citations
from app.config.logging_config import get_logger
from app.generation.ollama_client import OllamaClient, OllamaConnectionError
from app.generation.rag_pipeline import RAGPipeline
from app.retrieval.factory import build_default_reranker, build_default_retriever
from app.storage.indexing_service import index_pdf_file
from app.storage.vector_store import VectorStore
from evaluation.dataset.generate_dataset import DOCUMENTS, generate_dataset
from evaluation.metrics import aggregate, citation_correctness, groundedness_proxy, precision_recall

logger = get_logger(__name__)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
PDFS_DIR = os.path.join(DATASET_DIR, "pdfs")
QUESTIONS_PATH = os.path.join(DATASET_DIR, "questions.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _load_questions() -> list[dict]:
    if not os.path.isdir(PDFS_DIR) or not os.listdir(PDFS_DIR):
        print("Eval PDFs not found - generating dataset...")
        generate_dataset()
    with open(QUESTIONS_PATH) as f:
        return json.load(f)["questions"]


def _index_corpus(vector_store: VectorStore, collection_id: str, embedder=None) -> None:
    for filename in DOCUMENTS:
        path = os.path.join(PDFS_DIR, filename)
        result = index_pdf_file(
            path, filename, collection_id=collection_id, vector_store=vector_store, embedder=embedder
        )
        if not result.success:
            raise RuntimeError(f"Failed to index eval document '{filename}': {result.error}")
        logger.info("Indexed %s: %d chunks", filename, result.chunk_count)


def _run_one_question(
    question: dict,
    pipeline: RAGPipeline,
    collection_id: str,
    top_k: int,
    skip_generation: bool,
) -> dict:
    expected_sources = [tuple(s) for s in question["expected_sources"]]

    # --- Retrieval (always runs - no Ollama needed) ---
    t0 = time.perf_counter()
    hits = pipeline.search.search(question["question"], top_k=top_k, collection_id=collection_id)
    retrieval_latency_ms = (time.perf_counter() - t0) * 1000

    retrieved_sources = [(h["metadata"]["filename"], h["metadata"]["page_number"]) for h in hits]
    precision, recall = precision_recall(retrieved_sources, expected_sources)

    result = {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "precision": precision,
        "recall": recall,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_skipped": True,
        "declined": None,
        "answer_text": None,
        "citation_sources": [],
        "citation_correctness": None,
        "grounded": None,
        "answer_latency_ms": None,
        "error": None,
    }

    if skip_generation:
        return result

    # --- Full pipeline (retrieval + generation + citations) ---
    try:
        t0 = time.perf_counter()
        answer = pipeline.answer(question["question"], top_k=top_k, collection_id=collection_id)
        answer_latency_ms = (time.perf_counter() - t0) * 1000
    except OllamaConnectionError as exc:
        result["error"] = str(exc)
        return result

    citation_sources = [(c.filename, c.page_number) for c in answer.citations]

    result.update({
        "generation_skipped": False,
        "declined": answer.declined,
        "answer_text": answer.answer_text,
        "citation_sources": citation_sources,
        "citation_correctness": citation_correctness(citation_sources, expected_sources),
        "grounded": groundedness_proxy(answer.answer_text, question["expected_keywords"], answer.declined),
        "answer_latency_ms": answer_latency_ms,
    })
    return result


def run_evaluation(
    top_k: int = 5,
    use_hybrid: bool = True,
    use_rerank: bool = False,
    retrieval_only: bool = False,
    embedder=None,
    llm=None,
) -> tuple[list[dict], "object"]:
    """`embedder` and `llm` are injectable purely for testing without
    network access (see tests/test_eval_harness.py) - production runs
    (evaluation/run_eval.py's CLI) always use the real Embedder/OllamaClient."""
    questions = _load_questions()
    collection_id = f"eval_{uuid_suffix()}"

    tmp_dir = tempfile.mkdtemp(prefix="privatedocs_eval_")
    vector_store = VectorStore(persist_dir=tmp_dir)

    try:
        print(f"Indexing {len(DOCUMENTS)} eval documents into a temporary collection...")
        _index_corpus(vector_store, collection_id, embedder=embedder)

        if use_hybrid:
            from app.retrieval.hybrid_search import HybridSearch
            from app.retrieval.semantic_search import SemanticSearch
            search = HybridSearch(semantic_search=SemanticSearch(embedder=embedder, vector_store=vector_store))
            search.keyword_search.vector_store = vector_store
        else:
            from app.retrieval.semantic_search import SemanticSearch
            search = SemanticSearch(embedder=embedder, vector_store=vector_store)

        reranker = build_default_reranker() if use_rerank else None

        ollama_client = llm if llm is not None else OllamaClient()
        skip_generation = retrieval_only or not ollama_client.is_available()
        if skip_generation and not retrieval_only:
            print(
                "Ollama isn't reachable - running retrieval-only evaluation. "
                "Start `ollama serve` and re-run for full generation-based metrics."
            )

        pipeline = RAGPipeline(search=search, llm=ollama_client, reranker=reranker)

        results = []
        print(f"Running {len(questions)} questions (top_k={top_k}, hybrid={use_hybrid}, rerank={use_rerank})...")
        for i, q in enumerate(questions, 1):
            result = _run_one_question(q, pipeline, collection_id, top_k, skip_generation)
            results.append(result)
            if i % 10 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)} done")

        agg = aggregate(results)
        return results, agg
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def uuid_suffix() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def _print_report(agg, config: dict) -> None:
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Config: top_k={config['top_k']}, hybrid={config['use_hybrid']}, rerank={config['use_rerank']}")
    print(f"Questions: {agg.total_questions} total "
          f"({agg.known_answer_count} known-answer, {agg.no_answer_count} no-answer, "
          f"{agg.cross_document_count} cross-document)")
    if agg.generation_skipped_count:
        print(f"NOTE: generation skipped for {agg.generation_skipped_count} question(s) - Ollama unavailable.")
    print()
    print(f"Retrieval precision (mean, known+cross):  {agg.mean_precision:.2f}")
    print(f"Retrieval recall (mean, known+cross):     {agg.mean_recall:.2f}")
    print(f"Retrieval latency p50 / p95 (ms):          {agg.retrieval_latency_p50_ms:.0f} / {agg.retrieval_latency_p95_ms:.0f}")
    print()
    if agg.generation_skipped_count < agg.total_questions:
        print(f"Decline accuracy (no-answer questions):    {agg.decline_accuracy:.2f}")
        print(f"False-decline rate (known+cross):          {agg.false_decline_rate:.2f}")
        print(f"Citation correctness (mean):               {agg.citation_correctness_rate:.2f}")
        print(f"Groundedness proxy (mean, all categories): {agg.groundedness_rate:.2f}")
        print(f"Answer latency p50 / p95 (ms):              {agg.answer_latency_p50_ms:.0f} / {agg.answer_latency_p95_ms:.0f}")
    else:
        print("Generation-based metrics: not measured (Ollama unavailable or --retrieval-only).")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run the PrivateDocs AI evaluation suite.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-hybrid", action="store_true", help="Use semantic-only retrieval instead of hybrid.")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation even if Ollama is running.")
    args = parser.parse_args()

    config = {"top_k": args.top_k, "use_hybrid": not args.no_hybrid, "use_rerank": args.rerank}
    results, agg = run_evaluation(
        top_k=config["top_k"],
        use_hybrid=config["use_hybrid"],
        use_rerank=config["use_rerank"],
        retrieval_only=args.retrieval_only,
    )

    _print_report(agg, config)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": config,
            "summary": vars(agg),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
