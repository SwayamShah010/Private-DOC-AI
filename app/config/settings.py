"""
Centralized configuration for PrivateDocs AI.

Every other module reads settings from here instead of calling
os.getenv() directly - this keeps all tunables (Section 13 of the PRD:
"Simple settings for chunk size, overlap, top-k and model selection")
in one place, and makes it easy to expose them in the Streamlit
Settings screen later.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env if present; falls back to defaults below otherwise.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # Ollama / local LLM
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # OCR fallback (Phase 9) - kicks in per-page when normal text extraction
    # finds nothing (i.e. the page is likely a scan/image). Off switch is
    # here for machines without the Tesseract engine installed, or for
    # speed when you know your documents are all text-based PDFs.
    enable_ocr: bool = os.getenv("ENABLE_OCR", "true").lower() == "true"
    ocr_language: str = os.getenv("OCR_LANGUAGE", "eng")
    # Higher DPI = more accurate OCR but slower. 300 is the standard
    # sweet spot recommended by Tesseract's own docs.
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Retrieval - Phase 8 (advanced retrieval)
    # Hybrid search fuses semantic (embedding) results with BM25 keyword
    # results via Reciprocal Rank Fusion - catches exact terms (IDs, names,
    # acronyms) that embeddings alone sometimes blur past.
    enable_hybrid_search: bool = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
    # How many candidates each retriever (semantic + keyword) contributes to
    # the fusion pool before it's trimmed down to top_k. Higher = better
    # recall for fusion, more compute. PRD Section 15 Phase 8: "tune top-k
    # and thresholds" - this is one of the knobs to tune.
    hybrid_candidate_k: int = int(os.getenv("HYBRID_CANDIDATE_K", "20"))
    # Standard Reciprocal Rank Fusion constant (Cormack et al.) - dampens the
    # influence of rank 1 vs rank 2 so one retriever can't dominate just by
    # being first. 60 is the commonly used default; rarely needs tuning.
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Cross-encoder reranking - a second, more expensive but more accurate
    # relevance pass over the hybrid candidates before truncating to top_k.
    # Off by default: it downloads its own model from Hugging Face on first
    # use (like the embedding model), which not every environment has
    # network access for.
    enable_reranking: bool = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # How many fused candidates to feed the (slower) reranker before cutting
    # to top_k. Keep this well above top_k so reranking has room to promote
    # a good chunk that fusion ranked lower.
    rerank_candidate_k: int = int(os.getenv("RERANK_CANDIDATE_K", "20"))

    # Storage paths (resolved relative to project root)
    documents_dir: Path = PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "data/documents")
    vector_store_dir: Path = PROJECT_ROOT / os.getenv("VECTOR_STORE_DIR", "data/vector_store")

    # Privacy
    privacy_mode: str = os.getenv("PRIVACY_MODE", "local_only")

    def ensure_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
