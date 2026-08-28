"""
Local embedding generation using sentence-transformers.

This is the piece that keeps the "local-first" promise (PRD Section 12):
the embedding model downloads once from Hugging Face and then runs
entirely on your own CPU/GPU - no document text is ever sent to an
external API to be embedded.

Default model: all-MiniLM-L6-v2 - a small (~80MB), fast, well-regarded
general-purpose embedding model. Good balance of speed and quality for
a first version; swappable later via EMBEDDING_MODEL in .env.
"""
from sentence_transformers import SentenceTransformer

from app.config.logging_config import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Cache loaded models in memory - loading from disk/downloading is the
    expensive part, so we don't want to repeat it for every batch."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model '%s' (first load may download it)...", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.model = _get_model(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts. Used during indexing."""
        if not texts:
            return []
        vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query. Used at search time."""
        vector = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        return vector[0].tolist()
