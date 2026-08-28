"""
Thin wrapper around the Ollama Python client.

Ollama runs as a local server (default http://localhost:11434) and
serves models entirely on your own machine - this is what makes
"local LLM inference" (PRD Section 9) possible with zero API keys and
zero document content ever leaving the device.

This module deliberately does NOT catch-and-hide connection errors:
if Ollama isn't running, callers need to know clearly, because silently
returning an empty answer would look like the RAG pipeline failed
rather than "you forgot to start Ollama."
"""
import ollama

from app.config.logging_config import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama isn't reachable - almost always means the local
    Ollama server isn't running, or the model hasn't been pulled yet."""


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model
        self.client = ollama.Client(host=self.host)

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """Send a prompt to the local model and return its text response.

        temperature defaults low (0.1) - grounded QA wants faithful,
        low-variance answers, not creative ones.
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": temperature},
            )
            return response["response"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to reach Ollama at %s (model=%s): %s", self.host, self.model, exc)
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host} with model '{self.model}'. "
                f"Make sure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {self.model}`)."
            ) from exc

    def is_available(self) -> bool:
        """Quick health check - used by the UI's privacy/model status indicator
        (PRD Section 13)."""
        try:
            self.client.list()
            return True
        except Exception:  # noqa: BLE001
            return False
