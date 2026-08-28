# PrivateDocs AI - Docker image
#
# This packages the app itself. Ollama (the LLM server) is deliberately
# NOT included in this image - it's a large, separate piece of
# infrastructure (its own model downloads, ideally GPU access) that most
# people already run directly on their host or as its own container. See
# docker-compose.yml for the two ways to point this container at an
# Ollama server.
#
# Build:  docker build -t privatedocs-ai .
# Run:    docker run -p 8501:8501 -v $(pwd)/data:/app/data --env-file .env privatedocs-ai
# (docker-compose.yml wraps this with sensible defaults - see README's Docker section)

FROM python:3.12-slim

# tesseract-ocr: needed for Phase 9's OCR fallback on scanned PDFs.
# Everything else here are standard build tools some of chromadb's /
# sentence-transformers' dependencies need to compile from source on
# platforms without a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer) so `docker build`
# doesn't re-download everything just because application code changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persisted outside the image via a volume mount (see docker-compose.yml)
# so documents/embeddings survive a container rebuild - this is the same
# data/ directory the app uses when run outside Docker.
RUN mkdir -p /app/data/documents /app/data/vector_store

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# --server.address=0.0.0.0 is required for the app to be reachable from
# outside the container - `python run.py` (used for local, non-Docker
# runs) doesn't set this, since it binds to localhost by default which is
# correct there but wrong here.
CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
