"""
Local vector storage using ChromaDB (PRD Section 9: recommended for v1).

ChromaDB's PersistentClient writes straight to disk at
`data/vector_store/` - no server process required, and nothing here
ever makes a network call, which is what keeps this layer compliant
with the local-only privacy mode (PRD Section 12).

We deliberately accept pre-computed embeddings rather than raw text,
so this module has zero dependency on which embedding model is used
upstream - Phase 3's Embedder and this module are fully decoupled.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.ingestion.chunk_models import Chunk

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = str(persist_dir or settings.vector_store_dir)
        # anonymized_telemetry=False - no usage data leaves the machine, ever,
        # consistent with the privacy requirements in PRD Section 12.
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self, collection_id: str = "default"):
        return self.client.get_or_create_collection(name=collection_id)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Store chunks + their embeddings. Chunks may belong to different
        collection_ids in one call; each is routed to its own ChromaDB
        collection so collections stay fully isolated from each other."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) counts must match"
            )

        by_collection: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            by_collection.setdefault(c.collection_id, []).append(i)

        total_added = 0
        for collection_id, indices in by_collection.items():
            collection = self.get_or_create_collection(collection_id)
            collection.add(
                ids=[chunks[i].chunk_id for i in indices],
                embeddings=[embeddings[i] for i in indices],
                documents=[chunks[i].chunk_text for i in indices],
                metadatas=[chunks[i].to_metadata_dict() for i in indices],
            )
            total_added += len(indices)

        logger.info("Added %d chunks across %d collection(s).", total_added, len(by_collection))
        return total_added

    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        collection_id: str = "default",
        where: dict | None = None,
    ) -> list[dict]:
        """Return the top-k most similar chunks as plain dicts:
        {chunk_id, chunk_text, metadata, distance}."""
        top_k = top_k or settings.top_k
        collection = self.get_or_create_collection(collection_id)

        count = collection.count()
        if count == 0:
            return []

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
        )

        hits = []
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for i in range(len(ids)):
            hits.append({
                "chunk_id": ids[i],
                "chunk_text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            })
        return hits

    def delete_document(self, document_id: str, collection_id: str = "default") -> None:
        """Remove all chunks belonging to one document - used for re-indexing
        and document management (PRD Section 7.1: 'basic document management')."""
        collection = self.get_or_create_collection(collection_id)
        collection.delete(where={"document_id": document_id})
        logger.info("Deleted all chunks for document_id=%s from collection=%s", document_id, collection_id)

    def get_document_chunks(self, document_id: str, collection_id: str = "default") -> list[dict]:
        """Fetch every chunk belonging to one document, in correct reading
        order (by page, then position within the page). Used by
        summarization (Phase 6), which needs the whole document rather
        than a similarity-ranked subset."""
        collection = self.get_or_create_collection(collection_id)
        result = collection.get(where={"document_id": document_id})

        hits = []
        for i in range(len(result["ids"])):
            hits.append({
                "chunk_id": result["ids"][i],
                "chunk_text": result["documents"][i],
                "metadata": result["metadatas"][i],
            })

        hits.sort(key=lambda h: (h["metadata"]["page_number"], h["metadata"].get("chunk_index", 0)))
        return hits

    def list_documents(self, collection_id: str = "default") -> list[dict]:
        """Return one entry per distinct document in a collection: {document_id,
        filename, page_count}. Powers document management (PRD Section 7.1)."""
        collection = self.get_or_create_collection(collection_id)
        result = collection.get()

        docs: dict[str, dict] = {}
        for meta in result["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "page_numbers": set(),
                }
            docs[doc_id]["page_numbers"].add(meta["page_number"])

        return [
            {
                "document_id": d["document_id"],
                "filename": d["filename"],
                "page_count": len(d["page_numbers"]),
            }
            for d in docs.values()
        ]

    def get_all_chunks(self, collection_id: str = "default") -> list[dict]:
        """Fetch every chunk in a collection, across all documents - used by
        KeywordSearch (Phase 8) to build its BM25 index, which needs the
        full corpus rather than a similarity-ranked subset."""
        collection = self.get_or_create_collection(collection_id)
        result = collection.get()

        hits = []
        for i in range(len(result["ids"])):
            hits.append({
                "chunk_id": result["ids"][i],
                "chunk_text": result["documents"][i],
                "metadata": result["metadatas"][i],
            })
        return hits

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    def collection_chunk_count(self, collection_id: str = "default") -> int:
        try:
            return self.get_or_create_collection(collection_id).count()
        except Exception:  # noqa: BLE001
            return 0
