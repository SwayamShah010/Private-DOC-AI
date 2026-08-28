"""
Data model for a chunk (lives with ingestion since chunking is a text-prep step) - the atomic unit that gets embedded, stored,
retrieved, and cited.

Fields mirror the metadata model in PRD Section 10.3 exactly, so this
maps 1:1 onto what gets stored in ChromaDB later.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Chunk:
    chunk_id: str            # unique per chunk, e.g. "{document_id}_p{page}_c{index}"
    document_id: str
    filename: str
    page_number: int         # which page this chunk's text came from
    chunk_text: str
    chunk_index: int = 0     # position within the page (0-indexed) - lets us
                              # reconstruct correct reading order later, since
                              # chunk_id sorts lexically ("c10" < "c2") and can't
                              # be relied on for ordering
    collection_id: str = "default"
    embedding_model: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_metadata_dict(self) -> dict:
        """Metadata payload for ChromaDB - everything except the text itself
        and the vector, which ChromaDB stores separately."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "collection_id": self.collection_id,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
        }
