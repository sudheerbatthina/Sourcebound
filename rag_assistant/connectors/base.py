"""BaseConnector ABC shared by all connector implementations."""

import hashlib
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract base for all data-source connectors."""

    def __init__(self, config: dict, tenant_id: str):
        self.config = config
        self.tenant_id = tenant_id

    # ------------------------------------------------------------------
    # Override in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_data(self) -> list[dict]:
        """Return list of {"content": str, "source": str, "metadata": dict}."""

    def test_connection(self) -> dict:
        """Try a lightweight fetch. Returns {"ok": bool, "message": str}."""
        try:
            data = self.fetch_data()
            return {"ok": True, "message": f"Connected — {len(data)} item(s) found"}
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Shared sync logic
    # ------------------------------------------------------------------

    def sync(self) -> int:
        """Fetch, chunk, embed, and upsert to Chroma. Returns chunk count."""
        from rag_assistant.chunker import chunk_text
        from rag_assistant.vector_store import build_vector_store

        docs = self.fetch_data()
        if not docs:
            return 0

        chunks = []
        for doc in docs:
            content = doc.get("content", "").strip()
            if not content:
                continue
            source = doc.get("source", "connector")
            meta = doc.get("metadata", {})
            doc_chunks = chunk_text(content)
            for i, ch in enumerate(doc_chunks):
                chunk_id = hashlib.md5(f"{source}:{i}:{ch[:40]}".encode()).hexdigest()
                chunks.append({
                    "id": chunk_id,
                    "content": ch,
                    "metadata": {
                        "source": source,
                        "page_number": i,
                        "chunk_type": "connector",
                        "session_id": "global",
                        "tenant_id": self.tenant_id,
                        **meta,
                    },
                })

        if not chunks:
            return 0

        collection = build_vector_store(chunks, session_id="global", tenant_id=self.tenant_id)
        logger.info("Connector sync: upserted %d chunks for tenant %s", len(chunks), self.tenant_id)
        return len(chunks)
