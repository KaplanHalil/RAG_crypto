"""Persistent vector store backed by ChromaDB.

Chunks are stored with their metadata in a cosine-similarity collection.
Search returns the nearest neighbours above a similarity threshold together
with their metadata for citation-aware context assembly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import Settings
from .embeddings import ChromadbEmbeddingFunction, EmbeddingProvider


class VectorStore:
    def __init__(self, settings: Settings, provider: EmbeddingProvider):
        self.settings = settings
        self.provider = provider
        self.client = chromadb.PersistentClient(
            path=settings.db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=ChromadbEmbeddingFunction(
                provider, provider.__class__.__name__
            ),
            metadata={"hnsw:space": "cosine"},
        )

    # --- writes ------------------------------------------------------------
    def add_chunks(self, ids: List[str], texts: List[str],
                   metadatas: List[Dict[str, Any]]) -> int:
        if not texts:
            return 0
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(texts)

    def delete_document(self, source: str) -> bool:
        before = self.collection.count()
        self.collection.delete(where={"source": source})
        return self.collection.count() < before

    def reset(self) -> None:
        """Drop the collection so the index can be rebuilt deterministically."""
        self.client.delete_collection(self.settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            embedding_function=ChromadbEmbeddingFunction(
                self.provider, self.provider.__class__.__name__
            ),
            metadata={"hnsw:space": "cosine"},
        )

    # --- reads -------------------------------------------------------------
    def search(self, query: str, top_k: int = 5,
               doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        where = ({"type": doc_type_filter}
                 if doc_type_filter and doc_type_filter != "All" else None)
        try:
            results = self.collection.query(
                query_texts=[query], n_results=max(1, top_k), where=where
            )
        except Exception as exc:  # ChromaDB raises on empty/near-empty queries
            print(f"[VectorStore] query failed: {exc}")
            return []

        out: List[Dict[str, Any]] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1.0 - float(dist) if dist is not None else 1.0
            if similarity > 0.0:
                out.append({"content": doc, "metadata": meta,
                            "similarity": round(similarity, 4)})
        return out

    def list_documents(self) -> List[Dict[str, Any]]:
        recs = self.collection.get(include=["metadatas"])
        docs: Dict[str, Dict[str, Any]] = {}
        for meta in recs.get("metadatas", []):
            source = meta.get("source", "Unknown")
            entry = docs.setdefault(source, {
                "source": source,
                "title": meta.get("title", source),
                "type": meta.get("type", "Document"),
                "url": meta.get("url", ""),
                "chunk_count": 0,
            })
            entry["chunk_count"] += 1
        return sorted(docs.values(), key=lambda d: d["source"].lower())

    def get_document_text(self, source: str, max_chunks: int = 15) -> str:
        """Concatenate up to ``max_chunks`` of one document for summarization."""
        recs = self.collection.get(where={"source": source},
                                   include=["documents", "metadatas"])
        docs = recs.get("documents", [])
        if not docs:
            return ""
        return "\n\n".join(docs[:max_chunks])

    def get_stats(self) -> Dict[str, Any]:
        docs = self.list_documents()
        rfc = sum(1 for d in docs
                  if "RFC" in d["type"] or d["source"].startswith("RFC"))
        standards = sum(1 for d in docs if "NIST" in d["type"])
        papers = sum(1 for d in docs if "Paper" in d["type"]
                     or "Article" in d["type"])
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(docs),
            "rfc_count": rfc,
            "nist_count": standards,
            "paper_count": papers,
        }
