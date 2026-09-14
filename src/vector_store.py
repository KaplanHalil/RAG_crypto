import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from src.ollama_client import OllamaClient

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, ollama_client: OllamaClient, model_name: str = "nomic-embed-text"):
        self.ollama_client = ollama_client
        self.model_name = model_name

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        for text in input:
            vec = self.ollama_client.get_embedding(text, model=self.model_name)
            if not vec:
                # Fallback zero vector if embedding fails
                vec = [0.0] * 768
            embeddings.append(vec)
        return embeddings

class VectorStore:
    def __init__(self, db_path: str = "./data/chroma_db", ollama_client: Optional[OllamaClient] = None):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.ollama_client = ollama_client or OllamaClient()
        self.embedding_fn = OllamaEmbeddingFunction(self.ollama_client)
        
        self.collection = self.client.get_or_create_collection(
            name="crypto_cryptanalysis_rag",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Add text chunks with metadata into ChromaDB vector store."""
        if not chunks:
            return 0

        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        # Unique ID for each chunk: source + chunk_id
        ids = [f"{c['metadata']['source'].replace(' ', '_')}_{c['metadata']['chunk_id']}" for c in chunks]

        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5, doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search vector database for relevant chunks given a query."""
        where_clause = None
        if doc_type_filter and doc_type_filter != "All":
            where_clause = {"type": doc_type_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )

        formatted_results = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                similarity = round(1.0 - float(dist), 4) if dist is not None else 1.0
                if similarity <= 0.0:
                    continue
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "distance": dist
                })

        return formatted_results

    def list_ingested_documents(self) -> List[Dict[str, Any]]:
        """List distinct documents stored in the database."""
        all_records = self.collection.get(include=["metadatas"])
        docs_map = {}
        if all_records and all_records.get("metadatas"):
            for meta in all_records["metadatas"]:
                source = meta.get("source", "Unknown")
                if source not in docs_map:
                    docs_map[source] = {
                        "source": source,
                        "title": meta.get("title", source),
                        "type": meta.get("type", "Document"),
                        "url": meta.get("url", ""),
                        "chunk_count": 1
                    }
                else:
                    docs_map[source]["chunk_count"] += 1
        return list(docs_map.values())

    def delete_document(self, source: str) -> bool:
        """Delete all chunks belonging to a document source."""
        try:
            self.collection.delete(where={"source": source})
            return True
        except Exception as e:
            print(f"Error deleting document {source}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        count = self.collection.count()
        docs = self.list_ingested_documents()
        return {
            "total_chunks": count,
            "total_documents": len(docs),
            "rfc_count": sum(1 for d in docs if "RFC" in d.get("type", "") or "RFC" in d.get("source", "")),
            "article_count": sum(1 for d in docs if "Article" in d.get("type", "") or "Paper" in d.get("type", ""))
        }
