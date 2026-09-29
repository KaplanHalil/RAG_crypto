"""The RAG pipeline: ingestion, query, and summarization orchestration.

``RAGPipeline`` wires the chunker, the vector store, a chosen retriever and
the LLM provider into a single queryable object. It is the component invoked
by the CLI, the HTTP server, and the evaluation harness.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from .chunker import Chunker, Chunk
from .config import Settings
from .embeddings import EmbeddingProvider
from .ingestion import Document, DocumentLoader
from .llm import LLMProvider
from .prompt import (SYSTEM_PROMPT, assemble_context, build_summary_system_prompt,
                     build_user_prompt)
from .retriever import (DenseRetriever, HybridRetriever, Retriever)
from .vector_store import VectorStore

NO_ANSWER = (
    "No relevant cryptographic standard (RFC), NIST document, or cryptanalysis "
    "paper was found in the knowledge base for this question. Please rephrase "
    "or ingest a related document first."
)


def chunk_id(doc: Document, chunk: Chunk) -> str:
    """Stable unique id for a chunk (deduplicates ingests)."""
    raw = f"{doc.source}|{chunk.metadata['chunk_id']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


class RAGPipeline:
    def __init__(self, settings: Settings, store: VectorStore,
                 provider: EmbeddingProvider, llm: LLMProvider,
                 retriever: str = "hybrid"):
        self.settings = settings
        self.store = store
        self.provider = provider
        self.llm = llm
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        self.loader = DocumentLoader()
        self.retriever: Retriever = (
            HybridRetriever(store, threshold=settings.similarity_threshold)
            if retriever == "hybrid"
            else DenseRetriever(store, threshold=settings.similarity_threshold)
        )

    # ------------------------------------------------------------------ ingest
    def ingest_document(self, doc: Document) -> Dict[str, Any]:
        chunks = self.chunker.chunk(doc.content, doc.metadata())
        if not chunks:
            return {"error": "Document produced no text chunks."}
        ids = [chunk_id(doc, c) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        added = self.store.add_chunks(ids, texts, metadatas)
        return {"success": True, "source": doc.source, "title": doc.title,
                "type": doc.type, "chunks_added": added}

    def ingest_rfc(self, rfc_number: int) -> Dict[str, Any]:
        return self.ingest_document(self.loader.fetch_rfc(rfc_number))

    def ingest_url(self, url: str) -> Dict[str, Any]:
        return self.ingest_document(self.loader.fetch_url(url))

    def ingest_file(self, path: str) -> Dict[str, Any]:
        return self.ingest_document(self.loader.load(path))

    # ------------------------------------------------------------------- query
    def query(self, question: str, top_k: Optional[int] = None,
              doc_type_filter: Optional[str] = None,
              use_llm: bool = True) -> Dict[str, Any]:
        top_k = top_k or self.settings.default_top_k
        results = self.retriever.retrieve(question, top_k=top_k,
                                          doc_type_filter=doc_type_filter)
        ctx = assemble_context(results) if results else {"context": "",
                                                         "sources": []}
        if not results:
            return {"answer": NO_ANSWER, "sources": [], "retrieved_chunks": []}

        if not use_llm:
            return {"answer": None, "sources": ctx["sources"],
                    "retrieved_chunks": results, "context": ctx["context"]}

        user_prompt = build_user_prompt(question, ctx["context"])
        answer = self.llm.generate(
            user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_gen_tokens,
        )
        return {"answer": answer, "sources": ctx["sources"],
                "retrieved_chunks": results}

    # ------------------------------------------------------------ summarization
    def summarize_document(self, source: str, max_chunks: int = 15) -> Dict[str, Any]:
        docs = self.store.list_documents()
        match = next((d for d in docs if d["source"] == source), None)
        if match is None:
            return {"error": f"Document not found in the knowledge base: {source}"}
        text = self.store.get_document_text(source, max_chunks=max_chunks)
        if not text:
            return {"error": f"Document has no retrievable text: {source}"}
        summary = self.llm.generate(
            f"Document title: {match['title']} (file: {source})\n\n"
            f"DOCUMENT CONTENT:\n{text}\n\n"
            "Summarize the document above following the instructed structure.",
            system_prompt=build_summary_system_prompt(),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_gen_tokens,
        )
        return {"source": source, "title": match["title"],
                "type": match["type"], "chunk_count": match["chunk_count"],
                "summary": summary}

    # ---------------------------------------------------------------- utilities
    def stats(self) -> Dict[str, Any]:
        return self.store.get_stats()

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.store.list_documents()
