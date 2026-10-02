"""The RAG pipeline: ingestion, query, and summarization orchestration.

``RAGPipeline`` wires the chunker, the vector store, a chosen retriever and
the LLM provider into a single queryable object. It is the component invoked
by the CLI, the terminal UI, and the evaluation harness.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterator, List, Optional

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

# Common Turkish words/fragments that flag an ASCII-only question (e.g. türkçe
# written without diacritics) as needing translation before retrieval.
_TURKISH_MARKERS = (
    " nedir", " ne", " nasıl", " neden", " nerede", " niçin", " mı ", " mi ",
    " mu ", " mü ", " anlat", " açıkla", " acikla", " yaz", " örnek", " örn",
    " kaç", " kaç ", " hakkında", " hakkinda", " için", " icin", " ve ",
    " ile", " lineer", " kriptanaliz", " kriptoanaliz", " şifre", " sifre",
    " şifreleme", " sifreleme", " anahtar", " atak", " saldırı", " saldiri",
    " makale", " doküman", " dokuman", " inceleyelim", " açıklaması",
    " aciklamasi", " yapılır", " yapilir", " bulunur", " hangi", " nedir",
)


def _has_turkish_marker(question: str) -> bool:
    q = f" {question.strip().lower()} "
    return any(marker in q for marker in _TURKISH_MARKERS)


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
    def _translate_query(self, question: str) -> str:
        """Translate a non-English question to English for retrieval.

        The corpus and the embedding model are English; a non-Latin query
        (e.g. Turkish ``küp atağı``) retrieves almost nothing. When enabled,
        such questions are translated first. Already-ASCII questions pass
        through untouched UNLESS they contain a Turkish marker (for example
        "lineer kriptanalizin ne olduğunu anlat" is plain ASCII but clearly
        Turkish).
        """
        if not self.settings.translate_queries:
            return question
        if question.isascii() and not _has_turkish_marker(question):
            return question
        for _attempt in range(2):  # thinking models can echo the input; retry
            try:
                # NB: do NOT force temperature=0.0 here. Thinking-capable
                # models (e.g. qwen3.5:9b) burn the whole token budget in
                # their hidden thinking block under greedy decoding and reply
                # with an empty string, so the original question would be used.
                translated = self.llm.generate(
                    "Translate the following user question into English so it "
                    "can be used to search an English cryptography knowledge "
                    "base. Reply with ONLY the English translation, no "
                    "explanations.\n\n"
                f"Question: {question}",
                system_prompt="", max_tokens=4096,
            )
            except Exception:
                return question
            cleaned = (translated or "").strip()
            if self._acceptable_translation(cleaned, question):
                return cleaned
        return question

    @staticmethod
    def _acceptable_translation(translated: str, question: str) -> bool:
        """A translation is usable when it is non-empty, actually English
        (ASCII), and not just an echo of the original question."""
        if not translated:
            return False
        if translated.casefold() == question.casefold():
            return False
        return translated.isascii()

    def retrieve(self, question: str, top_k: Optional[int] = None,
                 doc_type_filter: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve sources for ``question`` without invoking the LLM."""
        top_k = top_k or self.settings.default_top_k
        results = self.retriever.retrieve(question, top_k=top_k,
                                          doc_type_filter=doc_type_filter)
        ctx = assemble_context(results) if results else {"context": "",
                                                         "sources": []}
        return {"sources": ctx["sources"], "context": ctx["context"],
                "retrieved_chunks": results}

    def query(self, question: str, top_k: Optional[int] = None,
              doc_type_filter: Optional[str] = None,
              use_llm: bool = True,
              history: Optional[List[dict]] = None) -> Dict[str, Any]:
        top_k = top_k or self.settings.default_top_k
        seek = self._translate_query(question)
        results = self.retriever.retrieve(seek, top_k=top_k,
                                          doc_type_filter=doc_type_filter)
        ctx = assemble_context(results) if results else {"context": "",
                                                         "sources": []}
        if not results:
            return {"answer": NO_ANSWER, "sources": [], "retrieved_chunks": []}

        if not use_llm:
            return {"answer": None, "sources": ctx["sources"],
                    "retrieved_chunks": results, "context": ctx["context"]}

        user_prompt = build_user_prompt(question, ctx["context"],
                                        history=history)
        answer = ""
        budgets = [self.settings.max_gen_tokens,
                   max(self.settings.max_gen_tokens * 2, 8192)]
        for budget in budgets:
            answer = self.llm.generate(
                user_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=self.settings.temperature,
                max_tokens=budget,
            ).strip()
            if answer:
                break
        if not answer:
            answer = ("The model produced no visible answer (it likely spent "
                      "its token budget on reasoning). Please retry.")
        return {"answer": answer, "sources": ctx["sources"],
                "retrieved_chunks": results}

    def query_stream(self, question: str, top_k: Optional[int] = None,
                     doc_type_filter: Optional[str] = None,
                     history: Optional[List[dict]] = None) -> Iterator[dict]:
        """Stream a query as a sequence of events.

        Yields dicts of the form:

        * ``{"type": "sources", "sources": [...], "context": "..."}`` first,
        * ``{"type": "token", "text": "..."}`` for every generated piece, and
        * ``{"type": "done", "answer": "...", "sources": [...],
             "retrieved_chunks": [...]}`` last.
        """
        top_k = top_k or self.settings.default_top_k
        seek = self._translate_query(question)
        results = self.retriever.retrieve(seek, top_k=top_k,
                                          doc_type_filter=doc_type_filter)
        ctx = assemble_context(results) if results else {"context": "",
                                                         "sources": []}
        yield {"type": "sources", "sources": ctx["sources"],
               "context": ctx["context"]}
        if not results:
            yield {"type": "done", "answer": NO_ANSWER, "sources": [],
                   "retrieved_chunks": []}
            return

        user_prompt = build_user_prompt(question, ctx["context"],
                                        history=history)
        parts: List[str] = []
        got_any = False
        budgets = [self.settings.max_gen_tokens,
                   max(self.settings.max_gen_tokens * 2, 8192)]
        for budget in budgets:
            for piece in self.llm.generate_stream(
                    user_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=self.settings.temperature,
                    max_tokens=budget):
                if piece:
                    got_any = True
                    parts.append(piece)
                    yield {"type": "token", "text": piece}
            if got_any:
                break

        answer = "".join(parts).strip()
        if not answer:
            answer = ("The model produced no visible answer (it likely spent "
                      "its token budget on reasoning). Please retry.")
        yield {"type": "done", "answer": answer,
               "sources": ctx["sources"], "retrieved_chunks": results,
               "tokens": getattr(self.llm, "last_tokens", None)}

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
