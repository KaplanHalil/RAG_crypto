"""Bulk knowledge-base indexing.

``index_corpus`` rebuilds (optionally) the vector index from the local
corpus directory (``kripto_makaleler/``) and supplements it with curated,
structurally-normalized entries for landmark cryptanalysis papers that are
frequently cited but for which we do not hold a machine-readable copy.

The curated entries are provided in ``landmark_papers.py`` and double as
evaluation ground truth for the QA benchmark.
"""

from __future__ import annotations

import os
from typing import Optional

from .config import get_settings
from .embeddings import build_embedding_provider
from .ingestion import DocumentLoader
from .landmark_papers import LANDMARK_PAPERS
from .pipeline import RAGPipeline, chunk_id
from .vector_store import VectorStore


def index_corpus(reset: bool = True, papers_dir: Optional[str] = None,
                 only_landmark: bool = False) -> None:
    settings = get_settings()
    provider = build_embedding_provider(settings)
    store = VectorStore(settings, provider)
    from .chunker import Chunker
    chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
    loader = DocumentLoader()

    if reset:
        print("[indexer] resetting collection ...")
        store.reset()

    total_chunks = 0
    total_docs = 0

    if not only_landmark:
        dir_path = papers_dir or settings.papers_dir
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Corpus directory not found: {dir_path}")
        files = sorted(
            f for f in os.listdir(dir_path)
            if f.lower().endswith((".pdf", ".txt", ".md"))
        )
        print(f"[indexer] found {len(files)} documents in {dir_path}")
        for filename in files:
            path = os.path.join(dir_path, filename)
            try:
                doc = loader.load(path)
            except Exception as exc:
                print(f"[indexer] SKIP {filename}: {exc}")
                continue
            if not doc.content.strip():
                print(f"[indexer] SKIP {filename}: empty content")
                continue
            chunks = chunker.chunk(doc.content, doc.metadata())
            if not chunks:
                continue
            ids = [chunk_id(doc, c) for c in chunks]
            store.add_chunks(ids, [c.text for c in chunks],
                             [c.metadata for c in chunks])
            total_chunks += len(chunks)
            total_docs += 1
            print(f"[indexer] + {filename} ({len(chunks)} chunks) [{doc.type}]")

    print(f"[indexer] indexing {len(LANDMARK_PAPERS)} curated landmark papers")
    for entry in LANDMARK_PAPERS:
        doc = entry["document"]
        chunks = chunker.chunk(doc.content, doc.metadata())
        ids = [chunk_id(doc, c) for c in chunks]
        store.add_chunks(ids, [c.text for c in chunks], [c.metadata for c in chunks])
        total_chunks += len(chunks)
        total_docs += 1
        print(f"[indexer] + landmark: {doc.title} ({len(chunks)} chunks)")

    print(f"[indexer] done: {total_docs} documents, {total_chunks} chunks")
    print(f"[indexer] stats: {store.get_stats()}")


if __name__ == "__main__":
    index_corpus()
