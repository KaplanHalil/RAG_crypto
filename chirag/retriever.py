"""Retrieval strategies.

Two retrievers are provided and compared in the evaluation:

* :class:`DenseRetriever` -- pure vector (dense) similarity search over the
  embedding index;
* :class:`HybridRetriever` -- reciprocal-rank fusion (RRF) of dense
  retrieval with a lightweight BM25 lexical scorer, which handles exact
  identifiers (``2^47``, ``RFC 8446``, ``ML-KEM``) that dense matching
  tends to miss.

Both return a list of ranked results with ``content``, ``metadata`` and a
normalised fusion score, so the rest of the pipeline is retriever-agnostic.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .vector_store import VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Scorer:
    """Compact Okapi-BM25 implementation over a fixed corpus of texts."""

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = [len(_tokenize(t)) for t in texts]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.doc_freq: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.tokenized = [_tokenize(t) for t in texts]
        n_docs = len(self.tokenized)
        for tokens in self.tokenized:
            self.doc_freq.update(set(tokens))
        for term, freq in self.doc_freq.items():
            self.idf[term] = math.log(
                1 + (n_docs - freq + 0.5) / (freq + 0.5)
            )

    def score(self, query: str) -> List[float]:
        qtokens = _tokenize(query)
        scores = [0.0] * len(self.tokenized)
        for i, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            dl = self.doc_len[i]
            denom = dl * self.b + (self.k1 * (1 - self.b))
            s = 0.0
            for term in qtokens:
                if term in self.idf and term in tf:
                    s += (self.idf[term] * tf[term] * (self.k1 + 1)
                          / (tf[term] + denom))
            scores[i] = s
        return scores


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5,
                 doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` ranked retrieval results."""


def _rrf(rankings: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    scores: Dict[tuple, float] = {}
    merged: Dict[tuple, Dict[str, Any]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            meta = item["metadata"]
            key = (meta.get("source", ""), meta.get("chunk_id"))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            merged[key] = item
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{**merged[k], "similarity": round(score, 5)}
            for k, score in ordered]


class DenseRetriever(Retriever):
    def __init__(self, store: VectorStore, threshold: float = 0.0):
        self.store = store
        self.threshold = threshold

    def retrieve(self, query: str, top_k: int = 5,
                 doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        results = self.store.search(query, top_k=top_k,
                                    doc_type_filter=doc_type_filter)
        return [r for r in results if r["similarity"] >= self.threshold]


class HybridRetriever(Retriever):
    """RRF fusion of dense retrieval and BM25 lexical scoring."""

    def __init__(self, store: VectorStore, dense_top_k: int = 40,
                 threshold: float = 0.0):
        self.store = store
        self.dense_top_k = dense_top_k
        self.threshold = threshold
        self._corpus: Optional[Dict[str, Any]] = None
        self._bm25: Optional[BM25Scorer] = None

    def _refresh_corpus(self) -> None:
        all_records = self.store.collection.get(include=["documents", "metadatas"])
        self._corpus = all_records
        self._bm25 = BM25Scorer(all_records["documents"])

    def retrieve(self, query: str, top_k: int = 5,
                 doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._bm25 is None:
            self._refresh_corpus()

        dense = self.store.search(query, top_k=self.dense_top_k,
                                  doc_type_filter=doc_type_filter)
        dense = [r for r in dense if r["similarity"] >= self.threshold]

        scores = self._bm25.score(query)  # type: ignore[union-attr]
        docs = self._corpus["documents"]  # type: ignore[index]
        metas = self._corpus["metadatas"]  # type: ignore[index]
        lexical = []
        for i, s in enumerate(scores):
            meta = metas[i]
            if doc_type_filter and doc_type_filter != "All":
                if meta.get("type") != doc_type_filter:
                    continue
            if s > 0:
                lexical.append({"content": docs[i], "metadata": meta,
                                "similarity": round(float(s), 5)})
        lexical.sort(key=lambda r: r["similarity"], reverse=True)
        lexical = lexical[:self.dense_top_k]

        return _rrf([dense, lexical])[:top_k]
