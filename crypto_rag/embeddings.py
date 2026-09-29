"""Embedding providers.

An :class:`EmbeddingProvider` turns text into dense vectors. Two
implementations are provided:

* :class:`OllamaEmbeddings` -- calls a local Ollama server (default),
* :class:`OpenAIEmbeddings` -- calls an OpenAI-compatible embeddings API.

A :class:`ChromadbEmbeddingFunction` adapter lets both providers be used
directly inside ChromaDB's ``get_or_create_collection`` flow.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import requests

from .config import Settings


class EmbeddingProvider(ABC):
    """Abstract embedding model."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return a dense vector for each input text."""


class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings served by a local Ollama instance."""

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 300, batch_size: int = 32):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:3000] for t in texts[i:i + self.batch_size]]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        with self._lock:  # Ollama processes one request at a time by default
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama embed error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()["embeddings"]


class OpenAIEmbeddings(EmbeddingProvider):
    """Embeddings from an OpenAI-compatible REST API (text-embedding-*)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1",
                 batch_size: int = 64):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": list(batch)},
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenAI embed error {resp.status_code}: {resp.text[:300]}"
                )
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
        return vectors


class ChromadbEmbeddingFunction:
    """Adapter exposing an EmbeddingProvider to ChromaDB."""

    def __init__(self, provider: EmbeddingProvider, name: str):
        self._provider = provider
        self._name = name

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self._provider.embed(list(input))

    def embed_documents(self, input, **kwargs):
        if isinstance(input, str):
            input = [input]
        return self._provider.embed(list(input))

    def embed_query(self, input, **kwargs):
        if isinstance(input, str):
            input = [input]
        return self._provider.embed(list(input))

    def name(self) -> str:
        return self._name


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Instantiate the configured embedding provider."""
    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "embedding_backend='openai' requires CRAG_OPENAI_API_KEY"
            )
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            base_url=settings.openai_base_url,
        )
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_request_timeout,
        batch_size=settings.embed_batch_size,
    )
