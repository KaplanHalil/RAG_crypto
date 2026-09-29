"""Central configuration for the CryptoRAG system.

All runtime knobs are read from environment variables (via pydantic-settings)
so that the same code base can be deployed fully offline with a local Ollama
server, or against a cloud API (OpenAI-compatible) without code changes.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CRAG_",
        extra="ignore",
    )

    # --- Backend selection -------------------------------------------------
    # "ollama" (default, fully local) or "openai" (OpenAI-compatible chat API).
    llm_backend: Literal["ollama", "openai"] = "ollama"
    embedding_backend: Literal["ollama", "openai"] = "ollama"

    # --- Ollama ------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embedding_model: str = "nomic-embed-text"
    ollama_request_timeout: int = 600

    # --- OpenAI-compatible cloud API --------------------------------------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Text processing ---------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_tokens_per_chunk: int = 512  # safety: refuse oversized chunks on ingest

    # --- Retrieval ---------------------------------------------------------
    default_top_k: int = 5
    similarity_threshold: float = 0.25  # chunks below this are not returned
    embed_batch_size: int = 32

    # --- Vector store ------------------------------------------------------
    db_path: str = "data/chroma_db"
    collection_name: str = "crypto_rag"

    # --- Generation --------------------------------------------------------
    temperature: float = 0.2
    max_gen_tokens: int = 2048

    # --- Corpus ------------------------------------------------------------
    papers_dir: str = "kripto_makaleler"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
