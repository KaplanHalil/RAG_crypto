"""Central configuration for the Chirag system.

All runtime knobs are read from environment variables (via pydantic-settings)
so that the same code base can be deployed fully offline with a local Ollama
server, or against a cloud API (OpenAI-compatible) without code changes.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    """Location of the repo (directory containing the ``chirag`` package).

    Overridable with the ``CHIRAG_ROOT`` environment variable. All data paths
    (corpus, ChromaDB, .env, sessions, eval artifacts) resolve from here so
    the ``chirag`` command works from any working directory.
    """
    env_root = os.environ.get("CHIRAG_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    pkg = Path(__file__).resolve().parent
    return pkg.parent


PROJECT_ROOT = project_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="CHIRAG_",
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

    # --- Query handling ----------------------------------------------------
    # Non-English questions (e.g. Turkish) are auto-translated to English
    # before retrieval, because the embedding model and the corpus are
    # English. The final answer is still generated from the ORIGINAL question,
    # so it comes back in the user's language.
    translate_queries: bool = True

    # --- Vector store ------------------------------------------------------
    db_path: str = "data/chroma_db"
    collection_name: str = "chirag"

    # --- Generation --------------------------------------------------------
    temperature: float = 0.2
    # Generous budget: thinking-capable local models (e.g. qwen3.5:9b) spend a
    # large share of the budget on their hidden reasoning before answering;
    # too small a budget yields an empty answer.
    max_gen_tokens: int = 8192

    # --- Corpus ------------------------------------------------------------
    papers_dir: str = "kripto_makaleler"

    @field_validator("db_path", "papers_dir")
    @classmethod
    def _root_relative(cls, value: str) -> str:
        """Resolve relative paths against the project root so the CLI works
        from any working directory. Absolute paths are left untouched."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        return str(PROJECT_ROOT / path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
