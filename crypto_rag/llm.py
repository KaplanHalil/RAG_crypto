"""LLM providers (generation backends).

:class:`LLMProvider` is the interface used by the RAG pipeline for answer
synthesis and document summarization. Implementations:

* :class:`OllamaLLM` -- local Ollama ``/api/generate``,
* :class:`OpenAILLM` -- OpenAI-compatible chat/completions API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import requests

from .config import Settings

SYSTEM_TAG = "<<SYS>>"


class LLMProvider(ABC):
    """Abstract text-generation backend."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.2,
                 max_tokens: int = 2048) -> str:
        """Generate a completion for ``prompt``."""


class OllamaLLM(LLMProvider):
    """LLM served by a local Ollama instance."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout: int = 600):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.2, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama generate error {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        if data.get("done_reason") == "length" and data.get("context"):
            pass  # truncated by num_predict is acceptable
        return (data.get("response") or "").strip()


class OpenAILLM(LLMProvider):
    """OpenAI-compatible chat completions backend."""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = 300):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.2, max_tokens: int = 2048) -> str:
        messages: list[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenAI generate error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()["choices"][0]["message"]["content"].strip()


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured LLM provider."""
    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("llm_backend='openai' requires CRAG_OPENAI_API_KEY")
        return OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.openai_llm_model,
            base_url=settings.openai_base_url,
        )
    return OllamaLLM(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_request_timeout,
    )
