"""LLM providers (generation backends).

:class:`LLMProvider` is the interface used by the RAG pipeline for answer
synthesis and document summarization. Implementations:

* :class:`OllamaLLM` -- local Ollama ``/api/generate``,
* :class:`OpenAILLM` -- OpenAI-compatible chat/completions API.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

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

    def generate_stream(self, prompt: str, system_prompt: str = "",
                        temperature: float = 0.2,
                        max_tokens: int = 2048) -> Iterator[str]:
        """Generate a completion, yielding incremental text pieces.

        The default implementation yields the whole result at once, so
        non-streaming providers keep working. Streaming providers override
        this to emit tokens as they arrive.
        """
        yield self.generate(prompt, system_prompt=system_prompt,
                            temperature=temperature, max_tokens=max_tokens)


class OllamaLLM(LLMProvider):
    """LLM served by a local Ollama instance."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout: int = 600,
                 extra_options: Optional[Dict[str, Any]] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_options = dict(extra_options or {})
        self.last_tokens: Optional[int] = None

    def _options(self, temperature: float, max_tokens: int) -> Dict[str, Any]:
        return {"temperature": temperature, "num_predict": max_tokens,
                **self.extra_options}

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.2, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": self._options(temperature, max_tokens),
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

    def generate_stream(self, prompt: str, system_prompt: str = "",
                        temperature: float = 0.2,
                        max_tokens: int = 2048) -> Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": self._options(temperature, max_tokens),
        }
        resp = requests.post(
            f"{self.base_url}/api/generate", json=payload,
            stream=True, timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama generate error {resp.status_code}: {resp.text[:300]}"
            )
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = data.get("response")
                if piece:
                    yield piece
                if data.get("done"):
                    self.last_tokens = int(data.get("eval_count") or 0)
                    break
        finally:
            resp.close()


class OpenAILLM(LLMProvider):
    """OpenAI-compatible chat completions backend."""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = 300,
                 extra_options: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_options = dict(extra_options or {})

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.2, max_tokens: int = 2048) -> str:
        messages: list[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **self.extra_options,
            },
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenAI generate error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()["choices"][0]["message"]["content"].strip()


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured LLM provider.

    When a ``chirag.json`` config exists, its model registry wins.
    Otherwise the legacy env-driven behaviour (``CHIRAG_LLM_BACKEND`` …) is
    preserved.
    """
    from .models import config_file_used, get_registry

    if config_file_used() is not None:
        return get_registry().build_llm()

    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("llm_backend='openai' requires CHIRAG_OPENAI_API_KEY")
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
