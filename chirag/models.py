"""Model registry: opencode-style provider/model configuration.

Chirag reads a JSON config (``chirag.json``) very much like opencode's
``opencode.json``. You can register any number of providers and models and
switch between them from the TUI or the CLI without touching code::

    {
      "provider": {
        "ollama": {
          "type": "ollama",
          "base_url": "http://localhost:11434",
          "models": { "qwen3.5:9b": { "name": "Qwen 3.5 9B" } }
        },
        "openai": {
          "type": "openai",
          "options": { "api_key": "env:OPENAI_API_KEY" },
          "models": { "gpt-4o-mini": {} }
        },
        "groq": {
          "type": "openai",
          "base_url": "https://api.groq.com/openai/v1",
          "options": { "api_key": "env:GROQ_API_KEY" },
          "models": { "llama-3.3-70b-versatile": {} }
        }
      },
      "embedding": { "provider": "ollama", "model": "nomic-embed-text" }
    }

Config files are searched in this order (first match wins), like opencode::

    $CHIRAG_CONFIG                      explicit override
    <project_root>/chirag.json     project config
    <project_root>/.chirag.json    project config (hidden)
    ~/.config/chirag/config.json   global config

The project file is deep-merged OVER the global file. API keys use the
opencode ``env:VAR_NAME`` syntax and are resolved from the environment; they
are never printed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import PROJECT_ROOT, Settings, get_settings
from .llm import LLMProvider, OllamaLLM, OpenAILLM
from .embeddings import EmbeddingProvider, OllamaEmbeddings, OpenAIEmbeddings

CONFIG_NAMES = ("chirag.json", ".chirag.json")
DEFAULT_GLOBAL_CONFIG = Path.home() / ".config" / "chirag" / "config.json"


# ----------------------------------------------------------------- config I/O
def config_paths() -> List[Path]:
    """Candidate config files, most specific first."""
    override = os.environ.get("CHIRAG_CONFIG")
    if override:
        return [Path(override).expanduser()]
    paths = [PROJECT_ROOT / name for name in CONFIG_NAMES]
    paths.append(DEFAULT_GLOBAL_CONFIG)
    return paths


def load_config() -> Dict[str, Any]:
    """Return the merged config dict (project over global). Empty if none."""
    merged: Dict[str, Any] = {}
    files = [p for p in config_paths() if p.is_file()]
    if not files:
        return merged
    for path in reversed(files):  # least specific (global) first
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid config {path}: {exc}")
        merged = _deep_merge(merged, data)
    return merged


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if (key in out and isinstance(out[key], dict)
                and isinstance(value, dict)):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_secret(value: Any) -> str:
    """Resolve ``env:VAR`` (or a literal string) to a secret value."""
    if isinstance(value, str) and value.startswith("env:"):
        name = value[4:]
        return os.environ.get(name, "")
    return str(value or "")


# ----------------------------------------------------------------- registry
class ModelRegistry:
    """Parsed view of the merged config with lookup helpers."""

    def __init__(self, raw: Dict[str, Any], settings: Settings):
        self.raw = raw
        self.settings = settings
        self.providers: Dict[str, Dict[str, Any]] = raw.get("provider", {})

    # -- providers -------------------------------------------------------
    def provider_for(self, model_id: str) -> Optional[Dict[str, Any]]:
        for provider in self.providers.values():
            if model_id in provider.get("models", {}):
                return provider
        return None

    def model_entry(self, model_id: str) -> Optional[dict]:
        provider = self.provider_for(model_id)
        if provider is None:
            return None
        return provider["models"].get(model_id, {})

    # -- chat models -----------------------------------------------------
    def chat_models(self) -> List[Tuple[str, str, str]]:
        """Return ``(model_id, display_name, provider_type)`` for every
        registered model. With no config, the default Ollama model is
        listed so the TUI / API always have a usable default."""
        if not self.providers:
            return [(self.settings.llm_model, self.settings.llm_model,
                     "ollama")]
        out: List[Tuple[str, str, str]] = []
        for provider in self.providers.values():
            ptype = provider.get("type", "ollama")
            for model_id, entry in provider.get("models", {}).items():
                name = entry.get("name", model_id) if isinstance(entry, dict) \
                    else model_id
                out.append((model_id, name, ptype))
        return out

    def default_llm(self) -> str:
        defaults = self.raw.get("defaults", {})
        model = defaults.get("llm") or self.settings.llm_model
        return model

    def effective_temperature(self, model_id: str) -> Optional[float]:
        entry = self.model_entry(model_id)
        if entry and isinstance(entry, dict) and entry.get("temperature"):
            return float(entry["temperature"])
        return None

    # -- builders --------------------------------------------------------
    def build_llm(self, model_id: Optional[str] = None) -> LLMProvider:
        model_id = model_id or self.default_llm()
        provider = self.provider_for(model_id)
        if provider is None:
            # fall back to the default Ollama backend
            return OllamaLLM(
                model=model_id,
                base_url=self.settings.ollama_base_url,
                timeout=self.settings.ollama_request_timeout,
            )
        base_url = provider.get("base_url")
        ptype = provider.get("type", "ollama")
        entry = self.model_entry(model_id)
        options = dict(provider.get("options", {}) or {})
        if isinstance(entry, dict):
            options.update(entry.get("options", {}) or {})
        if ptype == "openai":
            api_key = resolve_secret(options.get("api_key", ""))
            return OpenAILLM(
                api_key=api_key, model=model_id,
                base_url=base_url or self.settings.openai_base_url,
            )
        extra_options = {k: v for k, v in options.items() if k != "api_key"}
        return OllamaLLM(
            model=model_id,
            base_url=base_url or self.settings.ollama_base_url,
            timeout=self.settings.ollama_request_timeout,
            extra_options=extra_options,
        )

    def build_embedding(self) -> EmbeddingProvider:
        emb = self.raw.get("embedding", {}) or {}
        provider_id = emb.get("provider") or self.settings.embedding_backend
        model = emb.get("model")
        if provider_id == "openai":
            api_key = resolve_secret(self.raw.get("openai_api_key")
                                     or self.settings.openai_api_key)
            return OpenAIEmbeddings(
                api_key=api_key or self.settings.openai_api_key,
                model=model or self.settings.openai_embedding_model,
                base_url=self.settings.openai_base_url,
            )
        return OllamaEmbeddings(
            model=model or self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
            timeout=self.settings.ollama_request_timeout,
            batch_size=self.settings.embed_batch_size,
        )

    def effective(self, redact: bool = True) -> Dict[str, Any]:
        """Public view of the config (secrets redacted by default)."""
        out = json.loads(json.dumps(self.raw))
        if redact:
            _redact(out)
        return out


def _redact(node: Any) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "api_key":
                node[key] = "***"
            else:
                _redact(value)
    elif isinstance(node, list):
        for item in node:
            _redact(item)


# ----------------------------------------------------------------- singleton
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(load_config(), get_settings())
    return _registry


def reset_registry() -> None:
    """Drop the cached registry (used by tests)."""
    global _registry
    _registry = None


def config_file_used() -> Optional[Path]:
    for path in config_paths():
        if path.is_file():
            return path
    return None


# ------------------------------------------------------------------ sugar
def build_llm(model_id: Optional[str] = None) -> LLMProvider:
    return get_registry().build_llm(model_id)


def build_embedding() -> EmbeddingProvider:
    return get_registry().build_embedding()


def list_chat_models() -> List[Tuple[str, str, str]]:
    return get_registry().chat_models()
