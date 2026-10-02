"""Tests for the opencode-style model registry (chirag/models.py)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chirag.models as models
from chirag.llm import OllamaLLM, OpenAILLM


@pytest.fixture(autouse=True)
def _reset():
    models.reset_registry()
    yield
    models.reset_registry()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_config_paths_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "custom.json"
    monkeypatch.setenv("CHIRAG_CONFIG", str(cfg))
    assert models.config_paths() == [cfg]


def test_load_config_merges_global_then_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    global_dir = tmp_path / "global"
    monkeypatch.setattr(models, "PROJECT_ROOT", project)
    monkeypatch.setattr(models, "DEFAULT_GLOBAL_CONFIG",
                        global_dir / "config.json")
    monkeypatch.delenv("CHIRAG_CONFIG", raising=False)

    _write(global_dir / "config.json", {
        "provider": {"openai": {"type": "openai", "models": {"gpt-4o-mini": {}}}},
        "defaults": {"llm": "gpt-4o-mini"},
    })
    _write(project / "chirag.json", {
        "provider": {"ollama": {"type": "ollama",
                                "models": {"qwen3.5:9b": {}}}},
        "defaults": {"llm": "qwen3.5:9b"},
    })

    cfg = models.load_config()
    assert set(cfg["provider"]) == {"openai", "ollama"}
    assert cfg["defaults"]["llm"] == "qwen3.5:9b"  # project wins


def test_load_config_empty_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "PROJECT_ROOT", tmp_path / "empty")
    monkeypatch.delenv("CHIRAG_CONFIG", raising=False)
    assert models.load_config() == {}


def test_resolve_secret_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sekret")
    assert models.resolve_secret("env:MY_KEY") == "sekret"
    assert models.resolve_secret("literal") == "literal"
    assert models.resolve_secret("env:MISSING") == ""


def test_registry_build_ollama(tmp_path, monkeypatch):
    _write(tmp_path / "c.json", {
        "provider": {"ollama": {
            "type": "ollama", "base_url": "http://10.0.0.1:1234",
            "models": {"qwen3.5:9b": {}}}}})
    monkeypatch.setenv("CHIRAG_CONFIG", str(tmp_path / "c.json"))
    reg = models.get_registry()

    assert reg.chat_models() == [("qwen3.5:9b", "qwen3.5:9b", "ollama")]
    llm = reg.build_llm("qwen3.5:9b")
    assert isinstance(llm, OllamaLLM)
    assert llm.base_url == "http://10.0.0.1:1234"
    assert llm.model == "qwen3.5:9b"


def test_registry_build_openai_uses_env_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_TOKEN", "sk-test")
    _write(tmp_path / "c.json", {
        "provider": {"openai": {
            "type": "openai",
            "base_url": "https://api.example.com/v1",
            "options": {"api_key": "env:OPENAI_TOKEN"},
            "models": {"gpt-4o-mini": {}}}}})
    monkeypatch.setenv("CHIRAG_CONFIG", str(tmp_path / "c.json"))
    reg = models.get_registry()

    llm = reg.build_llm("gpt-4o-mini")
    assert isinstance(llm, OpenAILLM)
    assert llm.api_key == "sk-test"
    assert llm.base_url == "https://api.example.com/v1"


def test_registry_openai_without_key_is_allowed(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    _write(tmp_path / "c.json", {
        "provider": {"openai": {
            "type": "openai",
            "base_url": "https://open.example/v1",
            "options": {"api_key": "env:NOPE"},
            "models": {"gpt-5": {}}}}})
    monkeypatch.setenv("CHIRAG_CONFIG", str(tmp_path / "c.json"))
    llm = models.get_registry().build_llm("gpt-5")
    assert isinstance(llm, OpenAILLM)
    assert llm.api_key == ""  # keyless endpoints are allowed


def test_registry_unknown_model_falls_back_to_ollama(tmp_path, monkeypatch):
    _write(tmp_path / "c.json", {"provider": {}})
    monkeypatch.setenv("CHIRAG_CONFIG", str(tmp_path / "c.json"))
    reg = models.get_registry()
    llm = reg.build_llm("something-not-registered")
    assert isinstance(llm, OllamaLLM)
    assert llm.model == "something-not-registered"


def test_registry_default_llm(tmp_path, monkeypatch):
    _write(tmp_path / "c.json", {
        "defaults": {"llm": "my-default"},
        "provider": {"ollama": {"models": {"my-default": {}}}},
    })
    monkeypatch.setenv("CHIRAG_CONFIG", str(tmp_path / "c.json"))
    assert models.get_registry().default_llm() == "my-default"


def test_effective_redacts_api_keys(tmp_path, monkeypatch):
    _write(tmp_path / "c.json", {
        "provider": {"openai": {
            "type": "openai", "options": {"api_key": "env:OPENAI_API_KEY"},
            "models": {"gpt-4o-mini": {}}}}})
    monkeypatch.setenv("CHIRAG_CONFIG", str(tmp_path / "c.json"))
    eff = models.get_registry().effective(redact=True)
    assert eff["provider"]["openai"]["options"]["api_key"] == "***"


def test_build_llm_with_extra_options(tmp_path, monkeypatch):
    _write(tmp_path / "c.json", {
        "provider": {"ollama": {
            "type": "ollama",
            "models": {"deepseek:r1": {"options": {"think": False}}}}}})
    monkeypatch.setenv("CHIRAG_CONFIG", str(tmp_path / "c.json"))
    llm = models.get_registry().build_llm("deepseek:r1")
    assert llm.extra_options == {"think": False}
