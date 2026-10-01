"""Headless tests for the terminal UI (no network, no model inference).

These tests run fully offline: the app falls back to the configured default
model when the local Ollama server is unreachable, and composing/running the
UI does not require model inference.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crypto_rag.tui as tui_mod
from crypto_rag.tui import ChatApp
from textual.containers import VerticalScroll
from textual.widgets import Input, Select, Static


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _tmp_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(tui_mod, "SESSION_DIR", tmp_path / "sessions")


def test_chat_app_composes_with_settings():
    async def main():
        app = ChatApp()
        async with app.run_test():
            model_select = app.query_one("#model-select", Select)
            filter_select = app.query_one("#filter-select", Select)
            topk = app.query_one("#topk", Input)
            qinput = app.query_one("#qinput", Input)
            assert model_select.value  # at least the default model
            assert filter_select.value is not None
            assert int(topk.value) > 0
            assert qinput is not None
            # typing must go to the query input, not the model dropdown
            assert app.focused is qinput

    _run(main())


def test_help_command_renders_info_bubble():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/help"
            await qinput.action_submit()
            await pilot.pause()
            assert list(app.query(".md-info"))

    _run(main())


def test_clear_command_resets_chat():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/help"
            await qinput.action_submit()
            await pilot.pause()
            qinput.value = "/clear"
            await qinput.action_submit()
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            assert len(list(chat.children)) == 1

    _run(main())


def test_stats_command_updates_statsbar():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/stats"
            await qinput.action_submit()
            await pilot.pause()
            statsbar = app.query_one("#statsbar", Static)
            assert "chunks:" in str(statsbar.render())

    _run(main())


def test_save_and_load_session_roundtrips():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            app.history = [
                {"role": "user", "content": "What is AES?"},
                {"role": "assistant", "content": "AES is a block cipher."},
            ]
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/save"
            await qinput.action_submit()
            await pilot.pause()

            qinput.value = "/new"
            await qinput.action_submit()
            await pilot.pause()
            assert app.history == []

            qinput.value = "/load"
            await qinput.action_submit()
            await pilot.pause()
            assert [t["content"] for t in app.history] == [
                "What is AES?", "AES is a block cipher."]
            assert app._last_answer == "AES is a block cipher."

    _run(main())


def test_new_command_clears_history():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            app.history = [{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "a"}]
            app._last_answer = "a"
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/new"
            await qinput.action_submit()
            await pilot.pause()
            assert app.history == []
            assert app._last_answer == ""

    _run(main())


def test_query_stream_events_with_fake_llm():
    from crypto_rag.config import get_settings
    from crypto_rag.embeddings import build_embedding_provider
    from crypto_rag.pipeline import RAGPipeline
    from crypto_rag.vector_store import VectorStore

    class FakeLLM:
        last_tokens = 7

        def generate_stream(self, prompt, **kwargs):
            for piece in ("Hel", "lo ", "world"):
                yield piece

    class FakeRetriever:
        def retrieve(self, question, top_k=None, doc_type_filter=None):
            return [{"content": "chunk", "metadata": {
                "source": "X", "title": "Doc X", "type": "Paper"}}]

    settings = get_settings()
    provider = build_embedding_provider(settings)
    store = VectorStore(settings, provider)
    pipe = RAGPipeline(settings, store, provider, FakeLLM())
    pipe.retriever = FakeRetriever()

    events = list(pipe.query_stream("hello", history=[]))
    types = [e["type"] for e in events]
    assert types == ["sources", "token", "token", "token", "done"]
    assert [e["text"] for e in events
            if e["type"] == "token"] == ["Hel", "lo ", "world"]
    done = next(e for e in events if e["type"] == "done")
    assert done["answer"] == "Hello world"
    assert done["sources"][0]["title"] == "Doc X"
    assert done["tokens"] == 7


def test_bare_cryptorag_defaults_to_chat():
    from crypto_rag import cli

    parser = cli.build_parser()
    args = parser.parse_args([])
    assert args.func is cli.cmd_chat


def test_prompt_history_included():
    from crypto_rag.prompt import build_user_prompt

    prompt = build_user_prompt(
        "and 8 rounds?",
        "CTX",
        history=[{"role": "user", "content": "linear crypto?"},
                 {"role": "assistant", "content": "data complexity 2^43"}],
    )
    assert "CONVERSATION SO FAR" in prompt
    assert "linear crypto?" in prompt
    assert "Current question: and 8 rounds?" in prompt


def _make_pipeline_with_llm(llm):
    from crypto_rag.config import get_settings
    from crypto_rag.embeddings import build_embedding_provider
    from crypto_rag.pipeline import RAGPipeline
    from crypto_rag.vector_store import VectorStore

    settings = get_settings()
    provider = build_embedding_provider(settings)
    store = VectorStore(settings, provider)
    return RAGPipeline(settings, store, provider, llm)


def test_translate_query_retries_when_model_echoes_input():
    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return ("bana küp atağı anlat"
                    if self.calls == 1 else "Explain the cube attack")

    pipe = _make_pipeline_with_llm(FlakyLLM())
    assert pipe._translate_query("bana küp atağı anlat") == \
        "Explain the cube attack"


def test_translate_query_falls_back_to_original():
    class EchoLLM:
        def generate(self, prompt, **kwargs):
            return "bana küp atağı anlat"

    pipe = _make_pipeline_with_llm(EchoLLM())
    assert pipe._translate_query("bana küp atağı anlat") == \
        "bana küp atağı anlat"


def test_acceptable_translation_heuristic():
    from crypto_rag.pipeline import RAGPipeline

    assert RAGPipeline._acceptable_translation("", "x") is False
    assert RAGPipeline._acceptable_translation(
        "bana küp atağı anlat", "bana küp atağı anlat") is False
    assert RAGPipeline._acceptable_translation(
        "küp atağı", "bana küp atağı anlat") is False  # still Turkish
    assert RAGPipeline._acceptable_translation(
        "Explain the cube attack", "bana küp atağı anlat") is True

