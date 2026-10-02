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

import chirag.tui as tui_mod
from chirag.tui import ChatApp
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
    from chirag.config import get_settings
    from chirag.embeddings import build_embedding_provider
    from chirag.pipeline import RAGPipeline
    from chirag.vector_store import VectorStore

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


def test_bare_chirag_defaults_to_chat():
    from chirag import cli

    parser = cli.build_parser()
    args = parser.parse_args([])
    assert args.func is cli.cmd_chat


def test_prompt_history_included():
    from chirag.prompt import build_user_prompt

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
    from chirag.config import get_settings
    from chirag.embeddings import build_embedding_provider
    from chirag.pipeline import RAGPipeline
    from chirag.vector_store import VectorStore

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
    from chirag.pipeline import RAGPipeline

    assert RAGPipeline._acceptable_translation("", "x") is False
    assert RAGPipeline._acceptable_translation(
        "bana küp atağı anlat", "bana küp atağı anlat") is False
    assert RAGPipeline._acceptable_translation(
        "küp atağı", "bana küp atağı anlat") is False  # still Turkish
    assert RAGPipeline._acceptable_translation(
        "Explain the cube attack", "bana küp atağı anlat") is True


# --------------------------------------------------------------- documents
DOCS_STUB = [
    {"source": "RFC 1", "title": "RFC 1: TLS 1.3", "type": "RFC Document",
     "url": "", "chunk_count": 500},
    {"source": "RFC 2", "title": "RFC 2: HKDF", "type": "RFC Document",
     "url": "", "chunk_count": 25},
    {"source": "SP 1", "title": "SP 800-38D GCM", "type": "NIST Standard",
     "url": "", "chunk_count": 85},
    {"source": "a.pdf", "title": "Linear Cryptanalysis", "type": "Cryptanalysis Paper",
     "url": "", "chunk_count": 120},
]


class StubDocsPipeline:
    def list_documents(self):
        return [dict(d) for d in DOCS_STUB]


async def _open_docs(app, pilot):
    from chirag.tui import DocumentsScreen
    app.push_screen(DocumentsScreen(StubDocsPipeline()))
    for _ in range(30):
        await pilot.pause(0.05)
        if app.screen.query("#docs-table"):
            return


def _table(app):
    from textual.widgets import DataTable
    return app.screen.query_one("#docs-table", DataTable)


def test_documents_screen_lists_rows():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            await _open_docs(app, pilot)
            table = _table(app)
            assert table.row_count == 4
            # default sort: chunks descending -> TLS 1.3 first
            first = table.get_row_at(0)
            assert first[0] == "1"
            assert first[1] == "RFC 1: TLS 1.3"
            assert first[3] == "500"
            assert app.screen.query_one("#docs-type", Select)
            assert app.screen.query_one("#docs-search", Input)

    _run(main())


def test_documents_screen_type_filter():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            await _open_docs(app, pilot)
            sel = app.screen.query_one("#docs-type", Select)
            sel.value = "RFC Document"
            await pilot.pause()
            assert _table(app).row_count == 2

    _run(main())


def test_documents_screen_search():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            await _open_docs(app, pilot)
            search = app.screen.query_one("#docs-search", Input)
            search.value = "gcm"
            await pilot.pause()
            table = _table(app)
            assert table.row_count == 1
            assert table.get_row_at(0)[1] == "SP 800-38D GCM"

    _run(main())


def test_documents_screen_sorts_on_header_click():
    from rich.text import Text
    from textual.widgets import DataTable

    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            await _open_docs(app, pilot)
            table = _table(app)
            # simulate clicking the "Başlık" header -> screen sorts ascending
            table.post_message(DataTable.HeaderSelected(
                table, "title", 1, Text("Başlık")))
            await pilot.pause()
            rows = [table.get_row_at(i)[1] for i in range(table.row_count)]
            assert rows == ["Linear Cryptanalysis", "RFC 1: TLS 1.3",
                            "RFC 2: HKDF", "SP 800-38D GCM"]

    _run(main())


def test_documents_command_opens_screen():
    from chirag.tui import DocumentsScreen

    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            app.pipeline = StubDocsPipeline()
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/documents"
            await qinput.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, DocumentsScreen)

    _run(main())


def test_documents_screen_closes():
    from chirag.tui import DocumentsScreen

    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            await _open_docs(app, pilot)
            assert isinstance(app.screen, DocumentsScreen)
            app.screen.action_close_docs()
            await pilot.pause()
            assert not isinstance(app.screen, DocumentsScreen)

    _run(main())


# ------------------------------------------------------------ delete feature
class StubStore:
    def __init__(self, docs):
        self.docs = [dict(d) for d in docs]
        self.deleted: list = []

    def list_documents(self):
        return [dict(d) for d in self.docs]

    def delete_document(self, source):
        before = len(self.docs)
        self.docs = [d for d in self.docs if d["source"] != source]
        self.deleted.append(source)
        return len(self.docs) < before


class StubDeletePipeline:
    def __init__(self, docs=None):
        self.store = StubStore(docs if docs is not None else DOCS_STUB)

    def list_documents(self):
        return self.store.list_documents()


async def _open_delete_docs(app, pilot, docs=None):
    from chirag.tui import DocumentsScreen
    pipe = StubDeletePipeline(docs)
    app.push_screen(DocumentsScreen(pipe))
    for _ in range(30):
        await pilot.pause(0.05)
        if app.screen.query("#docs-table"):
            return pipe
    return pipe


def test_delete_requires_two_presses():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            pipe = await _open_delete_docs(app, pilot)
            screen = app.screen
            # first press -> confirm only, nothing deleted yet
            screen.action_delete_document()
            await pilot.pause()
            assert screen._confirm_source is not None
            assert pipe.store.deleted == []
            assert _table(app).row_count == 4
            assert "DİKKAT" in str(
                screen.query_one("#docs-summary", Static).render())

    _run(main())


def test_delete_removes_row_after_confirmation():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            pipe = await _open_delete_docs(app, pilot)
            screen = app.screen
            screen.action_delete_document()   # confirm
            screen.action_delete_document()   # execute
            await pilot.pause()
            assert pipe.store.deleted == ["RFC 1"]  # default sort: TLS 1.3 first
            assert _table(app).row_count == 3
            assert screen._confirm_source is None
            assert "Silindi" in str(screen.query_one("#docs-summary", Static).render())

    _run(main())


def test_escape_cancels_delete_confirmation():
    from chirag.tui import DocumentsScreen

    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            pipe = await _open_delete_docs(app, pilot)
            screen = app.screen
            screen.action_delete_document()   # confirm
            screen.action_close_docs()        # esc -> cancel, keep screen
            await pilot.pause()
            assert pipe.store.deleted == []
            assert isinstance(app.screen, DocumentsScreen)
            assert screen._confirm_source is None
            assert _table(app).row_count == 4

    _run(main())


def test_delete_removes_local_file(tmp_path, monkeypatch):
    async def main():
        monkeypatch.setattr(tui_mod, "CORPUS_DIR", tmp_path)
        (tmp_path / "RFC 1").write_text("x")  # source == filename
        app = ChatApp()
        async with app.run_test() as pilot:
            pipe = await _open_delete_docs(app, pilot)
            screen = app.screen
            screen.action_delete_document()
            screen.action_delete_document()
            await pilot.pause()
            assert pipe.store.deleted == ["RFC 1"]
            assert not (tmp_path / "RFC 1").exists()
            assert "Silindi" in str(screen.query_one("#docs-summary", Static).render())

    _run(main())


def test_delete_does_not_touch_missing_file(tmp_path, monkeypatch):
    async def main():
        monkeypatch.setattr(tui_mod, "CORPUS_DIR", tmp_path)
        app = ChatApp()
        async with app.run_test() as pilot:
            pipe = await _open_delete_docs(app, pilot)
            screen = app.screen
            screen.action_delete_document()
            screen.action_delete_document()
            await pilot.pause()
            assert pipe.store.deleted == ["RFC 1"]
            assert not (tmp_path / "RFC 1").exists()  # no error, no file

    _run(main())

