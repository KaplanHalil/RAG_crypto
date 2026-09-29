"""Interactive terminal chat UI for CryptoRAG, built with Textual.

Run with::

    python -m crypto_rag.cli chat

Features: ask questions, switch local Ollama model on the fly, set Top-K,
filter by document type, see inline sources, ingest RFCs/URLs/files, inspect
statistics, and keep an in-session chat history.
"""

from __future__ import annotations

from typing import Any, List, Optional

import requests

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Input, Label, Markdown, Select, Static

from .config import get_settings
from .embeddings import build_embedding_provider
from .llm import OllamaLLM
from .pipeline import RAGPipeline
from .vector_store import VectorStore

CSS = """
Screen {
    background: #0a0d14;
}

#settingsbar {
    height: auto;
    padding: 0 1;
    background: #121824;
    border-bottom: solid #1e293b;
    align: left middle;
}
#settingsbar .lbl {
    color: #94a3b8;
    margin: 0 0 0 1;
    padding: 0 0 0 1;
}
#settingsbar Select, #settingsbar Input {
    min-width: 22;
    margin: 0 0 0 1;
}
Input#topk { min-width: 6; max-width: 8; }

#statsbar {
    height: 1;
    color: #64748b;
    background: #0f1522;
    padding: 0 2;
    text-style: italic;
}

#chat { background: #0a0d14; }

.bubble {
    margin: 1 2;
    padding: 1 2;
    width: 1fr;
}
.bubble.user {
    border: round #00f2fe;
    background: #0b2430;
    dock: right;
}
.bubble.assistant {
    border: round #1e293b;
    background: #121824;
    width: 1fr;
}
.bubble.info {
    border: dashed #334155;
    background: #0d1420;
    width: 1fr;
}
.role {
    text-style: bold;
    margin-bottom: 1;
}
.role.user { color: #00f2fe; }
.role.assistant { color: #4facfe; }
.role.info { color: #ffb703; }
.thinking { color: #94a3b8; text-style: italic; }
.sources {
    margin-top: 1;
    padding-top: 1;
    border-top: solid #1e293b;
    color: #94a3b8;
}
.source-line { color: #94a3b8; }
.error { color: #f87171; }
#qinput {
    dock: bottom;
    margin: 0 1 1 1;
    border: round #1e293b;
    background: #121824;
}
"""

DOC_TYPES = ["All", "RFC Document", "NIST Standard", "Cryptanalysis Paper",
             "Cryptography Paper", "Local Document"]


def _ollama_models(base_url: str, embedding_model: str) -> List[str]:
    """List chat models available from the local Ollama server."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = []
            for m in resp.json().get("models", []):
                name = m.get("name", "")
                if name == embedding_model or "embed" in name.lower():
                    continue
                models.append(name)
            return models
    except requests.RequestException:
        pass
    return []


class ChatApp(App):
    """CryptoRAG terminal chat."""

    CSS = CSS
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear chat"),
    ]
    TITLE = "CryptoRAG — Terminal"

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.pipeline: Optional[RAGPipeline] = None
        self.top_k: int = self.settings.default_top_k
        self.doc_filter: str = "All"
        self._busy = False

    # ------------------------------------------------------------------ setup
    def compose(self) -> ComposeResult:
        models = _ollama_models(self.settings.ollama_base_url,
                                self.settings.embedding_model)
        if not models:
            models = [self.settings.llm_model]
        model_options = [(m, m) for m in models]
        default_model = (self.settings.llm_model
                         if self.settings.llm_model in models else models[0])

        with Horizontal(id="settingsbar"):
            yield Label("Model", classes="lbl")
            yield Select(model_options, value=default_model,
                         id="model-select")
            yield Label("Top-K", classes="lbl")
            yield Input(str(self.top_k), id="topk", type="integer")
            yield Label("Type", classes="lbl")
            yield Select([(t, t) for t in DOC_TYPES], value="All",
                         id="filter-select")
        yield Static("", id="statsbar")
        with VerticalScroll(id="chat"):
            yield Static(
                "Welcome to CryptoRAG.\n"
                "Ask a question below, or type /help for commands.",
                id="welcome", classes="bubble info",
            )
        yield Input(placeholder="Ask a question or type /help…",
                    id="qinput")
        yield Footer()

    def on_mount(self) -> None:
        try:
            provider = build_embedding_provider(self.settings)
            store = VectorStore(self.settings, provider)
            llm = OllamaLLM(self.settings.llm_model,
                            self.settings.ollama_base_url)
            self.pipeline = RAGPipeline(self.settings, store, provider, llm)
        except Exception as exc:
            self._info(f"Failed to initialize the RAG pipeline: {exc}")
            return
        self._refresh_stats()

    # -------------------------------------------------------------- settings
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select":
            self._set_model(event.value)
        elif event.select.id == "filter-select":
            self.doc_filter = event.value

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "topk":
            try:
                self.top_k = max(1, min(20, int(event.value or self.top_k)))
            except ValueError:
                pass

    def _set_model(self, model: str) -> None:
        if self.pipeline is not None:
            self.pipeline.llm = OllamaLLM(model, self.settings.ollama_base_url)
        if self.settings.llm_model != model:
            self._info(f"Switched to model: {model}")

    def _refresh_stats(self) -> None:
        try:
            s = self.pipeline.stats() if self.pipeline else {}
            self.query_one("#statsbar", Static).update(
                f"chunks: {s.get('total_chunks', 0)}  "
                f"documents: {s.get('total_documents', 0)}  "
                f"RFCs: {s.get('rfc_count', 0)}  "
                f"NIST: {s.get('nist_count', 0)}  "
                f"papers: {s.get('paper_count', 0)}"
            )
        except Exception:
            pass

    # ---------------------------------------------------------------- input
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "qinput":
            return
        raw = event.value.strip()
        event.input.value = ""
        if not raw or self._busy:
            return
        if raw.startswith("/"):
            self._run_command(raw)
        else:
            self._ask(raw)

    def _ask(self, question: str) -> None:
        self._busy = True
        self._append_message("user", question)
        self._append_thinking()
        self.run_worker(lambda: self._answer_worker(question),
                        thread=True, group="ask")

    def _answer_worker(self, question: str) -> None:
        try:
            result = self.pipeline.query(question, top_k=self.top_k,
                                         doc_type_filter=self.doc_filter)
            error = None
        except Exception as exc:
            result, error = {}, str(exc)
        self.call_from_thread(self._deliver, result, error)

    def _deliver(self, result: dict, error: Optional[str]) -> None:
        if error:
            self._finish_thinking(f"Query failed: {error}", is_error=True)
        else:
            answer = result.get("answer")
            if not answer:
                answer = "(No answer produced.)"
            sources = result.get("sources", [])
            self._finish_thinking(answer, sources)
        self._busy = False

    # -------------------------------------------------------------- messages
    def _append_message(self, role: str, text: str) -> Vertical:
        chat = self.query_one("#chat", VerticalScroll)
        label_text = "You" if role == "user" else "CryptoRAG"
        bubble = Vertical(Label(label_text, classes=f"role {role}"),
                          Markdown(text),
                          classes=f"bubble {role}")
        chat.mount(bubble)
        chat.scroll_end(animate=False)
        return bubble

    def _append_thinking(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        self._thinking = Vertical(
            Label("CryptoRAG", classes="role assistant"),
            Label("Thinking…", classes="thinking"),
            classes="bubble assistant",
        )
        chat.mount(self._thinking)
        chat.scroll_end(animate=False)

    def _finish_thinking(self, answer: str,
                         sources: Optional[List[dict]] = None) -> None:
        self._thinking.remove_children()
        if answer.lower().startswith("query failed"):
            self._thinking.mount(Label(answer, classes="error"))
        else:
            self._thinking.mount(Markdown(answer))
            if sources:
                lines = [f"[{s['id']}] {s['title']}  "
                         f"(file: {s['source']}"
                         + (f", sim={s['similarity']:.3f}"
                            if s.get("similarity") is not None else "")
                         + ")" for s in sources]
                self._thinking.mount(Static(
                    "\n".join(lines), classes="sources"))
        self.query_one("#chat", VerticalScroll).scroll_end(animate=True)

    # --------------------------------------------------------------- commands
    def _run_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if cmd in ("/help", "/?"):
            self._info(
                "Commands:\n"
                "  /help            show this help\n"
                "  /stats           knowledge-base statistics\n"
                "  /clear           clear the chat history\n"
                "  /ingest-rfc N    fetch and index IETF RFC N\n"
                "  /ingest-url URL  scrape and index a web page\n"
                "  /ingest-file P   index a local PDF/TXT/MD file\n"
                "  /quit            exit\n\n"
                "Shortcuts: ctrl+q quit, ctrl+l clear.\n"
                "Select the model, Top-K and document type in the top bar."
            )
        elif cmd == "/stats":
            try:
                stat = self.pipeline.stats() if self.pipeline else {}
                self._info(
                    f"chunks: {stat.get('total_chunks')}\n"
                    f"documents: {stat.get('total_documents')}\n"
                    f"RFCs: {stat.get('rfc_count')}\n"
                    f"NIST: {stat.get('nist_count')}\n"
                    f"papers: {stat.get('paper_count')}"
                )
                self._refresh_stats()
            except Exception as exc:
                self._info(f"/stats failed: {exc}")
        elif cmd == "/clear":
            self.query_one("#chat", VerticalScroll).remove_children()
            self.query_one("#chat", VerticalScroll).mount(
                Static("Chat cleared. Ask away!", classes="bubble info"))
        elif cmd == "/quit" or cmd == "/exit":
            self.exit()
        elif cmd in ("/ingest-rfc", "/ingest-url", "/ingest-file"):
            if not arg:
                self._info(f"Usage: {cmd} <argument>")
                return
            self._busy = True
            self._append_message("user", f"`{raw}`")
            if cmd == "/ingest-rfc":
                job = self._ingest_rfc
            elif cmd == "/ingest-url":
                job = self._ingest_url
            else:
                job = self._ingest_file
            self.run_worker(lambda a=arg: job(a), thread=True, group="ingest")
        else:
            self._info(f"Unknown command: {cmd} (try /help)")

    def _ingest_rfc(self, rfc: str) -> None:
        try:
            result = self.pipeline.ingest_rfc(int(rfc))
            msg = (f"Ingested {result.get('source')} — "
                   f"{result.get('chunks_added')} chunks.")
        except Exception as exc:
            msg = f"/ingest-rfc failed: {exc}"
        self.call_from_thread(self._info_done, msg)

    def _ingest_url(self, url: str) -> None:
        try:
            result = self.pipeline.ingest_url(url)
            msg = (f"Ingested '{result.get('title')}' — "
                   f"{result.get('chunks_added')} chunks.")
        except Exception as exc:
            msg = f"/ingest-url failed: {exc}"
        self.call_from_thread(self._info_done, msg)

    def _ingest_file(self, path: str) -> None:
        try:
            result = self.pipeline.ingest_file(path)
            msg = (f"Ingested '{result.get('title')}' — "
                   f"{result.get('chunks_added')} chunks.")
        except Exception as exc:
            msg = f"/ingest-file failed: {exc}"
        self.call_from_thread(self._info_done, msg)

    def _info_done(self, msg: str) -> None:
        self._info(msg)
        self._busy = False
        self._refresh_stats()

    # --------------------------------------------------------------- helpers
    def _info(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        bubble = Vertical(Label("CryptoRAG", classes="role info"),
                          Markdown(text),
                          classes="bubble info")
        chat.mount(bubble)
        chat.scroll_end(animate=False)

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        chat.remove_children()
        chat.mount(Static("Chat cleared. Ask away!", classes="bubble info"))


def run() -> None:
    ChatApp().run()
