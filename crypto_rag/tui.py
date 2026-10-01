"""Interactive terminal chat UI for CryptoRAG, built with Textual.

Run with::

    cryptorag                 # bare command launches the TUI
    python -m crypto_rag.cli chat

Features: streamed answers with inline sources, multi-turn chat memory,
on-the-fly model / Top-K / document-type controls, session save & load,
clipboard copy, generation cancellation, and answer statistics (model, time,
tokens, sources).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import requests

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Input, Label, Markdown, Select, Static

from .config import get_settings, PROJECT_ROOT
from .embeddings import build_embedding_provider
from .models import build_llm, list_chat_models
from .pipeline import RAGPipeline
from .vector_store import VectorStore

SESSION_DIR = PROJECT_ROOT / "data" / "sessions"

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

Markdown.md-user, Markdown.md-assistant, Markdown.md-info {
    margin: 1 2;
    padding: 1 2;
    width: 1fr;
    height: auto;
}
Markdown.md-user { border: round #00f2fe; background: #0b2430; }
Markdown.md-assistant { border: round #1e293b; background: #121824; }
Markdown.md-info { border: dashed #334155; background: #0d1420; }
Static.note {
    margin: 1 2;
    padding: 1 2;
    width: 1fr;
    background: #0d1420;
    border: dashed #334155;
}
.thinking { color: #94a3b8; text-style: italic; }
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


def _now() -> str:
    return datetime.now().strftime("%H:%M")


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
        ("ctrl+c", "cancel_generation", "Cancel"),
        ("escape", "cancel_generation", "Cancel"),
        ("ctrl+y", "copy_last", "Copy"),
    ]
    TITLE = "CryptoRAG — Terminal"

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.pipeline: Optional[RAGPipeline] = None
        self.top_k: int = self.settings.default_top_k
        self.doc_filter: str = "All"
        self.current_model: str = self.settings.llm_model
        self._busy = False
        self._cancel_requested = False
        self._ask_worker: Any = None

        self.history: List[dict] = []
        self._last_answer: str = ""
        self._stream_header: str = ""
        self._stream_markdown: Optional[Markdown] = None
        self._stream_buf: List[str] = []
        self._stream_flush_scheduled = False

        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ setup
    def compose(self) -> ComposeResult:
        model_options = []
        seen = set()
        for mid, name, _ptype in list_chat_models():
            model_options.append((name, mid))
            seen.add(mid)
        for m in _ollama_models(self.settings.ollama_base_url,
                                self.settings.embedding_model):
            if m not in seen:
                model_options.append((m, m))
                seen.add(m)
        if not model_options:
            model_options = [(self.settings.llm_model, self.settings.llm_model)]
        default_model = (self.settings.llm_model
                         if self.settings.llm_model in seen
                         else model_options[0][1])
        self.current_model = default_model

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
                "Ask a question below, or type /help for commands.\n"
                "Answers stream in as they are generated.",
                id="welcome", classes="note",
            )
        yield Input(placeholder="Ask a question or type /help…",
                    id="qinput")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#qinput", Input).focus()
        try:
            provider = build_embedding_provider(self.settings)
            store = VectorStore(self.settings, provider)
            llm = build_llm()
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
        self.current_model = model
        if self.pipeline is not None:
            self.pipeline.llm = build_llm(model)
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
        self._cancel_requested = False
        self.history.append({"role": "user", "content": question})
        self._append_message("user", question)
        self._append_thinking()
        self._ask_worker = self.run_worker(
            lambda q=question: self._answer_worker(q),
            thread=True, group="ask",
        )

    def _answer_worker(self, question: str) -> None:
        started = time.monotonic()
        buf: List[str] = []
        sources: List[dict] = []
        final: dict = {}
        try:
            events = self.pipeline.query_stream(
                question, top_k=self.top_k,
                doc_type_filter=self.doc_filter,
                history=list(self.history[:-1]))
            for event in events:
                if self._cancel_requested:
                    break
                et = event["type"]
                if et == "sources":
                    sources = event["sources"]
                elif et == "token":
                    buf.append(event["text"])
                    self.call_from_thread(self._on_stream_token, event["text"])
                elif et == "done":
                    final = event
        except Exception as exc:
            self.call_from_thread(self._stream_finished,
                                  {"error": str(exc)})
            return

        answer = "".join(buf).strip()
        if final.get("answer"):
            answer = final["answer"]
        self.call_from_thread(self._stream_finished, {
            "answer": answer,
            "sources": sources or final.get("sources", []),
            "tokens": getattr(self.pipeline.llm, "last_tokens", None),
            "elapsed": time.monotonic() - started,
            "cancelled": self._cancel_requested,
        })

    # -------------------------------------------------------------- streaming
    def _on_stream_token(self, text: str) -> None:
        self._stream_buf.append(text)
        if self._stream_flush_scheduled:
            return
        self._stream_flush_scheduled = True
        self.set_timer(0.05, self._render_stream)

    def _render_stream(self) -> None:
        self._stream_flush_scheduled = False
        if self._stream_markdown is None:
            return
        body = "".join(self._stream_buf)
        self._stream_markdown.update(
            f"{self._stream_header}\n\n{body}" if body else
            f"{self._stream_header}\n\n_Thinking…_")
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
    def _stream_finished(self, info: dict) -> None:
        self._stream_flush_scheduled = False
        if info.get("error"):
            self._finish_thinking(f"Query failed: {info['error']}",
                                  is_error=True)
            self._busy = False
            return

        answer = info.get("answer") or "(No answer produced.)"
        sources: List[dict] = info.get("sources", [])
        cancelled = bool(info.get("cancelled"))

        if cancelled:
            if answer.strip():
                answer += "\n\n_Generation cancelled._"
            else:
                answer = "_Generation cancelled._"
            self._finish_thinking(answer, sources=sources)
        else:
            self._finish_thinking(answer, sources=sources, stats=info)

        if not answer.startswith("_Generation cancelled._"):
            self.history.append({"role": "assistant", "content": answer})
            self._last_answer = answer
        self._busy = False
        self._refresh_stats()
        self.query_one("#qinput", Input).focus()

    def action_cancel_generation(self) -> None:
        if not self._busy or self._cancel_requested:
            return
        self._cancel_requested = True
        if self._ask_worker is not None:
            self._ask_worker.cancel()
        self._info("Cancelling generation…")

    def action_copy_last(self) -> None:
        if not self._last_answer:
            self._info("Nothing to copy yet.")
            return
        self.copy_to_clipboard(self._last_answer)
        self._info("Last answer copied to the clipboard.")

    # -------------------------------------------------------------- messages
    def _append_message(self, role: str, text: str) -> Markdown:
        chat = self.query_one("#chat", VerticalScroll)
        me = "You" if role == "user" else "CryptoRAG"
        cls = "md-user" if role == "user" else "md-assistant"
        md = Markdown(f"**{_now()} {me}**\n\n{text}", classes=cls)
        chat.mount(md)
        chat.scroll_end(animate=False)
        return md

    def _append_thinking(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        self._stream_buf = []
        self._stream_flush_scheduled = False
        self._stream_header = f"**{_now()} CryptoRAG**"
        self._stream_markdown = Markdown(
            f"{self._stream_header}\n\n_Thinking…_", classes="md-assistant")
        chat.mount(self._stream_markdown)
        chat.scroll_end(animate=False)

    def _finish_thinking(self, answer: str,
                         sources: Optional[List[dict]] = None,
                         stats: Optional[dict] = None,
                         is_error: bool = False) -> None:
        if self._stream_markdown is None:
            return
        header = getattr(self, "_stream_header", None) or f"**{_now()} CryptoRAG**"
        if is_error:
            body = f"**{header}**\n\n*{answer}*"
        else:
            answer = answer or "(No answer produced.)"
            parts = [header, "", answer]
            if sources:
                parts += ["", "---", "", "**Sources:**",
                          self._format_sources(sources)]
            if stats:
                parts += ["", f"*{self._format_stats(stats)}*"]
            body = "\n".join(parts)
        self._stream_markdown.update(body)
        self._stream_markdown = None
        self._stream_header = ""
        self.query_one("#chat", VerticalScroll).scroll_end(animate=True)

    def _format_sources(self, sources: List[dict]) -> str:
        lines = []
        for s in sources:
            sim = (f", sim={s['similarity']:.3f}"
                   if s.get("similarity") is not None else "")
            dtype = s.get("type", "")
            lines.append(f"- `[{s['id']}]` **{s['title']}**  "
                         f"({dtype}: {s['source']}{sim})")
        return "\n".join(lines)

    def _format_stats(self, info: dict) -> str:
        tokens = info.get("tokens")
        tok = f"{tokens} tokens" if tokens else "tokens: –"
        return (f"model: {self.current_model}    "
                f"{info.get('elapsed', 0):.1f}s    "
                f"{tok}    {len(info.get('sources', []))} sources")

    # --------------------------------------------------------------- commands
    def _run_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if cmd in ("/help", "/?"):
            self._info(
                "Commands:\n"
                "  /help            show this help\n"
                "  /stats           knowledge-base statistics\n"
                "  /new             start a fresh session (clears memory)\n"
                "  /clear           clear the chat & memory\n"
                "  /save            persist the session to disk\n"
                "  /load            restore the saved session\n"
                "  /copy            copy the last answer\n"
                "  /ingest-rfc N    fetch and index IETF RFC N\n"
                "  /ingest-url URL  scrape and index a web page\n"
                "  /ingest-file P   index a local PDF/TXT/MD file\n"
                "  /quit            exit\n\n"
                "Shortcuts: ctrl+q quit · ctrl+l clear · ctrl+c/esc cancel · "
                "ctrl+y copy.\n"
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
        elif cmd in ("/clear", "/new"):
            self.history = []
            self._last_answer = ""
            chat = self.query_one("#chat", VerticalScroll)
            chat.remove_children()
            chat.mount(Static("New session. Ask away!", classes="note"))
        elif cmd == "/save":
            self._save_session(arg)
        elif cmd == "/load":
            self._load_session(arg)
        elif cmd == "/copy":
            self.action_copy_last()
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

    def _session_path(self, name: str = "") -> Path:
        stem = name.strip().lower() or "default"
        if not stem.endswith(".json"):
            stem += ".json"
        return SESSION_DIR / stem

    def _save_session(self, name: str = "") -> None:
        payload = {
            "model": self.current_model,
            "top_k": self.top_k,
            "doc_filter": self.doc_filter,
            "history": self.history,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        path = self._session_path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._info(f"Session saved to {path.name} "
                       f"({len(self.history)} turns).")
        except Exception as exc:
            self._info(f"/save failed: {exc}")

    def _load_session(self, name: str = "") -> None:
        path = self._session_path(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._info(f"No saved session: {path.name}")
            return
        except Exception as exc:
            self._info(f"/load failed: {exc}")
            return

        self.current_model = payload.get("model", self.current_model)
        self.top_k = int(payload.get("top_k", self.top_k))
        self.doc_filter = payload.get("doc_filter", self.doc_filter)
        self.history = payload.get("history", [])

        chat = self.query_one("#chat", VerticalScroll)
        chat.remove_children()
        if not self.history:
            chat.mount(Static("Session is empty. Ask away!",
                              classes="note"))
        for turn in self.history:
            role = turn.get("role", "user")
            self._append_message(role, turn.get("content", ""))
            if role == "assistant":
                self._last_answer = turn.get("content", "")
        self._info(f"Loaded {len(self.history)} turns from {path.name}.")

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
        self.query_one("#qinput", Input).focus()

    # --------------------------------------------------------------- helpers
    def _info(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        md = Markdown(f"**{_now()} CryptoRAG**\n\n{text}", classes="md-info")
        chat.mount(md)
        chat.scroll_end(animate=False)

    def action_clear_chat(self) -> None:
        self.history = []
        self._last_answer = ""
        chat = self.query_one("#chat", VerticalScroll)
        chat.remove_children()
        chat.mount(Static("Chat cleared. Ask away!", classes="note"))


def run() -> None:
    ChatApp().run()
