"""HTTP API server for CryptoRAG (pure stdlib, no extra dependencies).

Two interfaces on one port:

* a small JSON REST API (``/api/...``) for querying, ingesting and stats, and
* an OpenAI-compatible ``/v1/chat/completions`` endpoint so any tool (opencode,
  curl, custom apps) can use CryptoRAG as a chat model.

Run with::

    cryptorag serve --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from .config import get_settings, PROJECT_ROOT
from .embeddings import build_embedding_provider
from .llm import build_llm_provider
from .models import build_llm, get_registry
from .pipeline import RAGPipeline
from .vector_store import VectorStore

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


def _build_pipeline() -> RAGPipeline:
    settings = get_settings()
    provider = build_embedding_provider(settings)
    store = VectorStore(settings, provider)
    llm = build_llm_provider(settings)
    return RAGPipeline(settings, store, provider, llm)


class CryptoRAGHandler(BaseHTTPRequestHandler):
    """Serves the REST + OpenAI-compatible API for one pipeline instance."""

    pipeline: Optional[RAGPipeline] = None  # set via make_handler()
    server_version = "CryptoRAG/1.0"

    # --------------------------------------------------------------- helpers
    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_event(self, data: str) -> None:
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[api] {self.address_string()} - {fmt % args}")

    # ---------------------------------------------------------------- routing
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        try:
            if path in ("/", "/api"):
                self._handle_index()
            elif path == "/api/stats":
                self._handle_stats()
            elif path == "/api/models":
                self._handle_models()
            elif path == "/api/config":
                self._handle_config()
            elif path == "/api/documents":
                self._handle_documents()
            elif path == "/v1/models":
                self._handle_openai_models()
            else:
                self._send_error_json(404, f"Not found: {path}")
        except Exception as exc:
            self._send_error_json(500, str(exc))

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        try:
            if path == "/api/documents":
                source = (query.get("source") or [""])[0]
                if not source:
                    self._send_error_json(400, "Missing ?source=...")
                    return
                self._send_json(200, {"deleted": self.pipeline.store
                                      .delete_document(source), "source": source})
            else:
                self._send_error_json(404, f"Not found: {path}")
        except Exception as exc:
            self._send_error_json(500, str(exc))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/query":
                self._handle_query(self._read_json(), stream=False)
            elif path == "/api/query/stream":
                self._handle_query(self._read_json(), stream=True)
            elif path == "/api/ingest/rfc":
                self._handle_ingest_rfc()
            elif path == "/api/ingest/url":
                self._handle_ingest_url()
            elif path == "/api/ingest/file":
                self._handle_ingest_file()
            elif path == "/api/summarize":
                self._handle_summarize()
            elif path == "/v1/chat/completions":
                self._handle_chat_completions(self._read_json())
            else:
                self._send_error_json(404, f"Not found: {path}")
        except Exception as exc:
            self._send_error_json(500, str(exc))

    # ---------------------------------------------------------------- REST
    def _handle_index(self) -> None:
        self._send_json(200, {
            "name": "CryptoRAG API",
            "endpoints": [
                "GET  /api/stats",
                "GET  /api/models",
                "GET  /api/config",
                "GET  /api/documents",
                "DELETE /api/documents?source=...",
                "POST /api/query",
                "POST /api/query/stream",
                "POST /api/ingest/rfc",
                "POST /api/ingest/url",
                "POST /api/ingest/file",
                "POST /api/summarize",
                "GET  /v1/models",
                "POST /v1/chat/completions",
            ],
            "config_file": str(self.pipeline.settings.env_file)
            if self.pipeline else None,
        })

    def _handle_stats(self) -> None:
        self._send_json(200, self.pipeline.stats())

    def _handle_models(self) -> None:
        models = [{"id": mid, "name": name, "type": ptype}
                  for mid, name, ptype in get_registry().chat_models()]
        active = self.pipeline.llm.model if self.pipeline else None
        self._send_json(200, {"models": models, "active": active})

    def _handle_config(self) -> None:
        self._send_json(200, get_registry().effective(redact=True))

    def _handle_documents(self) -> None:
        self._send_json(200, {"documents": self.pipeline.store.list_documents()})

    def _handle_ingest_rfc(self) -> None:
        data = self._read_json()
        num = data.get("rfc_number")
        if not isinstance(num, int):
            self._send_error_json(400, "Need an integer rfc_number")
            return
        self._send_json(200, self.pipeline.ingest_rfc(num))

    def _handle_ingest_url(self) -> None:
        data = self._read_json()
        url = data.get("url", "")
        if not url:
            self._send_error_json(400, "Need a url")
            return
        self._send_json(200, self.pipeline.ingest_url(url))

    def _handle_ingest_file(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        query = parse_qs(urlparse(self.path).query)
        name = (query.get("name") or [""])[0]
        if not name:
            name = self.headers.get("X-Filename", "upload.pdf")
        if length <= 0:
            self._send_error_json(400, "Empty body")
            return
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        path = UPLOAD_DIR / os.path.basename(name)
        with open(path, "wb") as f:
            f.write(self.rfile.read(length))
        self._send_json(200, self.pipeline.ingest_file(str(path)))

    def _handle_summarize(self) -> None:
        data = self._read_json()
        source = data.get("source", "")
        if not source:
            self._send_error_json(400, "Need a source")
            return
        result = self.pipeline.summarize_document(source)
        if "error" in result:
            self._send_error_json(404, result["error"])
            return
        self._send_json(200, result)

    def _handle_query(self, data: Dict[str, Any], stream: bool) -> None:
        question = (data.get("question") or "").strip()
        if not question:
            self._send_error_json(400, "Need a question")
            return
        model = data.get("model")
        active = self._swap_model(model)
        try:
            if stream:
                self._stream_query(question, data, active)
            else:
                result = self.pipeline.query(
                    question,
                    top_k=data.get("top_k"),
                    doc_type_filter=data.get("doc_type_filter"),
                    history=data.get("history"),
                )
                self._send_json(200, result)
        finally:
            if active:
                self.pipeline.llm = active

    def _stream_query(self, question: str, data: Dict[str, Any],
                      active_llm: Any) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        for event in self.pipeline.query_stream(
                question, top_k=data.get("top_k"),
                doc_type_filter=data.get("doc_type_filter"),
                history=data.get("history")):
            self._send_event(json.dumps(event))
        if active_llm:
            self.pipeline.llm = active_llm

    def _swap_model(self, model: Optional[str]) -> Any:
        if not model or model == self.pipeline.llm.model:
            return None
        old = self.pipeline.llm
        self.pipeline.llm = build_llm(model)
        return old

    # ------------------------------------------------- openai-compatible
    def _handle_openai_models(self) -> None:
        data = []
        for mid, name, ptype in get_registry().chat_models():
            data.append({"id": mid, "object": "model",
                         "created": int(time.time()),
                         "owned_by": ptype})
        self._send_json(200, {"object": "list", "data": data})

    def _handle_chat_completions(self, data: Dict[str, Any]) -> None:
        messages: List[Dict[str, str]] = data.get("messages", [])
        if not messages:
            self._send_error_json(400, "Need messages")
            return
        user = [m for m in messages if m.get("role") == "user"]
        if not user:
            self._send_error_json(400, "Need at least one user message")
            return
        question = user[-1].get("content", "").strip()
        history = [{"role": m["role"], "content": m.get("content", "")}
                   for m in messages[:-1]
                   if m.get("role") in ("user", "assistant")]
        model = data.get("model") or get_registry().default_llm()
        stream = bool(data.get("stream"))
        active = self._swap_model(model)
        try:
            if stream:
                self._stream_chat_completion(model, question, history)
            else:
                result = self.pipeline.query(
                    question, history=history, doc_type_filter=None)
                self._send_json(200, self._openai_completion(
                    model, result.get("answer") or "", sources=result.get("sources", [])))
        finally:
            if active:
                self.pipeline.llm = active

    def _stream_chat_completion(self, model: str, question: str,
                                history: List[dict]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        first = True
        for event in self.pipeline.query_stream(
                question, history=history, doc_type_filter=None):
            et = event["type"]
            if et == "sources":
                self._send_event(json.dumps({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0,
                                 "delta": {"role": "assistant"},
                                 "finish_reason": None}],
                    "crypto_rag": {"sources": event["sources"]},
                }))
            elif et == "token":
                self._send_event(json.dumps({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0,
                                 "delta": {"content": event["text"]},
                                 "finish_reason": None}],
                }))
                first = False
            elif et == "done":
                self._send_event(json.dumps({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}],
                }))
                self._send_event("[DONE]")
                return

    def _openai_completion(self, model: str, content: str,
                           sources: List[dict]) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": None, "completion_tokens": None,
                      "total_tokens": None},
            "crypto_rag": {"sources": sources},
        }


def make_handler(pipeline: RAGPipeline) -> type:
    """Return a handler class bound to a concrete pipeline (testable)."""
    class _Bound(CryptoRAGHandler):
        pass
    _Bound.pipeline = pipeline
    return _Bound


def serve(host: str = "0.0.0.0", port: int = 8000,
          pipeline: Optional[RAGPipeline] = None) -> None:
    """Start the API server (blocking)."""
    pipeline = pipeline or _build_pipeline()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), make_handler(pipeline))
    print(f"[api] CryptoRAG API listening on http://{host}:{port}")
    print(f"[api] REST:  /api/query  ·  OpenAI-compatible: /v1/chat/completions")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] shutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
