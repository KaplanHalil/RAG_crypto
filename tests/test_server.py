"""Tests for the stdlib HTTP API server (chirag/server.py).

Runs the server on an ephemeral port with an injected stub pipeline, so no
network / models / ChromaDB are required.
"""

import json
import os
import sys
import threading
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import ThreadingHTTPServer

import chirag.models as models


class _StubLLM:
    model = "stub-model"  # class attr so instances share it

    def __init__(self):
        self.model = "stub-model"


class _StubStore:
    def list_documents(self):
        return [{"source": "a", "title": "A", "type": "Paper",
                 "url": "", "chunk_count": 3}]

    def delete_document(self, source):
        return source == "a"

    def get_stats(self):
        return {"total_chunks": 1, "total_documents": 1, "rfc_count": 0,
                "nist_count": 0, "paper_count": 1}


class _StubPipeline:
    def __init__(self):
        self.llm = _StubLLM()
        self.store = _StubStore()
        self.settings = type("S", (), {"env_file": "x"})

    def stats(self):
        return self.store.get_stats()

    def query(self, question, top_k=None, doc_type_filter=None,
              history=None):
        return {"answer": f"A:{question}", "sources": [], "retrieved_chunks": []}

    def query_stream(self, question, top_k=None, doc_type_filter=None,
                     history=None):
        yield {"type": "sources", "sources": [{
            "id": 1, "title": "T", "source": "s", "type": "Paper",
            "similarity": 0.9}], "context": ""}
        for piece in ("Hel", "lo"):
            yield {"type": "token", "text": piece}
        yield {"type": "done", "answer": "Hello", "sources": [],
               "retrieved_chunks": [], "tokens": 2}

    def summarize_document(self, source):
        return {"error": "nope"}

    def ingest_rfc(self, n):
        return {"success": True, "source": f"RFC {n}", "chunks_added": 1}

    def ingest_url(self, url):
        return {"success": True, "title": url, "chunks_added": 1}

    def ingest_file(self, path):
        return {"success": True, "title": path, "chunks_added": 1}


@pytest.fixture(autouse=True)
def _reset_registry():
    models.reset_registry()
    yield
    models.reset_registry()


@pytest.fixture()
def server_url():
    from chirag.server import make_handler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                make_handler(_StubPipeline()))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _request(url, data=None, method=None, headers=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def test_get_stats(server_url):
    status, body = _request(f"{server_url}/api/stats")
    assert status == 200
    assert json.loads(body)["total_documents"] == 1


def test_post_query(server_url):
    status, body = _request(f"{server_url}/api/query",
                            {"question": "What is AES?"})
    assert status == 200
    assert json.loads(body)["answer"] == "A:What is AES?"


def test_query_requires_question(server_url):
    status, body = _request(f"{server_url}/api/query", {"question": ""})
    assert status == 400


def test_query_stream_sse(server_url):
    status, body = _request(f"{server_url}/api/query/stream",
                            {"question": "hi"})
    assert status == 200
    assert "data: " in body
    assert '"text": "Hel"' in body
    assert '"type": "done"' in body


def test_documents_list_and_delete(server_url):
    status, body = _request(f"{server_url}/api/documents")
    assert json.loads(body)["documents"][0]["source"] == "a"
    status, body = _request(f"{server_url}/api/documents?source=a",
                            method="DELETE")
    assert json.loads(body)["deleted"] is True


def test_ingest_rfc(server_url):
    status, body = _request(f"{server_url}/api/ingest/rfc", {"rfc_number": 9180})
    assert json.loads(body)["source"] == "RFC 9180"


def test_openai_models_list(server_url):
    status, body = _request(f"{server_url}/v1/models")
    assert status == 200
    assert json.loads(body)["object"] == "list"


def test_openai_chat_completions_non_stream(server_url):
    status, body = _request(f"{server_url}/v1/chat/completions", {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "explain des"}]})
    data = json.loads(body)
    assert status == 200
    assert data["choices"][0]["message"]["content"] == "A:explain des"


def test_openai_chat_completions_stream(server_url):
    status, body = _request(f"{server_url}/v1/chat/completions", {
        "model": "stub-model", "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    assert 'data: [DONE]' in body
    assert '"content": "Hel"' in body
    assert '"finish_reason": "stop"' in body


def test_openai_chat_completions_requires_user(server_url):
    status, body = _request(f"{server_url}/v1/chat/completions",
                            {"messages": [{"role": "system",
                                           "content": "x"}]})
    assert status == 400


def test_unknown_route(server_url):
    status, _ = _request(f"{server_url}/api/nope")
    assert status == 404
