"""FastAPI web server for the CryptoRAG system.

Endpoints:
    GET  /                      single-page English web UI
    GET  /api/stats             knowledge-base statistics
    GET  /api/models            available Ollama chat models
    GET  /api/documents         list indexed documents
    DELETE /api/documents       delete one document (?source=...)
    POST /api/ingest/rfc        ingest IETF RFC by number
    POST /api/ingest/url        ingest a web page
    POST /api/ingest/file       upload a PDF/TXT/MD file
    POST /api/query             retrieval-augmented question answering
    POST /api/summarize         summarize an indexed document
"""

from __future__ import annotations

import os
import shutil
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from crypto_rag.config import get_settings
from crypto_rag.embeddings import build_embedding_provider
from crypto_rag.llm import OllamaLLM, build_llm_provider
from crypto_rag.pipeline import RAGPipeline
from crypto_rag.vector_store import VectorStore

settings = get_settings()
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

provider = build_embedding_provider(settings)
store = VectorStore(settings, provider)
llm = build_llm_provider(settings)
pipeline = RAGPipeline(settings, store, provider, llm)

app = FastAPI(title="CryptoRAG - Cryptography & Cryptanalysis RAG System",
              version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    model: Optional[str] = None
    doc_type_filter: Optional[str] = None


class SumarizeRequest(BaseModel):
    source: str


class RFCRequest(BaseModel):
    rfc_number: int


class URLRequest(BaseModel):
    url: str


def _ollama_models() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = []
            for m in resp.json().get("models", []):
                name = m.get("name", "")
                if name == settings.embedding_model or "embed" in name.lower():
                    continue
                models.append({"name": name, "size": m.get("size", 0),
                               "family": m.get("details", {}).get("family", "")})
            return models
    except requests.RequestException:
        pass
    return []


@app.get("/api/stats")
def stats():
    return store.get_stats()


@app.get("/api/models")
def models():
    return {"models": _ollama_models(), "active": settings.llm_model}


@app.get("/api/documents")
def documents():
    return {"documents": store.list_documents()}


@app.delete("/api/documents")
def delete_document(source: str):
    if store.delete_document(source):
        return {"message": f"Deleted: {source}"}
    raise HTTPException(status_code=404, detail=f"Document not found: {source}")


@app.post("/api/ingest/rfc")
def ingest_rfc(req: RFCRequest):
    try:
        return pipeline.ingest_rfc(req.rfc_number)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ingest/url")
def ingest_url(req: URLRequest):
    try:
        return pipeline.ingest_url(req.url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return pipeline.ingest_file(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/query")
def query(req: QueryRequest):
    active_llm = pipeline.llm
    if req.model and req.model != settings.llm_model:
        if settings.llm_backend == "ollama":
            pipeline.llm = OllamaLLM(req.model, settings.ollama_base_url)
        else:
            raise HTTPException(status_code=400,
                                detail="Model override is only supported "
                                       "with the Ollama backend.")
    try:
        return pipeline.query(req.question, top_k=req.top_k,
                              doc_type_filter=req.doc_type_filter)
    finally:
        pipeline.llm = active_llm


@app.post("/api/summarize")
def summarize(req: SumarizeRequest):
    result = pipeline.summarize_document(req.source)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
