import os
import shutil
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ollama_client import OllamaClient
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.rfc_fetcher import POPULAR_CRYPTO_RFCS
from src.paper_catalog import FEATURED_CRYPTANALYSIS_PAPERS

app = FastAPI(title="Cryptanalysis & Cryptography RAG Engine")

# Setup upload directory
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Engine
ollama_client = OllamaClient()
vector_store = VectorStore(ollama_client=ollama_client)
rag_engine = RAGEngine(vector_store, ollama_client)

class QueryRequest(BaseModel):
    question: str
    model: str = "llama3.1:8b"
    top_k: int = 5
    doc_type_filter: Optional[str] = None

class SummarizeRequest(BaseModel):
    source: str
    model: str = "llama3.1:8b"
    summary_type: str = "general"

class RFCRequest(BaseModel):
    rfc_number: int

class URLRequest(BaseModel):
    url: str

@app.get("/api/models")
def get_models():
    models = ollama_client.get_models()
    return {"models": models}

@app.get("/api/rfcs/popular")
def get_popular_rfcs():
    return {"rfcs": POPULAR_CRYPTO_RFCS}

@app.get("/api/papers/featured")
def get_featured_papers():
    return {"papers": FEATURED_CRYPTANALYSIS_PAPERS}

@app.get("/api/stats")
def get_stats():
    return vector_store.get_stats()

@app.get("/api/documents")
def list_documents():
    return {"documents": vector_store.list_ingested_documents()}

@app.delete("/api/documents")
def delete_document(source: str):
    success = vector_store.delete_document(source)
    if success:
        return {"message": f"Successfully deleted {source}"}
    raise HTTPException(status_code=400, detail=f"Failed to delete {source}")

@app.post("/api/ingest/rfc")
def ingest_rfc(req: RFCRequest):
    result = rag_engine.ingest_rfc(req.rfc_number)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/api/ingest/url")
def ingest_url(req: URLRequest):
    result = rag_engine.ingest_url(req.url)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = rag_engine.ingest_file(file_path, file.filename)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/api/query")
def query_rag(req: QueryRequest):
    result = rag_engine.query(
        question=req.question,
        model=req.model,
        top_k=req.top_k,
        doc_type_filter=req.doc_type_filter
    )
    return result

@app.post("/api/summarize")
def summarize_doc(req: SummarizeRequest):
    result = rag_engine.summarize_document(
        source=req.source,
        model=req.model,
        summary_type=req.summary_type
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
