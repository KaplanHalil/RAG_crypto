# CryptoRAG

A modular, locally-deployable **Retrieval-Augmented Generation (RAG) system**
for cryptography and cryptanalysis. CryptoRAG grounds chat answers in a
curated knowledge base of IETF RFCs, NIST standards, and landmark
cryptanalysis papers, and answers with inline source citations. It runs fully
offline with local Ollama models, or optionally against an OpenAI-compatible
cloud API.

This repository is the implementation companion of the paper
*"CryptoRAG: A Locally-Deployable, Citation-Aware Retrieval-Augmented
Generation System for Cryptography and Cryptanalysis"* (see `paper/`).

---

## Highlights

- **Fully local & private** — embeddings (`nomic-embed-text`) and generation
  (`qwen3.5:9b`, `gemma4:26b`) run on your own machine via Ollama.
- **Curated knowledge base** — 40+ documents: NIST FIPS (AES, SHA-2/3, DSA,
  ML-KEM, ML-DSA, SLH-DSA, SP 800-22), IETF RFCs (TLS 1.3, ChaCha20-Poly1305,
  Curve25519, HMAC), and landmark papers (differential/linear cryptanalysis,
  DPA, padding oracles, Coppersmith, SHA-1 collisions, and more).
- **Citation-aware answers** — every answer cites the retrieved sources
  inline as `[SOURCE n: title]`, with a hallucination-guard system prompt.
- **Hybrid retrieval** — reciprocal-rank fusion of dense vector search and a
  BM25 lexical scorer, evaluated against a 49-question QA benchmark.
- **Evaluation harness** — retrieval metrics (hit@k, MRR@k, nDCG@k),
  LLM-judged answer faithfulness, and citation rate (see `eval/`, `results/`).
- **Three interfaces** — Python library, CLI, and a FastAPI web server with a
  single-page English UI.

## Architecture

```
                 +--------------------------------------------------+
  User (web / CLI / library)  --->  RAGPipeline (crypto_rag/pipeline.py)
                                   |  chunker | retriever | LLM      |
                 +--------------------------------------------------+
                              |                         |
                    VectorStore (ChromaDB)        LLM provider (Ollama/OpenAI)
                    cosine, nomic-embed-text      qwen3.5:9b / gemma4:26b
                              |
                 Knowledge base (46 docs / ~3.6k chunks)
                 RFCs | NIST standards | landmark papers
```

```
crypto_rag/
  config.py        # pydantic-settings, .env driven
  embeddings.py    # Ollama / OpenAI embedding providers + ChromaDB adapter
  llm.py           # Ollama / OpenAI generation providers
  chunker.py       # paragraph-aware overlapping chunker
  ingestion.py     # RFC / URL / PDF / TXT loaders + doc-type classification
  vector_store.py  # ChromaDB persistence: upsert / search / stats / delete
  retriever.py     # DenseRetriever, HybridRetriever (dense + BM25, RRF)
  prompt.py        # English prompts, citation rules, context assembly
  pipeline.py      # ingest / query / summarize orchestration
  indexer.py       # bulk knowledge-base construction
  landmark_papers.py  # curated entries for highly-cited papers
  evaluation.py    # benchmarks, metrics, LaTeX tables, plots
  cli.py           # command-line interface
app.py             # FastAPI server (REST API + web UI)
templates/index.html  # single-page English web UI
eval/qa_benchmark.json # 49-question grounded QA benchmark (gold sources)
results/           # generated evaluation artifacts
kripto_makaleler/  # local corpus (PDFs/TXT pulled from IETF & NIST)
```

## Quick start

```bash
# 1. Environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2. Start Ollama and pull models
ollama serve &
ollama pull nomic-embed-text
ollama pull qwen3.5:9b        # or any chat model you have

# 3. Build the knowledge base (embeds ~3.6k chunks once)
python -m crypto_rag.cli index-corpus --reset

# 4. Ask a question
python -m crypto_rag.cli query \
  "What is the data complexity of linear cryptanalysis against 16-round DES?"

# 5. Run the web UI at http://localhost:8000
python -m crypto_rag.cli serve
```

## CLI reference

```
python -m crypto_rag.cli stats
python -m crypto_rag.cli query "question" [--top-k 5] [--model qwen3.5:9b] [--type RFC Document]
python -m crypto_rag.cli ingest-rfc 9180
python -m crypto_rag.cli ingest-file paper.pdf
python -m crypto_rag.cli ingest-url https://...
python -m crypto_rag.cli summarize "RFC 8446"
python -m crypto_rag.cli index-corpus --reset
python -m crypto_rag.cli eval --top-k 1,3,5,10,20 --answer 15 --judge --judge-model gemma4:26b
```

## REST API

| Method   | Path               | Description                         |
| :------- | :----------------- | :---------------------------------- |
| `GET`    | `/api/stats`       | knowledge-base statistics           |
| `GET`    | `/api/models`      | available Ollama chat models        |
| `GET`    | `/api/documents`   | list indexed documents              |
| `DELETE` | `/api/documents`   | delete a document (`?source=...`)   |
| `POST`   | `/api/ingest/rfc`  | index an IETF RFC                   |
| `POST`   | `/api/ingest/url`  | index a web page                    |
| `POST`   | `/api/ingest/file` | upload a PDF/TXT/MD file            |
| `POST`   | `/api/query`       | RAG question answering              |
| `POST`   | `/api/summarize`   | summarize an indexed document       |

## Evaluation

```bash
python -m crypto_rag.cli eval --top-k 1,3,5,10,20              # retrieval metrics
python -m crypto_rag.cli eval --answer 15 --judge --judge-model gemma4:26b
```

Outputs land in `results/`: per-retriever JSON, a summary, LaTeX-ready tables
and `retrieval_curves.png`. Current results (49 questions, 46 documents):

| retriever | hit@5 | hit@10 | MRR@5 | nDCG@5 |
| :-------- | ----: | -----: | ----: | -----: |
| dense     | 0.776 | 0.796  | 0.695 | 0.706 |
| hybrid    | 0.837 | 0.857  | 0.713 | 0.743 |

Answer quality (15 generated answers, `qwen3.5:9b`; judged by `gemma4:26b`):

| metric | value |
| :----- | ----: |
| LLM-judged faithfulness (1–5) | 5.00  (12/15 scorable) |
| citation rate (cited lines) | 0.502 |
| citation validity (14/14 answers) | 100 % |

## Configuration

All knobs live in environment variables (prefix `CRAG_`). Switch backends with:

```bash
CRAG_LLM_BACKEND=openai CRAG_OPENAI_API_KEY=sk-... python -m crypto_rag.cli serve
```

## Paper

The LaTeX source for the accompanying paper is in `paper/` and is formatted
with the official `iacrcc` class (IACR Communications in Cryptology).

## License

MIT — see `LICENSE`.
