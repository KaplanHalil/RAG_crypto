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
- **Two interfaces** — a full-screen terminal chat (TUI) and a Python CLI
  (`cryptorag`).

## Terminal UI (TUI)

A full-screen, chat-style terminal interface built with Textual — run it with
plain `cryptorag` (no arguments) or `cryptorag chat`:

```bash
cryptorag           # equivalent to `cryptorag chat`
```

- Ask questions and get **streamed** answers rendered as markdown with inline
  `[SOURCE n]` citations and a source list (title, file, similarity) under
  each answer
- **Multi-turn memory** — follow-up questions refer to the ongoing
  conversation; `/new` resets it
- Switch models on the fly (top bar): all models from `cryptorag.json`
  (Ollama + any OpenAI-compatible provider) plus auto-discovered local Ollama
  models; set Top-K and filter by document type
- Answer footer shows model, response time, token count, and sources used
- Slash commands: `/help`, `/stats`, `/new`, `/clear`, `/save`, `/load`,
  `/copy`, `/ingest-rfc N`, `/ingest-url URL`, `/ingest-file PATH`, `/quit`
- Shortcuts: `ctrl+c` / `esc` cancel generation, `ctrl+y` copy last answer,
  `ctrl+q` quit, `ctrl+l` clear the chat
- Sessions persist under `data/sessions/` via `/save` and `/load`; generation
  runs on a worker thread so the UI stays responsive

## Architecture

```
                 +--------------------------------------------------+
  User (TUI / CLI / library)  --->  RAGPipeline (crypto_rag/pipeline.py)
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
  llm.py           # Ollama / OpenAI generation providers (+ streaming)
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
  tui.py           # interactive terminal chat UI
eval/qa_benchmark.json # 49-question grounded QA benchmark (gold sources)
results/           # generated evaluation artifacts
kripto_makaleler/  # local corpus (PDFs/TXT pulled from IETF & NIST)
```

## Quick start

`make install` does everything: it creates a virtual environment, installs the
package (with the `cryptorag` command), prepares `.env`, and links
`cryptorag` into `~/.local/bin` so you can run it from **any** directory.

```bash
# 1. One-time setup (installs `cryptorag` onto your PATH)
make install

# 2. Start Ollama and pull models
ollama serve &
make models          # ollama pull nomic-embed-text + qwen3.5:9b

# 3. Build the knowledge base (embeds ~3.6k chunks once)
make index           # or: make index-reset to rebuild from scratch

# 4. Ask a question (from anywhere)
cryptorag query \
  "What is the data complexity of linear cryptanalysis against 16-round DES?"

# 5. Or just start chatting
cryptorag
```

No virtualenv activation needed — the `cryptorag` command on your PATH handles
it. Under the hood the same things still work: `python -m crypto_rag.cli ...`
from the repo directory.

## Make targets

| Target               | What it does                                        |
| :------------------- | :-------------------------------------------------- |
| `make install`       | venv + deps + `.env` + link `cryptorag` to `~/.local/bin` |
| `make models`        | pull the Ollama embedding and chat models           |
| `make index`         | build the knowledge base                            |
| `make index-reset`   | rebuild the knowledge base from scratch             |
| `make query Q="..."` | ask a question                                       |
| `make chat`          | interactive terminal UI (same as `cryptorag`)          |
| `make serve`         | HTTP API server (REST + OpenAI-compatible)          |
| `make stats`         | knowledge-base statistics                           |
| `make test`          | run the test suite                                  |
| `make eval`          | run the evaluation harness                          |
| `make clean`         | remove Python caches                                |
| `make uninstall`     | remove the `cryptorag` symlink from PATH            |

## CLI reference

The `cryptorag` command is the same CLI as `python -m crypto_rag.cli`. Bare
`cryptorag` (no arguments) launches the TUI:

```
cryptorag                  # launch the TUI
cryptorag stats
cryptorag query "question" [--top-k 5] [--model qwen3.5:9b] [--type RFC Document]
cryptorag ingest-rfc 9180
cryptorag ingest-file paper.pdf
cryptorag ingest-url https://...
cryptorag summarize "RFC 8446"
cryptorag index-corpus --reset
cryptorag eval --top-k 1,3,5,10,20 --answer 15 --judge --judge-model gemma4:26b
cryptorag config          # show the effective model config
cryptorag serve           # HTTP API server (port 8000)
```

## Model configuration (opencode-style)

CryptoRAG reads an opencode-style JSON config so you can register as many
providers and models as you like and switch between them from the TUI, the
CLI (`--model`), or the API — no code changes. Config files are searched like
opencode:

1. `$CRAG_CONFIG` (explicit override)
2. `<repo>/cryptorag.json` (project)
3. `<repo>/.cryptorag.json`
4. `~/.config/cryptorag/config.json` (global)

A project file is **merged over** the global file. Copy
[`cryptorag.example.json`](cryptorag.example.json) to one of these locations:

```json
{
  "provider": {
    "ollama": {
      "type": "ollama",
      "base_url": "http://localhost:11434",
      "models": { "qwen3.5:9b": { "name": "Qwen 3.5 9B" } }
    },
    "openai": {
      "type": "openai",
      "options": { "api_key": "env:OPENAI_API_KEY" },
      "models": { "gpt-4o-mini": {} }
    },
    "groq": {
      "type": "openai",
      "base_url": "https://api.groq.com/openai/v1",
      "options": { "api_key": "env:GROQ_API_KEY" },
      "models": { "llama-3.3-70b-versatile": {} }
    }
  },
  "embedding": { "provider": "ollama", "model": "nomic-embed-text" },
  "defaults": { "llm": "qwen3.5:9b" }
}
```

- `type` is `ollama` or `openai` (any OpenAI-compatible endpoint: OpenAI,
  Groq, Together, vLLM, ...all work via a custom `base_url`).
- API keys use opencode's `env:VAR` syntax and are read from the environment;
  they are never printed (`cryptorag config` redacts them).
- Per-model `options` (e.g. `"think": false`) are passed through to the backend.
- Without any config file, the legacy `.env` (env-driven) behaviour is used
  unchanged.
- Ollama models are also auto-discovered locally, so models you `ollama pull`
  appear in the TUI dropdown even if not listed in the config.

## HTTP API

`cryptorag serve` (or `make serve`) exposes two interfaces on one port
(default `0.0.0.0:8000`), implemented with the Python standard library only:

```bash
cryptorag serve --host 127.0.0.1 --port 8000
```

**REST API** (JSON):

| Method   | Path                  | Description                          |
| :------- | :-------------------- | :----------------------------------- |
| `GET`    | `/api/stats`          | knowledge-base statistics            |
| `GET`    | `/api/models`         | registered chat models + active one  |
| `GET`    | `/api/config`         | effective config (keys redacted)     |
| `GET`    | `/api/documents`      | list indexed documents               |
| `DELETE` | `/api/documents`      | delete a document (`?source=...`)    |
| `POST`   | `/api/query`          | RAG question answering               |
| `POST`   | `/api/query/stream`   | RAG answering as server-sent events  |
| `POST`   | `/api/ingest/rfc`     | index an IETF RFC (`{"rfc_number":N}`) |
| `POST`   | `/api/ingest/url`     | index a web page (`{"url":...}`)     |
| `POST`   | `/api/ingest/file`    | upload raw body (`?name=paper.pdf`)  |
| `POST`   | `/api/summarize`      | summarize an indexed document        |

**OpenAI-compatible** — any tool (opencode, curl, your own apps) can use
CryptoRAG as a chat model:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:9b",
       "messages":[{"role":"user","content":"küp atağı nedir?"}]}'
```

- `GET /v1/models` lists registered models; `POST /v1/chat/completions`
  accepts `messages`, `model`, and `stream` (SSE chunks).
- Prior `user`/`assistant` messages become the conversation history, and the
  retrieved sources are returned under the `crypto_rag.sources` field.

## Evaluation

```bash
cryptorag eval --top-k 1,3,5,10,20              # retrieval metrics
cryptorag eval --answer 15 --judge --judge-model gemma4:26b
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
CRAG_LLM_BACKEND=openai CRAG_OPENAI_API_KEY=sk-... cryptorag
```

**Asking in other languages:** the corpus is English. If you type a question
with non-ASCII characters (e.g. Turkish *"küp atağı anlat"*), CryptoRAG
automatically translates it to English for retrieval (`CRAG_TRANSLATE_QUERIES=true`)
and still answers in your language, grounded in the English sources.

## Paper

The LaTeX source for the accompanying paper is in `paper/` and is formatted
with the official `iacrcc` class (IACR Communications in Cryptology).

## License

MIT — see `LICENSE`.
