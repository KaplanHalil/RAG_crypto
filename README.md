# Chirag

A modular, locally-deployable **Retrieval-Augmented Generation (RAG) system**
for cryptography and cryptanalysis. Chirag grounds chat answers in a
curated knowledge base of IETF RFCs, NIST standards, and landmark
cryptanalysis papers, and answers with inline source citations. It runs fully
offline with local Ollama models, or optionally against an OpenAI-compatible
cloud API.

This repository is the implementation companion of the paper
*"Chirag: A Locally-Deployable, Citation-Aware Retrieval-Augmented
Generation System for Cryptography and Cryptanalysis"* (see `paper/`).

---

## Highlights

- **Fully local & private** — embeddings (`nomic-embed-text`) and generation
  (`qwen3.5:9b`, `gemma4:26b`) run on your own machine via Ollama.
- **Curated knowledge base** — 80+ documents: NIST FIPS & SP 800 (AES, SHA-2/3,
  ML-KEM, ML-DSA, SLH-DSA, block-cipher modes GCM/CCM/XTS, DRBG, KDF, key
  management, SP 800-22/38/56/57/90/107/108/131/133/185/186/208), IETF RFCs
  (TLS 1.2/1.3, RSA-PKCS#1, Ed25519, HKDF, HPKE, SSH, X.509, JOSE, CMS, AES-CMAC,
  AEAD, BLAKE2, Curve25519, ChaCha20-Poly1305, HMAC), and landmark papers
  (differential/linear cryptanalysis, DPA, padding oracles, Coppersmith,
  SHA-1 collisions, and more).
- **Citation-aware answers** — every answer cites the retrieved sources
  inline as `[SOURCE n: title]`, with a hallucination-guard system prompt.
- **Hybrid retrieval** — reciprocal-rank fusion of dense vector search and a
  BM25 lexical scorer, evaluated against a 49-question QA benchmark.
- **Evaluation harness** — retrieval metrics (hit@k, MRR@k, nDCG@k),
  LLM-judged answer faithfulness, and citation rate (see `eval/`, `results/`).
- **Two interfaces** — a full-screen terminal chat (TUI) and a Python CLI
  (`chirag`).

## Terminal UI (TUI)

A full-screen, chat-style terminal interface built with Textual — run it with
plain `chirag` (no arguments) or `chirag chat`:

```bash
chirag           # equivalent to `chirag chat`
```

- Ask questions and get **streamed** answers rendered as markdown with inline
  `[SOURCE n]` citations and a source list (title, file, similarity) under
  each answer
- **Multi-turn memory** — follow-up questions refer to the ongoing
  conversation; `/new` resets it
- Switch models on the fly (top bar): all models from `chirag.json`
  (Ollama + any OpenAI-compatible provider) plus auto-discovered local Ollama
  models; set Top-K and filter by document type
- Answer footer shows model, response time, token count, and sources used
- **Document table** — `ctrl+o` or `/documents` opens a full-screen view of all
  indexed documents (title, type, chunk count, source) with type filter,
  live search, and sortable columns (click a header). Delete a document from
  the index by pressing `d` twice on its row (permanent: the source file under
  `kripto_makaleler/` is removed too); `esc` cancels
- Slash commands: `/help`, `/stats`, `/documents`, `/new`, `/clear`, `/save`,
  `/load`, `/copy`, `/ingest-rfc N`, `/ingest-url URL`, `/ingest-file PATH`,
  `/quit`
- Shortcuts: `ctrl+c` / `esc` cancel generation, `ctrl+y` copy last answer,
  `ctrl+o` documents, `ctrl+q` quit, `ctrl+l` clear the chat
- Sessions persist under `data/sessions/` via `/save` and `/load`; generation
  runs on a worker thread so the UI stays responsive

## Architecture

```
                 +--------------------------------------------------+
  User (TUI / CLI / library)  --->  RAGPipeline (chirag/pipeline.py)
                                   |  chunker | retriever | LLM      |
                 +--------------------------------------------------+
                              |                         |
                    VectorStore (ChromaDB)        LLM provider (Ollama/OpenAI)
                    cosine, nomic-embed-text      qwen3.5:9b / gemma4:26b
                              |
                 Knowledge base (82 docs / ~8k chunks)
                 RFCs | NIST standards | landmark papers
```

```
chirag/
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
package (with the `chirag` command), prepares `.env`, and links
`chirag` into `~/.local/bin` so you can run it from **any** directory.

```bash
# 1. One-time setup (installs `chirag` onto your PATH)
make install

# 2. Start Ollama and pull models
ollama serve &
make models          # ollama pull nomic-embed-text + qwen3.5:9b

# 3. Build the knowledge base (embeds ~8k chunks once)
make index           # or: make index-reset to rebuild from scratch

# 4. Ask a question (from anywhere)
chirag query \
  "What is the data complexity of linear cryptanalysis against 16-round DES?"

# 5. Or just start chatting
chirag
```

No virtualenv activation needed — the `chirag` command on your PATH handles
it. Under the hood the same things still work: `python -m chirag.cli ...`
from the repo directory.

## Make targets

| Target               | What it does                                        |
| :------------------- | :-------------------------------------------------- |
| `make install`       | venv + deps + `.env` + link `chirag` to `~/.local/bin` |
| `make models`        | pull the Ollama embedding and chat models           |
| `make index`         | build the knowledge base                            |
| `make index-reset`   | rebuild the knowledge base from scratch             |
| `make query Q="..."` | ask a question                                       |
| `make chat`          | interactive terminal UI (same as `chirag`)          |
| `make serve`         | HTTP API server (REST + OpenAI-compatible)          |
| `make stats`         | knowledge-base statistics                           |
| `make test`          | run the test suite                                  |
| `make eval`          | run the evaluation harness                          |
| `make clean`         | remove Python caches                                |
| `make uninstall`     | remove the `chirag` symlink from PATH            |

## CLI reference

The `chirag` command is the same CLI as `python -m chirag.cli`. Bare
`chirag` (no arguments) launches the TUI:

```
chirag                  # launch the TUI
chirag stats
chirag query "question" [--top-k 5] [--model qwen3.5:9b] [--type RFC Document]
chirag ingest-rfc 9180
chirag ingest-file paper.pdf
chirag ingest-url https://...
chirag summarize "RFC 8446"
chirag index-corpus --reset
chirag eval --top-k 1,3,5,10,20 --answer 15 --judge --judge-model gemma4:26b
chirag config          # show the effective model config
chirag serve           # HTTP API server (port 8000)
```

## Model configuration (opencode-style)

Chirag reads an opencode-style JSON config so you can register as many
providers and models as you like and switch between them from the TUI, the
CLI (`--model`), or the API — no code changes. Config files are searched like
opencode:

1. `$CHIRAG_CONFIG` (explicit override)
2. `<repo>/chirag.json` (project)
3. `<repo>/.chirag.json`
4. `~/.config/chirag/config.json` (global)

A project file is **merged over** the global file. Copy
[`chirag.example.json`](chirag.example.json) to one of these locations:

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
  they are never printed (`chirag config` redacts them).
- Per-model `options` (e.g. `"think": false`) are passed through to the backend.
- Without any config file, the legacy `.env` (env-driven) behaviour is used
  unchanged.
- Ollama models are also auto-discovered locally, so models you `ollama pull`
  appear in the TUI dropdown even if not listed in the config.

## HTTP API

`chirag serve` (or `make serve`) exposes two interfaces on one port
(default `0.0.0.0:8000`), implemented with the Python standard library only:

```bash
chirag serve --host 127.0.0.1 --port 8000
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
Chirag as a chat model:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:9b",
       "messages":[{"role":"user","content":"küp atağı nedir?"}]}'
```

- `GET /v1/models` lists registered models; `POST /v1/chat/completions`
  accepts `messages`, `model`, and `stream` (SSE chunks).
- Prior `user`/`assistant` messages become the conversation history, and the
  retrieved sources are returned under the `chirag.sources` field.

## Evaluation

```bash
chirag eval --top-k 1,3,5,10,20              # retrieval metrics
chirag eval --answer 15 --judge --judge-model gemma4:26b
```

Outputs land in `results/`: per-retriever JSON, a summary, LaTeX-ready tables
and `retrieval_curves.png`. Results (49 questions) measured on the earlier
46-document corpus:

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

All knobs live in environment variables (prefix `CHIRAG_`). Switch backends with:

```bash
CHIRAG_LLM_BACKEND=openai CHIRAG_OPENAI_API_KEY=sk-... chirag
```

**Asking in other languages:** the corpus is English. If you type a question
with non-ASCII characters (e.g. Turkish *"küp atağı anlat"*), Chirag
automatically translates it to English for retrieval (`CHIRAG_TRANSLATE_QUERIES=true`)
and still answers in your language, grounded in the English sources.

## Paper

The LaTeX source for the accompanying paper is in `paper/` and is formatted
with the official `iacrcc` class (IACR Communications in Cryptology).

## License

MIT — see `LICENSE`.
