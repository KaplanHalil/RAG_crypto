"""Command-line interface for Chirag.

Running ``chirag`` with no arguments launches the interactive terminal UI.

Usage::

    chirag                      # launch the TUI
    chirag index-corpus
    chirag query "How does linear cryptanalysis work?"
    chirag query --top-k 8 --model qwen3.5:9b "question"
    chirag ingest-rfc 9180
    chirag ingest-file path/to/paper.pdf
    chirag summarize "RFC 8446"
    chirag stats
    chirag eval
"""

from __future__ import annotations

import argparse
import sys

from .config import get_settings, PROJECT_ROOT
from .embeddings import build_embedding_provider
from .llm import build_llm_provider
from .pipeline import RAGPipeline
from .vector_store import VectorStore


def _build_pipeline():
    settings = get_settings()
    provider = build_embedding_provider(settings)
    store = VectorStore(settings, provider)
    llm = build_llm_provider(settings)
    return settings, RAGPipeline(settings, store, provider, llm)


def cmd_stats(_: argparse.Namespace) -> None:
    _, pipeline = _build_pipeline()
    import json
    print(json.dumps(pipeline.stats(), indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    settings, pipeline = _build_pipeline()
    question = " ".join(args.question)
    if args.model:
        from .models import build_llm
        pipeline.llm = build_llm(args.model)
    result = pipeline.query(
        question, top_k=args.top_k,
        doc_type_filter=args.type)
    print(f"\nANSWER ({len(result.get('retrieved_chunks', []))} chunks):\n")
    print(result.get("answer") or "[retrieval-only mode]")
    print("\nSOURCES:")
    for s in result.get("sources", []):
        print(f"  [{s['id']}] {s['title']}  ({s['source']})  sim={s['similarity']}")


def cmd_ingest_rfc(args: argparse.Namespace) -> None:
    _, pipeline = _build_pipeline()
    result = pipeline.ingest_rfc(args.rfc_number)
    print(result)


def cmd_ingest_file(args: argparse.Namespace) -> None:
    _, pipeline = _build_pipeline()
    result = pipeline.ingest_file(args.path)
    print(result)


def cmd_ingest_url(args: argparse.Namespace) -> None:
    _, pipeline = _build_pipeline()
    result = pipeline.ingest_url(args.url)
    print(result)


def cmd_summarize(args: argparse.Namespace) -> None:
    _, pipeline = _build_pipeline()
    result = pipeline.summarize_document(args.source)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(result["summary"])


def cmd_index_corpus(args: argparse.Namespace) -> None:
    from .indexer import index_corpus
    index_corpus(reset=args.reset, papers_dir=args.papers_dir,
                 only_landmark=args.only_landmark)


def cmd_chat(_: argparse.Namespace) -> None:
    from .tui import run
    run()


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve
    serve(host=args.host, port=args.port)


def cmd_config(args: argparse.Namespace) -> None:
    import json
    from .models import config_file_used, get_registry

    path = config_file_used()
    if path is None:
        print("No chirag.json found. Using environment-driven settings "
              "(see .env.example).")
        return
    print(f"Config file: {path}")
    print(json.dumps(get_registry().effective(redact=True), indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluation import main as eval_main
    sys.argv = ["evaluation", "--benchmark", args.benchmark, "--out",
                args.out, "--top-k", args.top_k, "--answer",
                str(args.answer)] + (["--judge"] if args.judge else []) + \
               (["--judge-model", args.judge_model]
                if args.judge_model else []) + \
               (["--judge-file", args.judge_file] if args.judge_file else [])
    eval_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chirag", description="Chirag CLI")
    p.set_defaults(func=cmd_chat)
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("stats", help="show knowledge-base statistics")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("query", help="ask the RAG system a question")
    sp.add_argument("question", nargs="+")
    sp.add_argument("--top-k", type=int, default=None)
    sp.add_argument("--model", default=None,
                    help="override LLM (any model from chirag.json)")
    sp.add_argument("--type", default=None,
                    help="filter: RFC Document | NIST Standard | "
                         "Cryptanalysis Paper | Cryptography Paper | All")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("ingest-rfc", help="ingest an IETF RFC by number")
    sp.add_argument("rfc_number", type=int)
    sp.set_defaults(func=cmd_ingest_rfc)

    sp = sub.add_parser("ingest-file", help="ingest a local PDF/TXT/MD file")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_ingest_file)

    sp = sub.add_parser("ingest-url", help="ingest a web page")
    sp.add_argument("url")
    sp.set_defaults(func=cmd_ingest_url)

    sp = sub.add_parser("summarize", help="summarize an indexed document")
    sp.add_argument("source")
    sp.set_defaults(func=cmd_summarize)

    sp = sub.add_parser("index-corpus", help="rebuild the knowledge base")
    sp.add_argument("--reset", action="store_true",
                    help="drop the collection before indexing")
    sp.add_argument("--papers-dir", default=None)
    sp.add_argument("--only-landmark", action="store_true",
                    help="index only the curated landmark papers")
    sp.set_defaults(func=cmd_index_corpus)

    sp = sub.add_parser("chat", help="launch the interactive terminal UI")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("serve", help="run the HTTP API server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("config", help="show the effective model config")
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("eval", help="run the evaluation harness")
    sp.add_argument("--benchmark",
                    default=str(PROJECT_ROOT / "eval" / "qa_benchmark.json"))
    sp.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    sp.add_argument("--top-k", default="1,3,5,10,20")
    sp.add_argument("--answer", type=int, default=0)
    sp.add_argument("--judge", action="store_true")
    sp.add_argument("--judge-model", default=None)
    sp.add_argument("--judge-file", default=None)
    sp.set_defaults(func=cmd_eval)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
