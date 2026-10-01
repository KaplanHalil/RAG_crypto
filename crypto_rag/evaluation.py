"""Evaluation harness.

The benchmark is a small, curated set of cryptography/cryptanalysis
questions, each annotated with the documents that are known to contain the
answer (document-level gold labels). From this we compute standard retrieval
metrics (hit@k, MRR@k, nDCG@k) for every configured retriever and top-k
value. Optionally, answers are generated and scored for faithfulness with an
LLM judge.

Run with::

    python -m crypto_rag.evaluation --out results/ --answer 10
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import PROJECT_ROOT, Settings, get_settings
from .embeddings import build_embedding_provider
from .llm import build_llm_provider
from .pipeline import RAGPipeline
from .retriever import DenseRetriever, HybridRetriever
from .vector_store import VectorStore

JUDGE_SYSTEM = """You are an evaluation judge for a retrieval-augmented
generation (RAG) system. Rate the faithfulness of an answer with respect to
the retrieved context that was given to the generator.

Faithfulness means: every factual claim in the answer is either directly
supported by the retrieved context, or explicitly flagged as outside the
retrieved context.

Return ONLY a single integer 1-5 and nothing else:
5 = fully faithful, every claim supported by the context
4 = minor unsupported detail of negligible impact
3 = some claims not supported by the context
2 = several key claims are not supported (hallucination)
1 = largely fabricated content
"""

JUDGE_USER = """Question: {question}

Retrieved context:
{context}

Model answer:
{answer}

Rate the faithfulness of the model answer with respect to the retrieved
context on a scale of 1 to 5. Output only the integer."""

_CITE_RE = re.compile(r"\[SOURCE\s*\d+")


@dataclass
class QAItem:
    question: str
    gold_sources: List[str]          # substrings matched against doc sources
    category: str = "general"


@dataclass
class RetrievalResult:
    query: str
    gold_sources: List[str]
    ranked_sources: List[str]        # doc-level, deduplicated, in rank order
    ranked_chunks: List[Dict[str, Any]]


def load_benchmark(path: str) -> List[QAItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [QAItem(q["question"], q.get("gold_sources", []),
                   q.get("category", "general")) for q in data]


def _is_gold(ranked_sources: Sequence[str], gold: List[str]) -> bool:
    for g in gold:
        for s in ranked_sources:
            if g.lower() in s.lower():
                return True
    return False


def _rank_position(ranked_sources: Sequence[str], gold: List[str]) -> int:
    for i, s in enumerate(ranked_sources):
        for g in gold:
            if g.lower() in s.lower():
                return i  # 0-based
    return -1


def evaluate_retrieval(items: List[QAItem], retriever, top_k_vals: List[int]
                       ) -> Dict[str, Any]:
    """Run document-level retrieval metrics for one retriever."""
    results: List[RetrievalResult] = []
    for item in items:
        chunks = retriever.retrieve(item.question, top_k=max(top_k_vals))
        ranked_sources: List[str] = []
        for c in chunks:
            s = c["metadata"].get("source", "Unknown")
            if s not in ranked_sources:
                ranked_sources.append(s)
        results.append(RetrievalResult(item.question, item.gold_sources,
                                       ranked_sources, chunks))

    metrics: Dict[str, Dict[str, float]] = {}
    for k in top_k_vals:
        hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0
        for r in results:
            top = r.ranked_sources[:k]
            if _is_gold(top, r.gold_sources):
                hits += 1
            pos = _rank_position(top, r.gold_sources)
            if pos >= 0:
                rr_sum += 1.0 / (pos + 1)
            ndcg_sum += _ndcg_at_k(r.ranked_sources, r.gold_sources, k)
        n = len(results)
        metrics[k] = {
            "hit@k": hits / n,
            "mrr@k": rr_sum / n,
            "ndcg@k": ndcg_sum / n,
        }
    return {"results": results, "metrics": metrics}


def _dcg(gains: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(ranked_sources: Sequence[str], gold: List[str], k: int) -> float:
    """nDCG@k with binary gains; ideal ranking places all gold docs first."""
    gains = [1.0 if any(g.lower() in t.lower() for g in gold) else 0.0
             for t in ranked_sources[:k]]
    if not gains:
        return 0.0
    n_relevant = min(len(gold), k)
    dcg = _dcg(gains)
    idcg = _dcg([1.0] * n_relevant)
    return dcg / idcg if idcg > 0 else 0.0


def judge_faithfulness(llm, item: QAItem, context: str, answer: str) -> int:
    prompt = JUDGE_USER.format(question=item.question, context=context,
                               answer=answer)
    for attempt in range(3):
        try:
            raw = llm.generate(prompt, system_prompt=JUDGE_SYSTEM,
                               temperature=0.0, max_tokens=2048).strip()
        except Exception as exc:  # pragma: no cover
            print(f"[judge] attempt {attempt + 1} error: {exc}", flush=True)
            continue
        m = re.search(r"(?<!\d)[1-5](?!\d)", raw)
        if m:
            return int(m.group(0))
        print(f"[judge] attempt {attempt + 1} no digit in {raw!r}",
              flush=True)
        if not raw:  # empty response: wait for the GPU to settle
            import time
            time.sleep(2)
    return 0


def citation_rate(answer: str) -> float:
    """Fraction of paragraphs (non-empty lines) containing a citation."""
    paragraphs = [p.strip() for p in answer.split("\n") if p.strip()]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if _CITE_RE.search(p))
    return cited / len(paragraphs)


def run_answer_eval(pipeline: RAGPipeline, items: List[QAItem], limit: int,
                    judge_llm=None) -> List[Dict[str, Any]]:
    """Generate answers and score faithfulness + citation rate."""
    out = []
    sample = items[:limit]
    for i, item in enumerate(sample, 1):
        print(f"[answer-eval] {i}/{len(sample)}: {item.question[:60]}...",
              flush=True)
        answer = ""
        for attempt in range(3):
            resp = pipeline.query(item.question, use_llm=True)
            answer = (resp.get("answer") or "").strip()
            if answer:
                break
            print(f"[answer-eval]  retry {attempt + 1} for empty answer",
                  flush=True)
        ctx = "\n\n".join(
            f"--- {c['metadata']['source']} ---\n{c['content']}"
            for c in resp["retrieved_chunks"])
        entry = {"question": item.question, "answer": answer,
                 "retrieved_sources": [s["source"] for s in resp["sources"]],
                 "context": ctx[:16000],
                 "citation_rate": citation_rate(answer) if answer else 0.0}
        if judge_llm is not None:
            entry["faithfulness"] = judge_faithfulness(
                judge_llm, item, ctx, answer)
        out.append(entry)
    return out


def latex_table(row_labels: Sequence[str], metric: str,
                col_labels: Sequence[str], grid: Sequence[Sequence[float]]
                ) -> str:
    """Render a LaTeX table body for one metric."""
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{ {metric} by retrievers and top-$k$. }}",
             "\\label{tab:" + metric.lower().replace("@", "at") + "}",
             "\\begin{tabular}{l" + "c" * len(col_labels) + "}",
             "\\toprule",
             " & " + " & ".join(col_labels) + " \\\\",
             "\\midrule"]
    for label, row in zip(row_labels, grid):
        lines.append(f"{label} & " + " & ".join(f"{v:.3f}" for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def plot_curves(retriever_metrics: Dict[str, Dict[int, Dict[str, float]]],
                out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    k_vals = sorted(next(iter(retriever_metrics.values())).keys())
    for name, m in retriever_metrics.items():
        ax[0].plot(k_vals, [m[k]["hit@k"] for k in k_vals], marker="o", label=name)
        ax[1].plot(k_vals, [m[k]["mrr@k"] for k in k_vals], marker="o", label=name)
    ax[0].set(title="Hit@k", xlabel="k", ylabel="Hit@k", ylim=(0, 1.05))
    ax[1].set(title="MRR@k", xlabel="k", ylabel="MRR@k", ylim=(0, 1.05))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="CryptoRAG evaluation harness")
    ap.add_argument("--benchmark",
                    default=str(PROJECT_ROOT / "eval" / "qa_benchmark.json"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--top-k", default="1,3,5,10,20")
    ap.add_argument("--answer", type=int, default=0,
                    help="generate answers for the first N questions (0=off)")
    ap.add_argument("--judge", action="store_true",
                    help="use the LLM as an extra faithfulness judge")
    ap.add_argument("--judge-model", default=None,
                    help="Ollama model to use as faithfulness judge "
                         "(defaults to the generation model)")
    ap.add_argument("--judge-file", default=None,
                    help="score the answers in an existing answer_eval.json "
                         "without regenerating them (separate pass)")
    args = ap.parse_args()

    settings: Settings = get_settings()
    provider = build_embedding_provider(settings)
    store = VectorStore(settings, provider)
    llm = build_llm_provider(settings)
    pipeline = RAGPipeline(settings, store, provider, llm)

    items = load_benchmark(args.benchmark)
    top_k_vals = [int(v) for v in args.top_k.split(",")]

    # --judge-file: score existing answers without regenerating them.
    if args.judge_file:
        with open(args.judge_file, "r", encoding="utf-8") as f:
            answer_log = json.load(f)
        from .llm import OllamaLLM
        judge_llm = OllamaLLM(args.judge_model, settings.ollama_base_url) \
            if args.judge_model else llm
        by_question = {i.question: i for i in items}
        for i, entry in enumerate(answer_log, 1):
            item = by_question.get(entry["question"])
            if item is None:
                print(f"[judge] skipping unknown question #{i}; continue")
                continue
            entry["faithfulness"] = judge_faithfulness(
                judge_llm, item, entry.get("context", ""), entry["answer"])
            print(f"[judge] {i}/{len(answer_log)} -> "
                  f"{entry['faithfulness']}/5")
        with open(args.judge_file, "w", encoding="utf-8") as f:
            json.dump(answer_log, f, indent=2)
        _print_answer_stats(answer_log)
        return

    retriever_metrics: Dict[str, Any] = {}
    for engine, retriever in (
            ("dense",
             DenseRetriever(store, threshold=settings.similarity_threshold)),
            ("hybrid",
             HybridRetriever(store, threshold=settings.similarity_threshold))):
        # NB: reuse the pipeline to keep caches and thresholds consistent
        pipeline.retriever = retriever
        print(f"[eval] running retriever: {engine} on {len(items)} queries")
        res = evaluate_retrieval(items, retriever, top_k_vals)
        retriever_metrics[engine] = res["metrics"]
        with open(f"{args.out}/retrieval_{engine}.json", "w") as f:
            json.dump({
                "metrics": res["metrics"],
                "results": [
                    {"query": r.query, "gold_sources": r.gold_sources,
                     "ranked_sources": r.ranked_sources}
                    for r in res["results"]
                ],
            }, f, indent=2)

    with open(f"{args.out}/retrieval_summary.json", "w") as f:
        json.dump(retriever_metrics, f, indent=2)

    for metric in ("hit@k", "mrr@k", "ndcg@k"):
        print(latex_table(
            list(retriever_metrics.keys()), metric, [f"$k={k}$" for k in top_k_vals],
            [[retriever_metrics[e][k][metric] for k in top_k_vals]
             for e in retriever_metrics]))

    plot_curves(retriever_metrics, f"{args.out}/retrieval_curves.png")

    answer_log = None
    if args.answer > 0:
        judge_llm = None
        if args.judge:
            from .llm import OllamaLLM
            judge_llm = (OllamaLLM(args.judge_model,
                                   settings.ollama_base_url)
                         if args.judge_model else llm)
        answer_log = run_answer_eval(pipeline, items, args.answer, judge_llm)
        with open(f"{args.out}/answer_eval.json", "w") as f:
            json.dump(answer_log, f, indent=2)
        _print_answer_stats(answer_log)


def _print_answer_stats(answer_log: list) -> None:
    """Print aggregate answer-quality statistics from an eval log."""
    import statistics

    nonempty = [a for a in answer_log if (a.get("answer") or "").strip()]
    print(f"[eval] answers: {len(answer_log)} total, "
          f"{len(nonempty)} non-empty")
    if nonempty:
        rates = [a["citation_rate"] for a in nonempty]
        print(f"[eval] citation rate over non-empty answers: "
              f"mean={statistics.mean(rates):.3f} "
              f"std={statistics.stdev(rates):.3f}")
    scores = [a["faithfulness"] for a in answer_log
              if a.get("faithfulness")]
    if scores:
        print(f"[eval] LLM-judge faithfulness: "
              f"mean={statistics.mean(scores):.2f}/5.0 "
              f"std={statistics.stdev(scores):.2f} "
              f"n={len(scores)}")


if __name__ == "__main__":
    main()
