"""Prompt templates and citation-aware context assembly.

All interaction with the generation model is English. Retrieved chunks are
assigned stable 1-based source identifiers (``[SOURCE 1]``...), the context
is assembled with explicit ``--- SOURCE n: <title> ---`` separators, and the
system prompt instructs the model to answer strictly from the provided
context, to cite sources inline, and to refuse to invent content not present
in the retrieved knowledge -- a hallucination guard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = """You are an expert assistant in cryptography and cryptanalysis.
Answer the user's question using ONLY the retrieved knowledge provided in the
'RETRIEVED KNOWLEDGE CONTEXT' section below. Follow these rules strictly:

1. GROUNDING: Base your answer exclusively on the retrieved context. Do not
   invent papers, RFCs, standards, attack complexities, or results that do not
   appear in the context. If the context does not contain the answer, say so
   clearly instead of guessing.
2. CITATIONS: Every important claim must be accompanied by an inline citation
   of the exact source it comes from, in the form [SOURCE n: Document Title]
   where n is the source number assigned in the context (e.g.
   [SOURCE 1: Linear Cryptanalysis Method for DES Cipher]).
3. NEVER cite a source number that is not present in the context.
4. Technical accuracy: include concrete parameters (number of rounds, data
   and time complexity, key sizes) exactly as reported in the context.
5. If the retrieved context only covers part of the question, answer that part
   and explicitly state which part is not covered by the retrieved sources."""

USER_PROMPT_TEMPLATE = """User question: {question}

RETRIEVED KNOWLEDGE CONTEXT:
{context}

Instructions: Using the sources above, answer the question with technical
precision. Attach the corresponding inline citation [SOURCE n: title] after
each important piece of information."""


def assemble_context(search_results: List[Dict[str, Any]],
                     doc_labels: Optional[Dict[str, int]] = None
                     ) -> Dict[str, Any]:
    """Turn raw retrieval results into (context_text, sources_meta).

    Assigns a stable 1-based index to each distinct document ``source`` so the
    model can cite ``[SOURCE n]`` and the UI can render the source list.
    """
    if doc_labels is None:
        doc_labels = {}
    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for res in search_results:
        meta = res["metadata"]
        source = meta.get("source", "Unknown")
        title = meta.get("title", source)
        if source not in doc_labels:
            doc_labels[source] = len(doc_labels) + 1
            sources.append({
                "id": doc_labels[source],
                "source": source,
                "title": title,
                "url": meta.get("url", ""),
                "type": meta.get("type", "Document"),
                "similarity": res.get("similarity"),
            })
        n = doc_labels[source]
        context_parts.append(
            f"--- SOURCE {n}: {title} (file: {source}) ---\n{res['content']}"
        )

    return {"context": "\n\n".join(context_parts), "sources": sources}


def build_user_prompt(question: str, context: str,
                      history: Optional[List[Dict[str, str]]] = None) -> str:
    """Assemble the user prompt.

    ``history`` is an optional list of prior turns (oldest first), each a dict
    with ``role`` ("user"|"assistant") and ``content``. When given, a bounded
    recent window is included so the model can answer follow-up questions in
    context. It must NOT contain the current question.
    """
    if not history:
        return USER_PROMPT_TEMPLATE.format(question=question, context=context)

    turns = []
    for turn in history[-8:]:
        label = "Question" if turn.get("role") == "user" else "Previous answer"
        turns.append(f"{label}: {turn.get('content', '')}")
    transcript = "\n\n".join(turns)

    return (
        "CONVERSATION SO FAR:\n"
        f"{transcript}\n\n"
        f"Current question: {question}\n\n"
        "RETRIEVED KNOWLEDGE CONTEXT:\n"
        f"{context}\n\n"
        "Instructions: Using the sources above, answer the current question "
        "with technical precision. Attach the corresponding inline citation "
        "[SOURCE n: title] after each important piece of information. You "
        "may refer to the conversation above when the current question "
        "builds on it, but stay grounded in the retrieved context."
    )


def build_summary_system_prompt() -> str:
    return (
        "You are a senior cryptography and cryptanalysis researcher. "
        "Summarize the provided document in English with the following "
        "sections: 1) Document identity and purpose; 2) Core technical "
        "method or protocol; 3) Attack/protocol details with time and data "
        "complexities or security parameters; 4) Cryptographic impact and "
        "countermeasures. Be precise and cite concrete numbers from the text."
    )
