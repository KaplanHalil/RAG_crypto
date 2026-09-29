"""Unit tests for the CryptoRAG core. Fast tests only (no network, no models)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_rag.chunker import Chunker
from crypto_rag.ingestion import Document, classify_doc_type
from crypto_rag.prompt import assemble_context, build_user_prompt
from crypto_rag.retriever import BM25Scorer, _rrf


def test_chunker_respects_size():
    text = "word " * 2000
    chunks = Chunker(1000, 150).chunk(text, {"source": "doc"})
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 1000 + 200  # allow overlap slack
        assert c.metadata["source"] == "doc"
        assert "chunk_id" in c.metadata
    # chunk ids are sequential
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


def test_chunker_overlap_carries_tail():
    text = " ".join(f"p{i} " + "word " * 100 for i in range(15))
    chunks = Chunker(1000, 150).chunk(text, {"source": "d"})
    assert len(chunks) >= 4
    # overlap region shared between consecutive chunks
    for a, b in zip(chunks, chunks[1:]):
        tail = " ".join(a.text.split()[-10:])
        assert tail in b.text


def test_document_metadata():
    doc = Document(source="s", title="t", url="u", type="Proxy",
                   content="x")
    md = doc.metadata()
    assert md["source"] == "s" and md["type"] == "Proxy"


def test_classify_doc_type():
    assert classify_doc_type("X_FIPS_197_Y.pdf") == "NIST Standard"
    assert classify_doc_type("rfc_8446.txt") == "RFC Document"
    assert classify_doc_type("Cube_Attacks.pdf") == "Cryptanalysis Paper"
    assert classify_doc_type("RSA_Cryptosystem.pdf") == "Cryptography Paper"


def test_assemble_context_assigns_stable_indices():
    results = [
        {"metadata": {"source": "A", "title": "Title A", "type": "P"},
         "content": "chunk a1"},
        {"metadata": {"source": "B", "title": "Title B", "type": "P"},
         "content": "chunk b1"},
        {"metadata": {"source": "A", "title": "Title A", "type": "P"},
         "content": "chunk a2"},
    ]
    ctx = assemble_context(results)
    assert len(ctx["sources"]) == 2
    assert ctx["sources"][0]["id"] == 1
    assert "--- SOURCE 1: Title A" in ctx["context"]
    assert "--- SOURCE 2: Title B" in ctx["context"]
    prompt = build_user_prompt("q?", ctx["context"])
    assert "q?" in prompt and "RETRIEVED KNOWLEDGE CONTEXT" in prompt


def test_bm25_ranks_term_matches():
    scorer = BM25Scorer([
        "linear cryptanalysis DES known plaintexts 2^43",
        "differential cryptanalysis chosen plaintexts 2^47",
        "AES block cipher S-box round function",
    ])
    scores = scorer.score("linear cryptanalysis data complexity")
    assert scores[0] > scores[1] > 0


def test_rrf_fuses_by_source_chunk():
    r1 = [{"metadata": {"source": "a", "chunk_id": 0}},
          {"metadata": {"source": "b", "chunk_id": 0}}]
    r2 = [{"metadata": {"source": "b", "chunk_id": 1}}]
    fused = _rrf([r1, r2])
    assert fused
    keys = [(i["metadata"]["source"], i["metadata"]["chunk_id"]) for i in fused]
    assert len(keys) == len(set(keys))  # no duplicates


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
