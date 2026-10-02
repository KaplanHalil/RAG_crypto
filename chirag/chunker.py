"""Paragraph-aware overlapping text chunker.

Chunking walks the document paragraph by paragraph (``\\n\\n`` separated),
accumulates words into chunks of at most ``chunk_size`` characters, and
carries a small overlapping tail into the next chunk so that information
split across a boundary remains retrievable. Oversized paragraphs are
split internally with the same overlap.

The chunker is deterministic: identical input produces identical chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_words = max(50, chunk_size // 5)
        self.overlap_words = max(10, chunk_overlap // 5)

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split ``text`` into overlapping chunks, tagged with ``metadata``."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        pieces: List[str] = []
        for para in paragraphs:
            words = para.split()
            if len(words) > self.max_words:
                step = max(1, self.max_words - self.overlap_words)
                for i in range(0, len(words), step):
                    pieces.append(" ".join(words[i:i + self.max_words]))
            else:
                pieces.append(para)

        chunks: List[Chunk] = []
        current: List[str] = []
        current_words = 0
        chunk_idx = 0

        def flush() -> None:
            nonlocal current, current_words, chunk_idx
            if not current:
                return
            text_joined = "\n\n".join(current).strip()
            if not text_joined:
                return
            chunks.append(Chunk(
                text=text_joined,
                metadata={**metadata, "chunk_id": chunk_idx,
                          "char_count": len(text_joined)},
            ))
            chunk_idx += 1
            # Keep the last overlap_words as a running head for the next chunk.
            tail_words = text_joined.split()[-self.overlap_words:]
            current = [" ".join(tail_words)]
            current_words = len(tail_words)

        for piece in pieces:
            piece_words = len(piece.split())
            if current and current_words + piece_words > self.max_words:
                flush()
            current.append(piece)
            current_words += piece_words

        flush()
        return chunks
