"""Fixed / sentence / semantic chunking strategies (Brief §4, L1 cache level).

Chunk boundaries must depend only on ``(text, chunk_size, chunk_overlap)`` —
never on the embedding model — since L1's cache key is
``corpus_id, chunking_strategy, chunk_size, chunk_overlap`` (Brief §5.3) and
doesn't include ``embedding_model``. "semantic" is therefore implemented as
paragraph-aware packing (blank-line-separated paragraphs as atomic units)
rather than embedding-similarity-based splitting, which would otherwise
depend on the model doing the splitting.

``chunk_size``/``chunk_overlap`` are measured in whitespace-split words, not
model tokens — a deterministic, model-independent proxy that keeps chunking
decoupled from any specific tokenizer.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text.strip() else []


def _pack(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack whole ``units`` (words/sentences/paragraphs) into chunks
    of at most ``chunk_size`` words, carrying ``chunk_overlap`` trailing words
    into the next chunk. A single unit longer than ``chunk_size`` is hard-split.
    """
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        unit_words = unit.split()
        if current and len(current) + len(unit_words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-chunk_overlap:] if chunk_overlap else []
        current.extend(unit_words)
        while len(current) > chunk_size:
            chunks.append(" ".join(current[:chunk_size]))
            tail_start = chunk_size - chunk_overlap if chunk_overlap else chunk_size
            current = current[tail_start:]

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return _pack(text.split(), chunk_size, chunk_overlap)


def chunk_sentence(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return _pack(_split_sentences(text), chunk_size, chunk_overlap)


def chunk_semantic(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return _pack(_split_paragraphs(text), chunk_size, chunk_overlap)


_STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}


def chunk_document(text: str, strategy: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    try:
        fn = _STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}"
        ) from None
    return fn(text, chunk_size, chunk_overlap)
