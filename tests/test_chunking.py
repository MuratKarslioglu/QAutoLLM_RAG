from qautollm.rag.chunking import chunk_document, chunk_fixed, chunk_semantic, chunk_sentence

LONG_TEXT = " ".join(f"word{i}" for i in range(1, 251))  # 250 words


def test_fixed_chunk_sizes_respect_budget():
    chunks = chunk_fixed(LONG_TEXT, chunk_size=100, chunk_overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert len(chunks) >= 3


def test_fixed_chunk_overlap_reconstructs_all_words():
    chunks = chunk_fixed(LONG_TEXT, chunk_size=100, chunk_overlap=20)
    # every original word must appear in at least one chunk, in order
    all_words = " ".join(chunks).split()
    seen = set(all_words)
    assert set(LONG_TEXT.split()) <= seen


def test_fixed_no_overlap_is_exact_partition():
    chunks = chunk_fixed(LONG_TEXT, chunk_size=100, chunk_overlap=0)
    rejoined = " ".join(chunks)
    assert rejoined == LONG_TEXT


def test_sentence_chunking_keeps_sentences_whole():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_sentence(text, chunk_size=6, chunk_overlap=0)
    for chunk in chunks:
        assert chunk.strip().endswith(".")


def test_semantic_chunking_respects_paragraph_boundaries_when_possible():
    text = "Para one sentence.\n\nPara two sentence.\n\nPara three sentence."
    chunks = chunk_semantic(text, chunk_size=100, chunk_overlap=0)
    # generous budget: all three paragraphs fit in a single chunk
    assert len(chunks) == 1
    assert "Para one" in chunks[0]
    assert "Para three" in chunks[0]


def test_chunk_document_dispatches_by_strategy_name():
    for strategy in ("fixed", "sentence", "semantic"):
        chunks = chunk_document(LONG_TEXT, strategy, chunk_size=100, chunk_overlap=10)
        assert chunks


def test_chunk_document_rejects_unknown_strategy():
    import pytest

    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_document(LONG_TEXT, "nonexistent", chunk_size=100, chunk_overlap=10)


def test_deterministic_across_calls():
    a = chunk_document(LONG_TEXT, "sentence", 100, 20)
    b = chunk_document(LONG_TEXT, "sentence", 100, 20)
    assert a == b
