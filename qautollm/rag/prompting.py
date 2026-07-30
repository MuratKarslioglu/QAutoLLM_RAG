"""Fixed prompt templates and context-budget packing (Brief §2: fixed
prompts for determinism — never construct a prompt ad hoc elsewhere).
"""

from __future__ import annotations

import re
from typing import Protocol

PROMPT_TEMPLATES: dict[str, str] = {
    "qwen_chat": (
        "You are a helpful assistant. Answer the question using only the "
        'provided context. If the answer is not in the context, say "I don\'t know". '
        "Reply with the shortest possible answer — a single word or short phrase, "
        "no explanation.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    # Same instructions as qwen_chat plus two fixed worked examples (one
    # yes/no, one short-entity) demonstrating the exact answer *format*
    # HotpotQA's EM/F1 scoring expects. Opt-in via a separate generator
    # catalog entry (configs/models/generator.yaml) — not the default,
    # since it hasn't been validated against the chat-template fix yet.
    "qwen_chat_fewshot": (
        "You are a helpful assistant. Answer the question using only the "
        'provided context. If the answer is not in the context, say "I don\'t know". '
        "Reply with the shortest possible answer — a single word or short phrase, "
        "no explanation. Follow the style of these examples:\n\n"
        "Context:\nScott Derrickson is an American director. Ed Wood was an American "
        "filmmaker.\n\nQuestion: Were Scott Derrickson and Ed Wood of the same "
        "nationality?\nAnswer: yes\n\n"
        "Context:\nThe Chief of Protocol is a position in the U.S. Department of "
        "State. Shirley Temple served as Chief of Protocol from 1976 to 1977.\n\n"
        "Question: What government position was held by the woman who portrayed "
        "Corliss Archer in the film Kiss and Tell?\nAnswer: Chief of Protocol\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    # Elicits a short graded answer *and* a full-sentence supporting claim in
    # one generation. Bare short spans ("yes", "Chief of Protocol") were
    # found to be un-scoreable by NLI-entailment faithfulness checking
    # (single words/phrases aren't well-formed hypotheses for an NLI
    # model — see the 2026-07-18 faithfulness investigation) regardless of
    # NLI model size; a full EVIDENCE sentence is. ANSWER still feeds
    # EM/F1 (unchanged scoring), EVIDENCE feeds faithfulness instead of the
    # bare answer (see ``parse_structured_answer`` and
    # ``evaluation/evaluator.py``). Distinct catalog entry/generator_id
    # (Brief cache-key discipline) so it never contaminates the base
    # entry's L4 cache.
    "qwen_chat_evidence": (
        "You are a helpful assistant. Answer the question using only the "
        'provided context. If the answer is not in the context, say "I don\'t know". '
        "Respond in exactly this two-line format, nothing else:\n"
        "ANSWER: <the shortest possible answer -- a single word or short phrase>\n"
        "EVIDENCE: <one full sentence, drawn from the context, that supports the answer>\n\n"
        "Follow the style of these examples:\n\n"
        "Context:\nScott Derrickson is an American director. Ed Wood was an American "
        "filmmaker.\n\nQuestion: Were Scott Derrickson and Ed Wood of the same "
        "nationality?\nANSWER: yes\nEVIDENCE: Scott Derrickson is an American director "
        "and Ed Wood was an American filmmaker, so both were American.\n\n"
        "Context:\nShirley Temple portrayed Corliss Archer in the film Kiss and Tell. "
        "Shirley Temple served as Chief of Protocol from 1976 to 1977.\n\n"
        "Question: What government position was held by the woman who portrayed "
        "Corliss Archer in the film Kiss and Tell?\nANSWER: Chief of Protocol\n"
        "EVIDENCE: Shirley Temple, who portrayed Corliss Archer, served as Chief of "
        "Protocol from 1976 to 1977.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n"
    ),
}

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"EVIDENCE:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def pack_context(chunks: list[str], context_budget: int, tokenizer: _Tokenizer) -> str:
    """Greedily include ``chunks`` (assumed already ranked best-first) up to
    ``context_budget`` tokens, dropping the rest.
    """
    pieces: list[str] = []
    used = 0
    for chunk in chunks:
        n_tokens = len(tokenizer.encode(chunk))
        if pieces and used + n_tokens > context_budget:
            break
        pieces.append(chunk)
        used += n_tokens
    return "\n\n".join(pieces)


def build_prompt(
    template_id: str,
    question: str,
    ranked_chunks: list[str],
    context_budget: int,
    tokenizer: _Tokenizer,
) -> str:
    try:
        template = PROMPT_TEMPLATES[template_id]
    except KeyError:
        raise ValueError(f"unknown prompt template {template_id!r}") from None
    context = pack_context(ranked_chunks, context_budget, tokenizer)
    return template.format(context=context, question=question)


def parse_structured_answer(raw_text: str) -> tuple[str, str]:
    """Splits a ``qwen_chat_evidence``-style ``ANSWER: .. / EVIDENCE: ..``
    generation into ``(short_answer, evidence_sentence)``.

    Templates that don't emit this format (``qwen_chat``, the toy fakes)
    have no ``ANSWER:``/``EVIDENCE:`` markers to match, so this falls back
    to ``(raw_text, "")`` — the same EM/F1-on-raw-text, faithfulness-on-
    nothing behavior those templates already had.

    In real ``qwen_chat_evidence`` generations the model very often skips
    the literal ``ANSWER:`` word (despite the few-shot examples) and just
    writes the short answer directly, but still reliably delimits it from
    the next line with ``EVIDENCE:`` -- confirmed by inspecting real cached
    L4 generations from the main_hotpot_v3 campaign, e.g. literally
    ``"Drifting\nEVIDENCE: D1NZ is a production car drifting series..."``
    with no ``ANSWER:`` anywhere. The old fallback (whole ``raw_text``) then
    fed the *entire* answer+evidence blob into F1/EM, which silently
    collapsed both metrics (observed: F1 ~0.52->0.14, EM ~0.43->0.02 between
    the old and new prompt template, despite retrieval/faithfulness staying
    healthy) even though the model's actual short answers were fine. When
    ``ANSWER:`` is missing but ``EVIDENCE:`` is present, take everything
    before ``EVIDENCE:`` as the answer instead of the whole blob. Only
    genuinely markerless text (no ``EVIDENCE:`` either) falls all the way
    back to ``raw_text`` -- a malformed reply degrades honestly rather than
    raising, it just no longer conflates two answer components as one.
    """
    answer_match = _ANSWER_RE.search(raw_text)
    evidence_match = _EVIDENCE_RE.search(raw_text)
    if answer_match:
        answer = answer_match.group(1).strip()
    elif evidence_match:
        answer = raw_text[: evidence_match.start()].strip()
    else:
        answer = raw_text.strip()
    evidence = evidence_match.group(1).strip() if evidence_match else ""
    return answer, evidence
