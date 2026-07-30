import pytest

from qautollm.rag.prompting import build_prompt, pack_context, parse_structured_answer


class _WordTokenizer:
    """Fake tokenizer: one token per whitespace-split word (fast, no deps)."""

    def encode(self, text: str) -> list[int]:
        return text.split()


def test_pack_context_stops_at_budget():
    tokenizer = _WordTokenizer()
    chunks = ["one two three", "four five", "six seven eight nine"]
    packed = pack_context(chunks, context_budget=5, tokenizer=tokenizer)
    assert packed == "one two three\n\nfour five"


def test_pack_context_always_includes_first_chunk_even_if_over_budget():
    tokenizer = _WordTokenizer()
    chunks = ["one two three four five six seven"]
    packed = pack_context(chunks, context_budget=2, tokenizer=tokenizer)
    assert packed == chunks[0]


def test_pack_context_empty_input():
    tokenizer = _WordTokenizer()
    assert pack_context([], context_budget=100, tokenizer=tokenizer) == ""


def test_build_prompt_fills_template():
    tokenizer = _WordTokenizer()
    prompt = build_prompt("qwen_chat", "what is quantum?", ["quantum is weird"], 100, tokenizer)
    assert "what is quantum?" in prompt
    assert "quantum is weird" in prompt


def test_build_prompt_rejects_unknown_template():
    tokenizer = _WordTokenizer()
    with pytest.raises(ValueError, match="unknown prompt template"):
        build_prompt("nonexistent", "q", ["c"], 100, tokenizer)


def test_build_prompt_evidence_template_fills_and_has_no_placeholder_leftovers():
    tokenizer = _WordTokenizer()
    prompt = build_prompt(
        "qwen_chat_evidence", "what is quantum?", ["quantum is weird"], 100, tokenizer
    )
    assert "what is quantum?" in prompt
    assert "quantum is weird" in prompt
    assert "ANSWER:" in prompt
    assert "EVIDENCE:" in prompt


def test_parse_structured_answer_well_formed():
    text = "ANSWER: Chief of Protocol\nEVIDENCE: Shirley Temple served as Chief of Protocol."
    answer, evidence = parse_structured_answer(text)
    assert answer == "Chief of Protocol"
    assert evidence == "Shirley Temple served as Chief of Protocol."


def test_parse_structured_answer_case_insensitive_and_extra_whitespace():
    text = "answer:   yes  \nevidence:   both were American.  \n"
    answer, evidence = parse_structured_answer(text)
    assert answer == "yes"
    assert evidence == "both were American."


def test_parse_structured_answer_falls_back_when_no_markers_present():
    # old-style templates (qwen_chat, toy fakes) never emit these markers.
    answer, evidence = parse_structured_answer("Chief of Protocol")
    assert answer == "Chief of Protocol"
    assert evidence == ""


def test_parse_structured_answer_missing_evidence_still_recovers_answer():
    answer, evidence = parse_structured_answer("ANSWER: yes\n")
    assert answer == "yes"
    assert evidence == ""


def test_parse_structured_answer_recovers_when_model_skips_answer_marker():
    # Real behavior observed in the main_hotpot_v3 campaign's cached L4
    # generations: the model very often omits the literal "ANSWER:" word
    # despite the few-shot examples, but still reliably writes "EVIDENCE:"
    # on the next line. Before this fix, the fallback used the *entire*
    # raw_text (answer + evidence) as the F1/EM answer, collapsing both
    # metrics even though the model's actual short answer was correct.
    text = "Drifting\nEVIDENCE: D1NZ is a production car drifting series in New Zealand."
    answer, evidence = parse_structured_answer(text)
    assert answer == "Drifting"
    assert evidence == "D1NZ is a production car drifting series in New Zealand."


def test_parse_structured_answer_truncated_generation_no_trailing_newline():
    # max_new_tokens can cut generation off mid-sentence with no newline.
    answer, evidence = parse_structured_answer(
        "ANSWER: yes\nEVIDENCE: Scott Derrickson is an American director and Ed"
    )
    assert answer == "yes"
    assert evidence == "Scott Derrickson is an American director and Ed"
