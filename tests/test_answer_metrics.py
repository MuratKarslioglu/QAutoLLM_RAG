from qautollm.evaluation.answer_metrics import exact_match, token_f1


def test_exact_match_ignores_case_punctuation_and_articles():
    assert exact_match("The Cat.", "cat") == 1.0
    assert exact_match("a dog", "dog") == 1.0
    assert exact_match("cat", "dog") == 0.0


def test_token_f1_perfect_match():
    assert token_f1("paris", "Paris") == 1.0


def test_token_f1_partial_overlap():
    f1 = token_f1("the capital city", "capital city of france")
    assert 0.0 < f1 < 1.0


def test_token_f1_no_overlap():
    assert token_f1("cats", "quantum computing") == 0.0


def test_token_f1_both_empty_after_normalization():
    assert token_f1("the a an", "the an a") == 1.0
