from qautollm.space.constraints import (
    ConditionalRule,
    ConstraintViolation,
    strip_inactive,
    validate,
)

HYBRID_RULE = ConditionalRule(variable="dense_weight", depends_on="hybrid_on", active_when=1)
RERANK_RULE = ConditionalRule(variable="reranker_top_n", depends_on="reranker_on", active_when=1)
RULES = [HYBRID_RULE, RERANK_RULE]


def test_is_active_true_and_false():
    assert HYBRID_RULE.is_active({"hybrid_on": 1}) is True
    assert HYBRID_RULE.is_active({"hybrid_on": 0}) is False


def test_is_active_missing_gate_raises():
    try:
        HYBRID_RULE.is_active({})
    except ConstraintViolation:
        pass
    else:
        raise AssertionError("expected ConstraintViolation for missing gate variable")


def test_validate_accepts_consistent_configs():
    validate({"hybrid_on": 1, "dense_weight": 0.5, "reranker_on": 0}, RULES)
    validate({"hybrid_on": 0, "reranker_on": 1, "reranker_top_n": 5}, RULES)


def test_validate_rejects_present_while_inactive():
    try:
        validate({"hybrid_on": 0, "dense_weight": 0.5}, RULES)
    except ConstraintViolation:
        pass
    else:
        raise AssertionError("expected rejection of dense_weight present with hybrid_on=0")


def test_validate_rejects_missing_while_active():
    try:
        validate({"hybrid_on": 1}, RULES)
    except ConstraintViolation:
        pass
    else:
        raise AssertionError("expected rejection of missing dense_weight with hybrid_on=1")


def test_strip_inactive_drops_stale_conditional_keys():
    raw = {"hybrid_on": 0, "dense_weight": 0.7, "reranker_on": 1, "reranker_top_n": 5}
    cleaned = strip_inactive(raw, RULES)
    assert cleaned == {"hybrid_on": 0, "reranker_on": 1, "reranker_top_n": 5}
