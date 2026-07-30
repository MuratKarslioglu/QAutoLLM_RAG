from qautollm.evaluation.retrieval_metrics import mrr, ndcg_at_k, recall_at_k

RETRIEVED = ["c3", "c1", "c4", "c2"]
GOLD = {"c1", "c2"}


def test_recall_at_k_partial():
    assert recall_at_k(RETRIEVED, GOLD, k=2) == 0.5  # only c1 in top-2
    assert recall_at_k(RETRIEVED, GOLD, k=4) == 1.0


def test_recall_at_k_no_gold_is_zero():
    assert recall_at_k(RETRIEVED, set(), k=2) == 0.0


def test_mrr_rank_of_first_relevant():
    assert mrr(RETRIEVED, GOLD) == 0.5  # c1 at rank 2


def test_mrr_no_hit_is_zero():
    assert mrr(RETRIEVED, {"nonexistent"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["c1", "c2"], {"c1", "c2"}, k=2) == 1.0


def test_ndcg_worse_ranking_is_lower():
    perfect = ndcg_at_k(["c1", "c2", "c3"], {"c1", "c2"}, k=3)
    worse = ndcg_at_k(["c3", "c1", "c2"], {"c1", "c2"}, k=3)
    assert worse < perfect


def test_ndcg_no_gold_is_zero():
    assert ndcg_at_k(RETRIEVED, set(), k=4) == 0.0
