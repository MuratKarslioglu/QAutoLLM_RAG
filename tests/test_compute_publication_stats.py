import json

import pytest

from scripts import compute_publication_stats as cps


@pytest.fixture(autouse=True)
def _fake_results_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results" / "main_hotpot_v3"
    results_dir.mkdir(parents=True)
    monkeypatch.setattr(cps, "RESULTS_DIR", results_dir)
    return results_dir


def test_holm_correction_orders_by_significance_and_flags_survivors(_fake_results_dir):
    sig = {
        "mean_f1": {"p_value": 0.04},
        "mean_faithfulness": {"p_value": 0.004},
        "mean_latency_s": {"p_value": 0.5},
    }
    (_fake_results_dir / "qkernel_vs_structured_significance.json").write_text(json.dumps(sig))

    holm = cps.holm_correction()

    # smallest raw p tested first (m=3 multiplier), largest tested last (x1)
    assert holm.iloc[0]["metric"] == "mean_faithfulness"
    assert holm.iloc[0]["holm_adjusted_p"] == pytest.approx(0.012, abs=1e-6)
    assert holm.iloc[0]["reject_at_0_05"]
    assert not holm.iloc[1]["reject_at_0_05"]  # mean_f1: 0.04 * 2 = 0.08
    assert not holm.iloc[2]["reject_at_0_05"]


def test_heldout_significance_flips_sign_and_pairs_by_seed(_fake_results_dir):
    # quantum (qkernel_bo) uniformly worse F1 (less negative internal objective
    # is objective is stored negated, so a *larger* -f1 magnitude means better
    # raw F1); construct two seeds where classical wins both.
    report = {
        "qkernel_bo_seed1": [{"test_objectives": [-0.30, -0.10, 0.80]}],
        "structured_bo_seed1": [{"test_objectives": [-0.35, -0.12, 0.79]}],
        "qkernel_bo_seed2": [{"test_objectives": [-0.31, -0.11, 0.81]}],
        "structured_bo_seed2": [{"test_objectives": [-0.36, -0.13, 0.78]}],
    }
    (_fake_results_dir / "test_split_report.json").write_text(json.dumps(report))

    out = cps.heldout_significance()

    f1_row = out[out["metric"] == "heldout_f1_mean"].iloc[0]
    assert f1_row["quantum_mean"] == pytest.approx(0.305)
    assert f1_row["classical_mean"] == pytest.approx(0.355)
    assert f1_row["wins_classical"] == 2
    assert f1_row["wins_quantum"] == 0
