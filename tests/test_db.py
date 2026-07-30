from qautollm.store.db import ArtifactDB


def test_artifact_roundtrip(tmp_path):
    db = ArtifactDB(tmp_path / "db.sqlite")
    assert db.get_artifact("L4", "missing") is None
    db.put_artifact("L4", "abc", {"text": "hello", "tokens": [1, 2, 3]}, latency_s=1.5)
    value, latency_s = db.get_artifact("L4", "abc")
    assert value == {"text": "hello", "tokens": [1, 2, 3]}
    assert latency_s == 1.5


def test_artifact_survives_reconnect(tmp_path):
    path = tmp_path / "db.sqlite"
    ArtifactDB(path).put_artifact("L5", "k1", 0.42, latency_s=0.01)
    reopened = ArtifactDB(path)
    value, latency_s = reopened.get_artifact("L5", "k1")
    assert value == 0.42
    assert latency_s == 0.01


def test_trial_write_and_read(tmp_path):
    db = ArtifactDB(tmp_path / "db.sqlite")
    row = {
        "experiment_id": "exp1",
        "trial_index": 0,
        "optimizer_name": "random",
        "seed": 42,
        "dataset_id": "toy",
        "config_hash": "deadbeef",
        "config_json": "{}",
        "objective_f1": -0.8,
        "objective_faithfulness": -0.9,
        "objective_latency_s": 1.2,
        "em": 0.5,
        "recall_at_k": 0.7,
        "ndcg": 0.6,
        "tokens_prompt": 100,
        "tokens_generated": 20,
        "peak_vram_mb": 512.0,
        "cache_hits_json": "{}",
        "duplicate_flag": 0,
        "wall_clock_s": 2.0,
        "status": "ok",
        "timestamp": "2026-01-01T00:00:00",
    }
    db.write_trial(row)
    trials = db.read_trials("exp1")
    assert len(trials) == 1
    assert trials[0]["config_hash"] == "deadbeef"


def test_trial_upsert_replaces_same_key(tmp_path):
    db = ArtifactDB(tmp_path / "db.sqlite")
    base = {"experiment_id": "exp1", "trial_index": 0, "status": "running"}
    db.write_trial(base)
    db.write_trial({**base, "status": "ok"})
    trials = db.read_trials("exp1")
    assert len(trials) == 1
    assert trials[0]["status"] == "ok"


def test_kernel_log_appends(tmp_path):
    db = ArtifactDB(tmp_path / "db.sqlite")
    db.write_kernel_log(
        {
            "experiment_id": "exp1",
            "trial_index": 0,
            "shots": None,
            "psd_clipped_mass": 0.0,
            "kernel_time_s": 0.1,
        }
    )
    db.write_kernel_log(
        {
            "experiment_id": "exp1",
            "trial_index": 1,
            "shots": 1024,
            "psd_clipped_mass": 0.02,
            "kernel_time_s": 0.2,
        }
    )
    rows = db._conn.execute("SELECT * FROM kernel_log ORDER BY trial_index").fetchall()
    assert len(rows) == 2


def test_experiment_write(tmp_path):
    db = ArtifactDB(tmp_path / "db.sqlite")
    db.write_experiment({"experiment_id": "exp1", "status": "running", "started_at": "now"})
    db.write_experiment({"experiment_id": "exp1", "status": "finished", "started_at": "now"})
    row = db._conn.execute(
        "SELECT status FROM experiments WHERE experiment_id = ?", ("exp1",)
    ).fetchone()
    assert row[0] == "finished"
