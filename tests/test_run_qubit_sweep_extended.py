import scripts.run_qubit_sweep_extended as rqse
from scripts.run_qubit_sweep_extended import run_one, run_sweep, select_scale

# Small n_candidates/mc_samples in every test below: the sweep script itself
# scales these down via HARDWARE_SCALE/QUICK_SCALE/GPU_SCALE, and tests want
# the cheapest values that still exercise the real code path. device_name
# is always the CPU backend here -- these tests must pass on a host with no
# GPU at all, and lightning.qubit is what every non-GPU scale uses anyway.
_SMALL = dict(n_candidates=8, mc_samples=4, device_name="lightning.qubit")


def test_run_one_structured_bo_above_brief_cap():
    row = run_one(
        n_groups=2,
        qubits_per_group=6,  # > Brief D3's 5-qubit cap
        interaction=0.0,
        optimizer_name="structured_bo",
        seed=0,
        budget=4,
        init_points=2,
        **_SMALL,
    )
    assert row["qubits_per_group"] == 6
    assert row["hypervolume"] >= 0.0
    assert row["failed"] is False


def test_run_one_qkernel_bo_above_brief_cap_uses_lightning():
    # small qubit count so this stays fast in the "not slow" suite --
    # this is the exact code path run_qubit_sweep_extended.py uses at
    # qubits_per_group up to 25, just at a size cheap enough for CI.
    row = run_one(
        n_groups=2,
        qubits_per_group=6,
        interaction=0.0,
        optimizer_name="qkernel_bo",
        seed=0,
        budget=4,
        init_points=2,
        **_SMALL,
    )
    assert row["qubits_per_group"] == 6
    assert row["hypervolume"] >= 0.0
    assert row["failed"] is False


def test_run_one_is_deterministic_given_seed():
    kwargs = dict(
        n_groups=2,
        qubits_per_group=6,
        interaction=1.0,
        optimizer_name="qkernel_bo",
        seed=1,
        budget=4,
        init_points=2,
        **_SMALL,
    )
    a = run_one(**kwargs)
    b = run_one(**kwargs)
    assert a["hypervolume"] == b["hypervolume"]


def test_run_one_calls_malloc_trim_every_trim_every_iterations(monkeypatch):
    calls = []
    monkeypatch.setattr(rqse, "_malloc_trim", lambda: calls.append(1))
    monkeypatch.setattr(rqse, "_TRIM_EVERY", 2)
    run_one(
        n_groups=2,
        qubits_per_group=6,
        interaction=0.0,
        optimizer_name="structured_bo",
        seed=0,
        budget=6,
        init_points=2,
        **_SMALL,
    )
    assert len(calls) == 3  # budget=6, trim every 2 -> after iterations 2, 4, 6


def test_run_sweep_covers_every_combination():
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6, 7),
        interactions=(0.0, 1.0),
        budget=3,
        init_points=2,
        n_seeds=2,
        **_SMALL,
    )
    # 2 qubit_counts x 2 interactions x 2 optimizers x 2 seeds
    assert len(df) == 16
    assert set(df["qubits_per_group"]) == {6, 7}
    assert set(df["optimizer"]) == {"structured_bo", "qkernel_bo"}
    assert not df["failed"].any()


class _FakePool:
    """Stands in for ``multiprocessing.Pool`` and always raises whatever
    exception it's constructed with, simulating a worker that hit a memory
    limit -- without actually spawning a process or touching any real
    rlimits.
    """

    def __init__(self, exc: BaseException):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def apply(self, fn, args):
        raise self._exc


class _FakeSpawnContext:
    def __init__(self, exc: BaseException):
        self._exc = exc

    def Pool(self, *args, **kwargs):
        return _FakePool(self._exc)


class _FlakyThenSucceedsPool:
    """Raises ``exc`` on the first ``n_failures`` calls to ``apply`` *for
    the qkernel_bo combo specifically* (matching run_one()'s real args
    order: index 3 is ``optimizer_name``), then returns a row built from
    the real args -- structured_bo combos in the same sweep always succeed
    immediately, matching reality (that path never touches lightning.gpu at
    all). Simulates lightning.gpu's observed transient-then-recovers
    behavior (see ``_GPU_ERROR_MAX_RETRIES``'s comment) without spawning
    real processes.
    """

    def __init__(self, exc: BaseException, n_failures: int):
        self._exc = exc
        self._remaining_failures = n_failures
        self.qkernel_call_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def apply(self, fn, args):
        n_groups, qubits_per_group, interaction, optimizer_name, seed, budget = args[:6]
        if optimizer_name == "qkernel_bo":
            self.qkernel_call_count += 1
            if self._remaining_failures > 0:
                self._remaining_failures -= 1
                raise self._exc
        return {
            "n_groups": n_groups,
            "qubits_per_group": qubits_per_group,
            "interaction": interaction,
            "optimizer": optimizer_name,
            "seed": seed,
            "budget": budget,
            "hypervolume": 1.23,
            "wall_clock_s": 0.5,
            "failed": False,
        }


class _FlakyThenSucceedsContext:
    def __init__(self, pool: _FlakyThenSucceedsPool):
        self._pool = pool

    def Pool(self, *args, **kwargs):
        return self._pool


def test_gpu_error_retried_and_succeeds_within_retry_budget(monkeypatch):
    gpu_flake = RuntimeError(
        "[cuStateVec_helpers.hpp][Method:make_shared_cusv_handle]: "
        "Error in PennyLane Lightning: custatevec not initialized"
    )
    # Fails on the first 2 attempts (< _GPU_ERROR_MAX_RETRIES=3), succeeds
    # on the 3rd -- the combo should end up recorded as a real success, not
    # "failed", and the retry loop should have stopped as soon as it did.
    fake_pool = _FlakyThenSucceedsPool(gpu_flake, n_failures=2)
    monkeypatch.setattr(rqse.mp, "get_context", lambda name: _FlakyThenSucceedsContext(fake_pool))
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6,),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        n_candidates=8,
        mc_samples=4,
        device_name="lightning.gpu",
    )
    assert fake_pool.qkernel_call_count == 3  # 2 failures + 1 success
    matching = df[df["optimizer"] == "qkernel_bo"]
    assert len(matching) == 1
    assert bool(matching.iloc[0]["failed"]) is False
    assert matching.iloc[0]["hypervolume"] == 1.23
    assert not df["failed"].any()  # structured_bo combo also present and never touched


def test_gpu_error_gives_up_after_max_retries(monkeypatch):
    gpu_flake = RuntimeError(
        "[cuStateVec_helpers.hpp][Method:make_shared_cusv_handle]: "
        "Error in PennyLane Lightning: custatevec not initialized"
    )
    # Fails on every attempt, including all _GPU_ERROR_MAX_RETRIES retries --
    # must eventually give up and mark the combo failed, not retry forever.
    fake_pool = _FlakyThenSucceedsPool(gpu_flake, n_failures=rqse._GPU_ERROR_MAX_RETRIES + 5)
    monkeypatch.setattr(rqse.mp, "get_context", lambda name: _FlakyThenSucceedsContext(fake_pool))
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6,),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        n_candidates=8,
        mc_samples=4,
        device_name="lightning.gpu",
    )
    assert fake_pool.qkernel_call_count == rqse._GPU_ERROR_MAX_RETRIES  # gave up, no infinite retry
    matching = df[df["optimizer"] == "qkernel_bo"]
    assert len(matching) == 1
    assert bool(matching.iloc[0]["failed"]) is True


def test_run_sweep_marks_combo_failed_on_memory_error(monkeypatch):
    monkeypatch.setattr(rqse.mp, "get_context", lambda name: _FakeSpawnContext(MemoryError()))
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6,),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        **_SMALL,
    )
    # 1 qubit_count x 1 interaction x 2 optimizers x 1 seed
    assert len(df) == 2
    assert df["failed"].all()
    assert df["hypervolume"].isna().all()


def test_run_sweep_marks_combo_failed_on_gpu_out_of_memory(monkeypatch):
    # cuStateVec's own OOM surfaces as a plain RuntimeError with "out of
    # memory" in the message (confirmed against lightning.gpu 2026-08-06) --
    # must be treated the same as the CPU-side MemoryError, not crash the
    # whole sweep.
    gpu_oom = RuntimeError(
        "[DevicePool.hpp][Method:getTotalDevices]: Error in PennyLane Lightning: out of memory"
    )
    monkeypatch.setattr(rqse.mp, "get_context", lambda name: _FakeSpawnContext(gpu_oom))
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6,),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        n_candidates=8,
        mc_samples=4,
        device_name="lightning.gpu",
    )
    assert len(df) == 2
    assert df["failed"].all()
    assert df["hypervolume"].isna().all()


def test_run_sweep_marks_combo_failed_on_gpu_cublas_allocation_failure(monkeypatch):
    # A second observed phrasing (from cuStateVec/cuBLAS's linear-algebra
    # path rather than its device-count check, confirmed 2026-08-06) --
    # must be recognized too, not just the first "out of memory" wording.
    cublas_oom = RuntimeError(
        "[LinearAlg.hpp][Method:CublasCaller]: Error in PennyLane Lightning: "
        "cuBLAS memory allocation failed"
    )
    monkeypatch.setattr(rqse.mp, "get_context", lambda name: _FakeSpawnContext(cublas_oom))
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6,),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        n_candidates=8,
        mc_samples=4,
        device_name="lightning.gpu",
    )
    assert len(df) == 2
    assert df["failed"].all()
    assert df["hypervolume"].isna().all()


def test_run_sweep_marks_combo_failed_on_gpu_custatevec_context_failure(monkeypatch):
    # A third observed phrasing, and not even memory-related this time (a
    # cuStateVec handle/context failure, confirmed 2026-08-06 live during a
    # real multi-combo run) -- proves the fix is matching on
    # _LIGHTNING_GPU_ERROR_TAG (the library's own error-tag convention)
    # rather than an enumerated list of memory-specific phrasings, since
    # this message contains neither "out of memory" nor "allocation
    # failed".
    context_failure = RuntimeError(
        "[cuStateVec_helpers.hpp][Method:make_shared_cusv_handle]: "
        "Error in PennyLane Lightning: custatevec not initialized"
    )
    monkeypatch.setattr(rqse.mp, "get_context", lambda name: _FakeSpawnContext(context_failure))
    df = run_sweep(
        n_groups=2,
        qubit_counts=(6,),
        interactions=(0.0,),
        budget=3,
        init_points=2,
        n_seeds=1,
        n_candidates=8,
        mc_samples=4,
        device_name="lightning.gpu",
    )
    assert len(df) == 2
    assert df["failed"].all()
    assert df["hypervolume"].isna().all()


def test_run_sweep_reraises_unrelated_runtime_error(monkeypatch):
    # A RuntimeError that *isn't* an out-of-memory message is a real bug --
    # must propagate, not get silently recorded as a "failed" (capacity
    # limit) row.
    monkeypatch.setattr(
        rqse.mp, "get_context", lambda name: _FakeSpawnContext(RuntimeError("boom"))
    )
    try:
        run_sweep(
            n_groups=2,
            qubit_counts=(6,),
            interactions=(0.0,),
            budget=3,
            init_points=2,
            n_seeds=1,
            **_SMALL,
        )
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected RuntimeError to propagate")


def test_select_scale_quick_flag_wins_regardless_of_ram(monkeypatch):
    monkeypatch.setattr(rqse, "ram_probe", lambda: {"total_gb": 4.0, "available_gb": 4.0})
    assert select_scale(quick=True, force_full=True) is rqse.QUICK_SCALE


def test_select_scale_force_full_wins_on_low_ram(monkeypatch):
    monkeypatch.setattr(rqse, "ram_probe", lambda: {"total_gb": 8.0, "available_gb": 4.0})
    assert select_scale(quick=False, force_full=True) is rqse.FULL_SCALE


def test_select_scale_never_auto_selects_gpu_even_if_available(monkeypatch):
    # GPU_SCALE is opt-in only via --gpu (2026-08-06, reverted after real
    # GPU_SCALE runs failed on effectively every qkernel_bo combo at
    # production BO scale) -- a working lightning.gpu device must NOT by
    # itself change which CPU tier gets picked.
    monkeypatch.setattr(rqse, "lightning_gpu_available", lambda: True)
    monkeypatch.setattr(rqse, "ram_probe", lambda: {"total_gb": 512.0, "available_gb": 500.0})
    assert select_scale(quick=False, force_full=False, force_gpu=False) is rqse.FULL_SCALE


def test_select_scale_force_gpu_selects_gpu_scale_when_available(monkeypatch):
    monkeypatch.setattr(rqse, "lightning_gpu_available", lambda: True)
    assert select_scale(quick=False, force_full=False, force_gpu=True) is rqse.GPU_SCALE


def test_select_scale_force_gpu_raises_when_gpu_unavailable(monkeypatch):
    monkeypatch.setattr(rqse, "lightning_gpu_available", lambda: False)
    try:
        select_scale(quick=False, force_full=False, force_gpu=True)
    except RuntimeError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when --gpu forced without a GPU")


def test_select_scale_low_ram_auto_selects_hardware_scale(monkeypatch):
    monkeypatch.setattr(
        rqse,
        "ram_probe",
        lambda: {"total_gb": 31.0, "available_gb": rqse._LOW_RAM_THRESHOLD_GB - 1},
    )
    assert select_scale(quick=False, force_full=False) is rqse.HARDWARE_SCALE


def test_select_scale_high_ram_auto_selects_full_scale(monkeypatch):
    monkeypatch.setattr(
        rqse,
        "ram_probe",
        lambda: {"total_gb": 64.0, "available_gb": rqse._LOW_RAM_THRESHOLD_GB + 10},
    )
    assert select_scale(quick=False, force_full=False) is rqse.FULL_SCALE


def test_compute_worker_ram_limit_bytes_scales_with_total_ram_for_cpu(monkeypatch):
    monkeypatch.setattr(rqse, "ram_probe", lambda: {"total_gb": 32.0, "available_gb": 20.0})
    limit_bytes = rqse._compute_worker_ram_limit_bytes("lightning.qubit")
    expected = int(32.0 * rqse._WORKER_RAM_FRACTION * 1024**3)
    assert limit_bytes == expected


def test_compute_worker_ram_limit_bytes_floors_for_gpu(monkeypatch):
    # On a low-RAM host, WORKER_RAM_FRACTION alone would land below what
    # cuStateVec's own context-init virtual-address-space needs (measured
    # ~20 GiB, 2026-08-06) -- GPU devices must get at least
    # _MIN_GPU_WORKER_RAM_LIMIT_GB regardless of the fraction.
    monkeypatch.setattr(rqse, "ram_probe", lambda: {"total_gb": 16.0, "available_gb": 10.0})
    limit_bytes = rqse._compute_worker_ram_limit_bytes("lightning.gpu")
    assert limit_bytes == int(rqse._MIN_GPU_WORKER_RAM_LIMIT_GB * 1024**3)


def test_compute_worker_ram_limit_bytes_gpu_uses_fraction_when_larger(monkeypatch):
    monkeypatch.setattr(rqse, "ram_probe", lambda: {"total_gb": 128.0, "available_gb": 100.0})
    limit_bytes = rqse._compute_worker_ram_limit_bytes("lightning.gpu")
    expected = int(128.0 * rqse._WORKER_RAM_FRACTION * 1024**3)
    assert limit_bytes == expected
