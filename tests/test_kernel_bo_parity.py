"""Brief P5 gate: "quantum and classical BO diff only in kernel factory
(assert in test)".
"""

import hashlib
from pathlib import Path

import numpy as np

from qautollm.optimizers.qkernel_bo import QKernelBO
from qautollm.optimizers.structured_bo import KernelBO, StructuredBO
from qautollm.space.encoding import encode
from qautollm.space.search_space import SearchSpace

RAG_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "search_spaces" / "rag_v2.yaml"
_SPACE = SearchSpace.from_yaml(RAG_V2_PATH)

_SHARED_METHODS = ["suggest", "update", "state_dict", "load_state_dict", "_suggest_via_ehvi"]


def _init_design(n=3, seed=1):
    return [encode(raw, _SPACE) for raw in _SPACE.sobol_design(n, seed=seed)]


def _fake_objectives(cfg) -> np.ndarray:
    seed = int(hashlib.sha256(cfg.config_hash.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=3)


def test_both_variants_subclass_the_same_kernel_bo():
    assert issubclass(StructuredBO, KernelBO)
    assert issubclass(QKernelBO, KernelBO)


def test_both_variants_share_every_kernel_bo_method_unmodified():
    """The entire GP/EHVI/archive/resume machinery must be literally the
    same function object for both — proving neither subclass reimplements
    or special-cases any of it.
    """
    for name in _SHARED_METHODS:
        assert getattr(StructuredBO, name) is getattr(KernelBO, name), name
        assert getattr(QKernelBO, name) is getattr(KernelBO, name), name


def test_variants_differ_only_in_group_kernel_factory():
    structured = StructuredBO(_SPACE, seed=1, init_design=[])
    quantum = QKernelBO(_SPACE, seed=1, init_design=[])

    assert structured.group_kernel_factory is not quantum.group_kernel_factory
    # every other piece of __init__ state comes from the identical
    # KernelBO.__init__ path, with identical defaults:
    assert structured._n_candidates == quantum._n_candidates
    assert structured._mc_samples == quantum._mc_samples
    assert structured._configs == quantum._configs == []
    assert structured._objectives == quantum._objectives == []
    assert structured.last_kernel_time_s == quantum.last_kernel_time_s is None
    assert structured.last_psd_clipped_mass == quantum.last_psd_clipped_mass == 0.0


def test_qkernel_bo_suggest_update_loop_produces_valid_configs():
    opt = QKernelBO(_SPACE, seed=42, init_design=_init_design(n=3))
    for _ in range(6):
        cfg = opt.suggest()
        _SPACE.validate(cfg.raw)
        opt.update(cfg, _fake_objectives(cfg))
    assert opt.last_kernel_time_s is not None  # went through the EHVI path at least once


def test_qkernel_bo_resume_reproduces_identical_continuation():
    init = _init_design(n=2)

    def _run(opt, n_steps):
        out = []
        for _ in range(n_steps):
            cfg = opt.suggest()
            out.append(cfg)
            opt.update(cfg, _fake_objectives(cfg))
        return out

    opt_a = QKernelBO(_SPACE, seed=42, init_design=init)
    _run(opt_a, 3)
    continuation_a = _run(opt_a, 2)

    opt_b = QKernelBO(_SPACE, seed=42, init_design=init)
    _run(opt_b, 3)
    checkpoint = opt_b.state_dict()

    opt_c = QKernelBO(_SPACE, seed=42, init_design=init)
    opt_c.load_state_dict(checkpoint)
    continuation_c = _run(opt_c, 2)

    assert [c.config_hash for c in continuation_a] == [c.config_hash for c in continuation_c]
