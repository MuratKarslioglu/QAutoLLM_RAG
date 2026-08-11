"""Quantum fidelity GroupKernel (Brief §5.4, D3): PennyLane angle-encoding
feature map + statevector simulation, ``K_g(z,z') = |<φ(z)|φ(z')>|^2``.

Same ``GroupKernel`` factory shape as ``classical.py`` so
``qkernel_bo.py``/``structured_bo.py`` differ *only* in which factory they
pass into :class:`qautollm.kernels.structural.StructuralKernel` (Brief §5.4).

D3: ≤5 qubits per group (enforced), statevector simulation on CPU,
``shots ∈ {None, 1024, 256}``.

**Finite-shot strategy.** Re-simulating a real overlap/SWAP-test circuit
``shots`` times for every one of the O(n²) pairs in a Gram matrix is not
feasible under the <5%-of-evaluation-time budget. Instead: compute the
*exact* fidelity once via statevector inner products, then sample shot
noise directly from ``Binomial(shots, p_exact) / shots``. This is not an
approximation of the measurement statistics — it *is* the measurement
statistics: estimating |<φ(z)|φ(z')>|^2 by measuring an overlap circuit
``shots`` times literally produces a ``Binomial(shots, p)/shots``-
distributed estimate, so sampling that distribution directly is exact in
distribution while being O(n²) cheaper than simulating gate-by-gate per
shot.

Noise is seeded from a hash of ``(seed, z1, z2)``, not call order: this
kernel has no learnable parameters, so GPyTorch/L-BFGS-B may call
``forward()`` on the *same* inputs many times while optimizing other
kernels' hyperparameters, and each such call must return the identical
Gram matrix or the optimizer's line search sees a noisy, inconsistent
objective.

**PSD projection** (finite shots only) applies eigenvalue clipping to the
*training* Gram matrix specifically (square, ``x1 is x2``) — that's the
one BoTorch actually Cholesky-factorizes; candidate-scoring kernel
evaluations aren't symmetric square matrices and don't need it. The
clipped mass (sum of |negative eigenvalues|) is exposed via
``last_psd_clipped_mass`` for ``kernel_log`` (Brief §8).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import pennylane as qml
import torch
from gpytorch.kernels import Kernel

MAX_QUBITS = 5  # Brief D3

# Statevectors are 2**n_qubits complex128 each (256 B at Brief D3's 5-qubit
# cap, 16 MiB at run_qubit_sweep_extended.py's 20-qubit exploratory
# override). _statevector_cache used to grow unboundedly for the lifetime
# of the optimizer -- harmless at 5 qubits, but a single 20-qubit BO run
# evaluates thousands of distinct candidate points during acquisition
# optimization and drove RSS past 29 GB, tripping the OOM killer
# (2026-08-03). This budget bounds cache memory regardless of n_qubits by
# deriving maxsize from the actual per-entry size.
_CACHE_BUDGET_BYTES = 512 * 1024 * 1024

# A second, independent memory blowup (2026-08-04, real 20-qubit run):
# GPyTorch broadcasts/tiles the training set to match a candidate batch's
# leading dimension before calling forward() -- a real BO iteration with 128
# candidates and 10 training points asked this kernel for 128*10 = 1280
# statevector rows, of which only 10 were actually distinct. The old
# forward()/_statevectors materialized one full 2**20-complex128 row (16
# MiB) per *requested* row regardless of duplicates -- a single 20 GiB
# `np.empty` call for that one acquisition-scoring pass. forward() below
# instead computes the small n_unique x n_unique overlap matrix once and
# fancy-indexes *that* (real scalars, not 16 MiB statevectors) back up to
# the full requested shape -- mathematically identical result, but the
# expensive 2**n_qubits-wide dimension never touches the duplicated axis.


def _n_qubits(dim: int, max_qubits: int = MAX_QUBITS) -> int:
    n = max(1, dim)
    if n > max_qubits:
        raise ValueError(f"group dimension {dim} exceeds the {max_qubits}-qubit cap")
    return n


# Per-layer frequency multiplier for the RY re-uploading angle (2026-08-11).
# Every group in this project's search spaces (Brief §5's categorical
# variables) one-hot-encodes to z in {0, 1}. RY(pi * k) for any integer k is,
# up to an irrelevant global phase, one of only two operators (k even -> I,
# k odd -> RY(pi)) -- both Clifford, as is CZ. Re-uploading the *same*
# RY(pi*z) angle every layer therefore builds a fixed Clifford unitary raised
# to the depth-th power, which has a finite, small period: a real
# scripts/run_depth_qubit_grid.py run confirmed depth=10 and depth=20 collapse
# back to depth=0's exact state (bit-for-bit identical downstream BO
# trajectories), even though generic (non-one-hot) z inputs never repeat.
# This is a data-reuploading expressivity bug, not a quantum finding.
#
# Fix: scale each layer's angle by a distinct multiplier so RY(pi*freq*z)
# stops landing on an integer multiple of pi for z=1 once depth > 1 -- the
# standard "Fourier-series" reuploading construction (Schuld, Sweke & Meyer,
# 2021), which is also independently motivated as increasing the accessible
# frequency spectrum with depth, not just a periodicity patch. freq=1.0 for
# the first layer keeps depth=1 (every production qkernel_bo caller's only
# configuration, incl. the completed main_hotpot_v3 campaign) byte-identical
# to the pre-fix circuit; only depth > 1 -- exploratory-sweep-only territory
# -- changes.
_LAYER_FREQ_STEP = 0.6180339887498949  # golden ratio conjugate: irrational,
# so no integer combination of layer frequencies is itself an integer,
# which is what ultimately kills the exact revival.


def _layer_frequency(layer: int) -> float:
    return 1.0 if layer == 0 else 1.0 + layer * _LAYER_FREQ_STEP


def _build_circuit(
    n_qubits: int, depth: int, device_name: str = "default.qubit"
) -> Callable[[np.ndarray], np.ndarray]:
    device = qml.device(device_name, wires=n_qubits)

    @qml.qnode(device)
    def circuit(z: np.ndarray) -> np.ndarray:
        for layer in range(depth):
            freq = _layer_frequency(layer)
            for i in range(n_qubits):
                qml.RY(np.pi * freq * z[i % len(z)], wires=i)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        return qml.state()

    return circuit


def _hash_seed(*parts: bytes, base_seed: int) -> int:
    h = hashlib.sha256(str(base_seed).encode())
    for p in parts:
        h.update(p)
    return int(h.hexdigest()[:8], 16)


class QuantumFidelityKernel(Kernel):
    has_lengthscale = False

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        shots: int | None = None,
        seed: int = 0,
        max_qubits: int = MAX_QUBITS,
        device_name: str = "default.qubit",
        **kwargs,
    ) -> None:
        """``max_qubits``/``device_name`` default to Brief D3's cap and
        PennyLane's reference statevector simulator -- unchanged for every
        production caller (``qkernel_bo`` always uses the defaults). They
        exist only for ``scripts/run_qubit_sweep_extended.py``'s
        exploratory, beyond-Brief-D3 qubit-count sweep (2026-07-18):
        ``lightning.qubit`` (already a declared dependency, PennyLane's own
        optimized statevector backend -- same physics, not an
        approximation) is ~18x faster than ``default.qubit`` at 20 qubits
        (measured live), which is what makes going past 5 qubits/group
        tractable at all.
        """
        super().__init__(**kwargs)
        self.n_qubits = _n_qubits(dim, max_qubits=max_qubits)
        self.depth = depth
        self.shots = shots
        self.seed = seed
        self.last_psd_clipped_mass = 0.0
        self._circuit = _build_circuit(self.n_qubits, depth, device_name=device_name)
        itemsize = np.dtype(np.complex128).itemsize * (2**self.n_qubits)
        self._cache_maxsize = max(1, _CACHE_BUDGET_BYTES // itemsize)
        self._statevector_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()

    def _unique_statevectors(self, z_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One circuit evaluation per *unique* point in ``z_batch``, cached
        by content across calls — this kernel has no learnable parameters,
        so the same z always yields the same statevector regardless of
        which GP-fit iteration or how many times it's queried. Bounded LRU
        (see ``_CACHE_BUDGET_BYTES``): evicts the least-recently-used entry
        once full instead of growing without limit.

        Returns ``(unique_statevectors, inverse)`` where ``inverse[i]`` is
        the row of ``unique_statevectors`` corresponding to ``z_batch[i]``
        — deliberately *not* ``unique_statevectors[inverse]`` pre-expanded
        back to ``z_batch``'s original row count. Callers should fancy-
        index whatever *small* result they compute from the unique rows
        (e.g. forward()'s overlap matrix) rather than re-expanding these
        statevectors themselves, which is exactly the allocation that blew
        up to 20 GiB for one real 20-qubit acquisition-scoring batch (see
        module-level comment above ``_CACHE_BUDGET_BYTES``).
        """
        unique_z, inverse = np.unique(z_batch, axis=0, return_inverse=True)
        out = np.empty((unique_z.shape[0], 2**self.n_qubits), dtype=np.complex128)
        for i, z in enumerate(unique_z):
            key = z.tobytes()
            cached = self._statevector_cache.get(key)
            if cached is None:
                cached = np.asarray(self._circuit(z))
                self._statevector_cache[key] = cached
                if len(self._statevector_cache) > self._cache_maxsize:
                    self._statevector_cache.popitem(last=False)
            else:
                self._statevector_cache.move_to_end(key)
            out[i] = cached
        return out, inverse.reshape(-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()
        is_training_gram = (
            not diag and x1_np.ndim == 2 and x2_np.ndim == 2 and np.array_equal(x1_np, x2_np)
        )

        batch_shape1, n1, d1 = x1_np.shape[:-2], x1_np.shape[-2], x1_np.shape[-1]
        batch_shape2, n2, d2 = x2_np.shape[:-2], x2_np.shape[-2], x2_np.shape[-1]

        psi1_u, inv1 = self._unique_statevectors(x1_np.reshape(-1, d1))
        if is_training_gram:
            psi2_u, inv2 = psi1_u, inv1
        else:
            psi2_u, inv2 = self._unique_statevectors(x2_np.reshape(-1, d2))

        # Small (n_unique1 x n_unique2) matrix of real work; expanding
        # *this* back up to the full batch via inv1/inv2 below is cheap
        # (real/complex scalars), unlike expanding the 2**n_qubits-wide
        # statevectors that produced it would be.
        overlap_u = psi1_u @ psi2_u.conj().T

        if diag:
            idx1 = inv1.reshape(*batch_shape1, n1)
            idx2 = inv2.reshape(*batch_shape2, n2)
            fidelity = np.abs(overlap_u[idx1, idx2]) ** 2
        else:
            idx1 = inv1.reshape(*batch_shape1, n1, 1)
            idx2 = inv2.reshape(*batch_shape2, 1, n2)
            fidelity = np.abs(overlap_u[idx1, idx2]) ** 2

        if self.shots is not None:
            noise_seed = _hash_seed(x1_np.tobytes(), x2_np.tobytes(), base_seed=self.seed)
            rng = np.random.default_rng(noise_seed)
            fidelity = rng.binomial(self.shots, np.clip(fidelity, 0.0, 1.0)) / self.shots

        if is_training_gram and self.shots is not None:
            fidelity, clipped_mass = _psd_project(fidelity)
            self.last_psd_clipped_mass = clipped_mass
        elif is_training_gram:
            self.last_psd_clipped_mass = 0.0

        return torch.tensor(fidelity, dtype=x1.dtype, device=x1.device)


def _psd_project(gram: np.ndarray) -> tuple[np.ndarray, float]:
    """Eigenvalue-clip a (possibly slightly non-PSD, finite-shot) Gram
    matrix back onto the PSD cone; returns the projected matrix and the
    clipped mass (sum of |negative eigenvalues|) for ``kernel_log``.
    """
    sym = (gram + gram.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped_mass = float(np.sum(np.abs(eigvals[eigvals < 0])))
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    projected = (eigvecs * eigvals_clipped) @ eigvecs.T
    return projected, clipped_mass


def make_quantum_group_kernel(
    group_name: str,
    variables: list,
    depth: int = 1,
    shots: int | None = None,
    seed: int = 0,
    max_qubits: int = MAX_QUBITS,
    device_name: str = "default.qubit",
) -> Kernel:
    dim = sum(len(v.values) if v.kind == "categorical" else 1 for v in variables)
    return QuantumFidelityKernel(
        dim=dim, depth=depth, shots=shots, seed=seed, max_qubits=max_qubits, device_name=device_name
    )


def quantum_group_kernel_factory(
    depth: int = 1,
    shots: int | None = None,
    seed: int = 0,
    max_qubits: int = MAX_QUBITS,
    device_name: str = "default.qubit",
) -> Callable[[str, list], Kernel]:
    """Bind ``depth``/``shots``/``seed`` (Brief §7's ``optimizer.quantum``
    config) into a two-arg ``(group_name, variables) -> Kernel`` factory —
    the exact shape :meth:`StructuralKernel.from_search_space` expects, and
    the one that ``classical.make_classical_group_kernel`` already has.
    ``max_qubits``/``device_name`` are exploratory-sweep-only overrides,
    see :class:`QuantumFidelityKernel`'s docstring.
    """

    def factory(group_name: str, variables: list) -> Kernel:
        return make_quantum_group_kernel(
            group_name,
            variables,
            depth=depth,
            shots=shots,
            seed=seed,
            max_qubits=max_qubits,
            device_name=device_name,
        )

    return factory
