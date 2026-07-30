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
from collections.abc import Callable

import numpy as np
import pennylane as qml
import torch
from gpytorch.kernels import Kernel

MAX_QUBITS = 5  # Brief D3


def _n_qubits(dim: int, max_qubits: int = MAX_QUBITS) -> int:
    n = max(1, dim)
    if n > max_qubits:
        raise ValueError(f"group dimension {dim} exceeds the {max_qubits}-qubit cap")
    return n


def _build_circuit(
    n_qubits: int, depth: int, device_name: str = "default.qubit"
) -> Callable[[np.ndarray], np.ndarray]:
    device = qml.device(device_name, wires=n_qubits)

    @qml.qnode(device)
    def circuit(z: np.ndarray) -> np.ndarray:
        for _ in range(depth):
            for i in range(n_qubits):
                qml.RY(np.pi * z[i % len(z)], wires=i)
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
        self._statevector_cache: dict[bytes, np.ndarray] = {}

    def _statevectors(self, z_batch: np.ndarray) -> np.ndarray:
        """One circuit evaluation per unique point, cached by content —
        this kernel has no learnable parameters, so the same z always
        yields the same statevector regardless of which GP-fit iteration
        or how many times it's queried.
        """
        out = np.empty((z_batch.shape[0], 2**self.n_qubits), dtype=np.complex128)
        for i, z in enumerate(z_batch):
            key = z.tobytes()
            cached = self._statevector_cache.get(key)
            if cached is None:
                cached = np.asarray(self._circuit(z))
                self._statevector_cache[key] = cached
            out[i] = cached
        return out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()
        is_training_gram = (
            not diag and x1_np.ndim == 2 and x2_np.ndim == 2 and np.array_equal(x1_np, x2_np)
        )

        batch_shape1, n1, d1 = x1_np.shape[:-2], x1_np.shape[-2], x1_np.shape[-1]
        batch_shape2, n2, d2 = x2_np.shape[:-2], x2_np.shape[-2], x2_np.shape[-1]

        psi1 = self._statevectors(x1_np.reshape(-1, d1)).reshape(*batch_shape1, n1, -1)
        if is_training_gram:
            psi2 = psi1
        else:
            psi2 = self._statevectors(x2_np.reshape(-1, d2)).reshape(*batch_shape2, n2, -1)

        if diag:
            fidelity = np.abs(np.einsum("...ik,...ik->...i", psi1, psi2.conj())) ** 2
        else:
            overlap = np.einsum("...ik,...jk->...ij", psi1, psi2.conj())
            fidelity = np.abs(overlap) ** 2

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
