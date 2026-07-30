import numpy as np
import pytest
import torch

from qautollm.kernels.quantum import MAX_QUBITS, QuantumFidelityKernel


def _features(n: int, dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(0, 1, size=(n, dim)), dtype=torch.float64)


def test_rejects_groups_above_max_qubits():
    with pytest.raises(ValueError, match="qubit cap"):
        QuantumFidelityKernel(dim=MAX_QUBITS + 1, depth=1)


def test_max_qubits_override_still_rejects_above_the_override():
    # scripts/run_qubit_sweep_extended.py's escape hatch -- still a hard
    # cap, just a higher one, not "no cap".
    with pytest.raises(ValueError, match="qubit cap"):
        QuantumFidelityKernel(dim=8, depth=1, max_qubits=6)


def test_max_qubits_override_allows_more_than_the_brief_default():
    kernel = QuantumFidelityKernel(dim=8, depth=1, shots=None, seed=0, max_qubits=10)
    assert kernel.n_qubits == 8
    x = _features(4, 8, seed=0)
    with torch.no_grad():
        mat = kernel.forward(x, x)
    assert mat.shape == (4, 4)
    assert torch.allclose(mat, mat.T, atol=1e-10)


def test_device_name_override_produces_the_same_fidelities_as_default():
    # lightning.qubit is a different simulator backend for the identical
    # circuit/physics -- results must match default.qubit, not just "run".
    x = _features(5, 3, seed=2)
    kernel_default = QuantumFidelityKernel(dim=3, depth=1, shots=None, seed=0)
    kernel_lightning = QuantumFidelityKernel(
        dim=3, depth=1, shots=None, seed=0, device_name="lightning.qubit"
    )
    with torch.no_grad():
        mat_default = kernel_default.forward(x, x)
        mat_lightning = kernel_lightning.forward(x, x)
    assert torch.allclose(mat_default, mat_lightning, atol=1e-8)


def test_exact_kernel_shape_and_symmetry():
    kernel = QuantumFidelityKernel(dim=3, depth=1, shots=None, seed=0)
    x = _features(8, 3, seed=0)
    with torch.no_grad():
        mat = kernel.forward(x, x)
    assert mat.shape == (8, 8)
    assert torch.allclose(mat, mat.T, atol=1e-10)


def test_exact_kernel_self_fidelity_is_one():
    kernel = QuantumFidelityKernel(dim=3, depth=2, shots=None, seed=0)
    x = _features(6, 3, seed=1)
    with torch.no_grad():
        mat = kernel.forward(x, x)
    assert torch.allclose(torch.diag(mat), torch.ones(6, dtype=torch.float64), atol=1e-8)


def test_exact_kernel_is_psd():
    kernel = QuantumFidelityKernel(dim=3, depth=1, shots=None, seed=0)
    x = _features(10, 3, seed=2)
    with torch.no_grad():
        mat = kernel.forward(x, x)
    assert torch.linalg.eigvalsh(mat).min().item() >= -1e-8


def test_asymmetric_shapes_supported():
    kernel = QuantumFidelityKernel(dim=3, depth=1, shots=None, seed=0)
    x1 = _features(4, 3, seed=3)
    x2 = _features(7, 3, seed=4)
    with torch.no_grad():
        mat = kernel.forward(x1, x2)
    assert mat.shape == (4, 7)


def test_diag_matches_full_matrix_diagonal():
    kernel = QuantumFidelityKernel(dim=3, depth=1, shots=None, seed=0)
    x = _features(6, 3, seed=5)
    with torch.no_grad():
        full = kernel.forward(x, x)
        diag = kernel.forward(x, x, diag=True)
    assert torch.allclose(torch.diag(full), diag, atol=1e-8)


def test_statevector_cache_gives_identical_repeated_calls():
    kernel = QuantumFidelityKernel(dim=3, depth=1, shots=None, seed=0)
    x = _features(5, 3, seed=6)
    with torch.no_grad():
        a = kernel.forward(x, x)
        b = kernel.forward(x, x)
    assert torch.equal(a, b)
    assert len(kernel._statevector_cache) == 5  # one entry per unique point, not per call


def test_finite_shots_are_deterministic_given_seed():
    x = _features(6, 3, seed=7)
    k_a = QuantumFidelityKernel(dim=3, depth=1, shots=256, seed=42)
    k_b = QuantumFidelityKernel(dim=3, depth=1, shots=256, seed=42)
    with torch.no_grad():
        mat_a = k_a.forward(x, x)
        mat_b = k_b.forward(x, x)
    assert torch.equal(mat_a, mat_b)


def test_finite_shots_differ_from_exact_but_stay_close():
    x = _features(6, 3, seed=8)
    exact = QuantumFidelityKernel(dim=3, depth=1, shots=None, seed=0)
    noisy = QuantumFidelityKernel(dim=3, depth=1, shots=1024, seed=0)
    with torch.no_grad():
        mat_exact = exact.forward(x, x)
        mat_noisy = noisy.forward(x, x)
    assert not torch.equal(mat_exact, mat_noisy)
    assert torch.allclose(mat_exact, mat_noisy, atol=0.15)


def test_psd_projection_fixes_negative_eigenvalues_and_logs_clipped_mass():
    """Brief P5 gate: PSD projection tested. Very few shots on n=20 points
    reliably produces a non-PSD raw Gram matrix (verified empirically with
    this exact seed before landing the test) — after projection the matrix
    must be PSD and the clipped mass must be logged as strictly positive.
    """
    x = _features(20, 3, seed=2)
    kernel = QuantumFidelityKernel(dim=3, depth=1, shots=16, seed=0)
    with torch.no_grad():
        mat = kernel.forward(x, x)
    assert torch.linalg.eigvalsh(mat).min().item() >= -1e-6
    assert kernel.last_psd_clipped_mass > 0.0


def test_psd_projection_only_applies_to_square_training_gram():
    """Candidate-scoring calls (x1 a different point set from x2) must not
    trigger PSD projection or update the clipped-mass log — only the
    actual (square, x1 is x2) training Gram matrix does.
    """
    kernel = QuantumFidelityKernel(dim=3, depth=1, shots=16, seed=0)
    train_x = _features(10, 3, seed=9)
    candidates = _features(5, 3, seed=10)
    with torch.no_grad():
        kernel.forward(train_x, candidates)
    assert kernel.last_psd_clipped_mass == 0.0
