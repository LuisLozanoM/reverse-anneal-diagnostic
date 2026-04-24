"""Exact diagonalization with unitary time evolution for closed S+E systems.

Memory: full state vector is 2^N complex128 → ~22 qubits = 64 MB, ~24 qubits = 256 MB.
Hamiltonian (sparse) is manageable up to ~24 qubits on M3 Max 128 GB.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

# ---------------------------------------------------------------------------
# Pauli helpers
# ---------------------------------------------------------------------------

_I2 = sp.eye(2, format="csr")
_SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
_SX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))


def _sigma_z(i: int, n: int) -> sp.csr_matrix:
    """Sparse sigma_z on qubit *i* in an *n*-qubit Hilbert space."""
    ops = [_I2] * n
    ops[i] = _SZ
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr")
    return out


def _sigma_x(i: int, n: int) -> sp.csr_matrix:
    """Sparse sigma_x on qubit *i* in an *n*-qubit Hilbert space."""
    ops = [_I2] * n
    ops[i] = _SX
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr")
    return out


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------


def build_ising_hamiltonian(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    s_p: float,
    A_s: float,
    B_s: float,
) -> sp.csr_matrix:
    """Build transverse-field Ising H at pause point s_p.

    H = -A(s_p)/2 * sum_i sx_i + B(s_p)/2 * (sum_i h_i sz_i + sum_{ij} J_ij sz_i sz_j)
    """
    qubits = set(h.keys())
    for i, j in J:
        qubits.update((i, j))
    n = max(qubits) + 1
    dim = 2**n

    H = sp.csr_matrix((dim, dim), dtype=np.complex128)

    # Transverse field: -A/2 * sum sx_i
    for i in sorted(qubits):
        H -= (A_s / 2.0) * _sigma_x(i, n)

    # Longitudinal field: B/2 * h_i sz_i
    for i, hi in h.items():
        H += (B_s / 2.0) * hi * _sigma_z(i, n)

    # Couplings: B/2 * J_ij sz_i sz_j
    for (i, j), Jij in J.items():
        H += (B_s / 2.0) * Jij * (_sigma_z(i, n) @ _sigma_z(j, n))

    return H


# ---------------------------------------------------------------------------
# State preparation
# ---------------------------------------------------------------------------


def product_state(spin_config: list[int], n_qubits: int) -> np.ndarray:
    """Build computational-basis product state. spin_config[i] = +1 (up) or -1 (down)."""
    psi = np.array([1.0], dtype=np.complex128)
    for s in spin_config[:n_qubits]:
        if s == 1:
            psi = np.kron(psi, np.array([1, 0], dtype=np.complex128))
        else:
            psi = np.kron(psi, np.array([0, 1], dtype=np.complex128))
    return psi


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------


def time_evolve(
    H: sp.spmatrix,
    psi_0: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evolve |psi_0> under H for time t via expm_multiply. Returns final state."""
    return expm_multiply(-1j * H * t, psi_0)


# ---------------------------------------------------------------------------
# Subsystem marginals
# ---------------------------------------------------------------------------


def compute_P_S_from_state(
    psi: np.ndarray,
    S_indices: list[int],
    n_qubits: int,
) -> dict[tuple[int, ...], float]:
    """Marginal probability distribution P_S by tracing out environment qubits."""
    probs = np.abs(psi) ** 2
    n_S = len(S_indices)
    E_indices = [i for i in range(n_qubits) if i not in S_indices]

    P_S: dict[tuple[int, ...], float] = {}

    for s_int in range(2**n_S):
        # Map subsystem integer to per-qubit bits
        s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
        p = 0.0
        for e_int in range(2 ** len(E_indices)):
            e_bits = [(e_int >> (len(E_indices) - 1 - k)) & 1 for k in range(len(E_indices))]
            # Build the full basis index
            full_bits = [0] * n_qubits
            for k, idx in enumerate(S_indices):
                full_bits[idx] = s_bits[k]
            for k, idx in enumerate(E_indices):
                full_bits[idx] = e_bits[k]
            basis_idx = sum(b << (n_qubits - 1 - i) for i, b in enumerate(full_bits))
            p += probs[basis_idx]
        # Convert 0/1 bits to +1/-1 spins
        config = tuple(1 - 2 * b for b in s_bits)
        P_S[config] = float(p)

    return P_S


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_ed_thermalization(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    initial_states: dict[str, np.ndarray],
    s_p: float,
    t_hold: float,
    A_s: float,
    B_s: float,
) -> dict[str, dict[tuple[int, ...], float]]:
    """Run ED pipeline: build H, evolve each initial state, compute P_S for each."""
    H = build_ising_hamiltonian(h, J, s_p, A_s, B_s)
    n_qubits = int(np.log2(H.shape[0]))

    results = {}
    for name, psi_0 in initial_states.items():
        psi_t = time_evolve(H, psi_0, t_hold)
        results[name] = compute_P_S_from_state(psi_t, S_indices, n_qubits)
    return results
