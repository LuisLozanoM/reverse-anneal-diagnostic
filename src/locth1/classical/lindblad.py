"""Lindblad master equation solver for open quantum system with Markovian bath.

Memory: density matrix is (2^N)^2 complex128, Liouvillian is (2^N)^2 x (2^N)^2 sparse.
~12 qubits: rho = 16 MB, L sparse ~ few GB.  ~14 qubits: rho = 256 MB, near limit.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .exact_diag import _sigma_x, _sigma_z, build_ising_hamiltonian

# ---------------------------------------------------------------------------
# Pauli raising/lowering
# ---------------------------------------------------------------------------

_I2 = sp.eye(2, format="csr")
_SP = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.complex128))  # sigma_+
_SM = sp.csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.complex128))  # sigma_-


def _sigma_pm(op_2x2: sp.spmatrix, i: int, n: int) -> sp.csr_matrix:
    """Embed a 2x2 operator on qubit *i* in an *n*-qubit space."""
    ops = [_I2] * n
    ops[i] = op_2x2
    out = ops[0]
    for o in ops[1:]:
        out = sp.kron(out, o, format="csr")
    return out


# ---------------------------------------------------------------------------
# Jump operators
# ---------------------------------------------------------------------------


def build_thermal_jump_operators(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    n_qubits: int,
    T: float,
    gamma_base: float = 1.0,
) -> list[sp.csr_matrix]:
    """Single-qubit thermal relaxation operators at temperature T.

    gamma_down = gamma_base; gamma_up = gamma_down * exp(-|Delta_E|/T) where
    Delta_E is the local field magnitude on qubit i.
    """
    jump_ops: list[sp.csr_matrix] = []

    for i in range(n_qubits):
        # Local field: h_i + sum_j J_ij (mean-field-like estimate)
        local = h.get(i, 0.0)
        for (a, b), Jij in J.items():
            if a == i or b == i:
                local += abs(Jij)
        delta_E = abs(local) if local != 0.0 else 0.1  # avoid div-by-zero

        gamma_down = gamma_base
        gamma_up = gamma_down * np.exp(-delta_E / max(T, 1e-12))

        # Lowering (relaxation): sqrt(gamma_down) * sigma_-
        jump_ops.append(np.sqrt(gamma_down) * _sigma_pm(_SM, i, n_qubits))
        # Raising (excitation): sqrt(gamma_up) * sigma_+
        jump_ops.append(np.sqrt(gamma_up) * _sigma_pm(_SP, i, n_qubits))

    return jump_ops


# ---------------------------------------------------------------------------
# Liouvillian
# ---------------------------------------------------------------------------


def build_lindblad_superoperator(
    H: sp.spmatrix,
    jump_ops: list[sp.spmatrix],
    n_qubits: int,
) -> sp.csr_matrix:
    """Build Liouvillian L s.t. d(vec(rho))/dt = L @ vec(rho).

    L = -i(H x I - I x H^T) + sum_k [ Lk x Lk* - 0.5*(Lk^dag Lk x I + I x (Lk^dag Lk)^T) ]
    """
    dim = 2**n_qubits
    I_d = sp.eye(dim, format="csr")

    # Hamiltonian part: -i (H x I - I x H^T)
    L = -1j * (sp.kron(H, I_d, format="csr") - sp.kron(I_d, H.T, format="csr"))

    # Dissipator
    for Lk in jump_ops:
        Lk_dag = Lk.conj().T
        LdL = Lk_dag @ Lk

        L += sp.kron(Lk, Lk.conj(), format="csr")
        L -= 0.5 * sp.kron(LdL, I_d, format="csr")
        L -= 0.5 * sp.kron(I_d, LdL.T, format="csr")

    return L.tocsr()


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------


def evolve_lindblad(
    L: sp.spmatrix,
    rho_0: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evolve vectorized rho_0 under L for time t. Returns density matrix."""
    dim = int(np.sqrt(L.shape[0]))
    vec_rho = rho_0.reshape(-1)
    vec_rho_t = expm_multiply(L * t, vec_rho)
    return vec_rho_t.reshape(dim, dim)


# ---------------------------------------------------------------------------
# Subsystem marginals
# ---------------------------------------------------------------------------


def compute_P_S_from_density(
    rho: np.ndarray,
    S_indices: list[int],
    n_qubits: int,
) -> dict[tuple[int, ...], float]:
    """Trace out environment from rho, extract diagonal probabilities."""
    diag = np.real(np.diag(rho))
    n_S = len(S_indices)
    E_indices = [i for i in range(n_qubits) if i not in S_indices]

    P_S: dict[tuple[int, ...], float] = {}
    for s_int in range(2**n_S):
        s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
        p = 0.0
        for e_int in range(2 ** len(E_indices)):
            e_bits = [(e_int >> (len(E_indices) - 1 - k)) & 1 for k in range(len(E_indices))]
            full_bits = [0] * n_qubits
            for k, idx in enumerate(S_indices):
                full_bits[idx] = s_bits[k]
            for k, idx in enumerate(E_indices):
                full_bits[idx] = e_bits[k]
            basis_idx = sum(b << (n_qubits - 1 - i) for i, b in enumerate(full_bits))
            p += diag[basis_idx]
        config = tuple(1 - 2 * b for b in s_bits)
        P_S[config] = float(p)

    return P_S


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_lindblad_thermalization(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    initial_states: dict[str, np.ndarray],
    s_p: float,
    t_hold: float,
    A_s: float,
    B_s: float,
    T_device: float,
    gamma_base: float = 1.0,
) -> dict[str, dict[tuple[int, ...], float]]:
    """Full Lindblad pipeline. initial_states values are state vectors or density matrices."""
    H = build_ising_hamiltonian(h, J, s_p, A_s, B_s)
    n_qubits = int(np.log2(H.shape[0]))

    jump_ops = build_thermal_jump_operators(h, J, n_qubits, T_device, gamma_base)
    L = build_lindblad_superoperator(H, jump_ops, n_qubits)

    results = {}
    for name, state in initial_states.items():
        # Accept state vector or density matrix
        if state.ndim == 1:
            rho_0 = np.outer(state, state.conj())
        else:
            rho_0 = state
        rho_t = evolve_lindblad(L, rho_0, t_hold)
        results[name] = compute_P_S_from_density(rho_t, S_indices, n_qubits)

    return results
