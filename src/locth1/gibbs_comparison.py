r"""Quantum reduced Gibbs marginal and effective subsystem Hamiltonian comparison.

For small systems ($N \leq 14$) we compute the quantum reduced Gibbs state of the
pause-point Hamiltonian $H(s_p)$ and project onto the subsystem $z$-basis.
Measured subsystem marginals $P_S^{\text{exp}}$ are then compared against three
theoretical predictions at the *externally calibrated* inverse temperature
$\beta_\text{eff}$:

  (a) the quantum reduced Gibbs marginal of $H(s_p)$
      (this module's ``quantum_reduced_gibbs_marginal``);
  (b) the classical diagonal marginal of $H_P$ alone
      (this module's ``classical_diagonal_marginal``);
  (c) a Glauber / stochastic baseline (see ``classical/glauber.py``).

For the full ``|S|=6`` subsystem in the on-chip-environment experiment, the
quantum reduced Gibbs of the *full* 56-qubit system is intractable, so we
instead fit an *effective* subsystem Hamiltonian

    $H_S^{\text{eff}}(s) = \sum_i \tilde h_i s_i + \sum_{ij} \tilde J_{ij} s_i s_j$

with the externally calibrated $\beta_\text{eff}$ fixed as the temperature
(``fit_effective_H_S``).  A successful conditional thermal-marginal claim
requires: (i) a single $\beta$ explains every relaxed ordered-bath condition,
(ii) the fitted $(\tilde h, \tilde J)$ is a small perturbation of the known
$(h + \lambda\, h_{\partial E}, J)$ induced by the ordered $E$ boundary field,
and (iii) the fit fails exactly where relaxation fails.

All functions operate in dimensionless problem-energy units where $\beta$ is
the already-dimensionless $\beta_\text{eff} = B(s_p) R / (2 k_B T)$ extracted
from the in-situ single-qubit probe calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .classical.exact_diag import build_ising_hamiltonian


def _compute_full_system_rho_diag(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    A_over_B: float,
    beta: float,
    S_indices: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    r"""Build $H(s_p)$, eigendecompose, and return the diagonal of the Gibbs
    density matrix in the z-basis along with the system size $n$.

    The transverse-field Ising Hamiltonian has no $\sigma^y$ terms and is real
    symmetric in the z-basis; we cast to float64 before calling ``eigh`` to
    avoid the 5--6x slowdown of the complex128 path on Apple Accelerate.
    """
    qubits: set[int] = set(h.keys())
    for i, j in J:
        qubits.update((i, j))
    if S_indices is not None:
        qubits.update(S_indices)
    n = (max(qubits) + 1) if qubits else 0
    if n == 0:
        return np.array([1.0]), 0

    H_sparse = build_ising_hamiltonian(
        h, J, s_p=0.0, A_s=2.0 * A_over_B, B_s=2.0,
    )
    H_dense_complex = H_sparse.toarray()
    imag_max = float(np.max(np.abs(np.imag(H_dense_complex))))
    if imag_max > 1e-12:
        raise ValueError(
            f"_compute_full_system_rho_diag: Hamiltonian has non-zero "
            f"imaginary part (max = {imag_max:.3e}); expected a real "
            f"symmetric TFIM matrix in the z-basis."
        )
    H_real = np.real(H_dense_complex).astype(np.float64, copy=False)
    asymmetry = float(np.max(np.abs(H_real - H_real.T)))
    if asymmetry > 1e-10:
        raise ValueError(
            f"_compute_full_system_rho_diag: Hamiltonian is not symmetric "
            f"(max |H - H^T| = {asymmetry:.3e})."
        )

    evals, evecs = np.linalg.eigh(H_real)
    evals_shifted = evals - evals.min()
    weights = np.exp(-beta * evals_shifted)
    Z = weights.sum()
    rho_diag = (evecs ** 2) @ (weights / Z)
    return rho_diag, n


# ---------------------------------------------------------------------------
# Quantum reduced Gibbs marginal
# ---------------------------------------------------------------------------


def quantum_reduced_gibbs_marginal(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    A_over_B: float,
    beta: float,
) -> dict[tuple[int, ...], float]:
    r"""Compute the diagonal subsystem marginal of the quantum reduced Gibbs state.

    The Gibbs state is
        $\rho \propto \exp\big(-\beta\,[\, -(A/B)\sum_i \sigma^x_i + H_P\,]\big)$
    where $H_P = \sum_i h_i \sigma^z_i + \sum_{ij} J_{ij} \sigma^z_i \sigma^z_j$.
    The subsystem marginal is the diagonal of $\mathrm{Tr}_E\,\rho$ in the
    $z$-basis, which is exactly what one obtains from $z$-basis readout after
    the QPU pause.

    Parameters
    ----------
    h, J
        Ising fields and couplings on the full $N$-qubit problem.
    S_indices
        Qubit indices comprising the subsystem (need not be contiguous).
    A_over_B
        Dimensionless ratio $A(s_p) / B(s_p)$ at the pause point.
    beta
        Externally calibrated dimensionless inverse temperature
        $\beta_\text{eff} = B(s_p) R / (2 k_B T)$.

    Returns
    -------
    P_S
        Mapping from spin tuples (``(+1, -1, ...)``) to probabilities.
        Keys span the full $2^{|S|}$ basis.
    """
    rho_diag, n = _compute_full_system_rho_diag(h, J, A_over_B, beta, S_indices)
    if n == 0:
        return {(): 1.0}
    return _trace_out_environment_diagonal(rho_diag, S_indices, n)


# ---------------------------------------------------------------------------
# Classical diagonal Gibbs marginal
# ---------------------------------------------------------------------------


def classical_diagonal_marginal(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    beta: float,
) -> dict[tuple[int, ...], float]:
    r"""Classical Ising Gibbs marginal of $H_P$ alone at inverse temperature $\beta$.

    This is the $A(s_p) / B(s_p) \to 0$ limit of
    :func:`quantum_reduced_gibbs_marginal` and serves as the transverse-field-
    free comparison baseline.  Use the full quantum version to distinguish
    genuinely quantum effects from classical thermal equilibration.
    """
    qubits: set[int] = set(h.keys())
    for i, j in J:
        qubits.update((i, j))
    qubits.update(S_indices)
    n = (max(qubits) + 1) if qubits else 0
    if n == 0:
        return {(): 1.0}
    if n > 24:
        raise ValueError(
            f"classical_diagonal_marginal enumerates 2^n states; n={n} is too large."
        )

    n_states = 2 ** n
    indices = np.arange(n_states, dtype=np.int64)
    # Bit k (from the LEFT, i.e. highest bit) corresponds to qubit k; spin = 1 - 2*bit.
    # This matches the convention in exact_diag.compute_P_S_from_state.
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    bits = (indices[:, None] >> shifts[None, :]) & 1
    spins = 1 - 2 * bits  # shape (n_states, n), values in {-1, +1}

    energies = np.zeros(n_states, dtype=np.float64)
    for i, hi in h.items():
        energies += hi * spins[:, i]
    for (i, j), Jij in J.items():
        energies += Jij * spins[:, i] * spins[:, j]

    E_min = energies.min()
    weights = np.exp(-beta * (energies - E_min))
    probs = weights / weights.sum()

    return _trace_out_environment_diagonal(probs, S_indices, n)


# ---------------------------------------------------------------------------
# Partial trace helper
# ---------------------------------------------------------------------------


def _trace_out_environment_diagonal(
    probs_diag: np.ndarray,
    S_indices: list[int],
    n: int,
) -> dict[tuple[int, ...], float]:
    r"""Marginalise a diagonal probability vector over the environment qubits.

    Mirrors the bit ordering in
    :func:`locth1.classical.exact_diag.compute_P_S_from_state` so that a
    probability array obtained from $|\psi|^2$, the diagonal of a density
    matrix, or an enumerated classical Gibbs distribution all trace out
    consistently.
    """
    n_S = len(S_indices)
    E_indices = [i for i in range(n) if i not in S_indices]
    n_E = len(E_indices)

    P_S: dict[tuple[int, ...], float] = {}
    for s_int in range(2 ** n_S):
        s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
        p = 0.0
        for e_int in range(2 ** n_E):
            e_bits = [(e_int >> (n_E - 1 - k)) & 1 for k in range(n_E)]
            full_bits = [0] * n
            for k, idx in enumerate(S_indices):
                full_bits[idx] = s_bits[k]
            for k, idx in enumerate(E_indices):
                full_bits[idx] = e_bits[k]
            basis_idx = sum(b << (n - 1 - i) for i, b in enumerate(full_bits))
            p += float(probs_diag[basis_idx])
        config = tuple(1 - 2 * b for b in s_bits)
        P_S[config] = p

    return P_S


# ---------------------------------------------------------------------------
# Subsystem observables
# ---------------------------------------------------------------------------


def one_point_magnetization(
    P_S: dict[tuple[int, ...], float],
) -> np.ndarray:
    r"""Expectation values $\langle s_i \rangle$ for each subsystem qubit."""
    keys = sorted(P_S.keys())
    if not keys:
        return np.array([])
    spins = np.array([list(k) for k in keys], dtype=np.float64)
    probs = np.array([P_S[k] for k in keys], dtype=np.float64)
    return spins.T @ probs


def two_point_correlations(
    P_S: dict[tuple[int, ...], float],
) -> dict[tuple[int, int], float]:
    r"""Connected-basis $\langle s_i s_j \rangle$ for each subsystem pair."""
    keys = sorted(P_S.keys())
    if not keys:
        return {}
    S_size = len(keys[0])
    spins = np.array([list(k) for k in keys], dtype=np.float64)
    probs = np.array([P_S[k] for k in keys], dtype=np.float64)
    out: dict[tuple[int, int], float] = {}
    for i in range(S_size):
        for j in range(i + 1, S_size):
            out[(i, j)] = float((spins[:, i] * spins[:, j]) @ probs)
    return out


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def compute_gibbs_fit_metrics(
    P_S_exp: dict[tuple[int, ...], float],
    P_S_th: dict[tuple[int, ...], float],
    *,
    epsilon: float = 1e-10,
) -> dict[str, float]:
    """Compare two subsystem distributions with multiple metrics.

    Returns the total-variation distance, symmetric KL divergence, and
    mean-absolute errors on one-point magnetisations and two-point
    correlations.  Keys missing from either input are treated as zero.
    """
    keys = sorted(set(P_S_exp) | set(P_S_th))
    if not keys:
        return {
            "d_tv": 0.0, "symmetric_kl": 0.0,
            "kl_exp_th": 0.0, "kl_th_exp": 0.0,
            "magnetization_mae": 0.0, "pair_correlation_mae": 0.0,
        }

    p_exp = np.array([P_S_exp.get(k, 0.0) for k in keys], dtype=np.float64)
    p_th = np.array([P_S_th.get(k, 0.0) for k in keys], dtype=np.float64)

    d_tv = 0.5 * float(np.sum(np.abs(p_exp - p_th)))

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0.0
        if not np.any(mask):
            return 0.0
        q_floor = np.maximum(q[mask], epsilon)
        return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q_floor))))

    kl_exp_th = _kl(p_exp, p_th)
    kl_th_exp = _kl(p_th, p_exp)
    sym_kl = 0.5 * (kl_exp_th + kl_th_exp)

    # Observables
    S_size = len(keys[0])
    spins = np.array([list(k) for k in keys], dtype=np.float64)
    mag_exp = spins.T @ p_exp
    mag_th = spins.T @ p_th
    mag_mae = float(np.mean(np.abs(mag_exp - mag_th)))

    if S_size >= 2:
        corr_diffs = []
        for i in range(S_size):
            for j in range(i + 1, S_size):
                ss = spins[:, i] * spins[:, j]
                corr_diffs.append(float(ss @ p_exp) - float(ss @ p_th))
        corr_mae = float(np.mean(np.abs(corr_diffs)))
    else:
        corr_mae = 0.0

    return {
        "d_tv": d_tv,
        "symmetric_kl": sym_kl,
        "kl_exp_th": kl_exp_th,
        "kl_th_exp": kl_th_exp,
        "magnetization_mae": mag_mae,
        "pair_correlation_mae": corr_mae,
    }


# ---------------------------------------------------------------------------
# Effective H_S fit at fixed beta
# ---------------------------------------------------------------------------


def fit_effective_H_S(
    P_S_measured: dict[tuple[int, ...], float],
    beta: float,
    *,
    edges: list[tuple[int, int]] | None = None,
    x0: np.ndarray | None = None,
    smoothing: float = 0.0,
    l2_regularization: float = 0.0,
) -> dict[str, Any]:
    r"""Fit an effective subsystem Hamiltonian $H_S^{\text{eff}}$ at fixed $\beta$.

    The model is
        $H_S^{\text{eff}}(s) = \sum_i \tilde h_i s_i + \sum_{(i,j) \in \text{edges}} \tilde J_{ij} s_i s_j$
    and the fit minimises the negative log-likelihood of the observed subsystem
    distribution under the Gibbs model at fixed externally calibrated
    $\beta_\text{eff}$.  Numerical stability uses the log-sum-exp trick on the
    partition function; all $2^{|S|}$ subsystem states are materialised so a
    missing bin in the input is filled with zero automatically.

    Parameters
    ----------
    P_S_measured
        Observed subsystem marginal over spin tuples.  Missing keys are
        interpreted as zero probability.
    beta
        Externally calibrated inverse temperature; held *fixed* to avoid
        the well-known $(\beta, H)$ gauge degeneracy.
    edges
        Optional list of subsystem pairs $(i, j)$ with $i < j$.  Defaults to
        the complete graph on the subsystem.
    x0
        Optional initial guess; defaults to zeros (infinite-temperature prior).
    smoothing
        Additive Laplace-style smoothing $\varepsilon$ applied as
        $p_\text{smooth} = (1-\varepsilon) p_\text{obs} + \varepsilon / K$.
        Use a small positive value when the measurement comes from finite
        shots and some subsystem states have zero observed counts; 0 disables.
    l2_regularization
        Optional $\ell_2$ penalty on $(\tilde h, \tilde J)$; useful when the
        distribution is near-delta and the likelihood is ill-conditioned.
    """
    if not P_S_measured:
        raise ValueError("P_S_measured is empty")

    first_key = next(iter(P_S_measured))
    S_size = len(first_key)
    n_states = 2 ** S_size

    # Materialise every 2^|S| subsystem state in canonical order (matches
    # the (bit-0 → spin +1) convention used throughout the module).
    all_keys: list[tuple[int, ...]] = []
    spins_list: list[list[int]] = []
    for s_int in range(n_states):
        bits = [(s_int >> (S_size - 1 - k)) & 1 for k in range(S_size)]
        config = tuple(1 - 2 * b for b in bits)
        all_keys.append(config)
        spins_list.append(list(config))
    spins = np.array(spins_list, dtype=np.float64)  # (n_states, S_size)

    p_obs = np.array(
        [P_S_measured.get(k, 0.0) for k in all_keys], dtype=np.float64,
    )
    # Normalise observed distribution (handles inputs that sum to != 1).
    total = p_obs.sum()
    if total <= 0.0:
        raise ValueError("P_S_measured sums to zero or is negative")
    p_obs = p_obs / total
    if smoothing > 0.0:
        if not 0.0 < smoothing < 1.0:
            raise ValueError("smoothing must lie in (0, 1)")
        p_obs = (1.0 - smoothing) * p_obs + smoothing / n_states

    if edges is None:
        edges = [(i, j) for i in range(S_size) for j in range(i + 1, S_size)]
    n_J = len(edges)
    n_params = S_size + n_J

    edge_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    if n_J > 0:
        edge_products = spins[:, edge_arr[:, 0]] * spins[:, edge_arr[:, 1]]  # (n_states, n_J)
    else:
        edge_products = np.zeros((n_states, 0), dtype=np.float64)

    def energies_for_params(params: np.ndarray) -> np.ndarray:
        h_tilde = params[:S_size]
        J_tilde = params[S_size:]
        return spins @ h_tilde + edge_products @ J_tilde

    def nll(params: np.ndarray) -> float:
        E = energies_for_params(params)
        log_Z = logsumexp(-beta * E)
        out = beta * float(p_obs @ E) + float(log_Z)
        if l2_regularization > 0.0:
            out += 0.5 * l2_regularization * float(params @ params)
        return out

    def nll_grad(params: np.ndarray) -> np.ndarray:
        E = energies_for_params(params)
        log_weights = -beta * E
        log_Z = logsumexp(log_weights)
        p_model = np.exp(log_weights - log_Z)
        diff = p_obs - p_model  # maximum-entropy matching condition
        grad_h = beta * (diff @ spins)
        grad_J = beta * (diff @ edge_products)
        grad = np.concatenate([grad_h, grad_J])
        if l2_regularization > 0.0:
            grad = grad + l2_regularization * params
        return grad

    if x0 is None:
        x0 = np.zeros(n_params, dtype=np.float64)

    result = minimize(
        nll, x0, jac=nll_grad, method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 1000},
    )

    h_tilde = result.x[:S_size]
    J_tilde_map = {edge: float(result.x[S_size + k]) for k, edge in enumerate(edges)}

    # Reconstruct fitted distribution for goodness-of-fit metrics.
    E_fit = energies_for_params(result.x)
    log_Z_fit = logsumexp(-beta * E_fit)
    p_fit = np.exp(-beta * E_fit - log_Z_fit)
    P_S_fit = {all_keys[i]: float(p_fit[i]) for i in range(n_states)}

    metrics = compute_gibbs_fit_metrics(P_S_measured, P_S_fit)

    return {
        "h_tilde": h_tilde.tolist(),
        "J_tilde": J_tilde_map,
        "beta": float(beta),
        "success": bool(result.success),
        "cost": float(result.fun),
        "n_params": n_params,
        "metrics": metrics,
        "P_S_fit": P_S_fit,
    }


# ---------------------------------------------------------------------------
# Conditional marginals (E held fixed)
# ---------------------------------------------------------------------------


def classical_conditional_marginal(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    E_state: dict[int, int],
    beta: float,
) -> dict[tuple[int, ...], float]:
    r"""Classical Gibbs marginal of $H_P$ conditioned on the environment spins
    being fixed in ``E_state``.

    This is the physically appropriate baseline for experiments that use a
    single fixed environment preparation (rather than averaging over an
    ensemble of environment states): it treats the boundary couplings as an
    effective field on the subsystem, rather than marginalising over the
    environment.
    """
    n_S = len(S_indices)
    if n_S == 0:
        return {(): 1.0}
    # Validate that E_state keys cover every non-S qubit that appears in h or J.
    involved: set[int] = set(h.keys())
    for i, j in J:
        involved.update((i, j))
    involved.update(S_indices)
    S_set = set(S_indices)
    missing_E = {q for q in involved if q not in S_set and q not in E_state}
    if missing_E:
        raise ValueError(
            f"classical_conditional_marginal: E_state missing spins {sorted(missing_E)}"
        )

    n_states = 2 ** n_S
    indices = np.arange(n_states, dtype=np.int64)
    shifts = np.arange(n_S - 1, -1, -1, dtype=np.int64)
    bits = (indices[:, None] >> shifts[None, :]) & 1
    spins_S = 1 - 2 * bits  # (n_states, n_S)

    s_pos = {q: idx for idx, q in enumerate(S_indices)}

    energies = np.zeros(n_states, dtype=np.float64)
    # On-site fields on subsystem qubits
    for i, hi in h.items():
        if i in s_pos:
            energies += hi * spins_S[:, s_pos[i]]
    # Pair interactions
    for (i, j), Jij in J.items():
        i_in_S = i in s_pos
        j_in_S = j in s_pos
        if i_in_S and j_in_S:
            energies += Jij * spins_S[:, s_pos[i]] * spins_S[:, s_pos[j]]
        elif i_in_S and not j_in_S:
            energies += Jij * spins_S[:, s_pos[i]] * float(E_state[j])
        elif j_in_S and not i_in_S:
            energies += Jij * spins_S[:, s_pos[j]] * float(E_state[i])
        else:
            # both in E — constant offset, absorbed into Z
            pass

    E_min = energies.min()
    weights = np.exp(-beta * (energies - E_min))
    probs = weights / weights.sum()

    P_S: dict[tuple[int, ...], float] = {}
    for s_int in range(n_states):
        s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
        config = tuple(1 - 2 * b for b in s_bits)
        P_S[config] = float(probs[s_int])
    return P_S


def quantum_conditional_marginal(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    E_state: dict[int, int],
    A_over_B: float,
    beta: float,
) -> dict[tuple[int, ...], float]:
    r"""Quantum reduced Gibbs marginal conditioned on a fixed z-basis
    environment state.

    Computes $P_S(z_S) = \mathrm{diag}(\rho)(z_S, z_E^{\text{fixed}})
    / \sum_{z_S'} \mathrm{diag}(\rho)(z_S', z_E^{\text{fixed}})$, where $\rho$
    is the full-system quantum Gibbs state at the pause point
    $\rho \propto \exp(-\beta\,[-(A/B)\sum\sigma^x + H_P])$.

    This is the natural "conditioning on a z-basis projective measurement of
    $E$" interpretation of the fixed-$E$ experimental protocol.
    """
    # Expand scope so that the full-system size includes E_state qubits too.
    qubits: set[int] = set(h.keys())
    for i, j in J:
        qubits.update((i, j))
    qubits.update(S_indices)
    qubits.update(E_state.keys())
    n_expected = (max(qubits) + 1) if qubits else 0
    if n_expected == 0:
        return {(): 1.0}

    # Compute diag(rho) on the same index space used by build_ising_hamiltonian;
    # _compute_full_system_rho_diag infers n from (h, J, S_indices) which may
    # miss isolated E qubits that don't appear in either dict, so we extend h
    # with zero fields for any missing qubits before calling the helper.
    h_full = dict(h)
    for q in E_state.keys():
        h_full.setdefault(q, 0.0)
    rho_diag, n = _compute_full_system_rho_diag(h_full, J, A_over_B, beta, S_indices)
    if n != n_expected:
        raise ValueError(
            f"quantum_conditional_marginal: Hamiltonian n={n} differs from "
            f"expected n={n_expected} (check that E_state covers all E qubits)."
        )

    # Condition on E = E_state: select basis indices where every E qubit
    # matches the fixed spin, renormalise.
    n_S = len(S_indices)
    E_indices = [i for i in range(n) if i not in set(S_indices)]
    # Validate E_state
    missing_E = [q for q in E_indices if q not in E_state]
    if missing_E:
        raise ValueError(
            f"quantum_conditional_marginal: E_state missing spins {missing_E}"
        )

    P_S: dict[tuple[int, ...], float] = {}
    total = 0.0
    for s_int in range(2 ** n_S):
        s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
        full_bits = [0] * n
        for k, idx in enumerate(S_indices):
            full_bits[idx] = s_bits[k]
        for idx in E_indices:
            # spin = +1 -> bit 0, spin = -1 -> bit 1
            full_bits[idx] = 0 if int(E_state[idx]) == 1 else 1
        basis_idx = sum(b << (n - 1 - i) for i, b in enumerate(full_bits))
        p = float(rho_diag[basis_idx])
        config = tuple(1 - 2 * b for b in s_bits)
        P_S[config] = p
        total += p

    if total <= 0.0:
        raise ValueError(
            "quantum_conditional_marginal: E_state lies in a zero-measure "
            "slice of the Gibbs state (check beta, A/B, and the Hamiltonian)."
        )
    return {k: v / total for k, v in P_S.items()}


__all__ = [
    "quantum_reduced_gibbs_marginal",
    "classical_diagonal_marginal",
    "classical_conditional_marginal",
    "quantum_conditional_marginal",
    "one_point_magnetization",
    "two_point_correlations",
    "compute_gibbs_fit_metrics",
    "fit_effective_H_S",
]
