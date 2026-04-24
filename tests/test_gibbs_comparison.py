"""Tests for gibbs_comparison module.

Validates the quantum reduced Gibbs marginal, classical diagonal marginal,
partial trace, observable helpers, comparison metrics, and effective H_S fit
against analytically known small cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from locth1.gibbs_comparison import (
    _trace_out_environment_diagonal,
    classical_diagonal_marginal,
    compute_gibbs_fit_metrics,
    fit_effective_H_S,
    one_point_magnetization,
    quantum_reduced_gibbs_marginal,
    two_point_correlations,
)


# ---------------------------------------------------------------------------
# quantum_reduced_gibbs_marginal
# ---------------------------------------------------------------------------


def test_single_qubit_no_transverse_field_matches_classical_boltzmann():
    """For h=1, A=0, B=2 and beta=1: the classical Ising gives P(+1) / P(-1) = exp(-2)."""
    h = {0: 1.0}
    J: dict[tuple[int, int], float] = {}
    S_indices = [0]

    P_S = quantum_reduced_gibbs_marginal(
        h, J, S_indices, A_over_B=0.0, beta=1.0,
    )

    # Energy of spin +1 is h*1 = +1; energy of spin -1 is -1.
    # exp(-β·(+1)) / exp(-β·(-1)) = exp(-2).
    ratio_expected = np.exp(-2.0)
    ratio_computed = P_S[(1,)] / P_S[(-1,)]
    assert pytest.approx(ratio_expected, rel=1e-10) == ratio_computed

    # Total probability sums to 1.
    assert pytest.approx(1.0, abs=1e-12) == sum(P_S.values())


def test_single_qubit_no_field_with_transverse_field_symmetric():
    """h=0, A/B=1 gives a purely transverse Hamiltonian.  The diagonal must be 50/50."""
    h: dict[int, float] = {0: 0.0}
    J: dict[tuple[int, int], float] = {}
    S_indices = [0]

    P_S = quantum_reduced_gibbs_marginal(
        h, J, S_indices, A_over_B=1.0, beta=1.0,
    )

    # Pure transverse field is off-diagonal in z-basis, so the ρ eigenbasis
    # is the σ_x eigenbasis, and the z-basis diagonal of rho is 50/50.
    assert pytest.approx(0.5, abs=1e-10) == P_S[(1,)]
    assert pytest.approx(0.5, abs=1e-10) == P_S[(-1,)]


def test_two_qubit_ferromagnet_classical_limit():
    """J=-1 ferromagnet, no transverse field: aligned states favoured.

    P(aligned) = 2 * exp(β) / (2 exp(β) + 2 exp(-β)); tracing out one qubit
    is 50/50 by symmetry.
    """
    h: dict[int, float] = {}
    J = {(0, 1): -1.0}

    # Joint distribution
    P_full = classical_diagonal_marginal(h, J, S_indices=[0, 1], beta=1.0)

    # By symmetry (h=0), P(↑↑) = P(↓↓) and P(↑↓) = P(↓↑).
    assert pytest.approx(P_full[(1, 1)], rel=1e-10) == P_full[(-1, -1)]
    assert pytest.approx(P_full[(1, -1)], rel=1e-10) == P_full[(-1, 1)]

    # Subsystem marginal over qubit 0 is 50/50 by symmetry.
    P_S = classical_diagonal_marginal(h, J, S_indices=[0], beta=1.0)
    assert pytest.approx(0.5, abs=1e-12) == P_S[(1,)]
    assert pytest.approx(0.5, abs=1e-12) == P_S[(-1,)]


def test_classical_matches_quantum_at_zero_transverse_field():
    """The classical marginal should match the quantum Gibbs at A/B = 0."""
    h = {0: 0.7, 1: -0.3}
    J = {(0, 1): -0.5}

    for S_indices in [[0], [1], [0, 1]]:
        p_q = quantum_reduced_gibbs_marginal(
            h, J, S_indices, A_over_B=0.0, beta=1.3,
        )
        p_c = classical_diagonal_marginal(h, J, S_indices, beta=1.3)

        assert set(p_q) == set(p_c)
        for k in p_q:
            assert pytest.approx(p_c[k], rel=1e-10, abs=1e-12) == p_q[k]


def test_quantum_reduced_gibbs_marginal_normalises():
    h = {0: 0.3, 1: -0.2, 2: 0.4}
    J = {(0, 1): -1.0, (1, 2): -0.5}

    P_S = quantum_reduced_gibbs_marginal(
        h, J, S_indices=[0, 2], A_over_B=0.5, beta=2.0,
    )
    assert pytest.approx(1.0, abs=1e-10) == sum(P_S.values())
    assert len(P_S) == 4  # 2^|S|


# ---------------------------------------------------------------------------
# classical_diagonal_marginal and _trace_out_environment_diagonal
# ---------------------------------------------------------------------------


def test_classical_marginal_uniform_at_zero_beta():
    h = {0: 1.0, 1: -0.5, 2: 0.3}
    J = {(0, 1): 0.2, (1, 2): -0.1}

    P_S = classical_diagonal_marginal(h, J, S_indices=[0, 1, 2], beta=0.0)
    # β=0 means every state has equal weight 1/2^n.
    for v in P_S.values():
        assert pytest.approx(1.0 / 8.0, abs=1e-12) == v


def test_trace_out_preserves_normalisation():
    n = 3
    rng = np.random.default_rng(0)
    probs = rng.random(2 ** n)
    probs = probs / probs.sum()

    P_S = _trace_out_environment_diagonal(probs, S_indices=[0, 2], n=n)
    assert pytest.approx(1.0, abs=1e-12) == sum(P_S.values())
    assert len(P_S) == 4


def test_trace_out_full_system_returns_input_distribution():
    """If S_indices spans all qubits, the 'marginal' is the full distribution."""
    n = 2
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    P_S = _trace_out_environment_diagonal(probs, S_indices=[0, 1], n=n)
    # Index 0 = bits 00 = spins (+1, +1), index 1 = bits 01 = spins (+1, -1)
    # index 2 = bits 10 = spins (-1, +1), index 3 = bits 11 = spins (-1, -1)
    assert pytest.approx(0.1, abs=1e-12) == P_S[(1, 1)]
    assert pytest.approx(0.2, abs=1e-12) == P_S[(1, -1)]
    assert pytest.approx(0.3, abs=1e-12) == P_S[(-1, 1)]
    assert pytest.approx(0.4, abs=1e-12) == P_S[(-1, -1)]


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


def test_one_point_magnetization_all_up():
    P_S = {(1, 1): 1.0, (1, -1): 0.0, (-1, 1): 0.0, (-1, -1): 0.0}
    mag = one_point_magnetization(P_S)
    assert np.allclose(mag, [1.0, 1.0])


def test_two_point_correlations_ferromagnet():
    # 50/50 aligned state: <s0 s1> = +1
    P_S = {(1, 1): 0.5, (1, -1): 0.0, (-1, 1): 0.0, (-1, -1): 0.5}
    corr = two_point_correlations(P_S)
    assert pytest.approx(1.0, abs=1e-12) == corr[(0, 1)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_identical_distributions_zero():
    P_S = {(1, 1): 0.25, (1, -1): 0.25, (-1, 1): 0.25, (-1, -1): 0.25}
    m = compute_gibbs_fit_metrics(P_S, P_S)
    assert pytest.approx(0.0, abs=1e-12) == m["d_tv"]
    assert pytest.approx(0.0, abs=1e-12) == m["symmetric_kl"]
    assert pytest.approx(0.0, abs=1e-12) == m["magnetization_mae"]


def test_metrics_disjoint_distributions_tvd_one():
    p = {(1,): 1.0, (-1,): 0.0}
    q = {(1,): 0.0, (-1,): 1.0}
    m = compute_gibbs_fit_metrics(p, q)
    assert pytest.approx(1.0, abs=1e-12) == m["d_tv"]
    assert pytest.approx(2.0, abs=1e-12) == m["magnetization_mae"]


# ---------------------------------------------------------------------------
# Effective H_S fit
# ---------------------------------------------------------------------------


def test_fit_recovers_synthetic_ferromagnet():
    """Generate a Gibbs distribution from a known (h, J), then fit and recover them."""
    # True parameters
    h_true = {0: 0.3, 1: -0.2, 2: 0.1}
    J_true = {(0, 1): -0.5, (1, 2): -0.5, (0, 2): 0.1}
    beta = 1.5

    # Build ground-truth classical marginal on the full 3-spin system (no partial trace).
    P_full = classical_diagonal_marginal(h_true, J_true, S_indices=[0, 1, 2], beta=beta)

    # Fit with fully-connected edge set (matches the synthetic model).
    result = fit_effective_H_S(P_full, beta=beta)

    assert result["success"]
    assert result["metrics"]["d_tv"] < 1e-6
    assert result["metrics"]["symmetric_kl"] < 1e-10

    # Reconstruct the fitted distribution and compare element-wise.
    for k, v in P_full.items():
        assert pytest.approx(v, abs=1e-6) == result["P_S_fit"][k]


def test_fit_at_experimental_temperature():
    """Fit at the experimentally relevant beta (~5), not in the near-delta limit.

    The Advantage2 calibration gives beta_eff ≈ 6.93; at that temperature the
    classical Gibbs is peaked but still multi-modal enough for the log-residual
    fit to converge cleanly.  Near-delta distributions at beta >> 10 are a
    separate regime where the fit is ill-conditioned and not used in practice.
    """
    h_true = {0: 1.0, 1: 0.5}
    J_true: dict[tuple[int, int], float] = {}
    beta = 5.0

    P_full = classical_diagonal_marginal(h_true, J_true, S_indices=[0, 1], beta=beta)
    # Lowest-energy state (both spins = -1 for positive h) dominates.
    assert P_full[(-1, -1)] > P_full[(1, 1)]

    result = fit_effective_H_S(P_full, beta=beta)
    assert result["success"]
    assert result["metrics"]["d_tv"] < 1e-3
