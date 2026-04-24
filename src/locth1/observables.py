"""Statistical observables for subsystem thermalization analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar, least_squares


# ---------------------------------------------------------------------------
# Distance measures
# ---------------------------------------------------------------------------


def total_variation_distance(P: dict[Any, float], Q: dict[Any, float]) -> float:
    """TVD = 0.5 * sum |P(x) - Q(x)| over the union of supports."""
    keys = set(P.keys()) | set(Q.keys())
    return 0.5 * sum(abs(P.get(k, 0.0) - Q.get(k, 0.0)) for k in keys)


def kl_divergence(
    P: dict[Any, float],
    Q: dict[Any, float],
    epsilon: float = 1e-10,
) -> float:
    """KL(P || Q) with floor epsilon on Q to avoid log(0)."""
    result = 0.0
    for k, p in P.items():
        if p <= 0.0:
            continue
        q = max(Q.get(k, 0.0), epsilon)
        result += p * np.log(p / q)
    return float(result)


# ---------------------------------------------------------------------------
# Gibbs distribution fitting
# ---------------------------------------------------------------------------


def _log_partition(beta: float, energies: np.ndarray) -> float:
    """log Z(beta) = log sum exp(-beta * E_k), numerically stable."""
    x = -beta * energies
    x_max = x.max()
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


def _gibbs_probs(beta: float, energies: np.ndarray) -> np.ndarray:
    """Boltzmann probabilities exp(-beta*E_k)/Z for a given beta."""
    x = -beta * energies
    x_max = x.max()
    log_probs = x - x_max - np.log(np.sum(np.exp(x - x_max)))
    return np.exp(log_probs)


def _neg_log_likelihood(beta: float, energies: np.ndarray, counts: np.ndarray) -> float:
    """Negative log-likelihood of observed counts under Gibbs(beta)."""
    log_Z = _log_partition(beta, energies)
    # sum_k n_k * (-beta * E_k - log_Z)
    ll = np.sum(counts * (-beta * energies - log_Z))
    return float(-ll)


def gibbs_fit(
    P_S: dict[Any, float],
    H_S: np.ndarray,
    *,
    method: str = "mle",
) -> dict[str, Any]:
    """Fit subsystem distribution P_S to Gibbs form exp(-beta*E)/Z.

    P_S maps configuration labels to probabilities. H_S is a 1D array of
    energies aligned with the sorted keys of P_S.
    """
    keys = sorted(P_S.keys())
    probs = np.array([P_S[k] for k in keys], dtype=np.float64)
    energies = np.asarray(H_S, dtype=np.float64).ravel()

    if len(probs) != len(energies):
        raise ValueError(
            f"P_S has {len(probs)} entries but H_S has {len(energies)} energies."
        )

    if method == "mle":
        # Treat probs as fractional counts (they sum to 1)
        result = minimize_scalar(
            lambda b: _neg_log_likelihood(b, energies, probs),
            bounds=(1e-4, 1e4),
            method="bounded",
        )
        beta_eff = float(result.x)
    elif method == "lstsq":
        # Least squares on log(P) ~ -beta*E - log Z
        mask = probs > 0
        if mask.sum() < 2:
            raise ValueError("Need at least 2 nonzero entries for lstsq fit.")
        log_p = np.log(probs[mask])
        E_masked = energies[mask]

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_val, log_Z = params
            return log_p - (-beta_val * E_masked - log_Z)

        x0 = np.array([1.0, np.log(len(energies))])
        sol = least_squares(residuals, x0)
        beta_eff = float(sol.x[0])
    else:
        raise ValueError(f"Unknown method: {method!r}")

    P_gibbs_arr = _gibbs_probs(beta_eff, energies)
    Z = float(np.sum(np.exp(-beta_eff * energies)))
    chi_sq = float(np.sum((probs - P_gibbs_arr) ** 2 / np.maximum(P_gibbs_arr, 1e-30)))
    residuals_arr = probs - P_gibbs_arr

    P_gibbs = {k: float(p) for k, p in zip(keys, P_gibbs_arr, strict=True)}

    return {
        "beta_eff": beta_eff,
        "partition_function": Z,
        "chi_squared": chi_sq,
        "residuals": residuals_arr,
        "P_gibbs": P_gibbs,
    }


# ---------------------------------------------------------------------------
# Initial-state dependence
# ---------------------------------------------------------------------------


def initial_state_distance(P_S_dict: dict[str, dict[Any, float]]) -> np.ndarray:
    """Pairwise TVD matrix between subsystem distributions from different initial states."""
    labels = sorted(P_S_dict.keys())
    n = len(labels)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = total_variation_distance(P_S_dict[labels[i]], P_S_dict[labels[j]])
            D[i, j] = d
            D[j, i] = d
    return D


def memory_order_parameter(P_S_dict: dict[str, dict[Any, float]]) -> float:
    """Max pairwise TVD: 0 = thermalized, 1 = full memory."""
    D = initial_state_distance(P_S_dict)
    if D.size == 0:
        return 0.0
    return float(D.max())
