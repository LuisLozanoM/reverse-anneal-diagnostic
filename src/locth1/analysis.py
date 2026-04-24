"""Data processing pipeline: marginals, crossover detection, scaling collapse."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Marginal distributions
# ---------------------------------------------------------------------------


def compute_marginal(
    samples: np.ndarray,
    S_indices: list[Any],
    variables: list[Any] | None = None,
) -> dict[tuple[int, ...], float]:
    """Compute the marginal distribution P_S from a sample array.

    samples: shape (n_reads, n_qubits), spin values in {-1, +1}.
    S_indices: qubit labels (or column indices if variables is None).
    variables: ordered list of variable labels matching sample columns.
        When provided, S_indices are looked up in this list to get column positions.
    """
    if variables is not None:
        var_to_col = {v: i for i, v in enumerate(variables)}
        col_indices = [var_to_col[s] for s in S_indices]
    else:
        col_indices = S_indices
    sub = samples[:, col_indices]
    n_reads = sub.shape[0]

    counts: dict[tuple[int, ...], int] = {}
    for row in sub:
        key = tuple(int(s) for s in row)
        counts[key] = counts.get(key, 0) + 1

    return {k: v / n_reads for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Crossover extraction
# ---------------------------------------------------------------------------


def extract_crossover(
    scan_data: list[dict[str, Any]],
    order_param_key: str,
    control_param_key: str,
) -> dict[str, Any]:
    """Find the crossover point where the order parameter crosses 0.5.

    Falls back to the inflection point (maximum |dO/dx|) if no 0.5 crossing exists.
    """
    if len(scan_data) < 2:
        raise ValueError("Need at least 2 data points for crossover extraction.")

    x = np.array([d[control_param_key] for d in scan_data], dtype=np.float64)
    y = np.array([d[order_param_key] for d in scan_data], dtype=np.float64)

    # Sort by control parameter
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Try to find 0.5 crossing via interpolation
    crossings = []
    for i in range(len(y) - 1):
        if (y[i] - 0.5) * (y[i + 1] - 0.5) <= 0:
            # Linear interpolation
            if abs(y[i + 1] - y[i]) < 1e-15:
                xc = 0.5 * (x[i] + x[i + 1])
            else:
                xc = x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            crossings.append(float(xc))

    if crossings:
        crossover_value = float(np.mean(crossings))
        ci_half = float(np.std(crossings)) if len(crossings) > 1 else float(
            0.5 * min(np.diff(x))
        )
        return {
            "crossover_value": crossover_value,
            "confidence_interval": (crossover_value - ci_half, crossover_value + ci_half),
            "method": "0.5_crossing",
        }

    # Fallback: inflection point (max |dy/dx|)
    if len(x) >= 3:
        f = interp1d(x, y, kind="linear")
        x_fine = np.linspace(x[0], x[-1], max(500, 10 * len(x)))
        y_fine = f(x_fine)
        dy = np.abs(np.gradient(y_fine, x_fine))
        idx = int(np.argmax(dy))
        crossover_value = float(x_fine[idx])
        dx = float(x_fine[1] - x_fine[0])
        return {
            "crossover_value": crossover_value,
            "confidence_interval": (crossover_value - 5 * dx, crossover_value + 5 * dx),
            "method": "inflection_point",
        }

    # Last resort: midpoint
    mid = float(0.5 * (x[0] + x[-1]))
    return {
        "crossover_value": mid,
        "confidence_interval": (float(x[0]), float(x[-1])),
        "method": "midpoint_fallback",
    }


# ---------------------------------------------------------------------------
# Scaling collapse
# ---------------------------------------------------------------------------


def _collapse_quality(
    exponents: np.ndarray,
    datasets: list[dict[str, Any]],
    exponent_names: list[str],
) -> float:
    """Measure scatter after rescaling: lower is better."""
    exp_dict = {name: val for name, val in zip(exponent_names, exponents, strict=True)}
    nu = exp_dict.get("nu", 1.0)
    beta_over_nu = exp_dict.get("beta_over_nu", 0.0)

    all_X: list[np.ndarray] = []
    all_Y: list[np.ndarray] = []

    for ds in datasets:
        x = np.asarray(ds["x"], dtype=np.float64)
        y = np.asarray(ds["y"], dtype=np.float64)
        L = float(ds["L"])

        X_scaled = (x - exp_dict.get("xc", 0.0)) * L ** (1.0 / nu)
        Y_scaled = y * L ** beta_over_nu
        all_X.append(X_scaled)
        all_Y.append(Y_scaled)

    X_cat = np.concatenate(all_X)
    Y_cat = np.concatenate(all_Y)

    # Sort by rescaled X
    order = np.argsort(X_cat)
    X_sorted = X_cat[order]
    Y_sorted = Y_cat[order]

    # Quality: sum of squared differences between consecutive points
    if len(X_sorted) < 2:
        return 0.0
    dY = np.diff(Y_sorted)
    dX = np.diff(X_sorted)
    # Weight by 1/(dX+eps) so nearby points contribute more
    weights = 1.0 / (np.abs(dX) + 1e-10)
    return float(np.sum(weights * dY ** 2) / np.sum(weights))


def scaling_collapse(
    datasets: list[dict[str, Any]],
    initial_exponents: dict[str, float],
) -> dict[str, Any]:
    """Optimize scaling exponents for data collapse.

    Each dataset dict has keys 'x' (control param), 'y' (order param), 'L' (system size).
    initial_exponents: starting values, e.g. {"nu": 1.0, "beta_over_nu": 0.0, "xc": 0.5}.
    """
    names = sorted(initial_exponents.keys())
    x0 = np.array([initial_exponents[n] for n in names])

    result = minimize(
        _collapse_quality,
        x0,
        args=(datasets, names),
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-10},
    )

    optimized = {n: float(v) for n, v in zip(names, result.x, strict=True)}
    return {
        "exponents": optimized,
        "quality": float(result.fun),
        "converged": bool(result.success),
        "n_iterations": int(result.nit),
    }
