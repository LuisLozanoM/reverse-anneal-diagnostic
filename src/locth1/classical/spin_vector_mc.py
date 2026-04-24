"""Classical spin-vector Monte Carlo for continuous O(3) Heisenberg spins.

Each spin is represented by a 3D unit vector ``n_i`` on the Bloch sphere and
evolves under the classical Hamiltonian

    H = sum_i h_i * n_i^z + sum_{ij} J_ij * (n_i . n_j).

Sampling uses single-spin Metropolis updates with random rotations on the unit
sphere. Subsystem marginals are obtained by projecting each sampled spin onto
the z axis, with ``P(up) = (1 + n_z) / 2``.
"""

from __future__ import annotations

from itertools import product

import numpy as np

_DEFAULT_MAX_ROTATION_ANGLE = np.pi / 2.0
_EPS = 1e-15


# ---------------------------------------------------------------------------
# Spin helpers
# ---------------------------------------------------------------------------


def _infer_n_spins(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> int:
    """Infer the number of spins from field and coupling dictionaries."""
    max_h = max(h.keys(), default=-1)
    max_J = max((max(i, j) for i, j in J), default=-1)
    return max(max_h, max_J) + 1


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Normalize an array of 3D vectors row-wise."""
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < _EPS):
        raise ValueError("Spin vectors must have nonzero norm.")
    return vectors / norms[:, None]


def _coerce_spin_vectors(
    spins: np.ndarray,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Convert input spins to an ``(n_spins, 3)`` unit-vector array.

    For compatibility, a 1D ``±1`` array is interpreted as z-polarized spins.
    """
    arr = np.array(spins, dtype=np.float64, copy=copy)
    if arr.ndim == 1:
        vectors = np.zeros((arr.shape[0], 3), dtype=np.float64)
        vectors[:, 2] = arr
        return _normalize_rows(vectors)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "Spins must be a 1D ±1 array or a 2D array with shape (n_spins, 3)."
        )

    return _normalize_rows(arr)


def _random_unit_vectors(
    n_spins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw unit vectors uniformly from the sphere."""
    vectors = rng.normal(size=(n_spins, 3))
    norms = np.linalg.norm(vectors, axis=1)
    zero_mask = norms < _EPS
    while np.any(zero_mask):
        vectors[zero_mask] = rng.normal(size=(int(zero_mask.sum()), 3))
        norms = np.linalg.norm(vectors, axis=1)
        zero_mask = norms < _EPS
    return vectors / norms[:, None]


def _random_tangent_unit_vector(
    spin: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random unit vector in the tangent plane at ``spin``."""
    tangent = rng.normal(size=3)
    tangent -= np.dot(tangent, spin) * spin
    norm = np.linalg.norm(tangent)
    while norm < _EPS:
        tangent = rng.normal(size=3)
        tangent -= np.dot(tangent, spin) * spin
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _propose_rotated_spin(
    spin: np.ndarray,
    rng: np.random.Generator,
    *,
    max_rotation_angle: float = _DEFAULT_MAX_ROTATION_ANGLE,
) -> np.ndarray:
    """Propose a random rotation of a spin on the unit sphere."""
    tangent = _random_tangent_unit_vector(spin, rng)
    theta = float(rng.uniform(0.0, max_rotation_angle))
    proposal = np.cos(theta) * spin + np.sin(theta) * tangent
    return proposal / np.linalg.norm(proposal)


def _build_neighbors(
    n_spins: int,
    J: dict[tuple[int, int], float],
) -> list[list[tuple[int, float]]]:
    """Build a symmetric adjacency list from pair couplings."""
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_spins)]
    for (i, j), Jij in J.items():
        if i >= n_spins or j >= n_spins:
            raise ValueError("Coupling indices exceed spin-array dimensions.")
        neighbors[i].append((j, Jij))
        neighbors[j].append((i, Jij))
    return neighbors


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------


def _total_energy(
    spins: np.ndarray,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> float:
    """Total Heisenberg energy for unit-vector spins."""
    vectors = _coerce_spin_vectors(spins, copy=False)
    e = sum(hi * vectors[i, 2] for i, hi in h.items())
    e += sum(Jij * float(np.dot(vectors[i], vectors[j])) for (i, j), Jij in J.items())
    return float(e)


def _local_field(
    spins: np.ndarray,
    i: int,
    h: dict[int, float],
    neighbors: list[list[tuple[int, float]]],
) -> np.ndarray:
    """Effective field seen by spin ``i`` in the classical Hamiltonian."""
    field = np.array([0.0, 0.0, h.get(i, 0.0)], dtype=np.float64)
    for j, Jij in neighbors[i]:
        field += Jij * spins[j]
    return field


def _delta_energy(
    spins: np.ndarray,
    i: int,
    proposal: np.ndarray,
    h: dict[int, float],
    neighbors: list[list[tuple[int, float]]],
) -> float:
    """Energy change from rotating spin ``i`` to ``proposal``."""
    field = _local_field(spins, i, h, neighbors)
    return float(np.dot(proposal - spins[i], field))


# ---------------------------------------------------------------------------
# Metropolis sweep
# ---------------------------------------------------------------------------


def metropolis_sweep(
    spins: np.ndarray,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    T: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """One Metropolis sweep of random single-spin rotations on the sphere."""
    vectors = _coerce_spin_vectors(spins, copy=True)
    n_spins = vectors.shape[0]
    if n_spins == 0:
        return vectors, 0.0

    neighbors = _build_neighbors(n_spins, J)
    n_accepted = 0

    for i in rng.permutation(n_spins):
        idx = int(i)
        proposal = _propose_rotated_spin(vectors[idx], rng)
        dE = _delta_energy(vectors, idx, proposal, h, neighbors)
        if dE <= 0.0 or (T > 0.0 and rng.random() < np.exp(-dE / T)):
            vectors[idx] = proposal
            n_accepted += 1

    return vectors, n_accepted / n_spins


# ---------------------------------------------------------------------------
# Equilibrium sampling
# ---------------------------------------------------------------------------


def run_equilibrium_sampling(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    T: float,
    n_equilibration: int,
    n_samples: int,
    n_interval: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Equilibrate and collect continuous-spin samples of shape ``(M, N, 3)``."""
    n_spins = _infer_n_spins(h, J)
    rng = np.random.default_rng(seed)
    spins = _random_unit_vectors(n_spins, rng)

    for _ in range(n_equilibration):
        spins, _ = metropolis_sweep(spins, h, J, T, rng)

    samples = np.empty((n_samples, n_spins, 3), dtype=np.float64)
    for s in range(n_samples):
        for _ in range(n_interval):
            spins, _ = metropolis_sweep(spins, h, J, T, rng)
        samples[s] = spins

    return samples


# ---------------------------------------------------------------------------
# Thermalization comparison
# ---------------------------------------------------------------------------


def _extract_P_S(
    samples: np.ndarray,
    S_indices: list[int],
) -> dict[tuple[int, ...], float]:
    """Compute a z-basis subsystem marginal from continuous spin samples."""
    array = np.asarray(samples, dtype=np.float64)

    if array.ndim == 2:
        # Backward-compatible path for legacy discrete ±1 samples.
        probs: dict[tuple[int, ...], float] = {}
        for config in product((1, -1), repeat=len(S_indices)):
            matches = np.ones(array.shape[0], dtype=bool)
            for offset, idx in enumerate(S_indices):
                matches &= np.isclose(array[:, idx], config[offset])
            probs[config] = float(matches.mean())
        return probs

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Samples must have shape (n_samples, n_spins, 3).")

    if not S_indices:
        return {(): 1.0}

    p_up = np.clip(0.5 * (1.0 + array[:, S_indices, 2]), 0.0, 1.0)
    probs: dict[tuple[int, ...], float] = {}

    for config in product((1, -1), repeat=len(S_indices)):
        selector = np.array(config, dtype=np.int8) == 1
        weights = np.where(selector[None, :], p_up, 1.0 - p_up)
        probs[config] = float(np.mean(np.prod(weights, axis=1)))

    total = sum(probs.values())
    if total > 0.0:
        probs = {config: value / total for config, value in probs.items()}
    return probs


def spin_vector_thermalization(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    initial_states: dict[str, np.ndarray],
    T: float,
    n_equilibration: int,
    n_samples: int,
    *,
    seed: int = 0,
) -> dict[str, dict[tuple[int, ...], float]]:
    """Equilibrate each initial state and estimate the projected subsystem marginal."""
    if not initial_states:
        return {}

    results: dict[str, dict[tuple[int, ...], float]] = {}

    for idx, (name, spins_0) in enumerate(initial_states.items()):
        rng = np.random.default_rng(seed + idx)
        spins = _coerce_spin_vectors(spins_0, copy=True)
        n_spins = spins.shape[0]

        for _ in range(n_equilibration):
            spins, _ = metropolis_sweep(spins, h, J, T, rng)

        samples = np.empty((n_samples, n_spins, 3), dtype=np.float64)
        for s in range(n_samples):
            spins, _ = metropolis_sweep(spins, h, J, T, rng)
            samples[s] = spins

        results[name] = _extract_P_S(samples, S_indices)

    return results


# ---------------------------------------------------------------------------
# Autocorrelation diagnostic
# ---------------------------------------------------------------------------


def compute_autocorrelation(
    samples: np.ndarray,
    max_lag: int = 100,
) -> np.ndarray:
    """Autocorrelation of the z-magnetization time series."""
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 3 and array.shape[2] == 3:
        mag = array[:, :, 2].mean(axis=1)
    elif array.ndim == 2:
        mag = array.mean(axis=1)
    else:
        raise ValueError(
            "Samples must have shape (n_samples, n_spins) or (n_samples, n_spins, 3)."
        )

    mag = mag - mag.mean()
    n = len(mag)
    max_lag = min(max_lag, n - 1)

    var = np.var(mag)
    if var < 1e-15:
        return np.zeros(max_lag + 1)

    acf = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        acf[lag] = np.mean(mag[: n - lag] * mag[lag:]) / var

    return acf
