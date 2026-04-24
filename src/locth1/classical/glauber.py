"""Glauber dynamics (single-spin-flip Metropolis) at device temperature.

THE critical classical baseline. Works at any system size. Correctness and
performance are paramount: inner loops are fully vectorized where possible.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Energy helpers (vectorized)
# ---------------------------------------------------------------------------


def _local_energy(
    spins: np.ndarray,
    i: int,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> float:
    """Compute the local energy contribution of spin i: E_i = h_i s_i + sum_j J_ij s_i s_j."""
    e = h.get(i, 0.0) * spins[i]
    for (a, b), Jij in J.items():
        if a == i:
            e += Jij * spins[i] * spins[b]
        elif b == i:
            e += Jij * spins[a] * spins[i]
    return e


def _delta_energy(
    spins: np.ndarray,
    i: int,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
) -> float:
    """Energy change from flipping spin i: Delta_E = -2 * E_local_i (since s -> -s)."""
    # E_before = h_i s_i + sum_j J_ij s_i s_j
    # E_after  = -E_before  (flip s_i)
    # Delta_E  = E_after - E_before = -2 E_before
    return -2.0 * _local_energy(spins, i, h, J)


# ---------------------------------------------------------------------------
# Single Monte Carlo sweep
# ---------------------------------------------------------------------------


def glauber_step(
    spins: np.ndarray,
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One MC sweep: N attempted single-spin flips with Glauber acceptance."""
    n = len(spins)
    spins = spins.copy()
    order = rng.permutation(n)

    for i in order:
        dE = _delta_energy(spins, int(i), h, J)
        # Glauber acceptance: p = 1 / (1 + exp(dE / T))
        if T > 0:
            p_accept = 1.0 / (1.0 + np.exp(dE / T))
        else:
            p_accept = 1.0 if dE < 0 else (0.5 if dE == 0 else 0.0)
        if rng.random() < p_accept:
            spins[int(i)] *= -1

    return spins


# ---------------------------------------------------------------------------
# Full Glauber run
# ---------------------------------------------------------------------------


def run_glauber(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    initial_spins: np.ndarray,
    T: float,
    n_sweeps: int,
    *,
    seed: int = 0,
) -> dict:
    """Run n_sweeps of Glauber dynamics. Returns trajectory, final_spins, acceptance_rate."""
    rng = np.random.default_rng(seed)
    spins = initial_spins.copy()
    trajectory = [spins.copy()]
    n_flips = 0

    for _ in range(n_sweeps):
        old = spins.copy()
        spins = glauber_step(spins, h, J, T, rng)
        n_flips += np.sum(old != spins)
        trajectory.append(spins.copy())

    acceptance_rate = n_flips / (n_sweeps * len(initial_spins))
    return {
        "trajectory": trajectory,
        "final_spins": spins,
        "acceptance_rate": float(acceptance_rate),
    }


# ---------------------------------------------------------------------------
# Subsystem marginal from samples
# ---------------------------------------------------------------------------


def _extract_P_S(
    samples: list[np.ndarray],
    S_indices: list[int],
) -> dict[tuple[int, ...], float]:
    """Compute P_S marginal from a list of spin-configuration samples."""
    counts: Counter[tuple[int, ...]] = Counter()
    for sample in samples:
        config = tuple(int(sample[i]) for i in S_indices)
        counts[config] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Thermalization comparison
# ---------------------------------------------------------------------------


def glauber_thermalization(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    initial_states: dict[str, np.ndarray],
    T: float,
    n_sweeps: int,
    n_samples: int,
    *,
    seed: int = 0,
) -> dict[str, dict[tuple[int, ...], float]]:
    """Run Glauber for each initial state, collect samples after equilibration, compute P_S."""
    results = {}
    for idx, (name, spins_0) in enumerate(initial_states.items()):
        rng = np.random.default_rng(seed + idx)
        spins = spins_0.copy()

        # Equilibrate
        for _ in range(n_sweeps):
            spins = glauber_step(spins, h, J, T, rng)

        # Sample (one sample per additional sweep)
        samples: list[np.ndarray] = []
        for _ in range(n_samples):
            spins = glauber_step(spins, h, J, T, rng)
            samples.append(spins.copy())

        results[name] = _extract_P_S(samples, S_indices)

    return results


# ---------------------------------------------------------------------------
# Thermalization time detection
# ---------------------------------------------------------------------------


def _tvd(p: dict[tuple, float], q: dict[tuple, float]) -> float:
    """Total variation distance between two distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def compute_thermalization_time(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    initial_states: dict[str, np.ndarray],
    T: float,
    max_sweeps: int,
    S_indices: list[int],
    *,
    threshold: float = 0.05,
    seed: int = 0,
    n_window: int = 50,
) -> dict:
    """Find sweep count where TVD between P_S from different initial states drops below threshold.

    Uses a sliding window of n_window samples to estimate P_S at each sweep checkpoint.
    """
    names = list(initial_states.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 initial states to compare.")

    rng_map = {name: np.random.default_rng(seed + i) for i, name in enumerate(names)}
    spins_map = {name: initial_states[name].copy() for name in names}
    sample_buffers: dict[str, list[np.ndarray]] = {name: [] for name in names}

    tvd_trajectory: list[tuple[int, float]] = []
    thermalization_sweep: int | None = None

    for sweep in range(1, max_sweeps + 1):
        for name in names:
            spins_map[name] = glauber_step(
                spins_map[name], h, J, T, rng_map[name]
            )
            sample_buffers[name].append(spins_map[name].copy())
            # Keep only last n_window samples
            if len(sample_buffers[name]) > n_window:
                sample_buffers[name].pop(0)

        # Compute TVD every n_window sweeps (after enough samples)
        if sweep >= n_window and sweep % (n_window // 5 or 1) == 0:
            p_s_all = {
                name: _extract_P_S(sample_buffers[name], S_indices)
                for name in names
            }
            # Average pairwise TVD
            tvd_vals = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    tvd_vals.append(_tvd(p_s_all[names[i]], p_s_all[names[j]]))
            avg_tvd = float(np.mean(tvd_vals))
            tvd_trajectory.append((sweep, avg_tvd))

            if thermalization_sweep is None and avg_tvd < threshold:
                thermalization_sweep = sweep

    return {
        "thermalization_sweep": thermalization_sweep,
        "tvd_trajectory": tvd_trajectory,
    }
