"""Tests for classical baselines — validates against known exact results."""

from __future__ import annotations

import numpy as np
import pytest

from locth1.classical.glauber import glauber_step, run_glauber
from locth1.classical.spin_vector_mc import (
    metropolis_sweep,
    run_equilibrium_sampling,
    spin_vector_thermalization,
)


def _complete_ferromagnet(n_spins: int) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    h = {i: 0.0 for i in range(n_spins)}
    J = {(i, j): -1.0 for i in range(n_spins) for j in range(i + 1, n_spins)}
    return h, J


class TestGlauber:
    def test_detailed_balance(self):
        """Glauber dynamics at T>0 should satisfy detailed balance.
        After many sweeps, the distribution should be Boltzmann."""
        # 2-qubit ferromagnet: H = -J*s1*s2, J=1
        h = {0: 0.0, 1: 0.0}
        J = {(0, 1): -1.0}
        T = 1.0

        # Run many sweeps from a fixed initial state
        initial = np.array([1, 1])
        result = run_glauber(h, J, initial, T, n_sweeps=10000, seed=42)

        # Count (1,1), (1,-1), (-1,1), (-1,-1) in trajectory
        traj = result["trajectory"]
        configs = [tuple(s) for s in traj[-5000:]]  # last half
        from collections import Counter
        counts = Counter(configs)
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}

        # Exact Boltzmann: P(aligned) ∝ exp(1/T), P(anti) ∝ exp(-1/T)
        Z = 2 * np.exp(1 / T) + 2 * np.exp(-1 / T)
        p_aligned = np.exp(1 / T) / Z
        p_anti = np.exp(-1 / T) / Z

        # Check roughly correct (within statistical noise)
        aligned = probs.get((1, 1), 0) + probs.get((-1, -1), 0)
        anti = probs.get((1, -1), 0) + probs.get((-1, 1), 0)
        assert aligned == pytest.approx(2 * p_aligned, abs=0.05)
        assert anti == pytest.approx(2 * p_anti, abs=0.05)

    def test_low_temperature_ferromagnet(self):
        """At very low T, ferromagnet should stay aligned."""
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): -1.0, (0, 2): -1.0}
        T = 0.01
        initial = np.array([1, 1, 1])
        result = run_glauber(h, J, initial, T, n_sweeps=1000, seed=42)
        # Should remain aligned
        final = result["final_spins"]
        assert np.all(final == 1) or np.all(final == -1)


class TestSpinVectorMC:
    def test_metropolis_sweep_preserves_unit_norms(self):
        """Continuous-spin updates should stay on the Bloch sphere."""
        h, J = _complete_ferromagnet(4)
        spins = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))

        updated, acceptance = metropolis_sweep(
            spins, h, J, T=0.5, rng=np.random.default_rng(7)
        )

        assert updated.shape == (4, 3)
        assert np.allclose(np.linalg.norm(updated, axis=1), 1.0, atol=1e-12)
        assert 0.0 <= acceptance <= 1.0

    def test_high_temperature_randomizes_spin_vectors(self):
        """At high temperature, O(3) spins should have near-zero net magnetization."""
        h, J = _complete_ferromagnet(8)
        samples = run_equilibrium_sampling(
            h,
            J,
            T=20.0,
            n_equilibration=2000,
            n_samples=400,
            n_interval=5,
            seed=42,
        )

        mean_vector = samples.mean(axis=(0, 1))
        assert np.allclose(np.linalg.norm(samples, axis=2), 1.0, atol=1e-10)
        assert np.linalg.norm(mean_vector) < 0.15

    def test_projected_subsystem_marginal_is_balanced_at_high_temperature(self):
        """Z-axis measurement of a hot single spin should give P(up) ~= P(down) ~= 1/2."""
        P_S = spin_vector_thermalization(
            h={0: 0.0},
            J={},
            S_indices=[0],
            initial_states={
                "all_up": np.array([[0.0, 0.0, 1.0]]),
                "all_down": np.array([[0.0, 0.0, -1.0]]),
            },
            T=5.0,
            n_equilibration=500,
            n_samples=500,
            seed=11,
        )

        for dist in P_S.values():
            assert dist[(1,)] == pytest.approx(0.5, abs=0.08)
            assert dist[(-1,)] == pytest.approx(0.5, abs=0.08)

    def test_low_temperature_ferromagnet_aligns(self):
        """At low temperature, a ferromagnet should develop strong vector alignment."""
        n_spins = 8
        h = {i: -0.5 for i in range(n_spins)}
        J = {(i, j): -1.0 for i in range(n_spins) for j in range(i + 1, n_spins)}
        samples = run_equilibrium_sampling(
            h,
            J,
            T=0.02,
            n_equilibration=3000,
            n_samples=200,
            n_interval=4,
            seed=123,
        )

        sample_magnetizations = np.linalg.norm(samples.mean(axis=1), axis=1)
        mean_z = samples[:, :, 2].mean()
        assert sample_magnetizations.mean() > 0.9
        assert mean_z > 0.75
