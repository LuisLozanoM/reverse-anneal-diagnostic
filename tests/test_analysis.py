"""Tests for analysis module — validates marginal computation."""

from __future__ import annotations

import numpy as np
import pytest

from locth1.analysis import compute_marginal


class TestComputeMarginal:
    def test_single_qubit_marginal(self):
        """3 qubits, marginal over qubit 0."""
        samples = np.array([
            [1, 1, 1],
            [1, -1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
        ])
        P = compute_marginal(samples, [0])
        assert (1,) in P
        assert (-1,) in P
        assert P[(1,)] == pytest.approx(0.5)
        assert P[(-1,)] == pytest.approx(0.5)

    def test_two_qubit_marginal(self):
        """4 qubits, marginal over qubits [0, 1]."""
        samples = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [-1, 1, 1, 1],
        ])
        P = compute_marginal(samples, [0, 1])
        assert sum(P.values()) == pytest.approx(1.0)
        assert P[(1, 1)] == pytest.approx(0.5)  # 2 out of 4

    def test_full_system_marginal(self):
        """Marginal over all qubits = full distribution."""
        samples = np.array([
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, 1],
        ])
        P = compute_marginal(samples, [0, 1])
        assert P[(1, 1)] == pytest.approx(0.25)
        assert P[(1, -1)] == pytest.approx(0.25)
        assert P[(-1, 1)] == pytest.approx(0.5)

    def test_normalization(self):
        """Marginal must be normalized."""
        rng = np.random.default_rng(42)
        samples = rng.choice([-1, 1], size=(1000, 10))
        P = compute_marginal(samples, [0, 3, 7])
        assert sum(P.values()) == pytest.approx(1.0, abs=1e-10)
