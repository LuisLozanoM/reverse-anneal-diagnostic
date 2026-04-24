"""Tests for observables module — validates against known distributions."""

from __future__ import annotations

import numpy as np
import pytest

from locth1.observables import (
    total_variation_distance,
    kl_divergence,
    gibbs_fit,
    initial_state_distance,
    memory_order_parameter,
)


class TestTotalVariationDistance:
    def test_identical_distributions(self):
        P = {(1, 1): 0.5, (-1, -1): 0.5}
        assert total_variation_distance(P, P) == pytest.approx(0.0)

    def test_disjoint_distributions(self):
        P = {(1,): 1.0}
        Q = {(-1,): 1.0}
        assert total_variation_distance(P, Q) == pytest.approx(1.0)

    def test_partial_overlap(self):
        P = {(1,): 0.7, (-1,): 0.3}
        Q = {(1,): 0.3, (-1,): 0.7}
        assert total_variation_distance(P, Q) == pytest.approx(0.4)

    def test_symmetry(self):
        P = {(1,): 0.6, (-1,): 0.4}
        Q = {(1,): 0.4, (-1,): 0.6}
        assert total_variation_distance(P, Q) == total_variation_distance(Q, P)

    def test_bounds(self):
        P = {(1,): 0.8, (-1,): 0.2}
        Q = {(1,): 0.1, (-1,): 0.9}
        tvd = total_variation_distance(P, Q)
        assert 0.0 <= tvd <= 1.0


class TestKLDivergence:
    def test_identical(self):
        P = {(1,): 0.5, (-1,): 0.5}
        assert kl_divergence(P, P) == pytest.approx(0.0, abs=1e-8)

    def test_positive(self):
        P = {(1,): 0.7, (-1,): 0.3}
        Q = {(1,): 0.5, (-1,): 0.5}
        assert kl_divergence(P, Q) > 0


class TestGibbsFit:
    def test_exact_gibbs_distribution(self):
        """A distribution that IS Gibbs should fit perfectly."""
        beta_true = 1.5
        # Configs in sorted order (gibbs_fit sorts keys)
        configs = sorted([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        energies = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.exp(-beta_true * energies)
        probs /= probs.sum()
        P_S = {c: float(p) for c, p in zip(configs, probs)}

        result = gibbs_fit(P_S, energies)
        assert result["beta_eff"] == pytest.approx(beta_true, rel=0.05)
        assert result["chi_squared"] < 0.01

    def test_uniform_distribution(self):
        """Uniform distribution corresponds to beta=0."""
        energies = np.array([0.0, 1.0, 2.0, 3.0])
        configs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        P_S = {c: 0.25 for c in configs}

        result = gibbs_fit(P_S, energies)
        assert result["beta_eff"] == pytest.approx(0.0, abs=0.1)


class TestInitialStateDistance:
    def test_identical_distributions(self):
        P = {"init1": {(1,): 0.5, (-1,): 0.5}, "init2": {(1,): 0.5, (-1,): 0.5}}
        D = initial_state_distance(P)
        assert D.shape == (2, 2)
        assert D[0, 1] == pytest.approx(0.0)

    def test_different_distributions(self):
        P = {"init1": {(1,): 1.0}, "init2": {(-1,): 1.0}}
        D = initial_state_distance(P)
        assert D[0, 1] == pytest.approx(1.0)


class TestMemoryOrderParameter:
    def test_thermalized(self):
        P = {"init1": {(1,): 0.5, (-1,): 0.5}, "init2": {(1,): 0.5, (-1,): 0.5}}
        assert memory_order_parameter(P) == pytest.approx(0.0)

    def test_full_memory(self):
        P = {"init1": {(1,): 1.0}, "init2": {(-1,): 1.0}}
        assert memory_order_parameter(P) == pytest.approx(1.0)
