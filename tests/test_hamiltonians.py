"""Tests for Hamiltonian builders."""

from __future__ import annotations

import numpy as np
import pytest

from locth1.hamiltonians import build_logical_graph, partition_subsystem


class TestBuildLogicalGraph:
    def test_random_regular_basic(self):
        result = build_logical_graph(
            graph_type="random_regular",
            n_qubits=10,
            subsystem_size=4,
            coupling_lambda=0.5,
            disorder_W=0.0,
            seed=42,
        )
        assert len(result["h"]) == 10
        assert len(result["S_qubits"]) == 4
        assert len(result["E_qubits"]) == 6
        assert set(result["S_qubits"]) & set(result["E_qubits"]) == set()

    def test_disorder(self):
        """Non-zero W should produce non-zero h fields."""
        result = build_logical_graph(
            graph_type="random_regular",
            n_qubits=10,
            subsystem_size=4,
            coupling_lambda=0.5,
            disorder_W=1.0,
            seed=42,
        )
        h_values = list(result["h"].values())
        assert any(abs(v) > 0 for v in h_values)
        assert all(abs(v) <= 1.0 for v in h_values)

    def test_lambda_scaling(self):
        """Boundary couplers should be scaled by lambda."""
        lam = 0.3
        result = build_logical_graph(
            graph_type="random_regular",
            n_qubits=10,
            subsystem_size=4,
            coupling_lambda=lam,
            disorder_W=0.0,
            seed=42,
        )
        S_set = set(result["S_qubits"])
        E_set = set(result["E_qubits"])
        for (i, j), v in result["J"].items():
            is_boundary = (i in S_set and j in E_set) or (j in S_set and i in E_set)
            if is_boundary:
                assert abs(v) == pytest.approx(abs(lam), rel=0.01)
            else:
                assert abs(v) == pytest.approx(1.0, rel=0.01)

    def test_no_partition(self):
        """subsystem_size=0 should give all qubits in E."""
        result = build_logical_graph(
            graph_type="random_regular",
            n_qubits=10,
            subsystem_size=0,
            coupling_lambda=1.0,
            disorder_W=0.0,
            seed=42,
        )
        assert len(result["S_qubits"]) == 0
        assert len(result["E_qubits"]) == 10

    def test_2d_lattice(self):
        result = build_logical_graph(
            graph_type="2d_lattice",
            n_qubits=16,
            subsystem_size=4,
            coupling_lambda=0.5,
            disorder_W=0.0,
            seed=42,
        )
        assert len(result["h"]) == 16
        assert len(result["S_qubits"]) == 4
