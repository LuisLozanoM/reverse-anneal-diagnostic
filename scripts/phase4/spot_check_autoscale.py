"""Spot-check: submit the same Phase 4-style problems with auto_scale True
and auto_scale False, and compare memory order parameters.

Purpose: validate that D-Wave's auto_scale was effectively a no-op for the
parameter ranges used in the paper (|h|<=W<=2, |J|<=1), so that the legacy
Phase 3/4 results (submitted with SDK default auto_scale=True) are
quantitatively comparable to classical baselines simulated at the nominal
(h, J).

Paired design: for each condition, the exact same (h, J, initial states,
embedding) is submitted twice back-to-back - once with auto_scale=True,
once with auto_scale=False.  Any statistically significant difference in
the memory order parameter between the two runs would indicate that
auto_scale materially changed the programmed Hamiltonian.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from locth1.analysis import compute_marginal  # noqa: E402
from locth1.observables import memory_order_parameter  # noqa: E402
from locth1.reverse_anneal import run_reverse_anneal  # noqa: E402


def _pick_live_subgraph(nodelist, edgelist, n_qubits, subsystem_size):
    """Pick a connected n-qubit subgraph from the *live* solver graph,
    with the subsystem chosen as a BFS ball from a high-degree node.
    Returns (S_qubits, E_qubits, edges)."""
    G = nx.Graph()
    G.add_nodes_from(nodelist)
    G.add_edges_from(edgelist)
    # Start BFS at a well-connected node
    start = max(G.nodes, key=lambda n: G.degree(n))
    seen = {start}
    visited = [start]
    queue = [start]
    while queue and len(visited) < n_qubits:
        node = queue.pop(0)
        for nbr in sorted(G.neighbors(node)):
            if nbr not in seen:
                seen.add(nbr)
                visited.append(nbr)
                queue.append(nbr)
                if len(visited) >= n_qubits:
                    break
    if len(visited) < n_qubits:
        raise RuntimeError(f"Could not assemble {n_qubits} connected live qubits.")
    sub_nodes = visited[:n_qubits]
    sub_G = G.subgraph(sub_nodes).copy()
    # Subsystem: BFS ball of size subsystem_size starting at the highest-degree node
    sub_start = max(sub_G.nodes, key=lambda n: sub_G.degree(n))
    S_seen = {sub_start}
    S_visited = [sub_start]
    S_queue = [sub_start]
    while S_queue and len(S_visited) < subsystem_size:
        node = S_queue.pop(0)
        for nbr in sorted(sub_G.neighbors(node)):
            if nbr not in S_seen:
                S_seen.add(nbr)
                S_visited.append(nbr)
                S_queue.append(nbr)
                if len(S_visited) >= subsystem_size:
                    break
    S_qubits = S_visited[:subsystem_size]
    E_qubits = [n for n in sub_nodes if n not in set(S_qubits)]
    return S_qubits, E_qubits, list(sub_G.edges())


def _build_hamiltonian(S_qubits, E_qubits, edges, coupling_lambda, disorder_W, seed):
    rng = np.random.default_rng(seed)
    S_set = set(S_qubits)
    h = {n: float(rng.uniform(-disorder_W, disorder_W)) for n in (S_qubits + E_qubits)}
    J = {}
    for u, v in edges:
        on_boundary = (u in S_set) != (v in S_set)
        J[(u, v)] = -1.0 * (coupling_lambda if on_boundary else 1.0)
    return h, J


# Conditions chosen to span the regions of the Phase 4 sweeps where
# auto_scale (if active) would be most likely to shift the memory order
# parameter: the environment-size crossover, the lambda crossover, and
# a mid-disorder point.
SPOT_CONDITIONS = [
    # (label,            n_env, lambda, W)
    # Fig 2a + Fig 3a regeneration: |E| sweep at lambda=0.5 and W sweep at
    # lambda=0.5, |E|=50.  All on native qubits.  Paired True/False arms.
    ("fig2a_E4_lam0.5", 4,  0.5, 0.0),
    ("fig2a_E8_lam0.5", 8,  0.5, 0.0),
    ("fig2a_E16_lam0.5", 16, 0.5, 0.0),
    ("fig2a_E32_lam0.5", 32, 0.5, 0.0),
    ("fig2a_E50_lam0.5", 50, 0.5, 0.0),
    ("fig3a_W0.0_E50_lam0.5", 50, 0.5, 0.0),
    ("fig3a_W0.5_E50_lam0.5", 50, 0.5, 0.5),
    ("fig3a_W1.0_E50_lam0.5", 50, 0.5, 1.0),
    ("fig3a_W1.5_E50_lam0.5", 50, 0.5, 1.5),
    ("fig3a_W2.0_E50_lam0.5", 50, 0.5, 2.0),
]
GAUGE_SEEDS = [0, 1, 2, 3]

SUBSYSTEM_SIZE = 6
S_P = 0.4
T_HOLD = 100.0
NUM_READS = 2000
DISORDER_SEED = 42
TOPOLOGY = "zephyr"


def _initial_states(S_qubits, E_qubits, seed=DISORDER_SEED):
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E_qubits}
    return {
        "S_up": {**{q: 1 for q in S_qubits}, **E_state},
        "S_down": {**{q: -1 for q in S_qubits}, **E_state},
        "S_random": {**{q: int(rng.choice([-1, 1])) for q in S_qubits}, **E_state},
    }


def _run_one(h, J, S_qubits, E_qubits, solver, auto_scale, sampler=None):
    """Submit the problem in PHYSICAL-qubit space (no embedding, no chains).

    build_native_graph returns physical Zephyr node IDs, so we bypass
    EmbeddingComposite with auto_embed=False; this keeps the effective
    Hamiltonian exactly nominal and eliminates embedding variability from
    the auto_scale comparison.
    """
    states = _initial_states(S_qubits, E_qubits)
    P_S = {}
    for name, init in states.items():
        result = run_reverse_anneal(
            h=h, J=J, initial_state=init,
            s_p=S_P, t_hold=T_HOLD,
            num_reads=NUM_READS,
            solver=solver,
            sampler=sampler,
            auto_embed=False,
            auto_scale=auto_scale,
            label=f"spot_autoscale_{'T' if auto_scale else 'F'}_{name}",
        )
        P_S[name] = compute_marginal(
            result["samples"], S_qubits, variables=result["variables"],
        )
        # polite delay between jobs
        time.sleep(0.2)
    memory = memory_order_parameter(P_S)
    return {"memory": float(memory), "P_S": {k: {str(kk): vv for kk, vv in v.items()} for k, v in P_S.items()}}


def _apply_gauge(h, J, init_states, gauge_bits):
    """Apply a spin-reversal gauge: flip spins in gauge_bits (set of qubit ids).

    Transformation: sigma_i' = g_i * sigma_i where g_i = -1 if i in gauge_bits.
    Under this transformation:
      h_i  -> g_i * h_i
      J_ij -> g_i * g_j * J_ij
      initial_state[i] -> g_i * initial_state[i]
    Memory order parameter is invariant because spin labels are flipped
    equally in every initial state.
    """
    g = {i: (-1 if i in gauge_bits else 1) for i in h}
    h_g = {i: g[i] * v for i, v in h.items()}
    J_g = {(u, v): g[u] * g[v] * w for (u, v), w in J.items()}
    init_g = {name: {i: g[i] * s for i, s in st.items()} for name, st in init_states.items()}
    return h_g, J_g, init_g


def _run_gauge_arm(sampler, solver, h, J, S, E, auto_scale, gauge_bits, gauge_seed):
    """Run one gauge-transformed submission and return M."""
    states = _initial_states(S, E)
    h_g, J_g, init_g = _apply_gauge(h, J, states, gauge_bits)
    P_S = {}
    for name, init in init_g.items():
        result = run_reverse_anneal(
            h=h_g, J=J_g, initial_state=init,
            s_p=S_P, t_hold=T_HOLD,
            num_reads=NUM_READS,
            solver=solver,
            sampler=sampler,
            auto_embed=False,
            auto_scale=auto_scale,
            label=f"spot_gauge{gauge_seed}_{name}",
        )
        # Unflip samples back to the original gauge before computing marginal
        variables = result["variables"]
        samples = result["samples"].copy()
        for col, q in enumerate(variables):
            if q in gauge_bits:
                samples[:, col] = -samples[:, col]
        P_S[name] = compute_marginal(samples, S, variables=variables)
        time.sleep(0.2)
    return float(memory_order_parameter(P_S))


def run_spot_check(solver: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    # Query live solver for actual active qubits/edges and reuse one shared sampler
    from dwave.system import DWaveSampler
    sampler = DWaveSampler(config_file="./dwave.conf", solver=solver)
    nodelist = list(sampler.nodelist)
    edgelist = list(sampler.edgelist)
    print(f"Solver {sampler.solver.name}: {len(nodelist)} qubits, {len(edgelist)} couplers")
    rows = []
    try:
        for label, n_env, lam, W in SPOT_CONDITIONS:
            n_total = SUBSYSTEM_SIZE + n_env
            S_qubits, E_qubits, edges = _pick_live_subgraph(nodelist, edgelist, n_total, SUBSYSTEM_SIZE)
            h, J = _build_hamiltonian(S_qubits, E_qubits, edges, lam, W, DISORDER_SEED)
            max_h = max((abs(v) for v in h.values()), default=0.0)
            max_J = max((abs(v) for v in J.values()), default=0.0)
            n_boundary = sum(1 for (u, v) in edges if (u in set(S_qubits)) != (v in set(S_qubits)))
            print(f"[{label}] n_env={n_env}, lam={lam}, W={W}; "
                  f"edges={len(edges)}, boundary={n_boundary}, max|h|={max_h:.3f}, max|J|={max_J:.3f}")

            # Paired auto_scale True/False submissions (no gauge averaging).
            res_true = _run_one(h, J, S_qubits, E_qubits, solver, auto_scale=True,
                                sampler=sampler)
            print(f"  auto_scale=True : M = {res_true['memory']:.4f}")
            res_false = _run_one(h, J, S_qubits, E_qubits, solver, auto_scale=False,
                                 sampler=sampler)
            print(f"  auto_scale=False: M = {res_false['memory']:.4f}")

            rows.append({
                "label": label,
                "n_env": n_env,
                "coupling_lambda": lam,
                "disorder_W": W,
                "max_abs_h": max_h,
                "max_abs_J": max_J,
                "memory_auto_scale_true": res_true["memory"],
                "memory_auto_scale_false": res_false["memory"],
                "delta_memory": res_true["memory"] - res_false["memory"],
            })
    finally:
        try:
            sampler.client.close()
        except Exception:
            pass

    summary = {
        "solver": solver,
        "num_reads": NUM_READS,
        "s_p": S_P,
        "t_hold": T_HOLD,
        "rows": rows,
    }
    out_file = output / "spot_check_autoscale.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_file}")

    # Summary table
    print("\nSummary (auto_scale True/False paired):")
    print(f"{'condition':<24} {'M_T':>8} {'M_F':>8} {'Δ M':>8}")
    for r in rows:
        print(f"{r['label']:<24} "
              f"{r.get('memory_auto_scale_true', float('nan')):>8.4f} "
              f"{r.get('memory_auto_scale_false', float('nan')):>8.4f} "
              f"{r.get('delta_memory', float('nan')):>+8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="Advantage2_system1",
                        help="QPU solver name (current Zephyr replacement for Advantage2_system1.13)")
    parser.add_argument("--output", default="data/raw/phase4/spot_check_autoscale")
    args = parser.parse_args()
    run_spot_check(args.solver, REPO / args.output)
