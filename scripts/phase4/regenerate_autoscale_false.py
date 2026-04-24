"""Regenerate Fig 2a/b/c and Fig 3a/b with auto_scale=False on native qubits.

Motivation: Kimi's addendum and Codex's independent pass both confirm the
legacy Phase 3/4 data (auto_scale=True through EmbeddingComposite) contain a
rescaling systematic that contaminates the lambda crossover in Fig. 2b and
the (lambda, W) grid in Fig. 3b.  Spot-checks already in the repo show the
effect vanishes for native-qubit direct submissions (no chains).  This
script re-runs the key panels on the live solvers with auto_scale=False and
no embedding composite, archiving the samplesets so the paper can report
the correct logical-Hamiltonian dynamics.

Scope:
  Fig 2b lambda sweep        - |E|=50, lam in 8 values, W=0, both QPUs
  Fig 2c sp sweep (Adv2)     - |E|=50, lam in 5 values, sp in 4 values, W=0
  Fig 2a |E| sweep spot     - lam=0.5, W=0, |E| in 4 values, Adv2
  Fig 3a W sweep spot       - lam=0.5, |E|=50, W in 5 values, Adv2
  Fig 3b (lam,W) reduced    - |E|=50, 5x5 grid, Adv2
  Phase 7 working point     - p_S=0.5, W=1, N=12, 20 seeds, both QPUs

All submissions are native-qubit (auto_embed=False) with auto_scale=False.
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


SUBSYSTEM_SIZE = 6
NUM_READS = 2000
DISORDER_SEED = 42
T_HOLD = 100.0
S_P_DEFAULT = 0.4


def _pick_subgraph(nodelist, edgelist, n_qubits, subsystem_size):
    G = nx.Graph()
    G.add_nodes_from(nodelist)
    G.add_edges_from(edgelist)
    start = max(G.nodes, key=lambda n: G.degree(n))
    visited = [start]; seen = {start}; queue = [start]
    while queue and len(visited) < n_qubits:
        node = queue.pop(0)
        for nbr in sorted(G.neighbors(node)):
            if nbr not in seen:
                seen.add(nbr); visited.append(nbr); queue.append(nbr)
                if len(visited) >= n_qubits:
                    break
    if len(visited) < n_qubits:
        raise RuntimeError(f"Could not assemble {n_qubits} connected live qubits.")
    sub_nodes = visited[:n_qubits]
    subG = G.subgraph(sub_nodes).copy()
    start_sub = max(subG.nodes, key=lambda n: subG.degree(n))
    sv = [start_sub]; ss = {start_sub}; sq = [start_sub]
    while sq and len(sv) < subsystem_size:
        nd = sq.pop(0)
        for nbr in sorted(subG.neighbors(nd)):
            if nbr not in ss:
                ss.add(nbr); sv.append(nbr); sq.append(nbr)
                if len(sv) >= subsystem_size:
                    break
    S = sv[:subsystem_size]
    E = [n for n in sub_nodes if n not in set(S)]
    return S, E, list(subG.edges())


def _build_h_J(S, E, edges, coupling_lambda, disorder_W, seed):
    rng = np.random.default_rng(seed)
    S_set = set(S)
    h = {n: float(rng.uniform(-disorder_W, disorder_W)) for n in (S + E)}
    J = {}
    for u, v in edges:
        on_boundary = (u in S_set) != (v in S_set)
        J[(u, v)] = -1.0 * (coupling_lambda if on_boundary else 1.0)
    return h, J


def _build_frustrated_h_J(S, E, edges, coupling_lambda, disorder_W, seed, sign_seed, p_flip=0.5):
    """Mixed frustration: subsystem couplers sign-flipped with prob p_flip, E ferromagnetic."""
    rng = np.random.default_rng(seed)
    sign_rng = np.random.default_rng(sign_seed)
    S_set = set(S)
    h = {n: float(rng.uniform(-disorder_W, disorder_W)) for n in (S + E)}
    J = {}
    for u, v in edges:
        if (u in S_set) != (v in S_set):
            Jij = -1.0 * coupling_lambda
        elif u in S_set and v in S_set:
            sign = 1.0 if sign_rng.random() < p_flip else -1.0
            Jij = sign * 1.0  # EA bond disorder in S
        else:
            Jij = -1.0
        J[(u, v)] = Jij
    return h, J


def _initial_states(S, E, seed=DISORDER_SEED):
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E}
    return {
        "S_up":     {**{q: 1 for q in S},  **E_state},
        "S_down":   {**{q: -1 for q in S}, **E_state},
        "S_random": {**{q: int(rng.choice([-1, 1])) for q in S}, **E_state},
    }


def _run_condition(sampler, solver, h, J, S, E, s_p, label_prefix):
    P_S = {}
    for name, init in _initial_states(S, E).items():
        result = run_reverse_anneal(
            h=h, J=J, initial_state=init,
            s_p=s_p, t_hold=T_HOLD,
            num_reads=NUM_READS,
            solver=solver, sampler=sampler,
            auto_embed=False, auto_scale=False,
            label=f"{label_prefix}_{name}",
        )
        P_S[name] = compute_marginal(result["samples"], S, variables=result["variables"])
        time.sleep(0.15)
    return float(memory_order_parameter(P_S)), P_S


def run_fig2b_lambda_sweep(sampler, solver, lambdas, n_env=50, out_dir=None):
    print(f"\n=== Fig 2b: lambda sweep, |E|={n_env}, W=0, {solver} ===")
    nodelist = list(sampler.nodelist); edgelist = list(sampler.edgelist)
    S, E, edges = _pick_subgraph(nodelist, edgelist, SUBSYSTEM_SIZE + n_env, SUBSYSTEM_SIZE)
    results = []
    for lam in lambdas:
        h, J = _build_h_J(S, E, edges, lam, 0.0, DISORDER_SEED)
        M, _ = _run_condition(sampler, solver, h, J, S, E, S_P_DEFAULT, f"fig2b_lam{lam}")
        print(f"  lam={lam:.3f}  M={M:.4f}")
        results.append({"lambda": lam, "memory": M})
    data = {
        "sweep": "lambda", "solver": solver, "n_env": n_env, "W": 0.0,
        "s_p": S_P_DEFAULT, "t_hold": T_HOLD, "num_reads": NUM_READS,
        "subsystem_size": SUBSYSTEM_SIZE, "auto_scale": False, "auto_embed": False,
        "results": results,
    }
    if out_dir:
        out = out_dir / f"fig2b_lambda_sweep_{solver}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"  wrote {out}")
    return data


def run_fig2c_sp_sweep(sampler, solver, sps, lambdas, n_env=50, out_dir=None):
    print(f"\n=== Fig 2c: (s_p, lambda) sweep, |E|={n_env}, W=0, {solver} ===")
    nodelist = list(sampler.nodelist); edgelist = list(sampler.edgelist)
    S, E, edges = _pick_subgraph(nodelist, edgelist, SUBSYSTEM_SIZE + n_env, SUBSYSTEM_SIZE)
    results = []
    for sp in sps:
        for lam in lambdas:
            h, J = _build_h_J(S, E, edges, lam, 0.0, DISORDER_SEED)
            M, _ = _run_condition(sampler, solver, h, J, S, E, sp, f"fig2c_sp{sp}_lam{lam}")
            print(f"  sp={sp:.3f} lam={lam:.3f}  M={M:.4f}")
            results.append({"s_p": sp, "lambda": lam, "memory": M})
    data = {
        "sweep": "sp_lambda", "solver": solver, "n_env": n_env, "W": 0.0,
        "t_hold": T_HOLD, "num_reads": NUM_READS, "auto_scale": False, "auto_embed": False,
        "results": results,
    }
    if out_dir:
        out = out_dir / f"fig2c_sp_sweep_{solver}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"  wrote {out}")
    return data


def run_fig2a_E_spotcheck(sampler, solver, E_values, coupling_lambda=0.5, out_dir=None):
    print(f"\n=== Fig 2a spot: |E| sweep, lambda={coupling_lambda}, W=0, {solver} ===")
    nodelist = list(sampler.nodelist); edgelist = list(sampler.edgelist)
    results = []
    for n_env in E_values:
        S, E, edges = _pick_subgraph(nodelist, edgelist, SUBSYSTEM_SIZE + n_env, SUBSYSTEM_SIZE)
        h, J = _build_h_J(S, E, edges, coupling_lambda, 0.0, DISORDER_SEED)
        M, _ = _run_condition(sampler, solver, h, J, S, E, S_P_DEFAULT, f"fig2a_E{n_env}")
        print(f"  |E|={n_env}  M={M:.4f}")
        results.append({"n_env": n_env, "memory": M})
    data = {
        "sweep": "n_env", "solver": solver, "coupling_lambda": coupling_lambda, "W": 0.0,
        "s_p": S_P_DEFAULT, "auto_scale": False, "auto_embed": False, "results": results,
    }
    if out_dir:
        out = out_dir / f"fig2a_E_spot_{solver}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"  wrote {out}")
    return data


def run_fig3a_W_spotcheck(sampler, solver, W_values, n_env=50, coupling_lambda=0.5, out_dir=None):
    print(f"\n=== Fig 3a spot: W sweep, lambda={coupling_lambda}, |E|={n_env}, {solver} ===")
    nodelist = list(sampler.nodelist); edgelist = list(sampler.edgelist)
    S, E, edges = _pick_subgraph(nodelist, edgelist, SUBSYSTEM_SIZE + n_env, SUBSYSTEM_SIZE)
    results = []
    for W in W_values:
        h, J = _build_h_J(S, E, edges, coupling_lambda, W, DISORDER_SEED)
        M, _ = _run_condition(sampler, solver, h, J, S, E, S_P_DEFAULT, f"fig3a_W{W}")
        print(f"  W={W:.3f}  M={M:.4f}")
        results.append({"W": W, "memory": M})
    data = {
        "sweep": "W", "solver": solver, "coupling_lambda": coupling_lambda, "n_env": n_env,
        "s_p": S_P_DEFAULT, "auto_scale": False, "auto_embed": False, "results": results,
    }
    if out_dir:
        out = out_dir / f"fig3a_W_spot_{solver}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"  wrote {out}")
    return data


def run_fig3b_grid(sampler, solver, lambdas, Ws, n_env=50, out_dir=None):
    print(f"\n=== Fig 3b reduced grid: {len(lambdas)}x{len(Ws)} ({solver}, |E|={n_env}) ===")
    nodelist = list(sampler.nodelist); edgelist = list(sampler.edgelist)
    S, E, edges = _pick_subgraph(nodelist, edgelist, SUBSYSTEM_SIZE + n_env, SUBSYSTEM_SIZE)
    results = []
    for W in Ws:
        for lam in lambdas:
            h, J = _build_h_J(S, E, edges, lam, W, DISORDER_SEED)
            M, _ = _run_condition(sampler, solver, h, J, S, E, S_P_DEFAULT, f"fig3b_lam{lam}_W{W}")
            print(f"  lam={lam:.3f} W={W:.3f}  M={M:.4f}")
            results.append({"lambda": lam, "W": W, "memory": M})
    data = {
        "sweep": "lambda_W", "solver": solver, "n_env": n_env, "s_p": S_P_DEFAULT,
        "auto_scale": False, "auto_embed": False, "results": results,
    }
    if out_dir:
        out = out_dir / f"fig3b_grid_{solver}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"  wrote {out}")
    return data


def run_phase7_working_point(sampler, solver, n_seeds=10, n_env=8, out_dir=None):
    print(f"\n=== Phase 7 mixed-frustration at p_S=0.5, W=1, |E|={n_env}, {solver} ===")
    nodelist = list(sampler.nodelist); edgelist = list(sampler.edgelist)
    S, E, edges = _pick_subgraph(nodelist, edgelist, SUBSYSTEM_SIZE - 2 + n_env, SUBSYSTEM_SIZE - 2)  # |S|=4
    # Override to |S|=4 for small-system comparison
    n_total = 4 + n_env
    Gfull = nx.Graph()
    Gfull.add_nodes_from(nodelist); Gfull.add_edges_from(edgelist)
    start = max(Gfull.nodes, key=lambda n: Gfull.degree(n))
    visited = [start]; seen = {start}; queue = [start]
    while queue and len(visited) < n_total:
        node = queue.pop(0)
        for nbr in sorted(Gfull.neighbors(node)):
            if nbr not in seen:
                seen.add(nbr); visited.append(nbr); queue.append(nbr)
                if len(visited) >= n_total:
                    break
    sub_nodes = visited[:n_total]
    subG = Gfull.subgraph(sub_nodes).copy()
    # |S|=4 BFS ball
    start_sub = max(subG.nodes, key=lambda n: subG.degree(n))
    sv = [start_sub]; ss = {start_sub}; sq = [start_sub]
    while sq and len(sv) < 4:
        nd = sq.pop(0)
        for nbr in sorted(subG.neighbors(nd)):
            if nbr not in ss:
                ss.add(nbr); sv.append(nbr); sq.append(nbr)
                if len(sv) >= 4:
                    break
    S = sv[:4]
    E = [n for n in sub_nodes if n not in set(S)]
    edges = list(subG.edges())

    results = []
    n_relaxed = 0
    for sd in range(n_seeds):
        h, J = _build_frustrated_h_J(S, E, edges, coupling_lambda=0.5, disorder_W=1.0,
                                     seed=DISORDER_SEED, sign_seed=100 + sd, p_flip=0.5)
        M, P_S = _run_condition(sampler, solver, h, J, S, E, S_P_DEFAULT, f"p7_seed{sd}")
        relaxed = M <= 0.05
        if relaxed:
            n_relaxed += 1
        print(f"  seed={sd:2d}  M={M:.4f}  relaxed={relaxed}")
        results.append({
            "seed": sd, "memory": M, "relaxed": relaxed,
            "P_S_pooled": {str(k): float(v) for name, ps in P_S.items() for k, v in ps.items()},
        })
    relaxation_rate = n_relaxed / n_seeds
    print(f"  relaxation rate: {n_relaxed}/{n_seeds} = {relaxation_rate:.3f}")
    data = {
        "experiment": "phase7_working_point", "solver": solver,
        "p_S": 0.5, "W": 1.0, "n_env": n_env, "subsystem_size": 4,
        "s_p": S_P_DEFAULT, "auto_scale": False, "auto_embed": False,
        "n_seeds": n_seeds, "n_relaxed": n_relaxed, "relaxation_rate": relaxation_rate,
        "results": results,
    }
    if out_dir:
        out = out_dir / f"phase7_working_point_{solver}.json"
        out.write_text(json.dumps(data, indent=2))
        print(f"  wrote {out}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-a2", default="Advantage2_system1")
    parser.add_argument("--solver-s64", default="Advantage_system6.4")
    parser.add_argument("--output", default="data/raw/phase4/regen_autoscale_false")
    parser.add_argument("--only", default="all",
                        help="Comma-separated: fig2b,fig2a,fig3a,fig3b,fig2c,phase7")
    args = parser.parse_args()
    out_dir = REPO / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only != "all" else set(
        ["fig2b", "fig2a", "fig3a", "fig3b", "fig2c", "phase7"])

    from dwave.system import DWaveSampler
    sampler_a2 = DWaveSampler(config_file="./dwave.conf", solver=args.solver_a2)
    sampler_s64 = None
    print(f"Adv2 solver: {sampler_a2.solver.name}  ({len(sampler_a2.nodelist)} qubits)")

    try:
        if "fig2b" in only:
            run_fig2b_lambda_sweep(
                sampler_a2, args.solver_a2,
                lambdas=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
                out_dir=out_dir,
            )
            if sampler_s64 is None:
                sampler_s64 = DWaveSampler(config_file="./dwave.conf", solver=args.solver_s64)
                print(f"S6.4 solver: {sampler_s64.solver.name}  ({len(sampler_s64.nodelist)} qubits)")
            run_fig2b_lambda_sweep(
                sampler_s64, args.solver_s64,
                lambdas=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
                out_dir=out_dir,
            )

        if "fig2a" in only:
            run_fig2a_E_spotcheck(
                sampler_a2, args.solver_a2,
                E_values=[4, 8, 16, 32, 50], out_dir=out_dir,
            )

        if "fig3a" in only:
            run_fig3a_W_spotcheck(
                sampler_a2, args.solver_a2,
                W_values=[0.0, 0.5, 1.0, 1.5, 2.0], out_dir=out_dir,
            )

        if "fig3b" in only:
            run_fig3b_grid(
                sampler_a2, args.solver_a2,
                lambdas=[0.1, 0.2, 0.3, 0.5, 0.7],
                Ws=[0.0, 0.5, 1.0, 1.5, 2.0],
                out_dir=out_dir,
            )

        if "fig2c" in only:
            run_fig2c_sp_sweep(
                sampler_a2, args.solver_a2,
                sps=[0.35, 0.40, 0.45, 0.50],
                lambdas=[0.1, 0.2, 0.3, 0.5, 0.7],
                out_dir=out_dir,
            )
            if sampler_s64 is None:
                sampler_s64 = DWaveSampler(config_file="./dwave.conf", solver=args.solver_s64)
                print(f"S6.4 solver: {sampler_s64.solver.name}")
            run_fig2c_sp_sweep(
                sampler_s64, args.solver_s64,
                sps=[0.35, 0.40, 0.45, 0.50],
                lambdas=[0.1, 0.2, 0.3, 0.5, 0.7],
                out_dir=out_dir,
            )

        if "phase7" in only:
            run_phase7_working_point(sampler_a2, args.solver_a2, n_seeds=15, out_dir=out_dir)
            if sampler_s64 is None:
                sampler_s64 = DWaveSampler(config_file="./dwave.conf", solver=args.solver_s64)
                print(f"S6.4 solver: {sampler_s64.solver.name}")
            run_phase7_working_point(sampler_s64, args.solver_s64, n_seeds=15, out_dir=out_dir)

    finally:
        try:
            sampler_a2.client.close()
        except Exception:
            pass
        if sampler_s64 is not None:
            try:
                sampler_s64.client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
