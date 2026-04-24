"""Reproducibility check: repeat the λ=0.2 embedded spot-check several times
back-to-back with each auto_scale setting, to distinguish a real auto_scale
effect from shot-noise / calibration-epoch fluctuation at the transition."""

from __future__ import annotations

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

SUBSYSTEM_SIZE = 6
S_P = 0.4
T_HOLD = 100.0
NUM_READS = 2000
DISORDER_SEED = 42
CHAIN_STRENGTH = 1.0
N_REPS = 4  # 4 runs of each arm


def _build_logical(n_env, lam, W, seed=DISORDER_SEED):
    n_total = SUBSYSTEM_SIZE + n_env
    G = nx.random_regular_graph(3, n_total, seed=seed)
    visited = [0]; seen = {0}; queue = [0]
    while queue and len(visited) < SUBSYSTEM_SIZE:
        node = queue.pop(0)
        for nbr in sorted(G.neighbors(node)):
            if nbr not in seen:
                seen.add(nbr); visited.append(nbr); queue.append(nbr)
                if len(visited) >= SUBSYSTEM_SIZE:
                    break
    S = visited[:SUBSYSTEM_SIZE]; S_set = set(S)
    E = [n for n in G.nodes() if n not in S_set]
    rng = np.random.default_rng(seed)
    h = {n: float(rng.uniform(-W, W)) for n in G.nodes()}
    J = {}
    for u, v in G.edges():
        boundary = (u in S_set) != (v in S_set)
        J[(u, v)] = -1.0 * (lam if boundary else 1.0)
    return h, J, S, E


def _states(S, E, seed=DISORDER_SEED):
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E}
    return {
        "S_up":    {**{q: 1 for q in S},  **E_state},
        "S_down":  {**{q: -1 for q in S}, **E_state},
        "S_random": {**{q: int(rng.choice([-1, 1])) for q in S}, **E_state},
    }


def _schedule():
    return [[0.0, 1.0], [5.0, S_P], [5.0 + T_HOLD, S_P], [10.0 + T_HOLD, 1.0]]


def _run_arm(target, h, J, S, E, auto_scale):
    P_S = {}
    for name, init in _states(S, E).items():
        ss = target.sample_ising(
            h, J,
            num_reads=NUM_READS,
            anneal_schedule=_schedule(),
            initial_state=dict(init),
            reinitialize_state=True,
            auto_scale=auto_scale,
            chain_strength=CHAIN_STRENGTH,
            answer_mode="raw",
            label=f"repeat_lam02_{'T' if auto_scale else 'F'}_{name}",
        )
        if hasattr(ss, "resolve"):
            ss.resolve()
        variables = list(ss.variables)
        samples = np.asarray(ss.record.sample, dtype=np.int8)
        P_S[name] = compute_marginal(samples, S, variables=variables)
        time.sleep(0.2)
    return float(memory_order_parameter(P_S))


def main():
    from dwave.system import DWaveSampler, FixedEmbeddingComposite
    from minorminer import find_embedding

    sampler = DWaveSampler(config_file="./dwave.conf", solver="Advantage2_system1")
    print(f"Solver {sampler.solver.name}")
    try:
        h, J, S, E = _build_logical(n_env=20, lam=0.2, W=0.0)
        embedding = find_embedding(list(J.keys()), sampler.edgelist, random_seed=DISORDER_SEED)
        target = FixedEmbeddingComposite(sampler, embedding=embedding)
        max_chain = max(len(c) for c in embedding.values())
        print(f"Embedding: {len(embedding)} logical, max_chain={max_chain}")

        # Interleave: T, F, T, F, ... so any time-trend shows as correlated pairs
        seq = []
        for _ in range(N_REPS):
            m_T = _run_arm(target, h, J, S, E, auto_scale=True)
            print(f"  auto_scale=True : M={m_T:.4f}")
            seq.append(("T", m_T))
            m_F = _run_arm(target, h, J, S, E, auto_scale=False)
            print(f"  auto_scale=False: M={m_F:.4f}")
            seq.append(("F", m_F))

        M_T = [m for tag, m in seq if tag == "T"]
        M_F = [m for tag, m in seq if tag == "F"]
        print(f"\nmean M_T = {np.mean(M_T):.4f}  std = {np.std(M_T):.4f}")
        print(f"mean M_F = {np.mean(M_F):.4f}  std = {np.std(M_F):.4f}")
        print(f"|mean Δ| = {abs(np.mean(M_T) - np.mean(M_F)):.4f}")
        out = REPO / "data/raw/phase4/spot_check_embedded/repeat_lam02.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "solver": sampler.solver.name,
            "n_reps": N_REPS,
            "lambda": 0.2,
            "n_env": 20,
            "max_chain": max_chain,
            "sequence": seq,
            "M_T_list": M_T, "M_F_list": M_F,
        }, indent=2))
        print(f"\nWrote {out}")
    finally:
        try:
            sampler.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
