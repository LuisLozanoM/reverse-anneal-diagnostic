"""Phase 4 |E| scan extension: fill |E|=75 and |E|=100 on Advantage2.

Mirrors scripts/phase4/scan_E_size.py for the two missing values.
Output: data/raw/phase4/scan_E_size/extension/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from locth1.analysis import compute_marginal
from locth1.hamiltonians import build_native_graph
from locth1.metadata import save_raw_results
from locth1.observables import (
    gibbs_fit,
    memory_order_parameter,
    total_variation_distance,
)
from locth1.reverse_anneal import run_reverse_anneal


def _build_subsystem_initial_states(S_qubits, E_qubits, seed=42):
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E_qubits}
    return {
        "S_up": {**{q: 1 for q in S_qubits}, **E_state},
        "S_down": {**{q: -1 for q in S_qubits}, **E_state},
        "S_random": {**{q: int(rng.choice([-1, 1])) for q in S_qubits}, **E_state},
    }


def main():
    with open("configs/phase4_subsystem.yaml") as f:
        config = yaml.safe_load(f)
    defaults = config["defaults"]
    fixed = config["scans"]["scan_E_size"]["fixed"]

    out = Path("data/raw/phase4/scan_E_size/extension")
    out.mkdir(parents=True, exist_ok=True)

    subsystem_size = defaults["subsystem_size"]
    coupling_lambda = fixed["coupling_lambda"]
    W = fixed["disorder_W"]
    s_p = defaults["s_p"]
    t_hold = defaults["t_hold"]
    solver = "Advantage2_system1"
    topology = "zephyr"
    E_values = [75, 100]

    results = []
    for n_env in E_values:
        n_total = subsystem_size + n_env
        problem = build_native_graph(
            topology=topology,
            n_qubits=n_total,
            subsystem_size=subsystem_size,
            coupling_lambda=coupling_lambda,
            disorder_W=W,
            seed=defaults["disorder_seed"],
        )
        h, J = problem["h"], problem["J"]
        S_qubits = problem["S_qubits"]
        E_qubits = problem["E_qubits"]
        initial_states = _build_subsystem_initial_states(S_qubits, E_qubits, defaults["disorder_seed"])

        P_S = {}
        for name, init_state in initial_states.items():
            result = run_reverse_anneal(
                h=h, J=J, initial_state=init_state,
                s_p=s_p, t_hold=t_hold,
                num_reads=defaults["num_reads"],
                solver=solver,
                label=f"scan_E_ext_{name}_E{n_env}",
            )
            save_raw_results(
                samples=result["samples"], energies=result["energies"],
                metadata=result["metadata"],
                path=out / f"E{n_env}_{name}.h5",
            )
            P_S[name] = compute_marginal(
                result["samples"], S_qubits, variables=result["variables"],
            )

        memory = memory_order_parameter(P_S)
        names = list(P_S.keys())
        tvd_pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                tvd_pairs[f"{names[i]}_vs_{names[j]}"] = total_variation_distance(
                    P_S[names[i]], P_S[names[j]],
                )

        import itertools
        n_S = len(S_qubits)
        configs = sorted(itertools.product([-1, 1], repeat=n_S))
        h_sub = {i: h.get(q, 0.0) for i, q in enumerate(S_qubits)}
        J_sub = {}
        S_set = set(S_qubits)
        for (a, b), Jij in J.items():
            if a in S_set and b in S_set:
                ia = S_qubits.index(a)
                ib = S_qubits.index(b)
                J_sub[(ia, ib)] = Jij
        H_S = np.array([
            sum(h_sub.get(i, 0.0) * c[i] for i in range(len(S_qubits)))
            + sum(Jij * c[ia] * c[ib] for (ia, ib), Jij in J_sub.items())
            for c in configs
        ])

        beta_effs = {}
        chi_sq = {}
        for name, P in P_S.items():
            P_sorted = {c: P.get(c, 0.0) for c in configs}
            fit = gibbs_fit(P_sorted, H_S)
            beta_effs[name] = fit["beta_eff"]
            chi_sq[name] = fit["chi_squared"]

        results.append({
            "n_environment": n_env, "n_total": n_total,
            "memory_order_param": memory,
            "tvd_pairs": tvd_pairs,
            "beta_effs": beta_effs,
            "gibbs_chi_squared": chi_sq,
        })
        print(f"  |E|={n_env}: memory={memory:.4f}, beta_eff={beta_effs}", flush=True)

    summary = {
        "scan": "E_size_extension",
        "qpu": "primary", "solver": solver,
        "subsystem_size": subsystem_size,
        "coupling_lambda": coupling_lambda,
        "disorder_W": W, "s_p": s_p, "t_hold": t_hold,
        "results": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Extension complete: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
