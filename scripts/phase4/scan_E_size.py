"""Phase 4 scan: environment size |E| sweep — the key Nature experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from locth1.hamiltonians import build_native_graph
from locth1.reverse_anneal import run_reverse_anneal
from locth1.observables import (
    total_variation_distance, gibbs_fit, memory_order_parameter,
)
from locth1.analysis import compute_marginal
from locth1.metadata import save_raw_results


def _build_subsystem_initial_states(
    S_qubits: list[int], E_qubits: list[int], seed: int = 42,
) -> dict[str, dict[int, int]]:
    """Build initial states that vary S but keep E fixed (all-up)."""
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E_qubits}  # E always starts all-up

    states = {}
    S_all_up = {q: 1 for q in S_qubits}
    S_all_down = {q: -1 for q in S_qubits}
    S_random = {q: int(rng.choice([-1, 1])) for q in S_qubits}

    for label, S_state in [("S_up", S_all_up), ("S_down", S_all_down), ("S_random", S_random)]:
        states[label] = {**S_state, **E_state}

    return states


def run_scan(config_path: str, output_dir: str, qpu: str = "primary"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    scan = config["scans"]["scan_E_size"]
    defaults = config["defaults"]
    qpu_config = config["qpus"][qpu]
    out = Path(output_dir) / "phase4" / "scan_E_size" / qpu
    out.mkdir(parents=True, exist_ok=True)

    subsystem_size = defaults["subsystem_size"]
    coupling_lambda = scan["fixed"]["coupling_lambda"]
    W = scan["fixed"]["disorder_W"]
    s_p = defaults["s_p"]
    t_hold = defaults["t_hold"]
    E_values = scan["values"]

    results = []
    for n_env in E_values:
        n_total = subsystem_size + n_env

        problem = build_native_graph(
            topology=qpu_config["topology"],
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
                solver=qpu_config["solver"],
                label=f"scan_E_{name}_E{n_env}",
            )
            save_raw_results(
                samples=result["samples"], energies=result["energies"],
                metadata=result["metadata"],
                path=out / f"E{n_env}_{name}.h5",
            )
            # Compute P_S (marginal over subsystem only)
            P_S[name] = compute_marginal(
                result["samples"], S_qubits, variables=result["variables"],
            )

        memory = memory_order_parameter(P_S)

        # Pairwise TVD on P_S
        names = list(P_S.keys())
        tvd_pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                tvd_pairs[f"{names[i]}_vs_{names[j]}"] = total_variation_distance(
                    P_S[names[i]], P_S[names[j]]
                )

        # Gibbs fit: enumerate ALL 2^|S| configurations (not just sampled support)
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
        for name, P in P_S.items():
            P_sorted = {c: P.get(c, 0.0) for c in configs}
            fit = gibbs_fit(P_sorted, H_S)
            beta_effs[name] = fit["beta_eff"]

        results.append({
            "n_environment": n_env,
            "n_total": n_total,
            "memory_order_param": memory,
            "tvd_pairs": tvd_pairs,
            "beta_effs": beta_effs,
            "gibbs_chi_squared": {
                name: gibbs_fit({c: P_S[name].get(c, 0.0) for c in configs}, H_S)["chi_squared"]
                for name in P_S
            },
        })
        print(f"  |E|={n_env}: memory={memory:.4f}, beta_eff={beta_effs}")

    summary = {
        "scan": "E_size", "qpu": qpu, "solver": qpu_config["solver"],
        "subsystem_size": subsystem_size,
        "coupling_lambda": coupling_lambda,
        "disorder_W": W, "s_p": s_p, "t_hold": t_hold,
        "results": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Scan complete. Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: |E| scan")
    parser.add_argument("--config", default="configs/phase4_subsystem.yaml")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--qpu", default="primary", choices=["primary", "secondary"])
    args = parser.parse_args()
    run_scan(args.config, args.output, args.qpu)
