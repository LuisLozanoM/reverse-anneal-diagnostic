"""Phase 4 scan: disorder strength W sweep with subsystem partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from locth1.hamiltonians import build_native_graph
from locth1.reverse_anneal import run_reverse_anneal
from locth1.observables import memory_order_parameter, total_variation_distance
from locth1.analysis import compute_marginal
from locth1.metadata import save_raw_results


def _build_subsystem_initial_states(S_qubits, E_qubits, seed=42):
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E_qubits}
    return {
        "S_up": {**{q: 1 for q in S_qubits}, **E_state},
        "S_down": {**{q: -1 for q in S_qubits}, **E_state},
        "S_random": {**{q: int(rng.choice([-1, 1])) for q in S_qubits}, **E_state},
    }


def run_scan(config_path: str, output_dir: str, qpu: str = "primary"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    scan = config["scans"]["scan_W_subsystem"]
    defaults = config["defaults"]
    qpu_config = config["qpus"][qpu]
    out = Path(output_dir) / "phase4" / "scan_W_subsystem" / qpu
    out.mkdir(parents=True, exist_ok=True)

    subsystem_size = defaults["subsystem_size"]
    n_env = scan["fixed"]["n_environment"]
    coupling_lambda = scan["fixed"]["coupling_lambda"]
    s_p = defaults["s_p"]
    t_hold = defaults["t_hold"]
    n_total = subsystem_size + n_env
    W_values = scan["values"]

    results = []
    for W in W_values:
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
                label=f"scan_Wsub_{name}_W{W}",
            )
            save_raw_results(
                samples=result["samples"], energies=result["energies"],
                metadata=result["metadata"],
                path=out / f"W{W}_{name}.h5",
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
                    P_S[names[i]], P_S[names[j]]
                )

        results.append({"disorder_W": W, "memory_order_param": memory, "tvd_pairs": tvd_pairs})
        print(f"  W={W}: memory={memory:.4f}")

    summary = {
        "scan": "W_subsystem", "qpu": qpu, "solver": qpu_config["solver"],
        "subsystem_size": subsystem_size, "n_environment": n_env,
        "coupling_lambda": coupling_lambda, "s_p": s_p, "t_hold": t_hold,
        "results": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Scan complete. Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: W scan (subsystem)")
    parser.add_argument("--config", default="configs/phase4_subsystem.yaml")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--qpu", default="primary", choices=["primary", "secondary"])
    args = parser.parse_args()
    run_scan(args.config, args.output, args.qpu)
