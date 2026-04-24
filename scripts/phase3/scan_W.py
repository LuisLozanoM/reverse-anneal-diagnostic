"""Phase 3 scan: disorder strength W sweep at fixed s_p, t_p, N."""

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


def _build_initial_states(qubit_labels: list, seed: int = 42) -> dict[str, dict[int, int]]:
    rng = np.random.default_rng(seed)
    return {
        "all_up": {q: 1 for q in qubit_labels},
        "all_down": {q: -1 for q in qubit_labels},
        "random": {q: int(rng.choice([-1, 1])) for q in qubit_labels},
        "neel": {q: (-1) ** i for i, q in enumerate(qubit_labels)},
    }


def run_scan(config_path: str, output_dir: str, qpu: str = "primary"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    scan = config["scans"]["scan_W"]
    defaults = config["defaults"]
    qpu_config = config["qpus"][qpu]
    out = Path(output_dir) / "phase3" / "scan_W" / qpu
    out.mkdir(parents=True, exist_ok=True)

    s_p = scan["fixed"]["s_p"]
    t_hold = scan["fixed"]["t_hold"]
    n_qubits = scan["fixed"]["n_qubits"]
    W_values = scan["values"]

    results = []
    for W in W_values:
        # Rebuild Hamiltonian for each W (disorder changes)
        problem = build_native_graph(
            topology=qpu_config["topology"],
            n_qubits=n_qubits,
            subsystem_size=0,
            coupling_lambda=1.0,
            disorder_W=W,
            seed=defaults["disorder_seed"],
        )
        h, J = problem["h"], problem["J"]
        all_qubits = sorted(h.keys())
        initial_states = _build_initial_states(all_qubits, seed=defaults["disorder_seed"])

        P_full = {}
        for name, init_state in initial_states.items():
            result = run_reverse_anneal(
                h=h, J=J, initial_state=init_state,
                s_p=s_p, t_hold=t_hold,
                num_reads=defaults["num_reads"],
                solver=qpu_config["solver"],
                label=f"scan_W_{name}_W{W}",
            )
            save_raw_results(
                samples=result["samples"], energies=result["energies"],
                metadata=result["metadata"],
                path=out / f"W{W}_{name}.h5",
            )
            P_full[name] = compute_marginal(
                result["samples"], all_qubits, variables=result["variables"],
            )

        memory = memory_order_parameter(P_full)
        names = list(P_full.keys())
        tvd_pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                tvd_pairs[f"{names[i]}_vs_{names[j]}"] = total_variation_distance(
                    P_full[names[i]], P_full[names[j]]
                )
        results.append({"disorder_W": W, "memory_order_param": memory, "tvd_pairs": tvd_pairs})
        print(f"  W={W}: memory={memory:.4f}")

    summary = {
        "scan": "W", "qpu": qpu, "solver": qpu_config["solver"],
        "s_p": s_p, "t_hold": t_hold, "n_qubits": n_qubits,
        "results": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Scan complete. Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: W scan")
    parser.add_argument("--config", default="configs/phase3_discovery.yaml")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--qpu", default="primary", choices=["primary", "secondary"])
    args = parser.parse_args()
    run_scan(args.config, args.output, args.qpu)
