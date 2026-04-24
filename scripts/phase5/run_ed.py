"""Phase 5: run exact diagonalization on small systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from locth1.hamiltonians import build_logical_graph
from locth1.classical.exact_diag import run_ed_thermalization, product_state
from locth1.observables import total_variation_distance, memory_order_parameter


def run_ed(config_path: str, output_dir: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ed_config = config["exact_diag"]
    out = Path(output_dir) / "phase5" / "exact_diag"
    out.mkdir(parents=True, exist_ok=True)

    # D-Wave Advantage2 energy scales (approximate, in GHz)
    # These should be replaced with actual per-QPU A(s)/B(s) from schedule spreadsheets
    A_s_values = {0.3: 3.5, 0.4: 2.0, 0.5: 0.8, 0.6: 0.2}
    B_s_values = {0.3: 0.5, 0.4: 1.5, 0.5: 2.5, 0.6: 3.5}

    # Unit conversion: H is in GHz, t_hold is in μs.
    # Schrödinger: |ψ(t)⟩ = exp(-i 2π H t)|ψ(0)⟩ with H in GHz, t in ns.
    # Factor: 2π × 1e3 converts (GHz, μs) → dimensionless phase.
    GHZ_US_TO_NATURAL = 2.0 * np.pi * 1e3

    results = []
    for n_qubits in ed_config["system_sizes"]:
        for sub_size in ed_config["subsystem_sizes"]:
            if sub_size >= n_qubits:
                continue
            for s_p in ed_config["s_p_values"]:
                for t_hold in ed_config["t_hold_values"]:
                    problem = build_logical_graph(
                        graph_type="random_regular",
                        n_qubits=n_qubits,
                        subsystem_size=sub_size,
                        coupling_lambda=0.5,
                        disorder_W=0.0,
                        seed=42,
                    )
                    h, J = problem["h"], problem["J"]
                    S_indices = problem["S_qubits"]

                    # Build initial states
                    initial_states = {
                        "all_up": product_state([1] * n_qubits, n_qubits),
                        "all_down": product_state([-1] * n_qubits, n_qubits),
                    }

                    A_s = A_s_values.get(s_p, 2.0)
                    B_s = B_s_values.get(s_p, 1.5)

                    t_natural = t_hold * GHZ_US_TO_NATURAL

                    P_S = run_ed_thermalization(
                        h=h, J=J, S_indices=S_indices,
                        initial_states=initial_states,
                        s_p=s_p, t_hold=t_natural,
                        A_s=A_s, B_s=B_s,
                    )

                    memory = memory_order_parameter(P_S)
                    tvd = total_variation_distance(P_S["all_up"], P_S["all_down"])

                    entry = {
                        "n_qubits": n_qubits,
                        "subsystem_size": sub_size,
                        "s_p": s_p,
                        "t_hold": t_hold,
                        "memory_order_param": memory,
                        "tvd_up_vs_down": tvd,
                    }
                    results.append(entry)
                    print(f"  N={n_qubits}, |S|={sub_size}, s_p={s_p}, t_p={t_hold}: memory={memory:.4f}")

    with open(out / "ed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"ED complete. Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: exact diagonalization")
    parser.add_argument("--config", default="configs/phase5_baselines.yaml")
    parser.add_argument("--output", default="data/classical")
    args = parser.parse_args()
    run_ed(args.config, args.output)
