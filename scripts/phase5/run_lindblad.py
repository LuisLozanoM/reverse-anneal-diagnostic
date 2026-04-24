"""Phase 5: run Lindblad master equation for open-system baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from locth1.hamiltonians import build_logical_graph
from locth1.classical.exact_diag import product_state
from locth1.classical.lindblad import run_lindblad_thermalization
from locth1.observables import total_variation_distance, memory_order_parameter


def run_lindblad(config_path: str, output_dir: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    lb_config = config["lindblad"]
    out = Path(output_dir) / "phase5" / "lindblad"
    out.mkdir(parents=True, exist_ok=True)

    A_s_values = {0.3: 3.5, 0.4: 2.0, 0.5: 0.8, 0.6: 0.2}
    B_s_values = {0.3: 0.5, 0.4: 1.5, 0.5: 2.5, 0.6: 3.5}

    results = []
    for n_qubits in lb_config["system_sizes"]:
        for bath_model in lb_config["bath_models"]:
            for T_label, T_device in [
                ("advantage2", lb_config["T_device_advantage2"]),
                ("system64", lb_config["T_device_system64"]),
            ]:
                problem = build_logical_graph(
                    graph_type="random_regular",
                    n_qubits=n_qubits,
                    subsystem_size=min(4, n_qubits // 2),
                    coupling_lambda=0.5,
                    disorder_W=0.0,
                    seed=42,
                )
                h, J = problem["h"], problem["J"]
                S_indices = problem["S_qubits"]

                initial_states = {
                    "all_up": product_state([1] * n_qubits, n_qubits),
                    "all_down": product_state([-1] * n_qubits, n_qubits),
                }

                s_p = 0.4
                t_hold = 100.0
                A_s = A_s_values[s_p]
                B_s = B_s_values[s_p]

                # Convert t_hold from μs to natural units: 2π × f_GHz × t_μs × 1e3
                t_natural = 2.0 * np.pi * t_hold * 1e3

                P_S = run_lindblad_thermalization(
                    h=h, J=J, S_indices=S_indices,
                    initial_states=initial_states,
                    s_p=s_p, t_hold=t_natural,
                    A_s=A_s, B_s=B_s,
                    T_device=T_device,
                    gamma_base=bath_model["gamma_base"],
                )

                memory = memory_order_parameter(P_S)
                tvd = total_variation_distance(P_S["all_up"], P_S["all_down"])

                results.append({
                    "n_qubits": n_qubits,
                    "bath_model": bath_model["name"],
                    "gamma_base": bath_model["gamma_base"],
                    "T_device": T_device,
                    "T_label": T_label,
                    "memory_order_param": memory,
                    "tvd_up_vs_down": tvd,
                })
                print(f"  N={n_qubits}, bath={bath_model['name']}, T={T_label}: memory={memory:.4f}")

    with open(out / "lindblad_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Lindblad complete. Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Lindblad simulation")
    parser.add_argument("--config", default="configs/phase5_baselines.yaml")
    parser.add_argument("--output", default="data/classical")
    args = parser.parse_args()
    run_lindblad(args.config, args.output)
