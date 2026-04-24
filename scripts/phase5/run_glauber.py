"""Phase 5: run Glauber dynamics — the critical classical baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from locth1.hamiltonians import build_native_graph, build_logical_graph
from locth1.classical.glauber import glauber_thermalization
from locth1.observables import total_variation_distance, memory_order_parameter


def _remap_contiguous(h, J, S_indices):
    """Remap arbitrary qubit labels to 0..N-1 for array-indexed backends."""
    labels = sorted(h.keys())
    label_map = {old: new for new, old in enumerate(labels)}
    h2 = {label_map[k]: v for k, v in h.items()}
    J2 = {(label_map[a], label_map[b]): v for (a, b), v in J.items()}
    S2 = [label_map[s] for s in S_indices]
    return h2, J2, S2, len(labels)


def run_glauber(config_path: str, output_dir: str, graph_type: str = "native"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gl_config = config["glauber"]
    out = Path(output_dir) / "phase5" / "glauber" / graph_type
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for n_qubits in gl_config["system_sizes"]:
        for T_label, T_device in [
            ("advantage2", gl_config["T_device_advantage2"]),
            ("system64", gl_config["T_device_system64"]),
        ]:
            build_fn = build_native_graph if graph_type == "native" else build_logical_graph
            build_kwargs = dict(
                n_qubits=n_qubits,
                subsystem_size=min(6, n_qubits // 2),
                coupling_lambda=0.5,
                disorder_W=0.0,
                seed=42,
            )
            if graph_type == "native":
                build_kwargs["topology"] = "zephyr" if T_label == "advantage2" else "pegasus"
            else:
                build_kwargs["graph_type"] = "random_regular"

            problem = build_fn(**build_kwargs)
            h, J, S_indices, n_actual = _remap_contiguous(
                problem["h"], problem["J"], problem["S_qubits"],
            )

            initial_states = {
                "all_up": np.ones(n_actual, dtype=int),
                "all_down": -np.ones(n_actual, dtype=int),
                "random": np.random.default_rng(42).choice([-1, 1], size=n_actual),
            }

            for seed in gl_config["seeds"]:
                P_S = glauber_thermalization(
                    h=h, J=J, S_indices=S_indices,
                    initial_states=initial_states,
                    T=T_device,
                    n_sweeps=gl_config["n_sweeps"],
                    n_samples=gl_config["n_samples"],
                    seed=seed,
                )

                memory = memory_order_parameter(P_S)
                tvd = total_variation_distance(P_S["all_up"], P_S["all_down"])

                results.append({
                    "n_qubits": n_qubits,
                    "T_device": T_device,
                    "T_label": T_label,
                    "seed": seed,
                    "memory_order_param": memory,
                    "tvd_up_vs_down": tvd,
                })
                print(f"  N={n_qubits}, T={T_label}, seed={seed}: memory={memory:.4f}")

    with open(out / "glauber_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Glauber complete. Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Glauber dynamics")
    parser.add_argument("--config", default="configs/phase5_baselines.yaml")
    parser.add_argument("--output", default="data/classical")
    parser.add_argument("--graph", default="native", choices=["native", "logical"])
    args = parser.parse_args()
    run_glauber(args.config, args.output, args.graph)
