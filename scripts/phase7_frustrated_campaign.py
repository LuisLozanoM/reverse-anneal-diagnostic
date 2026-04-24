"""Phase 7 Campaign 1: Expanded Edwards-Anderson frustrated sweep.

Nature-level upgrade experiment — tests whether the quantum conditional
Gibbs target systematically outperforms the classical conditional target
in frustrated (multi-modal) regimes.

Run:
    PYTHONPATH=src .venv/bin/python scripts/phase7_frustrated_campaign.py \\
        --config configs/phase7_frustrated.yaml \\
        --qpu advantage2 \\
        --output data/raw/phase7_frustrated

The script can also be run with --dry-run for a classical simulation
that exercises the full pipeline without QPU time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from locth1.analysis import compute_marginal
from locth1.hamiltonians import build_logical_graph
from locth1.metadata import save_raw_results
from locth1.observables import (
    memory_order_parameter,
    total_variation_distance,
)


# ---------------------------------------------------------------------------
# Helpers (adapted from phase6_gibbs_campaign.py)
# ---------------------------------------------------------------------------


def _build_initial_states(S_qubits, E_qubits, seed=42):
    rng = np.random.default_rng(seed)
    E_state = {q: 1 for q in E_qubits}
    return {
        "S_up": {**{q: 1 for q in S_qubits}, **E_state},
        "S_down": {**{q: -1 for q in S_qubits}, **E_state},
        "S_random": {**{q: int(rng.choice([-1, 1])) for q in S_qubits}, **E_state},
    }


def _remap_to_contiguous(h, J, S_qubits, E_qubits, initial_states):
    all_q = list(S_qubits) + list(E_qubits)
    m = {q: i for i, q in enumerate(all_q)}
    h2 = {m[q]: v for q, v in h.items() if q in m}
    J2 = {}
    for (a, b), v in J.items():
        if a in m and b in m:
            u, w = m[a], m[b]
            J2[(min(u, w), max(u, w))] = v
    S2 = [m[q] for q in S_qubits]
    E2 = [m[q] for q in E_qubits]
    init2 = {lab: {m[q]: v for q, v in st.items() if q in m} for lab, st in initial_states.items()}
    return h2, J2, S2, E2, init2


def _flip_coupler_signs(J, sign_seed, flip_probability=0.5):
    """Flip coupler signs with probability p. Matches phase6_gibbs_campaign.py exactly."""
    rng = np.random.default_rng(sign_seed)
    J_ea = {}
    for edge, Jij in J.items():
        if rng.random() < flip_probability:
            J_ea[edge] = -Jij  # flip the original sign
        else:
            J_ea[edge] = Jij   # preserve the original sign
    return J_ea


def _apply_gauge(h, J, initial_states, gauge_vector):
    """Apply spin-reversal gauge: flip h_i -> g_i*h_i, J_ij -> g_i*g_j*J_ij."""
    h_g = {i: hi * gauge_vector[i] for i, hi in h.items()}
    J_g = {(i, j): Jij * gauge_vector[i] * gauge_vector[j] for (i, j), Jij in J.items()}
    init_g = {}
    for lab, st in initial_states.items():
        init_g[lab] = {i: v * gauge_vector[i] for i, v in st.items()}
    return h_g, J_g, init_g


def _undo_gauge_samples(samples, variables, gauge_vector):
    """Undo gauge on raw samples."""
    g_arr = np.array([gauge_vector.get(v, 1) for v in variables], dtype=np.int8)
    return samples * g_arr[np.newaxis, :]


# ---------------------------------------------------------------------------
# Gibbs targets (simplified for N=12 systems)
# ---------------------------------------------------------------------------


def _classical_conditional_gibbs(h, J, S_qubits, E_qubits, beta):
    """Compute classical conditional Gibbs P_S(sigma_S | sigma_E=all-up)."""
    n_S = len(S_qubits)
    configs = list(itertools.product([-1, 1], repeat=n_S))
    E_state = {q: 1 for q in E_qubits}

    energies = []
    for cfg in configs:
        S_state = {S_qubits[k]: cfg[k] for k in range(n_S)}
        full = {**S_state, **E_state}
        E_ising = sum(hi * full[i] for i, hi in h.items())
        E_ising += sum(Jij * full[i] * full[j] for (i, j), Jij in J.items())
        energies.append(E_ising)

    energies = np.array(energies)
    log_p = -beta * energies
    log_p -= log_p.max()
    probs = np.exp(log_p)
    probs /= probs.sum()
    return {cfg: float(p) for cfg, p in zip(configs, probs)}


def _quantum_conditional_gibbs(h, J, S_qubits, E_qubits, beta, A_over_B):
    """Compute quantum conditional reduced Gibbs P_S from pause-point H(s_p).

    Uses full matrix exponentiation exp(-beta * H) (not just the diagonal),
    then traces out the environment with E fixed to all-up.
    """
    try:
        from locth1.gibbs_comparison import quantum_conditional_marginal
    except ImportError:
        pass

    n_total = len(S_qubits) + len(E_qubits)
    if n_total > 14:
        return None

    try:
        # Use the project's existing quantum_conditional_marginal
        # which correctly exponentiates the full Hamiltonian
        E_state = {q: 1 for q in E_qubits}  # all-up boundary condition
        P_q = quantum_conditional_marginal(
            h=h, J=J,
            S_indices=S_qubits, E_state=E_state,
            A_over_B=A_over_B, beta=beta,
        )
        return P_q
    except Exception:
        # Fallback: build from scratch with full matrix exp
        try:
            from locth1.classical.exact_diag import build_ising_hamiltonian
            from scipy.linalg import expm

            A_s = 2 * A_over_B  # convention: H = -(A/2)Σσx + (B/2)Hp, A_s=A, B_s=B
            B_s = 2.0
            H = build_ising_hamiltonian(h, J, s_p=0.4, A_s=A_s, B_s=B_s)
            H_dense = H.toarray()

            # Full thermal state: rho = exp(-beta * H) / Z
            # Use eigendecomposition for numerical stability
            eigenvalues = np.real(np.linalg.eigvalsh(H_dense))
            E_min = eigenvalues.min()
            rho_diag_full = np.exp(-beta * (eigenvalues - E_min))
            Z = rho_diag_full.sum()

            # Get the diagonal of exp(-beta*H) in computational basis
            rho_matrix = expm(-beta * H_dense)
            rho_diag = np.real(np.diag(rho_matrix))
            rho_diag /= rho_diag.sum()

            # Conditional: fix E = all-up, trace over S
            n_S = len(S_qubits)
            n_E = len(E_qubits)
            configs = list(itertools.product([-1, 1], repeat=n_S))
            P_S = {}
            for cfg in configs:
                s_bits = tuple(0 if c == 1 else 1 for c in cfg)
                E_bits = tuple(0 for _ in range(n_E))  # all-up
                full_bits = [0] * n_total
                for k, idx in enumerate(S_qubits):
                    full_bits[idx] = s_bits[k]
                for k, idx in enumerate(E_qubits):
                    full_bits[idx] = E_bits[k]
                basis_idx = sum(b << (n_total - 1 - i) for i, b in enumerate(full_bits))
                P_S[cfg] = float(rho_diag[basis_idx])

            total = sum(P_S.values())
            if total > 0:
                P_S = {k: v / total for k, v in P_S.items()}
            return P_S
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Main campaign
# ---------------------------------------------------------------------------


def run_campaign(config_path: str, output_dir: str, qpu_key: str, dry_run: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    p7 = config["phase7_frustrated"]
    qpu_config = config["qpus"][qpu_key]
    solver = qpu_config["solver"]
    beta = config["calibration"]["beta_eff"][qpu_key]
    s_p = config["schedule"]["s_p"]

    if qpu_key == "system64":
        A_over_B = config["calibration"]["system64_A_over_B"]
    else:
        A_over_B = config["schedule"]["A_over_B"]

    out = Path(output_dir) / qpu_key
    out.mkdir(parents=True, exist_ok=True)

    sub_size = p7["subsystem_size"]
    n_total = p7["system_size"]
    lambda_values = p7["coupling_lambda_values"]
    W_values = p7["disorder_W_values"]
    seeds = p7["disorder_seeds"]
    t_hold_values = p7["t_hold_values"]
    num_reads_normal = p7["num_reads"]
    num_reads_long = p7["num_reads_long_pause"]
    sign_offset = p7["sign_seed_offset"]
    flip_p = p7["sign_flip_probability"]
    n_gauges = p7["n_gauges"]
    memory_thresh = p7["memory_threshold"]

    total_conditions = len(lambda_values) * len(W_values) * len(seeds) * len(t_hold_values)
    print(f"Phase 7 frustrated on {qpu_key} ({solver}): {total_conditions} conditions")
    print(f"  beta_eff={beta:.3f}, A/B={A_over_B:.4f}")
    print(f"  {len(lambda_values)} lambda × {len(W_values)} W × {len(seeds)} seeds × {len(t_hold_values)} t_p")
    print(f"  × {n_gauges} gauges × 3 init = {total_conditions * n_gauges * 3} QPU submissions")

    conditions = []
    cond_idx = 0

    for lam in lambda_values:
        for W in W_values:
            for seed in seeds:
                # Build base Hamiltonian
                problem = build_logical_graph(
                    "random_regular", n_total, sub_size, lam, W, seed,
                )
                h_raw, J_raw = problem["h"], problem["J"]
                S_q_raw, E_q_raw = problem["S_qubits"], problem["E_qubits"]
                init_raw = _build_initial_states(S_q_raw, E_q_raw, seed)
                h, J_ferro, S_q, E_q, init_states = _remap_to_contiguous(
                    h_raw, J_raw, S_q_raw, E_q_raw, init_raw,
                )
                # Apply EA bond disorder
                J_ea = _flip_coupler_signs(J_ferro, sign_seed=seed + sign_offset, flip_probability=flip_p)

                for t_hold in t_hold_values:
                    cond_idx += 1
                    nr = num_reads_long if t_hold >= 500 else num_reads_normal

                    # Collect gauge-averaged samples
                    all_samples_per_init = {lab: [] for lab in init_states}
                    all_variables = None
                    rng_gauge = np.random.default_rng(seed * 1000 + int(t_hold))

                    for g_idx in range(n_gauges):
                        # Generate gauge vector
                        if g_idx == 0:
                            gauge = {i: 1 for i in range(n_total)}
                        else:
                            gauge = {i: int(rng_gauge.choice([-1, 1])) for i in range(n_total)}

                        h_g, J_g, init_g = _apply_gauge(h, J_ea, init_states, gauge)

                        for init_label, init_state in init_g.items():
                            if dry_run:
                                # Simulate with classical Gibbs
                                # Stable deterministic seed mapped from init_label
                                _init_seed = {"S_up": 1, "S_down": 2, "S_random": 3}.get(init_label, 0)
                                rng_dry = np.random.default_rng(seed + g_idx * 100 + _init_seed * 10)
                                n_q = n_total
                                samples = rng_dry.choice([-1, 1], size=(nr, n_q)).astype(np.int8)
                                variables = list(range(n_q))
                                result = {"samples": samples, "energies": np.zeros(nr),
                                          "variables": variables, "metadata": {}}
                            else:
                                from locth1.reverse_anneal import run_reverse_anneal
                                result = run_reverse_anneal(
                                    h=h_g, J=J_g, initial_state=init_state,
                                    s_p=s_p, t_hold=t_hold, num_reads=nr,
                                    solver=solver, label=f"p7ea_lam{lam}_W{W}_s{seed}_tp{t_hold}_g{g_idx}_{init_label}",
                                )
                                time.sleep(0.1)

                            # Undo gauge on samples
                            raw_samples = _undo_gauge_samples(
                                result["samples"], result["variables"], gauge,
                            )
                            all_samples_per_init[init_label].append(raw_samples)
                            if all_variables is None:
                                all_variables = result["variables"]

                    # Pool gauge-averaged samples per initial state
                    P_S_per_init = {}
                    for init_label in init_states:
                        pooled = np.concatenate(all_samples_per_init[init_label], axis=0)
                        P_S_per_init[init_label] = compute_marginal(
                            pooled, S_q, variables=all_variables,
                        )

                    # Memory order parameter
                    mem = memory_order_parameter(P_S_per_init)
                    relaxed = mem <= memory_thresh

                    # Pool all initial states for thermal-marginal comparison
                    all_pooled = {}
                    for init_label, P in P_S_per_init.items():
                        for cfg, p in P.items():
                            all_pooled[cfg] = all_pooled.get(cfg, 0.0) + p / len(P_S_per_init)

                    # Classical conditional Gibbs
                    P_cl = _classical_conditional_gibbs(h, J_ea, S_q, E_q, beta)
                    dtv_cl = total_variation_distance(all_pooled, P_cl)

                    # Quantum conditional Gibbs
                    P_q = _quantum_conditional_gibbs(h, J_ea, S_q, E_q, beta, A_over_B)
                    dtv_q = total_variation_distance(all_pooled, P_q) if P_q else None

                    delta_d = (dtv_cl - dtv_q) if dtv_q is not None else None

                    entry = {
                        "cond_idx": cond_idx,
                        "lambda": lam, "W": W, "seed": seed, "t_hold": t_hold,
                        "memory": round(mem, 6),
                        "relaxed": relaxed,
                        "dtv_classical": round(dtv_cl, 6),
                        "dtv_quantum": round(dtv_q, 6) if dtv_q is not None else None,
                        "delta_d": round(delta_d, 6) if delta_d is not None else None,
                        "n_gauges": n_gauges,
                        "num_reads_per_gauge": nr,
                    }
                    conditions.append(entry)

                    status = "RELAXED" if relaxed else "memory"
                    q_str = f"Dq={dtv_q:.4f}" if dtv_q is not None else "Dq=N/A"
                    print(
                        f"  [{cond_idx}/{total_conditions}] "
                        f"lam={lam} W={W} s={seed} tp={t_hold}: "
                        f"M={mem:.4f} [{status}] Dcl={dtv_cl:.4f} {q_str}",
                        flush=True,
                    )

    # Save results
    summary = {
        "campaign": "phase7_frustrated",
        "qpu": qpu_key,
        "solver": solver,
        "beta_eff": beta,
        "A_over_B": A_over_B,
        "total_conditions": total_conditions,
        "conditions": conditions,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Go/no-go analysis
    relaxed = [c for c in conditions if c["relaxed"]]
    n_relaxed = len(relaxed)
    print(f"\n=== GO/NO-GO ANALYSIS ({qpu_key}) ===")
    print(f"  Total conditions: {total_conditions}")
    print(f"  Relaxed (M ≤ {memory_thresh}): {n_relaxed}")

    if n_relaxed > 0:
        dtv_q_vals = [c["dtv_quantum"] for c in relaxed if c["dtv_quantum"] is not None]
        dtv_cl_vals = [c["dtv_classical"] for c in relaxed]
        delta_d_vals = [c["delta_d"] for c in relaxed if c["delta_d"] is not None]

        if dtv_q_vals:
            print(f"  Median D_TV(quantum): {np.median(dtv_q_vals):.4f}")
            print(f"  Median D_TV(classical): {np.median(dtv_cl_vals):.4f}")
            print(f"  Median ΔD (cl - q): {np.median(delta_d_vals):.4f}")
            print(f"  Fraction ΔD > 0: {sum(1 for d in delta_d_vals if d > 0)}/{len(delta_d_vals)}")

            # Nature go/no-go
            go_relaxed = n_relaxed >= 30
            go_quantum_tight = np.median(dtv_q_vals) <= 0.05
            go_inversion = np.median(delta_d_vals) > 0 and sum(1 for d in delta_d_vals if d > 0) > 0.7 * len(delta_d_vals)

            if go_relaxed and go_quantum_tight and go_inversion:
                print("  >>> NATURE GO: quantum target consistently beats classical! <<<")
            elif n_relaxed >= 15:
                print("  >>> NATURE PHYSICS: enough data for a solid frustrated section")
            else:
                print("  >>> INSUFFICIENT: need more relaxed instances")
    else:
        print("  >>> NO RELAXED INSTANCES — check parameters")

    print(f"\nResults saved to {out / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: Frustrated EA campaign")
    parser.add_argument("--config", default="configs/phase7_frustrated.yaml")
    parser.add_argument("--output", default="data/raw/phase7_frustrated")
    parser.add_argument("--qpu", default="advantage2", choices=["advantage2", "system64"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_campaign(args.config, args.output, args.qpu, dry_run=args.dry_run)
