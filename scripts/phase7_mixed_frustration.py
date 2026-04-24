"""Phase 7 Campaign 2b: Mixed-frustration design.

GPT Pro insight: full EA everywhere makes E stop being a bath.
Better design: frustrated S, ferromagnetic E, scan frustration density.

Protocol:
  - N=12, |S|=4, |E|=8, random 3-regular graph
  - E couplers: always ferromagnetic (J=-1)
  - S couplers: EA bond disorder with tunable flip probability p_S
  - S-E boundary couplers: EA with tunable flip probability p_boundary
  - Scan p_S from 0 (ferro) to 0.5 (full EA)
  - Scan p_boundary from 0 to 0.5
  - This finds the regime where S is frustrated but E still acts as a bath

Run:
    PYTHONPATH=src .venv/bin/python scripts/phase7_mixed_frustration.py \\
        --qpu advantage2 --output data/raw/phase7_mixed_frustration
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import yaml

from locth1.analysis import compute_marginal
from locth1.hamiltonians import build_logical_graph
from locth1.observables import memory_order_parameter, total_variation_distance


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


def _selective_sign_flip(J, S_set, E_set, p_S, p_boundary, seed):
    """Flip coupler signs selectively: p_S for S-S, p_boundary for S-E, 0 for E-E."""
    rng = np.random.default_rng(seed)
    J_new = {}
    for (a, b), Jij in J.items():
        a_in_S = a in S_set
        b_in_S = b in S_set
        a_in_E = a in E_set
        b_in_E = b in E_set

        if a_in_S and b_in_S:
            # S-S coupler: flip with probability p_S
            flip = rng.random() < p_S
        elif (a_in_S and b_in_E) or (a_in_E and b_in_S):
            # S-E boundary: flip with probability p_boundary
            flip = rng.random() < p_boundary
        else:
            # E-E coupler: keep ferromagnetic
            flip = False

        J_new[(a, b)] = -Jij if flip else Jij
    return J_new


def _classical_conditional_gibbs(h, J, S_qubits, E_qubits, beta):
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
    try:
        from locth1.gibbs_comparison import quantum_conditional_marginal
        E_state = {q: 1 for q in E_qubits}
        return quantum_conditional_marginal(
            h=h, J=J, S_indices=S_qubits, E_state=E_state,
            A_over_B=A_over_B, beta=beta,
        )
    except Exception:
        return None


def run_campaign(output_dir: str, qpu_key: str, dry_run: bool = False):
    # Config
    solvers = {
        "advantage2": {"solver": "Advantage2_system1", "beta": 7.219, "A_over_B": 0.2603},
        "system64": {"solver": "Advantage_system6.4", "beta": 4.289, "A_over_B": 0.2589},
    }
    cfg = solvers[qpu_key]
    solver, beta, A_over_B = cfg["solver"], cfg["beta"], cfg["A_over_B"]

    out = Path(output_dir) / qpu_key
    out.mkdir(parents=True, exist_ok=True)

    # Parameters
    N, sub_size = 12, 4
    lam_values = [0.5, 0.7, 1.0]
    # Frustration density scan: p_S controls S frustration, p_boundary controls interface
    p_S_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    p_boundary_values = [0.0, 0.25, 0.5]
    W_values = [0.0, 0.5, 1.0]
    seeds = [42, 123, 456, 789, 1024]
    t_hold = 100.0
    num_reads = 2000
    s_p = 0.4

    total = len(lam_values) * len(p_S_values) * len(p_boundary_values) * len(W_values) * len(seeds)
    print(f"Mixed-frustration on {qpu_key} ({solver}): {total} conditions")
    print(f"  {len(lam_values)} λ × {len(p_S_values)} p_S × {len(p_boundary_values)} p_bnd × {len(W_values)} W × {len(seeds)} seeds")

    conditions = []
    idx = 0

    for lam in lam_values:
        for p_S in p_S_values:
            for p_bnd in p_boundary_values:
                for W in W_values:
                    for seed in seeds:
                        idx += 1

                        # Build base graph
                        problem = build_logical_graph("random_regular", N, sub_size, lam, W, seed)
                        h_raw, J_raw = problem["h"], problem["J"]
                        S_q_raw, E_q_raw = problem["S_qubits"], problem["E_qubits"]

                        # Build initial states
                        rng = np.random.default_rng(seed)
                        E_state = {q: 1 for q in E_q_raw}
                        init_raw = {
                            "S_up": {**{q: 1 for q in S_q_raw}, **E_state},
                            "S_down": {**{q: -1 for q in S_q_raw}, **E_state},
                            "S_random": {**{q: int(rng.choice([-1, 1])) for q in S_q_raw}, **E_state},
                        }

                        # Remap to contiguous
                        h, J_ferro, S_q, E_q, init_states = _remap_to_contiguous(
                            h_raw, J_raw, S_q_raw, E_q_raw, init_raw,
                        )
                        S_set = set(S_q)
                        E_set = set(E_q)

                        # Apply selective frustration
                        J = _selective_sign_flip(J_ferro, S_set, E_set, p_S, p_bnd, seed + 7777)

                        # Run QPU or dry-run
                        P_S_per_init = {}
                        for init_label, init_state in init_states.items():
                            if dry_run:
                                # Stable deterministic seed mapped from init_label
                                _init_seed = {"S_up": 1, "S_down": 2, "S_random": 3}.get(init_label, 0)
                                rng_dry = np.random.default_rng(seed + _init_seed * 1000)
                                samples = rng_dry.choice([-1, 1], size=(num_reads, N)).astype(np.int8)
                                variables = list(range(N))
                                result = {"samples": samples, "variables": variables}
                            else:
                                from locth1.reverse_anneal import run_reverse_anneal
                                result = run_reverse_anneal(
                                    h=h, J=J, initial_state=init_state,
                                    s_p=s_p, t_hold=t_hold, num_reads=num_reads,
                                    solver=solver,
                                    label=f"mixfrust_lam{lam}_pS{p_S}_pB{p_bnd}_W{W}_s{seed}_{init_label}",
                                )
                                time.sleep(0.1)

                            P_S_per_init[init_label] = compute_marginal(
                                result["samples"], S_q, variables=result["variables"],
                            )

                        mem = memory_order_parameter(P_S_per_init)
                        relaxed = mem <= 0.05

                        # Pool and compare to targets
                        all_pooled = {}
                        for P in P_S_per_init.values():
                            for cfg, p in P.items():
                                all_pooled[cfg] = all_pooled.get(cfg, 0.0) + p / 3

                        P_cl = _classical_conditional_gibbs(h, J, S_q, E_q, beta)
                        dtv_cl = total_variation_distance(all_pooled, P_cl)

                        P_q = _quantum_conditional_gibbs(h, J, S_q, E_q, beta, A_over_B)
                        dtv_q = total_variation_distance(all_pooled, P_q) if P_q else None
                        delta_d = (dtv_cl - dtv_q) if dtv_q is not None else None

                        entry = {
                            "idx": idx, "lambda": lam, "p_S": p_S, "p_boundary": p_bnd,
                            "W": W, "seed": seed,
                            "memory": round(mem, 6), "relaxed": relaxed,
                            "dtv_classical": round(dtv_cl, 6),
                            "dtv_quantum": round(dtv_q, 6) if dtv_q is not None else None,
                            "delta_d": round(delta_d, 6) if delta_d is not None else None,
                        }
                        conditions.append(entry)

                        status = "RELAXED" if relaxed else "memory"
                        q_str = f"Dq={dtv_q:.4f}" if dtv_q is not None else "Dq=N/A"
                        dd_str = f"ΔD={delta_d:+.4f}" if delta_d is not None else ""
                        print(
                            f"  [{idx}/{total}] λ={lam} pS={p_S} pB={p_bnd} W={W} s={seed}: "
                            f"M={mem:.4f} [{status}] Dcl={dtv_cl:.4f} {q_str} {dd_str}",
                            flush=True,
                        )

    # Save
    summary = {
        "campaign": "phase7_mixed_frustration",
        "qpu": qpu_key, "solver": solver, "beta_eff": beta, "A_over_B": A_over_B,
        "total_conditions": total, "conditions": conditions,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Analysis
    relaxed_list = [c for c in conditions if c["relaxed"]]
    n_relaxed = len(relaxed_list)
    print(f"\n=== RESULTS ({qpu_key}) ===")
    print(f"  Total: {total}, Relaxed: {n_relaxed}")

    if n_relaxed > 0:
        dd_vals = [c["delta_d"] for c in relaxed_list if c["delta_d"] is not None]
        dq_vals = [c["dtv_quantum"] for c in relaxed_list if c["dtv_quantum"] is not None]
        if dq_vals:
            print(f"  Median Dq (relaxed): {np.median(dq_vals):.4f}")
            print(f"  Median Dcl (relaxed): {np.median([c['dtv_classical'] for c in relaxed_list]):.4f}")
            print(f"  Median ΔD: {np.median(dd_vals):.4f}")
            print(f"  Fraction ΔD > 0: {sum(1 for d in dd_vals if d > 0)}/{len(dd_vals)}")

        # Best regime: which (p_S, p_boundary) gives relaxation + quantum inversion?
        print("\n  Best regimes (relaxed, ΔD > 0):")
        for c in sorted(relaxed_list, key=lambda x: -(x.get("delta_d") or -999)):
            if c.get("delta_d", 0) > 0:
                print(f"    λ={c['lambda']} pS={c['p_S']} pB={c['p_boundary']} W={c['W']}: "
                      f"M={c['memory']:.4f} ΔD={c['delta_d']:+.4f}")
                if len([x for x in relaxed_list if x.get("delta_d", 0) > 0]) > 10:
                    break  # show top 10

    print(f"\nSaved to {out / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: Mixed-frustration design")
    parser.add_argument("--output", default="data/raw/phase7_mixed_frustration")
    parser.add_argument("--qpu", default="advantage2", choices=["advantage2", "system64"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_campaign(args.output, args.qpu, dry_run=args.dry_run)
