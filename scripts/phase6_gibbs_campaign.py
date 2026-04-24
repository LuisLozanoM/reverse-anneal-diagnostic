"""Phase 6: Gibbs-fit campaign driver.

Subcommands
-----------
  calibrate_readout       - Readout confusion-matrix calibration on the 6-qubit S.
  run_phase2_exact        - Exact small-system QPU runs (|S|=4, N in {8,10,12}).
  analyze_phase2_exact    - Compare measured P_S to quantum reduced Gibbs, classical Gibbs, Glauber.
  run_phase3_bath         - Bath-ensemble QPU runs (|S|=6, |E|=50, ordered E).
  analyze_phase3_bath     - Pool samples, fit effective H_S^eff at fixed beta.

Every ``run_*`` subcommand supports ``--dry-run`` to skip the QPU submission
and substitute a classical_diagonal_marginal sample draw, which exercises the
full analysis pipeline without consuming QPU time.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from locth1.analysis import compute_marginal
from locth1.classical.exact_diag import run_ed_thermalization, product_state
from locth1.gibbs_comparison import (
    _compute_full_system_rho_diag,
    _trace_out_environment_diagonal,
    classical_conditional_marginal,
    classical_diagonal_marginal,
    compute_gibbs_fit_metrics,
    fit_effective_H_S,
    one_point_magnetization,
    quantum_conditional_marginal,
    quantum_reduced_gibbs_marginal,
    two_point_correlations,
)
from locth1.hamiltonians import build_logical_graph, build_native_graph
from locth1.metadata import save_raw_results
from locth1.observables import memory_order_parameter, total_variation_distance

GHZ_US_TO_NATURAL = 2.0 * np.pi * 1e3  # GHz * microseconds -> dimensionless phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _beta_for_qpu(config: dict[str, Any], qpu_name: str) -> float:
    return float(config["calibration"]["beta_eff"][qpu_name])


def _schedule_params(config: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (s_p, A_s, B_s, A_over_B) from the config schedule block."""
    sch = config["schedule"]
    return float(sch["s_p"]), float(sch["A_s"]), float(sch["B_s"]), float(sch["A_over_B"])


def _load_calibration(
    output_dir: Path,
    config: dict[str, Any],
    qpu_name: str,
) -> tuple[float, float, float, float, str]:
    """Return (beta, A_s, B_s, A_over_B, source_label).

    Searches for a Phase 1 calibration JSON in this order:
      1. ``phase1_calibration/{qpu_name}_beta_eff.json`` — QPU-specific
      2. ``phase1_calibration/beta_eff.json`` — legacy, check solver field
      3. Config fallback
    """
    calib_dir = output_dir / "phase6_gibbs" / "phase1_calibration"

    qpu_specific = calib_dir / f"{qpu_name}_beta_eff.json"
    if qpu_specific.exists():
        with open(qpu_specific) as f:
            calib = json.load(f)
        return (
            float(calib["beta_eff"]),
            float(calib["A_s"]),
            float(calib["B_s"]),
            float(calib["A_over_B"]) if calib.get("A_over_B") is not None else float(calib["A_s"]) / float(calib["B_s"]),
            f"phase1_calibration/{qpu_name}_beta_eff.json",
        )

    legacy = calib_dir / "beta_eff.json"
    if legacy.exists():
        with open(legacy) as f:
            calib = json.load(f)
        # Only accept if the legacy file's solver matches the requested QPU
        legacy_solver = str(calib.get("solver", "")).lower()
        # Rough match: "advantage2_system1" contains "advantage2", etc.
        if qpu_name.lower().replace("_", "") in legacy_solver.replace("_", ""):
            return (
                float(calib["beta_eff"]),
                float(calib["A_s"]),
                float(calib["B_s"]),
                float(calib["A_over_B"]) if calib.get("A_over_B") is not None else float(calib["A_s"]) / float(calib["B_s"]),
                "phase1_calibration/beta_eff.json (legacy)",
            )

    beta = _beta_for_qpu(config, qpu_name)
    _s_p, A_s, B_s, A_over_B = _schedule_params(config)
    return beta, A_s, B_s, A_over_B, "config_fallback"


def _build_initial_states(
    S_qubits: list[int],
    E_qubits: list[int],
    E_preparation: str = "all_up",
    seed: int = 42,
) -> dict[str, dict[int, int]]:
    """Build a set of (S, E) initial state dicts for QPU submission."""
    rng = np.random.default_rng(seed)

    if E_preparation == "all_up":
        E_state = {q: 1 for q in E_qubits}
    elif E_preparation == "all_down":
        E_state = {q: -1 for q in E_qubits}
    elif E_preparation == "random":
        E_state = {q: int(rng.choice([-1, 1])) for q in E_qubits}
    elif E_preparation == "domain":
        mid = len(E_qubits) // 2
        E_state = {q: (1 if i < mid else -1) for i, q in enumerate(E_qubits)}
    else:
        raise ValueError(f"Unknown E_preparation: {E_preparation!r}")

    return {
        "S_up": {**{q: 1 for q in S_qubits}, **E_state},
        "S_down": {**{q: -1 for q in S_qubits}, **E_state},
        "S_random": {
            **{q: int(rng.choice([-1, 1])) for q in S_qubits},
            **E_state,
        },
    }


def _remap_to_contiguous(
    h: dict[Any, float],
    J: dict[tuple[Any, Any], float],
    S_qubits: list[Any],
    E_qubits: list[Any],
    initial_states: dict[str, dict[Any, int]],
) -> tuple[
    dict[int, float],
    dict[tuple[int, int], float],
    list[int],
    list[int],
    dict[str, dict[int, int]],
    dict[Any, int],
]:
    """Remap arbitrary qubit labels (e.g. Zephyr-native) to contiguous 0..N-1.

    Native hardware graphs use sparse integer labels drawn from the topology;
    the ED / classical Gibbs enumerators infer system size as ``max(qubits)+1``
    and would otherwise try to enumerate 2^(max_label+1) states.  Remapping at
    the driver level keeps the theoretical functions simple and does not
    affect the QPU run: the D-Wave ``EmbeddingComposite`` embeds whatever
    logical labels it receives, so 0..N-1 is equivalent to the native labels
    for a small subsystem with trivial embedding.
    """
    all_qubits = list(S_qubits) + list(E_qubits)
    label_to_idx: dict[Any, int] = {q: i for i, q in enumerate(all_qubits)}

    h_new: dict[int, float] = {label_to_idx[q]: float(v) for q, v in h.items() if q in label_to_idx}
    J_new: dict[tuple[int, int], float] = {}
    for (a, b), Jij in J.items():
        if a in label_to_idx and b in label_to_idx:
            u, v = label_to_idx[a], label_to_idx[b]
            if u > v:
                u, v = v, u
            J_new[(u, v)] = float(Jij)

    S_new = [label_to_idx[q] for q in S_qubits]
    E_new = [label_to_idx[q] for q in E_qubits]

    initial_states_new: dict[str, dict[int, int]] = {}
    for label, state in initial_states.items():
        initial_states_new[label] = {
            label_to_idx[q]: int(v) for q, v in state.items() if q in label_to_idx
        }
    return h_new, J_new, S_new, E_new, initial_states_new, label_to_idx


def _simulate_qpu_classical(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    initial_state: dict[int, int],
    num_reads: int,
    beta: float,
    rng: np.random.Generator,
    *,
    enumeration_cap: int = 20,
) -> dict[str, Any]:
    """Dry-run stand-in for a QPU job.

    For $n \\leq 20$ qubits we enumerate the full $2^n$ basis and sample from
    $\\exp(-\\beta H_P)/Z$, independently of the initial state (the classical
    Gibbs distribution has no memory).  For larger systems we fall back to
    uniform random samples, which give a physically meaningless but cheap
    stand-in that still exercises the analysis pipeline (the effective
    $H_S^{\\text{eff}}$ fit should converge to $\\tilde h = \\tilde J = 0$
    for the uniform distribution).
    """
    qubits: set[int] = set(h.keys())
    for i, j in J:
        qubits.update((i, j))
    n = (max(qubits) + 1) if qubits else 0

    if n > enumeration_cap:
        # Uniform-random stand-in for large systems.
        samples = rng.choice([-1, 1], size=(num_reads, n)).astype(np.int8)
        energies_out = np.zeros(num_reads, dtype=np.float64)
        solver_tag = "dry_run_uniform_large_n"
    else:
        n_states = 2 ** n
        indices = np.arange(n_states, dtype=np.int64)
        shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        spins = 1 - 2 * bits

        energies = np.zeros(n_states, dtype=np.float64)
        for i, hi in h.items():
            energies += hi * spins[:, i]
        for (i, j), Jij in J.items():
            energies += Jij * spins[:, i] * spins[:, j]

        E_min = energies.min()
        weights = np.exp(-beta * (energies - E_min))
        probs = weights / weights.sum()
        draws = rng.choice(n_states, size=num_reads, p=probs)
        samples = spins[draws].astype(np.int8)
        energies_out = energies[draws]
        solver_tag = "dry_run_classical"

    return {
        "samples": samples,
        "energies": energies_out,
        "variables": list(range(n)),
        "timing": {},
        "metadata": {
            "solver": solver_tag,
            "graph_id": "synthetic",
            "calibration_epoch": "N/A",
            "anneal_schedule": [],
            "initial_state": dict(initial_state),
        },
    }


def _spin_tuple_keys_from_samples(
    samples: np.ndarray,
    S_indices: list[int],
) -> dict[tuple[int, ...], float]:
    """Compute the subsystem marginal from raw +/-1 samples."""
    sub = samples[:, S_indices]  # shape (num_reads, |S|)
    keys, counts = np.unique(sub, axis=0, return_counts=True)
    total = counts.sum()
    return {tuple(int(s) for s in k): float(c) / float(total) for k, c in zip(keys, counts)}


# ---------------------------------------------------------------------------
# Phase 2: exact small-system benchmark
# ---------------------------------------------------------------------------


def run_phase2_exact(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Submit QPU jobs for the |S|=4 exact small-system benchmark.

    3D sweep over (system size N, disorder W, disorder seed).  For each
    (N, W, seed) combination we submit three QPU jobs (S_up, S_down, S_random)
    with the environment fixed in all-up, save raw h5 samples, and record the
    per-initial-state subsystem marginal.  The analysis step computes the
    memory order parameter and the conditional Gibbs comparisons.

    With ``dry_run=True`` the QPU submission is replaced by classical sampling
    from the thermal distribution of $H_P$.
    """
    p2 = config["phase2"]
    qpu_name = p2["qpu"]
    beta, _A_s, _B_s, A_over_B, calib_source = _load_calibration(output_dir, config, qpu_name)
    s_p, _, _, _ = _schedule_params(config)
    print(f"Using beta_eff={beta:.3f}, A/B={A_over_B:.4f} from {calib_source}")

    out = output_dir / "phase6_gibbs" / "phase2_exact"
    out.mkdir(parents=True, exist_ok=True)

    sub_size = int(p2["subsystem_size"])
    t_hold = float(p2["t_hold"])
    num_reads = int(p2["num_reads"])
    graph_type = str(p2["graph_type"])
    sizes = list(p2["system_sizes"])
    lambda_values = [float(lam) for lam in p2["coupling_lambda_values"]]
    W_values = [float(w) for w in p2["disorder_W_values"]]
    seeds_per_W = {
        float(k): list(v) for k, v in p2["disorder_seeds_per_W"].items()
    }

    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical"}
    solver_name = qpu_config["solver"]

    conditions: list[dict[str, Any]] = []
    total_jobs = 0
    for lam in lambda_values:
        for W in W_values:
            for _ in seeds_per_W[W]:
                for _n in sizes:
                    total_jobs += 3  # S_up, S_down, S_random

    print(
        f"Phase 2 sweep: {len(lambda_values)} lambda × {len(W_values)} W × varying seeds × "
        f"{len(sizes)} sizes × 3 S-initial states = {total_jobs} QPU jobs"
    )

    job_counter = 0
    native_topology = config["qpus"][qpu_name].get("topology", "zephyr") if not dry_run else "zephyr"
    for lam in lambda_values:
        for W in W_values:
            for seed in seeds_per_W[W]:
                for n_total in sizes:
                    if graph_type == "native":
                        problem = build_native_graph(
                            topology=native_topology,
                            n_qubits=n_total,
                            subsystem_size=sub_size,
                            coupling_lambda=lam,
                            disorder_W=W,
                            seed=int(seed),
                        )
                    else:
                        problem = build_logical_graph(
                            graph_type=graph_type,
                            n_qubits=n_total,
                            subsystem_size=sub_size,
                            coupling_lambda=lam,
                            disorder_W=W,
                            seed=int(seed),
                        )
                    h_raw, J_raw = problem["h"], problem["J"]
                    S_qubits_raw = problem["S_qubits"]
                    E_qubits_raw = problem["E_qubits"]
                    initial_states_raw = _build_initial_states(
                        S_qubits_raw, E_qubits_raw, "all_up", int(seed),
                    )
                    initial_states_raw = {
                        k: v for k, v in initial_states_raw.items()
                        if k in ("S_up", "S_down", "S_random")
                    }
                    # Remap arbitrary labels (especially sparse native-Zephyr
                    # integers) to contiguous 0..N-1 for the ED / classical
                    # marginal code, and also for the QPU submission; the
                    # embedding composite handles either mapping.
                    h, J, S_qubits, E_qubits, initial_states, _ = _remap_to_contiguous(
                        h_raw, J_raw, S_qubits_raw, E_qubits_raw, initial_states_raw,
                    )

                    per_init_entries: list[dict[str, Any]] = []
                    rng_dryrun = np.random.default_rng(int(seed))
                    for init_label, init_state in initial_states.items():
                        job_counter += 1
                        tag = f"lam{lam}_W{W}_seed{seed}_N{n_total}_{init_label}"
                        if dry_run:
                            result = _simulate_qpu_classical(
                                h=h, J=J, initial_state=init_state,
                                num_reads=num_reads, beta=beta, rng=rng_dryrun,
                            )
                        else:
                            from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
                            result = run_reverse_anneal(
                                h=h, J=J, initial_state=init_state,
                                s_p=s_p, t_hold=t_hold,
                                num_reads=num_reads,
                                solver=solver_name,
                                label=f"phase6_phase2_{tag}",
                            )

                        raw_path = out / f"{tag}.h5"
                        save_raw_results(
                            samples=result["samples"],
                            energies=result["energies"],
                            metadata=result["metadata"],
                            path=raw_path,
                        )
                        P_S = compute_marginal(
                            result["samples"], S_qubits, variables=result["variables"],
                        )
                        per_init_entries.append({
                            "initial_state": init_label,
                            "raw_path": str(raw_path),
                            "P_S": {str(k): v for k, v in P_S.items()},
                        })
                        print(f"  [{job_counter}/{total_jobs}] {tag}")

                    conditions.append({
                        "N": n_total,
                        "W": W,
                        "seed": int(seed),
                        "coupling_lambda": lam,
                        "subsystem_size": sub_size,
                        "S_qubits": S_qubits,
                        "E_qubits": E_qubits,
                        "h": {str(k): v for k, v in h.items()},
                        "J": {str(k): v for k, v in J.items()},
                        "s_p": s_p,
                        "A_over_B": A_over_B,
                        "beta_eff": beta,
                        "per_initial_state": per_init_entries,
                    })

    summary = {
        "beta_eff": beta,
        "A_over_B": A_over_B,
        "s_p": s_p,
        "solver": solver_name,
        "calibration_source": calib_source,
        "conditions": conditions,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase 2 run complete ({'dry-run' if dry_run else 'live'}). Summary at {summary_path}")


def _shannon_entropy(P: dict[tuple[int, ...], float]) -> float:
    """Shannon entropy of a distribution in nats."""
    total = 0.0
    for p in P.values():
        if p > 1e-15:
            total -= p * float(np.log(p))
    return total


def analyze_phase2_exact(
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Load Phase 2 summary and compute theoretical predictions for each condition.

    For every (N, W, seed) combination we compute the memory order parameter
    across the three S initial states, pool the measured marginals, and
    compare the pooled distribution against four theoretical predictions:
    classical conditional / unconditional and quantum conditional /
    unconditional.  Conditions with memory order parameter above the threshold
    are flagged as not fully relaxed (their Gibbs comparison is less meaningful).
    """
    memory_threshold = float(config["phase2"].get("memory_threshold", 0.05))
    ed_cap_n = int(config["phase2"].get("ed_cap_n", 12))

    in_path = output_dir / "phase6_gibbs" / "phase2_exact" / "summary.json"
    with open(in_path) as f:
        summary = json.load(f)

    conditions = summary["conditions"]
    out = output_dir / "phase6_gibbs" / "phase2_exact"
    comparison: list[dict[str, Any]] = []

    for condition in conditions:
        n_total = int(condition["N"])
        W = float(condition["W"])
        seed_val = int(condition["seed"])
        lam = float(condition.get("coupling_lambda", 0.5))
        S_indices = list(condition["S_qubits"])
        E_indices_meta = list(condition.get("E_qubits", []))
        beta = float(condition["beta_eff"])
        A_over_B = float(condition["A_over_B"])

        h = {int(k): float(v) for k, v in condition["h"].items()}
        J: dict[tuple[int, int], float] = {}
        for k, v in condition["J"].items():
            k_clean = k.strip("()").replace(" ", "")
            parts = k_clean.split(",")
            J[(int(parts[0]), int(parts[1]))] = float(v)

        # Parse per-initial-state P_S dicts
        per_init: dict[str, dict[tuple[int, ...], float]] = {}
        for res in condition["per_initial_state"]:
            label = res["initial_state"]
            P = {}
            for k_str, v in res["P_S"].items():
                k_tuple = ast.literal_eval(k_str)
                P[k_tuple] = float(v)
            per_init[label] = P

        # Memory order parameter = max pairwise TVD across initial states
        labels = sorted(per_init.keys())
        max_tvd = 0.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
                tvd = 0.5 * sum(
                    abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                    for k in keys
                )
                if tvd > max_tvd:
                    max_tvd = tvd
        M = float(max_tvd)
        relaxed = M <= memory_threshold

        # Pool across initial states (simple average of the normalised P_S).
        pooled: dict[tuple[int, ...], float] = {}
        n_initial = len(per_init)
        for P in per_init.values():
            for k, v in P.items():
                pooled[k] = pooled.get(k, 0.0) + v
        if n_initial > 0:
            pooled = {k: v / n_initial for k, v in pooled.items()}
        pooled_entropy = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}

        # Classical comparisons are cheap for any N (they enumerate only 2^|S|).
        P_S_classical_cond = classical_conditional_marginal(h, J, S_indices, E_state, beta)
        P_S_classical_uncond = classical_diagonal_marginal(h, J, S_indices, beta)
        metrics_classical_cond = compute_gibbs_fit_metrics(pooled, P_S_classical_cond)
        metrics_classical_uncond = compute_gibbs_fit_metrics(pooled, P_S_classical_uncond)

        # Quantum comparisons require dense eigh of 2^N x 2^N; skip above ED cap.
        metrics_quantum_cond: dict[str, float] | None = None
        metrics_quantum_uncond: dict[str, float] | None = None
        P_S_quantum_cond: dict[tuple[int, ...], float] | None = None
        if n_total <= ed_cap_n:
            h_full = dict(h)
            for q in E_indices_meta:
                h_full.setdefault(q, 0.0)
            rho_diag, n_full = _compute_full_system_rho_diag(
                h_full, J, A_over_B, beta, S_indices,
            )

            P_S_quantum_uncond = _trace_out_environment_diagonal(rho_diag, S_indices, n_full)

            n_S = len(S_indices)
            P_S_quantum_cond = {}
            total_q = 0.0
            for s_int in range(2 ** n_S):
                s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
                full_bits = [0] * n_full
                for k, idx in enumerate(S_indices):
                    full_bits[idx] = s_bits[k]
                basis_idx = sum(b << (n_full - 1 - i) for i, b in enumerate(full_bits))
                p = float(rho_diag[basis_idx])
                cfg = tuple(1 - 2 * b for b in s_bits)
                P_S_quantum_cond[cfg] = p
                total_q += p
            if total_q > 0:
                P_S_quantum_cond = {k: v / total_q for k, v in P_S_quantum_cond.items()}

            metrics_quantum_cond = compute_gibbs_fit_metrics(pooled, P_S_quantum_cond)
            metrics_quantum_uncond = compute_gibbs_fit_metrics(pooled, P_S_quantum_uncond)

        comparison.append({
            "N": n_total,
            "W": W,
            "seed": seed_val,
            "coupling_lambda": lam,
            "beta_eff": beta,
            "A_over_B": A_over_B,
            "memory_order_param": M,
            "relaxed": relaxed,
            "pooled_entropy_nats": pooled_entropy,
            "classical_cond_entropy_nats": _shannon_entropy(P_S_classical_cond),
            "vs_quantum_conditional": metrics_quantum_cond,
            "vs_classical_conditional": metrics_classical_cond,
            "vs_quantum_unconditional": metrics_quantum_uncond,
            "vs_classical_unconditional": metrics_classical_uncond,
            "measured_magnetization": one_point_magnetization(pooled).tolist(),
            "classical_cond_magnetization": one_point_magnetization(P_S_classical_cond).tolist(),
        })
        flag = "RELAXED" if relaxed else "MEMORY"
        q_txt = (
            f"D_TV(q|E)={metrics_quantum_cond['d_tv']:.3f}"
            if metrics_quantum_cond is not None else "D_TV(q|E)=[ED skipped]"
        )
        print(
            f"  lam={lam:.2f} N={n_total:2d} W={W:.2f} seed={seed_val:4d}: "
            f"M={M:.3f} [{flag}], H={pooled_entropy:.3f}, "
            f"{q_txt}, "
            f"D_TV(c|E)={metrics_classical_cond['d_tv']:.3f}"
        )

    out_struct = {
        "beta_eff": summary.get("beta_eff"),
        "A_over_B": summary.get("A_over_B"),
        "s_p": summary.get("s_p"),
        "solver": summary.get("solver"),
        "memory_threshold": memory_threshold,
        "conditions": comparison,
    }
    out_path = out / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(out_struct, f, indent=2)
    print(f"Phase 2 analysis complete. Comparison at {out_path}")


# ---------------------------------------------------------------------------
# Phase 6 robustness panel: return-ramp + gauge averaging
# ---------------------------------------------------------------------------


def _build_test_case_problem(
    test_case: dict[str, Any],
    qpu_topology: str,
) -> tuple[dict[int, float], dict[tuple[int, int], float], list[int], list[int]]:
    """Rebuild the (h, J, S, E) problem for a named test case.

    Uses the same builder and seed as Phase 2/3 so the result is the
    identical Ising instance. Applies ``hamiltonian_scale`` and the
    contiguous-label remap in the same order as the original sweep.
    """
    graph_type = str(test_case["graph_type"])
    if graph_type == "native":
        problem = build_native_graph(
            topology=qpu_topology,
            n_qubits=int(test_case["N"]),
            subsystem_size=int(test_case.get("subsystem_size", 4)),
            coupling_lambda=float(test_case["coupling_lambda"]),
            disorder_W=float(test_case["W"]),
            seed=int(test_case["seed"]),
        )
    else:
        problem = build_logical_graph(
            graph_type=graph_type,
            n_qubits=int(test_case["N"]),
            subsystem_size=int(test_case.get("subsystem_size", 4)),
            coupling_lambda=float(test_case["coupling_lambda"]),
            disorder_W=float(test_case["W"]),
            seed=int(test_case["seed"]),
        )
    h_raw, J_raw = problem["h"], problem["J"]
    S_qubits_raw = problem["S_qubits"]
    E_qubits_raw = problem["E_qubits"]
    initial_states_raw = _build_initial_states(
        S_qubits_raw, E_qubits_raw, "all_up", int(test_case["seed"]),
    )
    initial_states_raw = {
        k: v for k, v in initial_states_raw.items()
        if k in ("S_up", "S_down", "S_random")
    }
    h_rm, J_rm, S_qubits, E_qubits, _, _ = _remap_to_contiguous(
        h_raw, J_raw, S_qubits_raw, E_qubits_raw, initial_states_raw,
    )
    scale = float(test_case.get("hamiltonian_scale", 1.0))
    h = {q: scale * v for q, v in h_rm.items()}
    J = {edge: scale * v for edge, v in J_rm.items()}
    return h, J, S_qubits, E_qubits


def _gauge_transform_problem(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    gauge: dict[int, int],
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    r"""Apply a spin-reversal gauge: $s'_i = \sigma_i s_i$ with $\sigma_i \in \{\pm 1\}$.

    Under this transformation $h_i \to \sigma_i h_i$ and
    $J_{ij} \to \sigma_i \sigma_j J_{ij}$.  The physical energy landscape is
    invariant; gauge-averaging samples over random $\sigma$ removes
    per-qubit hardware biases.
    """
    h_new = {i: float(gauge.get(i, 1)) * float(v) for i, v in h.items()}
    J_new: dict[tuple[int, int], float] = {}
    for (i, j), Jij in J.items():
        sigma = float(gauge.get(i, 1)) * float(gauge.get(j, 1))
        J_new[(i, j)] = sigma * float(Jij)
    return h_new, J_new


def _inverse_gauge_samples(
    samples: np.ndarray,
    gauge: dict[int, int],
    variables: list[int],
) -> np.ndarray:
    """Un-gauge measured samples back to the physical frame.

    ``samples`` is a ``(num_reads, num_cols)`` array indexed by the sampleset
    variable order; ``variables`` maps column index to logical qubit.  The
    inverse transform is simply multiplication by the gauge vector again
    (since $\sigma^2 = 1$).
    """
    out = np.asarray(samples, dtype=np.int8).copy()
    for col, var in enumerate(variables):
        g = int(gauge.get(int(var), 1))
        if g == -1:
            out[:, col] = -out[:, col]
    return out


def _gauge_initial_state(
    initial_state: dict[int, int],
    gauge: dict[int, int],
) -> dict[int, int]:
    """Transform a physical initial state into the gauged frame."""
    return {q: int(gauge.get(q, 1)) * int(v) for q, v in initial_state.items()}


def _calibrate_beta_eff_at_ramp(
    sampler,
    s_p: float,
    t_ramp_up: float,
    t_ramp_down: float = 5.0,
    t_hold: float = 100.0,
    num_reads: int = 5000,
    probe_qubits: list[int] | None = None,
    h_probe: float = 0.5,
) -> dict[str, Any]:
    """Run the single-qubit probe calibration with a custom return-ramp time.

    Returns a dict with beta_eff (mean and std) and per-probe counts.
    """
    if probe_qubits is None:
        probe_qubits = list(sampler.nodelist)[:3]
    schedule = [
        (0.0, 1.0),
        (t_ramp_down, s_p),
        (t_ramp_down + t_hold, s_p),
        (t_ramp_down + t_hold + t_ramp_up, 1.0),
    ]
    per_probe: list[dict[str, Any]] = []
    for q in probe_qubits:
        n_up = 0
        n_down = 0
        for init_val in (1, -1):
            sampleset = sampler.sample_ising(
                {q: h_probe}, {},
                num_reads=num_reads,
                anneal_schedule=[list(pt) for pt in schedule],
                initial_state={q: init_val},
                reinitialize_state=True,
                auto_scale=False,
                label=f"ramp_calib_q{q}_ramp{t_ramp_up}_init{init_val}",
            )
            try:
                sampleset.resolve()
            except Exception:  # noqa: BLE001
                pass
            arr = np.asarray(sampleset.record.sample, dtype=np.int8)
            variables = list(sampleset.variables)
            col = variables.index(q)
            col_vals = arr[:, col]
            n_up += int(np.sum(col_vals == 1))
            n_down += int(np.sum(col_vals == -1))
        if n_up == 0 or n_down == 0:
            continue
        beta_q = float(np.log(n_down / n_up) / (2.0 * h_probe))
        per_probe.append({"qubit": int(q), "n_up": n_up, "n_down": n_down, "beta_eff": beta_q})
    if not per_probe:
        raise RuntimeError("All probes gave degenerate populations.")
    betas = [p["beta_eff"] for p in per_probe]
    return {
        "beta_eff": float(np.mean(betas)),
        "beta_eff_std": float(np.std(betas)),
        "t_ramp_up": t_ramp_up,
        "per_probe": per_probe,
    }


def run_phase6_robustness(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Run the Phase 6 robustness panel.

    Two sweeps on a small set of representative test cases:
      (a) return-ramp sweep: submit the same problem with different
          ``t_ramp_up`` values and recalibrate ``beta_eff`` per ramp.
      (b) gauge sweep: submit random spin-reversal gauges of the same
          problem, un-gauge the samples, pool across gauges.

    Both tests target: (i) a clean relaxed case at W=0, (ii) a relaxed case
    at W=1.0, (iii) the Phase 3A dynamical trapping anomaly.
    """
    p6 = config["phase6_robustness"]
    qpu_name = p6["qpu"]
    s_p, _A_s, _B_s, A_over_B_default = _schedule_params(config)
    num_reads = int(p6["num_reads"])
    ramp_values = [float(r) for r in p6["ramp_values_us"]]
    num_gauges = int(p6["num_gauges"])
    gauge_seed = int(p6.get("gauge_seed", 42))
    recalibrate = bool(p6.get("recalibrate_per_ramp", True))
    test_cases = list(p6["test_cases"])

    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical", "topology": "zephyr"}
    solver_name = qpu_config["solver"]
    qpu_topology = qpu_config.get("topology", "zephyr")

    out = output_dir / "phase6_gibbs" / "phase6_robustness"
    out.mkdir(parents=True, exist_ok=True)

    # Default beta_eff from the existing Phase 1 calibration.
    beta_default, _, _, _, _ = _load_calibration(output_dir, config, qpu_name)

    # ------------------------------------------------------------------
    # Step 1: recalibrate beta_eff at each ramp value (reverse-anneal probes)
    # ------------------------------------------------------------------
    beta_eff_per_ramp: dict[float, float] = {}
    if recalibrate and not dry_run:
        from dwave.system import DWaveSampler  # noqa: WPS433
        config_kwargs: dict[str, Any] = {"solver": solver_name}
        if Path("./dwave.conf").exists():
            config_kwargs["config_file"] = "./dwave.conf"
        sampler = DWaveSampler(**config_kwargs)
        try:
            for t_ramp in ramp_values:
                print(f"Calibrating beta_eff at t_ramp_up={t_ramp} μs...")
                calib = _calibrate_beta_eff_at_ramp(
                    sampler, s_p=s_p, t_ramp_up=t_ramp, num_reads=5000,
                )
                beta_eff_per_ramp[t_ramp] = calib["beta_eff"]
                calib_path = out / f"beta_eff_ramp{t_ramp}.json"
                with open(calib_path, "w") as f:
                    json.dump(calib, f, indent=2)
                print(
                    f"  beta_eff(ramp={t_ramp} μs) = {calib['beta_eff']:.3f} "
                    f"± {calib['beta_eff_std']:.3f}"
                )
        finally:
            try:
                sampler.client.close()
            except Exception:  # noqa: BLE001
                pass
    else:
        for t_ramp in ramp_values:
            beta_eff_per_ramp[t_ramp] = beta_default

    # ------------------------------------------------------------------
    # Step 2: ramp sweep
    # ------------------------------------------------------------------
    ramp_results: list[dict[str, Any]] = []
    for case in test_cases:
        case_name = str(case["name"])
        h, J, S_qubits, E_qubits = _build_test_case_problem(case, qpu_topology)
        initial_states = _build_initial_states(S_qubits, E_qubits, "all_up", int(case["seed"]))
        initial_states = {
            k: v for k, v in initial_states.items()
            if k in ("S_up", "S_down", "S_random")
        }
        for t_ramp in ramp_values:
            beta_eff = beta_eff_per_ramp[t_ramp]
            per_init_entries: list[dict[str, Any]] = []
            for init_label, init_state in initial_states.items():
                tag = f"ramp{t_ramp}_{case_name}_{init_label}"
                if dry_run:
                    rng = np.random.default_rng(int(case["seed"]))
                    result = _simulate_qpu_classical(
                        h=h, J=J, initial_state=init_state,
                        num_reads=num_reads, beta=beta_eff, rng=rng,
                    )
                else:
                    from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
                    result = run_reverse_anneal(
                        h=h, J=J, initial_state=init_state,
                        s_p=s_p, t_hold=float(case.get("t_hold", 100.0)),
                        num_reads=num_reads, solver=solver_name,
                        t_ramp_down=5.0, t_ramp_up=t_ramp,
                        label=f"phase6_robustness_{tag}",
                    )
                raw_path = out / f"{tag}.h5"
                save_raw_results(
                    samples=result["samples"], energies=result["energies"],
                    metadata=result["metadata"], path=raw_path,
                )
                P_S = compute_marginal(
                    result["samples"], S_qubits, variables=result["variables"],
                )
                per_init_entries.append({
                    "initial_state": init_label,
                    "raw_path": str(raw_path),
                    "P_S": {str(k): v for k, v in P_S.items()},
                })
                print(f"  ramp {t_ramp} μs / {case_name} / {init_label}")
            ramp_results.append({
                "test_case": case_name,
                "t_ramp_up": t_ramp,
                "beta_eff": beta_eff,
                "S_qubits": S_qubits,
                "E_qubits": E_qubits,
                "h": {str(k): v for k, v in h.items()},
                "J": {str(k): v for k, v in J.items()},
                "per_initial_state": per_init_entries,
            })

    # ------------------------------------------------------------------
    # Step 3: gauge sweep at default ramp (5 μs)
    # ------------------------------------------------------------------
    gauge_results: list[dict[str, Any]] = []
    gauge_rng = np.random.default_rng(gauge_seed)
    for case in test_cases:
        case_name = str(case["name"])
        h, J, S_qubits, E_qubits = _build_test_case_problem(case, qpu_topology)
        all_qubits = list(S_qubits) + list(E_qubits)
        physical_initial_states = _build_initial_states(
            S_qubits, E_qubits, "all_up", int(case["seed"]),
        )
        physical_initial_states = {
            k: v for k, v in physical_initial_states.items()
            if k in ("S_up", "S_down", "S_random")
        }

        # Identity gauge + num_gauges random gauges
        gauges: list[dict[int, int]] = [{q: 1 for q in all_qubits}]
        for _ in range(num_gauges):
            sigma = gauge_rng.choice([-1, 1], size=len(all_qubits))
            gauges.append({all_qubits[i]: int(sigma[i]) for i in range(len(all_qubits))})

        for g_idx, gauge in enumerate(gauges):
            h_g, J_g = _gauge_transform_problem(h, J, gauge)
            per_init_entries: list[dict[str, Any]] = []
            for init_label, physical_init in physical_initial_states.items():
                gauged_init = _gauge_initial_state(physical_init, gauge)
                tag = f"gauge{g_idx}_{case_name}_{init_label}"
                if dry_run:
                    rng = np.random.default_rng(int(case["seed"]) + g_idx)
                    result = _simulate_qpu_classical(
                        h=h_g, J=J_g, initial_state=gauged_init,
                        num_reads=num_reads, beta=beta_default, rng=rng,
                    )
                else:
                    from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
                    result = run_reverse_anneal(
                        h=h_g, J=J_g, initial_state=gauged_init,
                        s_p=s_p, t_hold=float(case.get("t_hold", 100.0)),
                        num_reads=num_reads, solver=solver_name,
                        t_ramp_down=5.0, t_ramp_up=5.0,
                        label=f"phase6_robustness_{tag}",
                    )
                # Un-gauge the measured samples back to the physical frame.
                ungauged_samples = _inverse_gauge_samples(
                    result["samples"], gauge, list(result["variables"]),
                )
                raw_path = out / f"{tag}.h5"
                save_raw_results(
                    samples=ungauged_samples, energies=result["energies"],
                    metadata={
                        **result["metadata"],
                        "gauge": {str(k): int(v) for k, v in gauge.items()},
                    },
                    path=raw_path,
                )
                P_S = compute_marginal(
                    ungauged_samples, S_qubits, variables=list(result["variables"]),
                )
                per_init_entries.append({
                    "initial_state": init_label,
                    "raw_path": str(raw_path),
                    "P_S": {str(k): v for k, v in P_S.items()},
                })
                print(f"  gauge {g_idx}/{len(gauges)-1} / {case_name} / {init_label}")
            gauge_results.append({
                "test_case": case_name,
                "gauge_index": g_idx,
                "gauge": {str(k): int(v) for k, v in gauge.items()},
                "S_qubits": S_qubits,
                "E_qubits": E_qubits,
                "per_initial_state": per_init_entries,
            })

    summary = {
        "s_p": s_p,
        "A_over_B": A_over_B_default,
        "solver": solver_name,
        "beta_eff_per_ramp": {str(k): v for k, v in beta_eff_per_ramp.items()},
        "test_cases": test_cases,
        "ramp_results": ramp_results,
        "gauge_results": gauge_results,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase 6 robustness complete ({'dry-run' if dry_run else 'live'}). Summary at {summary_path}")


def analyze_phase6_robustness(
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Analyse the robustness panel: per-ramp and per-gauge D_TV + M."""
    memory_threshold = float(
        config["phase6_robustness"].get("memory_threshold", 0.05)
    )
    in_path = output_dir / "phase6_gibbs" / "phase6_robustness" / "summary.json"
    with open(in_path) as f:
        summary = json.load(f)

    out_struct: dict[str, Any] = {
        "beta_eff_per_ramp": summary["beta_eff_per_ramp"],
        "ramp_analysis": [],
        "gauge_analysis": {},
    }

    def _analyze_condition(
        case_block: dict[str, Any],
        beta: float,
    ) -> dict[str, Any]:
        S_indices = list(case_block["S_qubits"])
        E_indices_meta = list(case_block.get("E_qubits", []))
        h = {int(k): float(v) for k, v in case_block["h"].items()}
        J: dict[tuple[int, int], float] = {}
        for k, v in case_block["J"].items():
            k_clean = k.strip("()").replace(" ", "")
            parts = k_clean.split(",")
            J[(int(parts[0]), int(parts[1]))] = float(v)

        per_init: dict[str, dict[tuple[int, ...], float]] = {}
        for res in case_block["per_initial_state"]:
            P: dict[tuple[int, ...], float] = {}
            for k_str, v in res["P_S"].items():
                k_tuple = ast.literal_eval(k_str)
                P[k_tuple] = float(v)
            per_init[res["initial_state"]] = P

        labels = sorted(per_init.keys())
        max_tvd = 0.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
                tvd = 0.5 * sum(
                    abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                    for k in keys
                )
                if tvd > max_tvd:
                    max_tvd = tvd
        M = float(max_tvd)

        pooled: dict[tuple[int, ...], float] = {}
        for P in per_init.values():
            for k, v in P.items():
                pooled[k] = pooled.get(k, 0.0) + v
        if per_init:
            pooled = {k: v / len(per_init) for k, v in pooled.items()}
        H = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}
        P_S_classical_cond = classical_conditional_marginal(
            h, J, S_indices, E_state, beta,
        )
        metrics = compute_gibbs_fit_metrics(pooled, P_S_classical_cond)
        return {
            "M": M,
            "relaxed": bool(M <= memory_threshold),
            "pooled_entropy": H,
            "d_tv_classical_cond": metrics["d_tv"],
            "sym_kl": metrics["symmetric_kl"],
        }

    # Load gauged test cases to rebuild (h, J) for gauge analysis
    # (summary stores only per_initial_state P_S; we need h/J for each case).
    # We rebuild via _build_test_case_problem.
    qpu_name = config["phase6_robustness"]["qpu"]
    qpu_topology = config["qpus"][qpu_name].get("topology", "zephyr")

    print("=== Ramp sweep ===")
    for rr in summary["ramp_results"]:
        beta = float(rr["beta_eff"])
        stats = _analyze_condition(rr, beta)
        out_struct["ramp_analysis"].append({
            "test_case": rr["test_case"],
            "t_ramp_up": rr["t_ramp_up"],
            "beta_eff": beta,
            **stats,
        })
        flag = "RELAXED" if stats["relaxed"] else "MEMORY"
        print(
            f"  {rr['test_case']:25s} ramp={rr['t_ramp_up']:5.1f} μs β={beta:.2f}: "
            f"M={stats['M']:.3f} [{flag}], H={stats['pooled_entropy']:.3f}, "
            f"D_TV={stats['d_tv_classical_cond']:.4f}"
        )

    print()
    print("=== Gauge sweep ===")
    for case in config["phase6_robustness"]["test_cases"]:
        case_name = case["name"]
        case_gauge_entries = [
            g for g in summary["gauge_results"] if g["test_case"] == case_name
        ]
        h_phys, J_phys, S_indices, E_indices = _build_test_case_problem(case, qpu_topology)
        beta = float(summary["beta_eff_per_ramp"].get("5.0", 0.0))
        if beta == 0.0:
            beta = float(next(iter(summary["beta_eff_per_ramp"].values())))
        per_gauge_stats = []
        for g_entry in case_gauge_entries:
            g_entry_plus = {
                **g_entry,
                "h": {str(k): v for k, v in h_phys.items()},
                "J": {str(k): v for k, v in J_phys.items()},
            }
            stats = _analyze_condition(g_entry_plus, beta)
            per_gauge_stats.append({
                "gauge_index": g_entry["gauge_index"],
                **stats,
            })
            flag = "RELAXED" if stats["relaxed"] else "MEMORY"
            print(
                f"  {case_name:25s} gauge={g_entry['gauge_index']}: "
                f"M={stats['M']:.3f} [{flag}], H={stats['pooled_entropy']:.3f}, "
                f"D_TV={stats['d_tv_classical_cond']:.4f}"
            )
        # Aggregate
        dtvs = np.array([s["d_tv_classical_cond"] for s in per_gauge_stats])
        Ms = np.array([s["M"] for s in per_gauge_stats])
        out_struct["gauge_analysis"][case_name] = {
            "beta_eff": beta,
            "per_gauge": per_gauge_stats,
            "D_TV_mean": float(np.mean(dtvs)),
            "D_TV_std": float(np.std(dtvs)),
            "M_mean": float(np.mean(Ms)),
            "M_std": float(np.std(Ms)),
        }
        print(
            f"  {case_name:25s} over {len(per_gauge_stats)} gauges: "
            f"D_TV = {np.mean(dtvs):.4f} ± {np.std(dtvs):.4f}, "
            f"M = {np.mean(Ms):.4f} ± {np.std(Ms):.4f}"
        )

    out_path = (
        output_dir / "phase6_gibbs" / "phase6_robustness" / "comparison.json"
    )
    with open(out_path, "w") as f:
        json.dump(out_struct, f, indent=2)
    print(f"Phase 6 robustness analysis complete. Comparison at {out_path}")


# ---------------------------------------------------------------------------
# Phase 3A: scaled Hamiltonian at |S|=4
# ---------------------------------------------------------------------------


def run_phase3a_scaled(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    scale_override: float | None = None,
    subdir_override: str | None = None,
) -> None:
    """Run the |S|=4 sweep with a global Hamiltonian scaling factor.

    Same 4D sweep as Phase 2 but with every (h, J) value multiplied by
    ``hamiltonian_scale`` before submission.  Reduces effective beta x E
    so that the relaxed distributions have measurable entropy rather than
    being delta functions on the ferromagnetic ground state.
    """
    p3a = config["phase3a"]
    qpu_name = p3a["qpu"]
    beta, _A_s, _B_s, A_over_B, calib_source = _load_calibration(output_dir, config, qpu_name)
    s_p, _, _, _ = _schedule_params(config)
    scale = float(scale_override if scale_override is not None else p3a.get("hamiltonian_scale", 1.0))
    subdir = subdir_override or f"phase3a_scale{scale}"
    print(f"Using beta_eff={beta:.3f}, A/B={A_over_B:.4f} from {calib_source}; hamiltonian_scale={scale}")

    out = output_dir / "phase6_gibbs" / subdir
    out.mkdir(parents=True, exist_ok=True)

    sub_size = int(p3a["subsystem_size"])
    t_hold = float(p3a["t_hold"])
    num_reads = int(p3a["num_reads"])
    graph_type = str(p3a["graph_type"])
    sizes = list(p3a["system_sizes"])
    lambda_values = [float(lam) for lam in p3a["coupling_lambda_values"]]
    W_values = [float(w) for w in p3a["disorder_W_values"]]
    seeds_per_W = {float(k): list(v) for k, v in p3a["disorder_seeds_per_W"].items()}

    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical", "topology": "zephyr"}
    solver_name = qpu_config["solver"]
    native_topology = qpu_config.get("topology", "zephyr")

    conditions: list[dict[str, Any]] = []
    total_jobs = 3 * sum(len(seeds_per_W[w]) * len(sizes) for w in W_values) * len(lambda_values)
    print(
        f"Phase 3A sweep: {len(lambda_values)} lambda x {len(W_values)} W x varying seeds x "
        f"{len(sizes)} sizes x 3 S-initial states = {total_jobs} QPU jobs"
    )

    job_counter = 0
    for lam in lambda_values:
        for W in W_values:
            for seed in seeds_per_W[W]:
                for n_total in sizes:
                    if graph_type == "native":
                        problem = build_native_graph(
                            topology=native_topology, n_qubits=n_total,
                            subsystem_size=sub_size, coupling_lambda=lam,
                            disorder_W=W, seed=int(seed),
                        )
                    else:
                        problem = build_logical_graph(
                            graph_type=graph_type, n_qubits=n_total,
                            subsystem_size=sub_size, coupling_lambda=lam,
                            disorder_W=W, seed=int(seed),
                        )
                    h_raw, J_raw = problem["h"], problem["J"]
                    S_qubits_raw = problem["S_qubits"]
                    E_qubits_raw = problem["E_qubits"]
                    initial_states_raw = _build_initial_states(
                        S_qubits_raw, E_qubits_raw, "all_up", int(seed),
                    )
                    initial_states_raw = {
                        k: v for k, v in initial_states_raw.items()
                        if k in ("S_up", "S_down", "S_random")
                    }
                    h_rm, J_rm, S_qubits, E_qubits, initial_states, _ = _remap_to_contiguous(
                        h_raw, J_raw, S_qubits_raw, E_qubits_raw, initial_states_raw,
                    )
                    # Apply the scaling factor on the remapped (h, J).
                    h = {q: scale * v for q, v in h_rm.items()}
                    J = {edge: scale * v for edge, v in J_rm.items()}

                    per_init_entries: list[dict[str, Any]] = []
                    rng_dryrun = np.random.default_rng(int(seed))
                    for init_label, init_state in initial_states.items():
                        job_counter += 1
                        tag = f"lam{lam}_W{W}_seed{seed}_N{n_total}_{init_label}"
                        if dry_run:
                            result = _simulate_qpu_classical(
                                h=h, J=J, initial_state=init_state,
                                num_reads=num_reads, beta=beta, rng=rng_dryrun,
                            )
                        else:
                            from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
                            result = run_reverse_anneal(
                                h=h, J=J, initial_state=init_state,
                                s_p=s_p, t_hold=t_hold, num_reads=num_reads,
                                solver=solver_name, label=f"phase6_phase3a_{tag}",
                            )

                        raw_path = out / f"{tag}.h5"
                        save_raw_results(
                            samples=result["samples"], energies=result["energies"],
                            metadata=result["metadata"], path=raw_path,
                        )
                        P_S = compute_marginal(
                            result["samples"], S_qubits, variables=result["variables"],
                        )
                        per_init_entries.append({
                            "initial_state": init_label,
                            "raw_path": str(raw_path),
                            "P_S": {str(k): v for k, v in P_S.items()},
                        })
                        print(f"  [{job_counter}/{total_jobs}] {tag}")

                    conditions.append({
                        "N": n_total, "W": W, "seed": int(seed),
                        "coupling_lambda": lam,
                        "subsystem_size": sub_size,
                        "S_qubits": S_qubits, "E_qubits": E_qubits,
                        "h": {str(k): v for k, v in h.items()},
                        "J": {str(k): v for k, v in J.items()},
                        "s_p": s_p, "A_over_B": A_over_B,
                        "beta_eff": beta, "hamiltonian_scale": scale,
                        "per_initial_state": per_init_entries,
                    })

    summary = {
        "beta_eff": beta, "A_over_B": A_over_B, "s_p": s_p,
        "hamiltonian_scale": scale, "solver": solver_name,
        "calibration_source": calib_source, "conditions": conditions,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase 3A run complete ({'dry-run' if dry_run else 'live'}). Summary at {summary_path}")


def analyze_phase3a_scaled(
    config: dict[str, Any],
    output_dir: Path,
    *,
    subdir: str = "phase3a_scaled",
) -> None:
    """Phase 3A analysis — same pipeline as Phase 2 but reading from a
    scale-specific subdir."""
    p3a_config = dict(config["phase2"])
    p3a_config["memory_threshold"] = float(config["phase3a"].get("memory_threshold", 0.05))
    p3a_config["ed_cap_n"] = int(config["phase3a"].get("ed_cap_n", 12))

    in_path = output_dir / "phase6_gibbs" / subdir / "summary.json"
    with open(in_path) as f:
        summary = json.load(f)

    memory_threshold = p3a_config["memory_threshold"]
    ed_cap_n = p3a_config["ed_cap_n"]

    conditions_list = summary["conditions"]
    out = output_dir / "phase6_gibbs" / subdir
    comparison: list[dict[str, Any]] = []

    for condition in conditions_list:
        n_total = int(condition["N"])
        W = float(condition["W"])
        seed_val = int(condition["seed"])
        lam = float(condition.get("coupling_lambda", 0.5))
        scale = float(condition.get("hamiltonian_scale", 1.0))
        S_indices = list(condition["S_qubits"])
        E_indices_meta = list(condition.get("E_qubits", []))
        beta = float(condition["beta_eff"])
        A_over_B = float(condition["A_over_B"])

        h = {int(k): float(v) for k, v in condition["h"].items()}
        J: dict[tuple[int, int], float] = {}
        for k, v in condition["J"].items():
            k_clean = k.strip("()").replace(" ", "")
            parts = k_clean.split(",")
            J[(int(parts[0]), int(parts[1]))] = float(v)

        per_init: dict[str, dict[tuple[int, ...], float]] = {}
        for res in condition["per_initial_state"]:
            P: dict[tuple[int, ...], float] = {}
            for k_str, v in res["P_S"].items():
                k_tuple = ast.literal_eval(k_str)
                P[k_tuple] = float(v)
            per_init[res["initial_state"]] = P

        labels = sorted(per_init.keys())
        max_tvd = 0.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
                tvd = 0.5 * sum(
                    abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                    for k in keys
                )
                if tvd > max_tvd:
                    max_tvd = tvd
        M = float(max_tvd)
        relaxed = M <= memory_threshold

        pooled: dict[tuple[int, ...], float] = {}
        n_initial = len(per_init)
        for P in per_init.values():
            for k, v in P.items():
                pooled[k] = pooled.get(k, 0.0) + v
        if n_initial > 0:
            pooled = {k: v / n_initial for k, v in pooled.items()}
        pooled_entropy = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}

        P_S_classical_cond = classical_conditional_marginal(h, J, S_indices, E_state, beta)
        P_S_classical_uncond = classical_diagonal_marginal(h, J, S_indices, beta)
        metrics_classical_cond = compute_gibbs_fit_metrics(pooled, P_S_classical_cond)
        metrics_classical_uncond = compute_gibbs_fit_metrics(pooled, P_S_classical_uncond)

        metrics_quantum_cond = None
        metrics_quantum_uncond = None
        if n_total <= ed_cap_n:
            h_full = dict(h)
            for q in E_indices_meta:
                h_full.setdefault(q, 0.0)
            rho_diag, n_full = _compute_full_system_rho_diag(
                h_full, J, A_over_B, beta, S_indices,
            )
            P_S_quantum_uncond = _trace_out_environment_diagonal(rho_diag, S_indices, n_full)
            n_S = len(S_indices)
            P_S_quantum_cond: dict[tuple[int, ...], float] = {}
            total_q = 0.0
            for s_int in range(2 ** n_S):
                s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
                full_bits = [0] * n_full
                for k, idx in enumerate(S_indices):
                    full_bits[idx] = s_bits[k]
                basis_idx = sum(b << (n_full - 1 - i) for i, b in enumerate(full_bits))
                p = float(rho_diag[basis_idx])
                cfg = tuple(1 - 2 * b for b in s_bits)
                P_S_quantum_cond[cfg] = p
                total_q += p
            if total_q > 0:
                P_S_quantum_cond = {k: v / total_q for k, v in P_S_quantum_cond.items()}
            metrics_quantum_cond = compute_gibbs_fit_metrics(pooled, P_S_quantum_cond)
            metrics_quantum_uncond = compute_gibbs_fit_metrics(pooled, P_S_quantum_uncond)

        comparison.append({
            "N": n_total, "W": W, "seed": seed_val,
            "coupling_lambda": lam,
            "hamiltonian_scale": scale,
            "beta_eff": beta, "A_over_B": A_over_B,
            "memory_order_param": M, "relaxed": relaxed,
            "pooled_entropy_nats": pooled_entropy,
            "classical_cond_entropy_nats": _shannon_entropy(P_S_classical_cond),
            "vs_quantum_conditional": metrics_quantum_cond,
            "vs_classical_conditional": metrics_classical_cond,
            "vs_quantum_unconditional": metrics_quantum_uncond,
            "vs_classical_unconditional": metrics_classical_uncond,
            "measured_magnetization": one_point_magnetization(pooled).tolist(),
            "classical_cond_magnetization": one_point_magnetization(P_S_classical_cond).tolist(),
        })
        q_txt = (
            f"D_TV(q|E)={metrics_quantum_cond['d_tv']:.3f}"
            if metrics_quantum_cond is not None else "D_TV(q|E)=[ED skipped]"
        )
        flag = "RELAXED" if relaxed else "MEMORY"
        print(
            f"  lam={lam:.2f} N={n_total:2d} W={W:.2f} seed={seed_val:4d}: "
            f"M={M:.3f} [{flag}], H={pooled_entropy:.3f}, {q_txt}, "
            f"D_TV(c|E)={metrics_classical_cond['d_tv']:.3f}"
        )

    out_struct = {
        "beta_eff": summary.get("beta_eff"),
        "A_over_B": summary.get("A_over_B"),
        "s_p": summary.get("s_p"),
        "hamiltonian_scale": summary.get("hamiltonian_scale"),
        "solver": summary.get("solver"),
        "memory_threshold": memory_threshold,
        "conditions": comparison,
    }
    out_path = out / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(out_struct, f, indent=2)
    print(f"Phase 3A analysis complete. Comparison at {out_path}")


# ---------------------------------------------------------------------------
# Phase 3B: |S|=6 bath-ensemble fit at main-experiment scale
# ---------------------------------------------------------------------------


def run_phase3b_bath_ensemble(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    scale: float = 1.0,
) -> None:
    """Phase 3B — |S|=6, |E|=50 bath-ensemble sweep.

    Submits three reverse-anneal jobs per (W, seed) condition with different
    S initial states, saves raw h5, and records the pooled marginal.  The
    analysis step fits an effective subsystem Hamiltonian at fixed beta_eff
    only for conditions that pass the entropy gate (Codex Q3).
    """
    p3b = config["phase3b"]
    qpu_name = p3b["qpu"]
    beta, _A_s, _B_s, A_over_B, calib_source = _load_calibration(output_dir, config, qpu_name)
    s_p, _, _, _ = _schedule_params(config)
    subdir_tag = "phase3b_bath_unscaled" if scale == 1.0 else f"phase3b_bath_scale{scale}"
    print(f"Phase 3B (scale={scale}): beta_eff={beta:.3f} from {calib_source}")

    out = output_dir / "phase6_gibbs" / subdir_tag
    out.mkdir(parents=True, exist_ok=True)

    sub_size = int(p3b["subsystem_size"])
    n_env = int(p3b["n_environment"])
    n_total = sub_size + n_env
    t_hold = float(p3b["t_hold"])
    num_reads = int(p3b["num_reads"])
    coupling_lambda = float(p3b["coupling_lambda"])
    graph_type = str(p3b["graph_type"])
    W_values = [float(w) for w in p3b["disorder_W_values"]]
    seeds_per_W = {float(k): list(v) for k, v in p3b["disorder_seeds_per_W"].items()}
    E_preparation = str(p3b.get("E_preparation", "all_up"))

    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical", "topology": "zephyr"}
    solver_name = qpu_config["solver"]
    native_topology = qpu_config.get("topology", "zephyr")

    conditions: list[dict[str, Any]] = []
    total_jobs = 3 * sum(len(seeds_per_W[w]) for w in W_values)
    print(f"Phase 3B sweep: {len(W_values)} W x seeds x 3 S-initial states = {total_jobs} QPU jobs")

    job_counter = 0
    for W in W_values:
        for seed in seeds_per_W[W]:
            if graph_type == "native":
                problem = build_native_graph(
                    topology=native_topology, n_qubits=n_total,
                    subsystem_size=sub_size, coupling_lambda=coupling_lambda,
                    disorder_W=W, seed=int(seed),
                )
            else:
                problem = build_logical_graph(
                    graph_type=graph_type, n_qubits=n_total,
                    subsystem_size=sub_size, coupling_lambda=coupling_lambda,
                    disorder_W=W, seed=int(seed),
                )
            h_raw, J_raw = problem["h"], problem["J"]
            S_qubits_raw = problem["S_qubits"]
            E_qubits_raw = problem["E_qubits"]
            initial_states_raw = _build_initial_states(
                S_qubits_raw, E_qubits_raw, E_preparation, int(seed),
            )
            initial_states_raw = {
                k: v for k, v in initial_states_raw.items()
                if k in ("S_up", "S_down", "S_random")
            }
            h_rm, J_rm, S_qubits, E_qubits, initial_states, _ = _remap_to_contiguous(
                h_raw, J_raw, S_qubits_raw, E_qubits_raw, initial_states_raw,
            )
            h = {q: scale * v for q, v in h_rm.items()}
            J = {edge: scale * v for edge, v in J_rm.items()}

            per_init_entries: list[dict[str, Any]] = []
            rng_dryrun = np.random.default_rng(int(seed))
            for init_label, init_state in initial_states.items():
                job_counter += 1
                tag = f"W{W}_seed{seed}_{init_label}"
                if dry_run:
                    result = _simulate_qpu_classical(
                        h=h, J=J, initial_state=init_state,
                        num_reads=num_reads, beta=beta, rng=rng_dryrun,
                    )
                else:
                    from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
                    result = run_reverse_anneal(
                        h=h, J=J, initial_state=init_state,
                        s_p=s_p, t_hold=t_hold, num_reads=num_reads,
                        solver=solver_name, label=f"phase6_phase3b_{tag}",
                    )

                raw_path = out / f"{tag}.h5"
                save_raw_results(
                    samples=result["samples"], energies=result["energies"],
                    metadata=result["metadata"], path=raw_path,
                )
                P_S = compute_marginal(
                    result["samples"], S_qubits, variables=result["variables"],
                )
                per_init_entries.append({
                    "initial_state": init_label,
                    "raw_path": str(raw_path),
                    "P_S": {str(k): v for k, v in P_S.items()},
                })
                print(f"  [{job_counter}/{total_jobs}] {tag}")

            conditions.append({
                "N": n_total, "W": W, "seed": int(seed),
                "coupling_lambda": coupling_lambda,
                "subsystem_size": sub_size,
                "n_environment": n_env,
                "S_qubits": S_qubits, "E_qubits": E_qubits,
                "h": {str(k): v for k, v in h.items()},
                "J": {str(k): v for k, v in J.items()},
                "s_p": s_p, "A_over_B": A_over_B,
                "beta_eff": beta,
                "hamiltonian_scale": scale,
                "E_preparation": E_preparation,
                "per_initial_state": per_init_entries,
            })

    summary = {
        "beta_eff": beta, "A_over_B": A_over_B, "s_p": s_p,
        "hamiltonian_scale": scale, "solver": solver_name,
        "calibration_source": calib_source, "conditions": conditions,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase 3B run complete ({'dry-run' if dry_run else 'live'}). Summary at {summary_path}")


def analyze_phase3b_bath_ensemble(
    config: dict[str, Any],
    output_dir: Path,
    *,
    subdir_tag: str = "phase3b_bath_unscaled",
) -> None:
    """Phase 3B analysis: classical conditional Gibbs + effective H_S fit.

    The effective-H fit is gated by a pre-registered entropy threshold
    (Codex Q3); conditions below the gate are analysed with the classical
    Gibbs D_TV only.
    """
    p3b = config["phase3b"]
    memory_threshold = float(p3b.get("memory_threshold", 0.05))
    fit_min_entropy = float(p3b.get("fit_min_entropy", 0.05))
    l2_reg = float(p3b.get("fit_l2_regularization", 0.01))

    in_path = output_dir / "phase6_gibbs" / subdir_tag / "summary.json"
    with open(in_path) as f:
        summary = json.load(f)

    out = output_dir / "phase6_gibbs" / subdir_tag
    comparison: list[dict[str, Any]] = []

    for condition in summary["conditions"]:
        W = float(condition["W"])
        seed_val = int(condition["seed"])
        S_indices = list(condition["S_qubits"])
        E_indices_meta = list(condition.get("E_qubits", []))
        beta = float(condition["beta_eff"])

        h = {int(k): float(v) for k, v in condition["h"].items()}
        J: dict[tuple[int, int], float] = {}
        for k, v in condition["J"].items():
            k_clean = k.strip("()").replace(" ", "")
            parts = k_clean.split(",")
            J[(int(parts[0]), int(parts[1]))] = float(v)

        per_init: dict[str, dict[tuple[int, ...], float]] = {}
        for res in condition["per_initial_state"]:
            P: dict[tuple[int, ...], float] = {}
            for k_str, v in res["P_S"].items():
                k_tuple = ast.literal_eval(k_str)
                P[k_tuple] = float(v)
            per_init[res["initial_state"]] = P

        labels = sorted(per_init.keys())
        max_tvd = 0.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
                tvd = 0.5 * sum(
                    abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                    for k in keys
                )
                if tvd > max_tvd:
                    max_tvd = tvd
        M = float(max_tvd)
        relaxed = M <= memory_threshold

        pooled: dict[tuple[int, ...], float] = {}
        for P in per_init.values():
            for k, v in P.items():
                pooled[k] = pooled.get(k, 0.0) + v
        if per_init:
            pooled = {k: v / len(per_init) for k, v in pooled.items()}
        pooled_entropy = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}
        P_S_classical_cond = classical_conditional_marginal(h, J, S_indices, E_state, beta)
        metrics_classical_cond = compute_gibbs_fit_metrics(pooled, P_S_classical_cond)

        # Effective H_S fit only for conditions that pass the entropy gate.
        fit_result = None
        fit_skipped_reason = None
        if pooled_entropy >= fit_min_entropy and relaxed:
            try:
                fit_result = fit_effective_H_S(
                    pooled, beta=beta, l2_regularization=l2_reg,
                )
            except Exception as err:  # noqa: BLE001
                fit_skipped_reason = f"fit failed: {err}"
        else:
            if not relaxed:
                fit_skipped_reason = f"not relaxed (M={M:.4f} > {memory_threshold})"
            else:
                fit_skipped_reason = f"pooled_entropy {pooled_entropy:.4f} < fit_gate {fit_min_entropy}"

        comparison.append({
            "W": W, "seed": seed_val,
            "beta_eff": beta,
            "memory_order_param": M, "relaxed": relaxed,
            "pooled_entropy_nats": pooled_entropy,
            "vs_classical_conditional": metrics_classical_cond,
            "measured_magnetization": one_point_magnetization(pooled).tolist(),
            "fit_result": None if fit_result is None else {
                "h_tilde": fit_result["h_tilde"],
                "J_tilde": {str(k): v for k, v in fit_result["J_tilde"].items()},
                "success": fit_result["success"],
                "cost": fit_result["cost"],
                "metrics": fit_result["metrics"],
            },
            "fit_skipped_reason": fit_skipped_reason,
        })
        fit_marker = "FIT" if fit_result is not None else "skip"
        print(
            f"  W={W:.2f} seed={seed_val:4d}: M={M:.3f}, H={pooled_entropy:.3f}, "
            f"D_TV(c|E)={metrics_classical_cond['d_tv']:.3f}, fit={fit_marker}"
        )

    out_path = out / "comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "beta_eff": summary.get("beta_eff"),
            "hamiltonian_scale": summary.get("hamiltonian_scale"),
            "memory_threshold": memory_threshold,
            "fit_min_entropy": fit_min_entropy,
            "conditions": comparison,
        }, f, indent=2)
    print(f"Phase 3B analysis complete. Comparison at {out_path}")


# ---------------------------------------------------------------------------
# Phase 3C: disorder-averaging sweep at fixed N, lambda
# ---------------------------------------------------------------------------


def run_phase3c_disorder_sweep(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    qpu_override: str | None = None,
) -> None:
    """Run a disorder-averaged sweep over W at fixed (N, lambda).

    For the paper's main "D_TV vs W" figure: 10 disorder seeds per W
    (Codex Q4), 6 W levels, 3 S initial states per (W, seed) — 180 QPU
    jobs on random_regular (more reliable relaxation than native at N=12).
    """
    p3c = config["phase3c"]
    qpu_name = qpu_override or p3c["qpu"]
    beta, _A_s, _B_s, A_over_B, calib_source = _load_calibration(output_dir, config, qpu_name)
    s_p, _, _, _ = _schedule_params(config)
    print(f"Phase 3C on {qpu_name}: beta_eff={beta:.3f} from {calib_source}")

    subdir_name = "phase3c_disorder_sweep" if qpu_name == p3c["qpu"] else f"phase3c_disorder_sweep_{qpu_name}"
    out = output_dir / "phase6_gibbs" / subdir_name
    out.mkdir(parents=True, exist_ok=True)

    sub_size = int(p3c["subsystem_size"])
    n_total = int(p3c["system_size"])
    t_hold = float(p3c["t_hold"])
    num_reads = int(p3c["num_reads"])
    coupling_lambda = float(p3c["coupling_lambda"])
    graph_type = str(p3c["graph_type"])
    W_values = [float(w) for w in p3c["disorder_W_values"]]
    seeds = [int(s) for s in p3c["disorder_seeds"]]

    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical", "topology": "zephyr"}
    solver_name = qpu_config["solver"]
    native_topology = qpu_config.get("topology", "zephyr")

    conditions: list[dict[str, Any]] = []
    total_jobs = 3 * len(W_values) * len(seeds)
    print(
        f"Phase 3C sweep: {len(W_values)} W x {len(seeds)} seeds x 3 S-initial states = {total_jobs} QPU jobs"
    )

    job_counter = 0
    for W in W_values:
        for seed in seeds:
            if graph_type == "native":
                problem = build_native_graph(
                    topology=native_topology, n_qubits=n_total,
                    subsystem_size=sub_size, coupling_lambda=coupling_lambda,
                    disorder_W=W, seed=seed,
                )
            else:
                problem = build_logical_graph(
                    graph_type=graph_type, n_qubits=n_total,
                    subsystem_size=sub_size, coupling_lambda=coupling_lambda,
                    disorder_W=W, seed=seed,
                )
            h_raw, J_raw = problem["h"], problem["J"]
            S_qubits_raw = problem["S_qubits"]
            E_qubits_raw = problem["E_qubits"]
            initial_states_raw = _build_initial_states(
                S_qubits_raw, E_qubits_raw, "all_up", seed,
            )
            initial_states_raw = {
                k: v for k, v in initial_states_raw.items()
                if k in ("S_up", "S_down", "S_random")
            }
            h, J, S_qubits, E_qubits, initial_states, _ = _remap_to_contiguous(
                h_raw, J_raw, S_qubits_raw, E_qubits_raw, initial_states_raw,
            )

            per_init_entries: list[dict[str, Any]] = []
            rng_dryrun = np.random.default_rng(seed)
            for init_label, init_state in initial_states.items():
                job_counter += 1
                tag = f"W{W}_seed{seed}_{init_label}"
                if dry_run:
                    result = _simulate_qpu_classical(
                        h=h, J=J, initial_state=init_state,
                        num_reads=num_reads, beta=beta, rng=rng_dryrun,
                    )
                else:
                    from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
                    result = run_reverse_anneal(
                        h=h, J=J, initial_state=init_state,
                        s_p=s_p, t_hold=t_hold, num_reads=num_reads,
                        solver=solver_name, label=f"phase6_phase3c_{tag}",
                    )

                raw_path = out / f"{tag}.h5"
                save_raw_results(
                    samples=result["samples"], energies=result["energies"],
                    metadata=result["metadata"], path=raw_path,
                )
                P_S = compute_marginal(
                    result["samples"], S_qubits, variables=result["variables"],
                )
                per_init_entries.append({
                    "initial_state": init_label,
                    "raw_path": str(raw_path),
                    "P_S": {str(k): v for k, v in P_S.items()},
                })
                print(f"  [{job_counter}/{total_jobs}] {tag}")

            conditions.append({
                "N": n_total, "W": W, "seed": seed,
                "coupling_lambda": coupling_lambda,
                "subsystem_size": sub_size,
                "S_qubits": S_qubits, "E_qubits": E_qubits,
                "h": {str(k): v for k, v in h.items()},
                "J": {str(k): v for k, v in J.items()},
                "s_p": s_p, "A_over_B": A_over_B,
                "beta_eff": beta,
                "per_initial_state": per_init_entries,
            })

    summary = {
        "beta_eff": beta, "A_over_B": A_over_B, "s_p": s_p,
        "solver": solver_name, "calibration_source": calib_source,
        "conditions": conditions,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase 3C run complete ({'dry-run' if dry_run else 'live'}). Summary at {summary_path}")


def analyze_phase3c_disorder_sweep(
    config: dict[str, Any],
    output_dir: Path,
    *,
    subdir: str = "phase3c_disorder_sweep",
) -> None:
    """Phase 3C analysis: per-condition D_TV + per-W disorder averaging."""
    p3c = config["phase3c"]
    memory_threshold = float(p3c.get("memory_threshold", 0.05))

    in_path = output_dir / "phase6_gibbs" / subdir / "summary.json"
    with open(in_path) as f:
        summary = json.load(f)

    out = output_dir / "phase6_gibbs" / subdir
    comparison: list[dict[str, Any]] = []

    for condition in summary["conditions"]:
        W = float(condition["W"])
        seed_val = int(condition["seed"])
        S_indices = list(condition["S_qubits"])
        E_indices_meta = list(condition.get("E_qubits", []))
        beta = float(condition["beta_eff"])

        h = {int(k): float(v) for k, v in condition["h"].items()}
        J: dict[tuple[int, int], float] = {}
        for k, v in condition["J"].items():
            k_clean = k.strip("()").replace(" ", "")
            parts = k_clean.split(",")
            J[(int(parts[0]), int(parts[1]))] = float(v)

        per_init: dict[str, dict[tuple[int, ...], float]] = {}
        for res in condition["per_initial_state"]:
            P: dict[tuple[int, ...], float] = {}
            for k_str, v in res["P_S"].items():
                k_tuple = ast.literal_eval(k_str)
                P[k_tuple] = float(v)
            per_init[res["initial_state"]] = P

        labels = sorted(per_init.keys())
        max_tvd = 0.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
                tvd = 0.5 * sum(
                    abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                    for k in keys
                )
                if tvd > max_tvd:
                    max_tvd = tvd
        M = float(max_tvd)
        relaxed = M <= memory_threshold

        pooled: dict[tuple[int, ...], float] = {}
        for P in per_init.values():
            for k, v in P.items():
                pooled[k] = pooled.get(k, 0.0) + v
        if per_init:
            pooled = {k: v / len(per_init) for k, v in pooled.items()}
        pooled_entropy = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}
        P_S_classical_cond = classical_conditional_marginal(h, J, S_indices, E_state, beta)
        metrics_classical_cond = compute_gibbs_fit_metrics(pooled, P_S_classical_cond)

        comparison.append({
            "W": W, "seed": seed_val,
            "memory_order_param": M, "relaxed": relaxed,
            "pooled_entropy_nats": pooled_entropy,
            "vs_classical_conditional": metrics_classical_cond,
        })

    # Per-W aggregation
    per_w_stats: list[dict[str, Any]] = []
    for W in sorted(set(c["W"] for c in comparison)):
        sub = [c for c in comparison if c["W"] == W]
        dtvs = np.array([c["vs_classical_conditional"]["d_tv"] for c in sub])
        Ms = np.array([c["memory_order_param"] for c in sub])
        relax_count = sum(1 for c in sub if c["relaxed"])
        relaxed_dtvs = np.array(
            [c["vs_classical_conditional"]["d_tv"] for c in sub if c["relaxed"]]
        )
        per_w_stats.append({
            "W": W, "n_seeds": len(sub),
            "n_relaxed": int(relax_count),
            "relax_rate": float(relax_count / len(sub)),
            "M_mean": float(np.mean(Ms)),
            "M_std": float(np.std(Ms)),
            "d_tv_all_mean": float(np.mean(dtvs)),
            "d_tv_all_std": float(np.std(dtvs)),
            "d_tv_relaxed_mean": float(np.mean(relaxed_dtvs)) if len(relaxed_dtvs) else None,
            "d_tv_relaxed_std": float(np.std(relaxed_dtvs)) if len(relaxed_dtvs) else None,
        })
        print(
            f"  W={W:.2f}: {relax_count:2d}/{len(sub):2d} relaxed, "
            f"D_TV all = {np.mean(dtvs):.3f} ± {np.std(dtvs):.3f}, "
            f"D_TV relaxed = "
            + (f"{np.mean(relaxed_dtvs):.4f} ± {np.std(relaxed_dtvs):.4f}" if len(relaxed_dtvs) else "n/a")
        )

    out_path = out / "comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "beta_eff": summary.get("beta_eff"),
            "memory_threshold": memory_threshold,
            "per_seed_conditions": comparison,
            "per_w_stats": per_w_stats,
        }, f, indent=2)
    print(f"Phase 3C analysis complete. Comparison at {out_path}")


# ---------------------------------------------------------------------------
# Phase 6 frustrated pilot (mixed-sign Edwards-Anderson)
# ---------------------------------------------------------------------------


def _flip_coupler_signs(
    J: dict[tuple[int, int], float],
    sign_seed: int,
    flip_probability: float = 0.5,
) -> dict[tuple[int, int], float]:
    """Return a copy of J with each coupler sign-flipped independently with
    probability ``flip_probability`` using a separate deterministic RNG.

    At 50/50, this produces the Edwards-Anderson spin-glass bond disorder
    pattern on top of whatever topology and magnitude the original J had.
    """
    rng = np.random.default_rng(sign_seed)
    J_new: dict[tuple[int, int], float] = {}
    for edge, Jij in J.items():
        if rng.random() < flip_probability:
            J_new[edge] = -float(Jij)
        else:
            J_new[edge] = float(Jij)
    return J_new


def run_phase6_frustrated(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Run the Phase 6 frustrated (mixed-sign J) pilot sweep.

    Same structure as Phase 3C but with Edwards-Anderson bond disorder
    (50/50 sign flips on every coupler).  Produces naturally multi-modal
    classical Gibbs targets that can distinguish the thermal-marginal
    agreement from a ground-state tautology.
    """
    p6f = config["phase6_frustrated"]
    qpu_name = p6f["qpu"]
    beta, _A_s, _B_s, A_over_B, calib_source = _load_calibration(output_dir, config, qpu_name)
    s_p, _, _, _ = _schedule_params(config)
    print(f"Phase 6 frustrated on {qpu_name}: beta_eff={beta:.3f} from {calib_source}")

    out = output_dir / "phase6_gibbs" / "phase6_frustrated"
    out.mkdir(parents=True, exist_ok=True)

    sub_size = int(p6f["subsystem_size"])
    n_total = int(p6f["system_size"])
    t_hold = float(p6f["t_hold"])
    num_reads = int(p6f["num_reads"])
    coupling_lambda = float(p6f["coupling_lambda"])
    graph_type = str(p6f["graph_type"])
    W_values = [float(w) for w in p6f["disorder_W_values"]]
    seeds = [int(s) for s in p6f["disorder_seeds"]]
    sign_offset = int(p6f.get("sign_seed_offset", 7777))
    flip_p = float(p6f.get("sign_flip_probability", 0.5))

    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical", "topology": "zephyr"}
    solver_name = qpu_config["solver"]
    native_topology = qpu_config.get("topology", "zephyr")

    conditions: list[dict[str, Any]] = []
    total_jobs = 3 * len(W_values) * len(seeds)
    print(
        f"Phase 6 frustrated sweep: {len(W_values)} W × {len(seeds)} seeds × 3 S-initial = {total_jobs} jobs"
    )

    job_counter = 0
    for W in W_values:
        for seed in seeds:
            if graph_type == "native":
                problem = build_native_graph(
                    topology=native_topology, n_qubits=n_total,
                    subsystem_size=sub_size, coupling_lambda=coupling_lambda,
                    disorder_W=W, seed=seed,
                )
            else:
                problem = build_logical_graph(
                    graph_type=graph_type, n_qubits=n_total,
                    subsystem_size=sub_size, coupling_lambda=coupling_lambda,
                    disorder_W=W, seed=seed,
                )
            h_raw, J_raw = problem["h"], problem["J"]
            S_qubits_raw = problem["S_qubits"]
            E_qubits_raw = problem["E_qubits"]
            initial_states_raw = _build_initial_states(
                S_qubits_raw, E_qubits_raw, "all_up", seed,
            )
            initial_states_raw = {
                k: v for k, v in initial_states_raw.items()
                if k in ("S_up", "S_down", "S_random")
            }
            h, J_ferro, S_qubits, E_qubits, initial_states, _ = _remap_to_contiguous(
                h_raw, J_raw, S_qubits_raw, E_qubits_raw, initial_states_raw,
            )
            # Apply Edwards-Anderson bond sign flips.
            J = _flip_coupler_signs(J_ferro, sign_seed=seed + sign_offset, flip_probability=flip_p)

            per_init_entries: list[dict[str, Any]] = []
            rng_dryrun = np.random.default_rng(seed)
            for init_label, init_state in initial_states.items():
                job_counter += 1
                tag = f"W{W}_seed{seed}_{init_label}"
                if dry_run:
                    result = _simulate_qpu_classical(
                        h=h, J=J, initial_state=init_state,
                        num_reads=num_reads, beta=beta, rng=rng_dryrun,
                    )
                else:
                    from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
                    result = run_reverse_anneal(
                        h=h, J=J, initial_state=init_state,
                        s_p=s_p, t_hold=t_hold, num_reads=num_reads,
                        solver=solver_name, label=f"phase6_frustrated_{tag}",
                    )
                raw_path = out / f"{tag}.h5"
                save_raw_results(
                    samples=result["samples"], energies=result["energies"],
                    metadata=result["metadata"], path=raw_path,
                )
                P_S = compute_marginal(
                    result["samples"], S_qubits, variables=result["variables"],
                )
                per_init_entries.append({
                    "initial_state": init_label,
                    "raw_path": str(raw_path),
                    "P_S": {str(k): v for k, v in P_S.items()},
                })
                print(f"  [{job_counter}/{total_jobs}] {tag}")

            conditions.append({
                "N": n_total, "W": W, "seed": seed,
                "coupling_lambda": coupling_lambda,
                "subsystem_size": sub_size,
                "S_qubits": S_qubits, "E_qubits": E_qubits,
                "h": {str(k): v for k, v in h.items()},
                "J": {str(k): v for k, v in J.items()},
                "s_p": s_p, "A_over_B": A_over_B,
                "beta_eff": beta,
                "sign_seed": seed + sign_offset,
                "flip_probability": flip_p,
                "per_initial_state": per_init_entries,
            })

    summary = {
        "beta_eff": beta, "A_over_B": A_over_B, "s_p": s_p,
        "solver": solver_name, "calibration_source": calib_source,
        "flip_probability": flip_p,
        "conditions": conditions,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase 6 frustrated run complete ({'dry-run' if dry_run else 'live'}).")


def analyze_phase6_frustrated(
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Analyze the frustrated pilot using the same pipeline as Phase 3C."""
    p6f = config["phase6_frustrated"]
    memory_threshold = float(p6f.get("memory_threshold", 0.05))

    in_path = output_dir / "phase6_gibbs" / "phase6_frustrated" / "summary.json"
    with open(in_path) as f:
        summary = json.load(f)

    out = output_dir / "phase6_gibbs" / "phase6_frustrated"
    comparison: list[dict[str, Any]] = []

    for condition in summary["conditions"]:
        W = float(condition["W"])
        seed_val = int(condition["seed"])
        S_indices = list(condition["S_qubits"])
        E_indices_meta = list(condition.get("E_qubits", []))
        beta = float(condition["beta_eff"])

        h = {int(k): float(v) for k, v in condition["h"].items()}
        J: dict[tuple[int, int], float] = {}
        for k, v in condition["J"].items():
            k_clean = k.strip("()").replace(" ", "")
            parts = k_clean.split(",")
            J[(int(parts[0]), int(parts[1]))] = float(v)

        per_init: dict[str, dict[tuple[int, ...], float]] = {}
        for res in condition["per_initial_state"]:
            P: dict[tuple[int, ...], float] = {}
            for k_str, v in res["P_S"].items():
                k_tuple = ast.literal_eval(k_str)
                P[k_tuple] = float(v)
            per_init[res["initial_state"]] = P

        labels = sorted(per_init.keys())
        max_tvd = 0.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
                tvd = 0.5 * sum(
                    abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                    for k in keys
                )
                if tvd > max_tvd:
                    max_tvd = tvd
        M = float(max_tvd)
        relaxed = M <= memory_threshold

        pooled: dict[tuple[int, ...], float] = {}
        for P in per_init.values():
            for k, v in P.items():
                pooled[k] = pooled.get(k, 0.0) + v
        if per_init:
            pooled = {k: v / len(per_init) for k, v in pooled.items()}
        H = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}
        P_S_classical_cond = classical_conditional_marginal(h, J, S_indices, E_state, beta)
        metrics = compute_gibbs_fit_metrics(pooled, P_S_classical_cond)

        comparison.append({
            "W": W, "seed": seed_val,
            "memory_order_param": M, "relaxed": relaxed,
            "pooled_entropy_nats": H,
            "vs_classical_conditional": metrics,
            "classical_cond_entropy_nats": _shannon_entropy(P_S_classical_cond),
        })
        flag = "RELAXED" if relaxed else "MEMORY"
        print(
            f"  W={W:.2f} seed={seed_val:4d}: M={M:.3f} [{flag}], H={H:.3f}, "
            f"H_Gibbs={_shannon_entropy(P_S_classical_cond):.3f}, "
            f"D_TV={metrics['d_tv']:.4f}"
        )

    # Per-W aggregates
    per_w = []
    for W in sorted(set(c["W"] for c in comparison)):
        sub = [c for c in comparison if c["W"] == W]
        rel = [c for c in sub if c["relaxed"]]
        dtvs = np.array([c["vs_classical_conditional"]["d_tv"] for c in rel])
        per_w.append({
            "W": W, "n_seeds": len(sub), "n_relaxed": len(rel),
            "d_tv_relaxed_mean": float(np.mean(dtvs)) if len(dtvs) else None,
            "d_tv_relaxed_std": float(np.std(dtvs)) if len(dtvs) else None,
            "mean_H_Gibbs": float(np.mean([c["classical_cond_entropy_nats"] for c in sub])),
        })
    print()
    print("Per-W frustrated summary:")
    for w in per_w:
        print(
            f"  W={w['W']:.2f}: {w['n_relaxed']:2d}/{w['n_seeds']:2d} relaxed, "
            f"H_Gibbs={w['mean_H_Gibbs']:.3f}, "
            f"D_TV_relaxed = "
            + (f"{w['d_tv_relaxed_mean']:.4f} ± {w['d_tv_relaxed_std']:.4f}" if w['d_tv_relaxed_mean'] is not None else "n/a")
        )

    with open(out / "comparison.json", "w") as f:
        json.dump({
            "beta_eff": summary["beta_eff"],
            "flip_probability": summary["flip_probability"],
            "conditions": comparison,
            "per_w": per_w,
        }, f, indent=2)
    print(f"Phase 6 frustrated analysis complete.")


# ---------------------------------------------------------------------------
# Phase 3 (legacy): bath-ensemble effective H_S fit
# ---------------------------------------------------------------------------


def run_phase3_bath(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Submit Phase 3 QPU jobs for the |S|=6, |E|=50 bath-ensemble experiment."""
    p3 = config["phase3"]
    qpu_name = p3["qpu"]
    beta, _A_s, _B_s, _A_over_B, calib_source = _load_calibration(output_dir, config, qpu_name)
    s_p, _, _, _ = _schedule_params(config)
    print(f"Using beta_eff={beta:.3f} from {calib_source}")

    out = output_dir / "phase6_gibbs" / "phase3_bath"
    out.mkdir(parents=True, exist_ok=True)

    sub_size = int(p3["subsystem_size"])
    n_env = int(p3["n_environment"])
    t_hold = float(p3["t_hold"])
    num_reads = int(p3["num_reads"])
    coupling_lambda = float(p3["coupling_lambda"])
    disorder_W = float(p3["disorder_W"])
    graph_type = str(p3["graph_type"])
    seed = int(p3["disorder_seed"])
    E_preparation = str(p3["E_preparation"])
    initial_state_labels = list(p3["initial_states"])
    n_total = sub_size + n_env

    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical", "topology": "zephyr"}
    solver_name = qpu_config["solver"]

    if graph_type == "native":
        problem = build_native_graph(
            topology=qpu_config["topology"],
            n_qubits=n_total,
            subsystem_size=sub_size,
            coupling_lambda=coupling_lambda,
            disorder_W=disorder_W,
            seed=seed,
        )
    else:
        problem = build_logical_graph(
            graph_type=graph_type,
            n_qubits=n_total,
            subsystem_size=sub_size,
            coupling_lambda=coupling_lambda,
            disorder_W=disorder_W,
            seed=seed,
        )

    h, J = problem["h"], problem["J"]
    S_qubits = problem["S_qubits"]
    E_qubits = problem["E_qubits"]

    initial_states = _build_initial_states(S_qubits, E_qubits, E_preparation, seed)
    initial_states = {k: v for k, v in initial_states.items() if k in ("S_up", "S_down", "S_random")}

    rng = np.random.default_rng(seed)
    entries: list[dict[str, Any]] = []
    for init_label, init_state in initial_states.items():
        if dry_run:
            result = _simulate_qpu_classical(
                h=h, J=J, initial_state=init_state,
                num_reads=num_reads, beta=beta, rng=rng,
            )
        else:
            from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
            result = run_reverse_anneal(
                h=h, J=J, initial_state=init_state,
                s_p=s_p, t_hold=t_hold,
                num_reads=num_reads,
                solver=solver_name,
                label=f"phase6_phase3_{init_label}",
            )

        raw_path = out / f"{init_label}.h5"
        save_raw_results(
            samples=result["samples"],
            energies=result["energies"],
            metadata=result["metadata"],
            path=raw_path,
        )
        P_S = compute_marginal(
            result["samples"], S_qubits, variables=result["variables"],
        )
        entries.append({
            "initial_state": init_label,
            "raw_path": str(raw_path),
            "P_S": {str(k): v for k, v in P_S.items()},
        })
        print(f"  {init_label}: {raw_path}")

    summary = {
        "subsystem_size": sub_size,
        "n_environment": n_env,
        "S_qubits": S_qubits,
        "E_qubits": E_qubits,
        "h": {str(k): v for k, v in h.items()},
        "J": {str(k): v for k, v in J.items()},
        "E_preparation": E_preparation,
        "s_p": s_p,
        "beta_eff": beta,
        "results": entries,
    }
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase 3 run complete ({'dry-run' if dry_run else 'live'}). Summary at {summary_path}")


def analyze_phase3_bath(
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Pool Phase 3 subsystem samples and fit an effective H_S^eff at fixed beta."""
    _ = config  # unused here

    in_path = output_dir / "phase6_gibbs" / "phase3_bath" / "summary.json"
    with open(in_path) as f:
        summary = json.load(f)

    beta = float(summary["beta_eff"])

    # Pool across S initial states (the measured "relaxed" marginal)
    pooled: dict[tuple[int, ...], float] = {}
    n_initial = 0
    for res in summary["results"]:
        n_initial += 1
        for k_str, v in res["P_S"].items():
            k_tuple = ast.literal_eval(k_str)
            pooled[k_tuple] = pooled.get(k_tuple, 0.0) + float(v)
    if n_initial > 0:
        pooled = {k: v / n_initial for k, v in pooled.items()}

    fit_result = fit_effective_H_S(pooled, beta=beta)

    out_path = output_dir / "phase6_gibbs" / "phase3_bath" / "effective_H_S_fit.json"
    with open(out_path, "w") as f:
        json.dump({
            "beta_eff": beta,
            "n_S": len(summary["S_qubits"]),
            "pooled_magnetization": one_point_magnetization(pooled).tolist(),
            "fit": {
                "h_tilde": fit_result["h_tilde"],
                "J_tilde": {str(k): v for k, v in fit_result["J_tilde"].items()},
                "success": fit_result["success"],
                "cost": fit_result["cost"],
                "metrics": fit_result["metrics"],
            },
        }, f, indent=2)
    print(
        f"Phase 3 fit complete. beta_eff={beta:.3f}, "
        f"D_TV={fit_result['metrics']['d_tv']:.4f}, "
        f"sym-KL={fit_result['metrics']['symmetric_kl']:.4f}"
    )
    print(f"Effective-H_S fit saved to {out_path}")


# ---------------------------------------------------------------------------
# Phase 1: beta_eff calibration via single-qubit probes
# ---------------------------------------------------------------------------


def _interp_schedule(points: list[Any], s_p: float) -> tuple[float, float] | None:
    """Parse an anneal_schedule entry from sampler.properties and linearly
    interpolate (A(s_p), B(s_p)) in the native units reported by the solver.

    D-Wave solvers with a schedule in ``properties`` typically provide a list
    of 3- or 4-tuples of the form ``[s, A(s), B(s)]`` or ``[s, A(s), B(s), c]``
    with energies in GHz.  We interpolate linearly between the two bracketing
    rows.
    """
    if not points:
        return None
    rows: list[tuple[float, float, float]] = []
    for row in points:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            return None
        rows.append((float(row[0]), float(row[1]), float(row[2])))
    rows.sort(key=lambda r: r[0])
    if s_p <= rows[0][0]:
        return (rows[0][1], rows[0][2])
    if s_p >= rows[-1][0]:
        return (rows[-1][1], rows[-1][2])
    for i in range(1, len(rows)):
        s_hi = rows[i][0]
        if s_hi >= s_p:
            s_lo = rows[i - 1][0]
            t = (s_p - s_lo) / (s_hi - s_lo) if s_hi > s_lo else 0.0
            A = rows[i - 1][1] + t * (rows[i][1] - rows[i - 1][1])
            B = rows[i - 1][2] + t * (rows[i][2] - rows[i - 1][2])
            return (A, B)
    return None


def calibrate_beta_eff(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    n_probes: int = 3,
    probe_h: float = 0.5,
    probe_reads: int = 5000,
    probe_t_hold: float = 100.0,
    qpu_override: str | None = None,
) -> None:
    """Calibrate beta_eff on the primary QPU via single-qubit probes.

    Submits Ising problems with a single probe qubit carrying bias ``probe_h``
    and no couplings, reverse-anneals at the pause point, and extracts
    ``beta_eff = ln(n_down / n_up) / (2*h)`` from the spin populations after
    many reads.  The schedule is derived from ``sampler.properties`` when
    available, otherwise it falls back to the ``schedule`` block in the YAML
    config with an explicit warning.

    Results (beta_eff mean/std, per-probe values, interpolated A/B and
    A/B ratio) are saved to
    ``data/raw/phase6_gibbs/phase1_calibration/beta_eff.json`` alongside a
    full dump of the solver properties.
    """
    s_p = float(config["schedule"]["s_p"])
    fallback_A = float(config["schedule"]["A_s"])
    fallback_B = float(config["schedule"]["B_s"])

    qpu_name = qpu_override or config.get("phase2", {}).get("qpu", "advantage2")
    qpu_config = config["qpus"][qpu_name] if not dry_run else {
        "solver": "dry_run_calibration", "topology": "zephyr",
    }
    solver_name = qpu_config["solver"]

    out = output_dir / "phase6_gibbs" / "phase1_calibration"
    out.mkdir(parents=True, exist_ok=True)

    if dry_run:
        calibration = {
            "solver": solver_name,
            "graph_id": "synthetic",
            "s_p": s_p,
            "beta_eff": 6.93,
            "beta_eff_std": 0.20,
            "per_probe_beta_eff": [6.8, 6.93, 7.05],
            "probe_h": probe_h,
            "A_s": fallback_A,
            "B_s": fallback_B,
            "A_over_B": fallback_A / fallback_B,
            "schedule_from_properties": False,
            "dry_run": True,
        }
        with open(out / "beta_eff.json", "w") as f:
            json.dump(calibration, f, indent=2)
        print("Dry-run calibration written.")
        return

    from dwave.system import DWaveSampler  # noqa: WPS433  (optional dep)

    config_kwargs: dict[str, Any] = {"solver": solver_name}
    if Path("./dwave.conf").exists():
        config_kwargs["config_file"] = "./dwave.conf"
    sampler = DWaveSampler(**config_kwargs)
    try:
        # Dump sampler properties in full so we can audit what's available.
        props = dict(sampler.properties)

        def _safe(obj: Any) -> Any:
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, dict):
                return {str(k): _safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_safe(v) for v in obj]
            return str(obj)

        props_safe = _safe(props)
        props_path = out / f"{qpu_name}_properties.json"
        with open(props_path, "w") as f:
            json.dump(props_safe, f, indent=2, default=str)
        print(f"Sampler properties dumped to {props_path}")

        # Try to recover A(s_p), B(s_p) from the properties.
        schedule_points = props.get("anneal_schedule") or props.get("schedule")
        schedule_found = False
        if schedule_points is not None:
            ab = _interp_schedule(schedule_points, s_p)
            if ab is not None:
                A_s, B_s = ab
                schedule_found = True
                print(f"Schedule interpolated at s_p={s_p}: A={A_s:.4f}, B={B_s:.4f}")
        if not schedule_found:
            print(
                "WARNING: anneal_schedule not found in sampler.properties; "
                f"falling back to config values A={fallback_A}, B={fallback_B}."
            )
            A_s, B_s = fallback_A, fallback_B

        # Pick probe qubits from the active nodelist.
        nodelist = list(sampler.nodelist)
        if len(nodelist) < n_probes:
            raise RuntimeError(f"Only {len(nodelist)} active qubits on {solver_name}.")
        probe_qubits = nodelist[:n_probes]
        print(f"Using probe qubits: {probe_qubits}")

        schedule = [
            (0.0, 1.0),
            (5.0, s_p),
            (5.0 + probe_t_hold, s_p),
            (10.0 + probe_t_hold, 1.0),
        ]

        per_probe_beta: list[float] = []
        per_probe_counts: list[dict[str, Any]] = []
        for q in probe_qubits:
            n_up = 0
            n_down = 0
            for init_val in (1, -1):
                sampleset = sampler.sample_ising(
                    {q: probe_h}, {},
                    num_reads=probe_reads,
                    anneal_schedule=[list(pt) for pt in schedule],
                    initial_state={q: init_val},
                    reinitialize_state=True,
                    auto_scale=False,
                    label=f"beta_eff_probe_q{q}_init{init_val}",
                )
                try:
                    sampleset.resolve()
                except Exception:
                    pass
                samples = np.asarray(sampleset.record.sample, dtype=np.int8)
                variables = list(sampleset.variables)
                col = variables.index(q)
                col_vals = samples[:, col]
                n_up += int(np.sum(col_vals == 1))
                n_down += int(np.sum(col_vals == -1))

            if n_up == 0 or n_down == 0:
                print(f"  probe q{q}: degenerate populations (n_up={n_up}, n_down={n_down}); skipping")
                continue

            beta_q = float(np.log(n_down / n_up) / (2.0 * probe_h))
            per_probe_beta.append(beta_q)
            per_probe_counts.append({"qubit": int(q), "n_up": n_up, "n_down": n_down, "beta_eff": beta_q})
            print(f"  probe q{q}: n_up={n_up}, n_down={n_down}, beta_eff={beta_q:.3f}")

        if not per_probe_beta:
            raise RuntimeError("All probe qubits had degenerate populations; calibration failed.")

        beta_mean = float(np.mean(per_probe_beta))
        beta_std = float(np.std(per_probe_beta))

        calibration = {
            "solver": solver_name,
            "graph_id": props.get("chip_id", "unknown"),
            "calibration_epoch": props.get("calibration_epoch"),
            "s_p": s_p,
            "probe_h": probe_h,
            "probe_t_hold_us": probe_t_hold,
            "probe_reads": probe_reads,
            "beta_eff": beta_mean,
            "beta_eff_std": beta_std,
            "per_probe": per_probe_counts,
            "A_s": A_s,
            "B_s": B_s,
            "A_over_B": (A_s / B_s) if B_s != 0.0 else None,
            "schedule_from_properties": schedule_found,
            "dry_run": False,
        }

        # Save to QPU-specific path so cross-QPU calibrations don't overwrite
        # each other; also save to the legacy default path so existing
        # loaders without QPU awareness continue to work for the default
        # (phase2) QPU.
        qpu_specific_path = out / f"{qpu_name}_beta_eff.json"
        with open(qpu_specific_path, "w") as f:
            json.dump(calibration, f, indent=2)
        default_qpu = config.get("phase2", {}).get("qpu", "advantage2")
        if qpu_name == default_qpu:
            legacy_path = out / "beta_eff.json"
            with open(legacy_path, "w") as f:
                json.dump(calibration, f, indent=2)
        print(
            f"Calibration saved to {qpu_specific_path}\n"
            f"  beta_eff = {beta_mean:.3f} ± {beta_std:.3f}\n"
            f"  A(s_p) = {A_s:.4f} GHz, B(s_p) = {B_s:.4f} GHz, A/B = {A_s / B_s:.4f}"
        )

    finally:
        try:
            sampler.client.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Phase 1b: readout confusion-matrix calibration
# ---------------------------------------------------------------------------


def calibrate_readout(
    config: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Pin each 6-qubit basis state and measure the readout response."""
    p1 = config["phase1_readout"]
    qpu_name = p1["qpu"]
    sub_size = int(p1["subsystem_size"])
    num_reads = int(p1["num_reads_per_basis_state"])
    qpu_config = config["qpus"][qpu_name] if not dry_run else {"solver": "dry_run_classical"}
    solver_name = qpu_config["solver"]

    out = output_dir / "phase6_gibbs" / "phase1_readout"
    out.mkdir(parents=True, exist_ok=True)

    confusion: dict[str, dict[tuple[int, ...], float]] = {}
    rng = np.random.default_rng(0)

    # REVERSE-annealing readout calibration: use the same reverse-anneal
    # protocol as the main experiment, but with a minimal non-zero pause
    # time so the transverse field has essentially no time to mix states.
    # This matches the physical measurement protocol (unlike forward
    # annealing) so the confusion matrix is directly comparable.
    calib_s_p = float(p1.get("s_p_calibration", 0.95))  # very shallow pause
    calib_t_hold = float(p1.get("t_hold_calibration", 1.0))  # 1 μs minimal hold

    for bits in itertools.product((0, 1), repeat=sub_size):
        label = "".join(str(b) for b in bits)
        initial_state = {i: (1 if b == 0 else -1) for i, b in enumerate(bits)}
        # Trivial problem: zero fields and couplings so the initial state is
        # a degenerate minimum. The reverse-anneal return ramp should leave
        # each qubit in its prepared state modulo readout error.
        h = {i: 0.0 for i in range(sub_size)}
        J: dict[tuple[int, int], float] = {}

        if dry_run:
            samples = np.array(
                [[1 if b == 0 else -1 for b in bits]] * num_reads, dtype=np.int8,
            )
            energies = np.zeros(num_reads)
            variables = list(range(sub_size))
        else:
            from locth1.reverse_anneal import run_reverse_anneal  # noqa: WPS433
            result = run_reverse_anneal(
                h=h, J=J, initial_state=initial_state,
                s_p=calib_s_p, t_hold=calib_t_hold,
                num_reads=num_reads,
                solver=solver_name,
                label=f"readout_calib_{label}",
            )
            samples = result["samples"]
            energies = result["energies"]
            variables = result["variables"]

        raw_path = out / f"basis_{label}.h5"
        save_raw_results(
            samples=samples, energies=energies,
            metadata={"initial_state": initial_state, "solver": solver_name},
            path=raw_path,
        )
        # Use compute_marginal with the variables list so logical qubit 0..n-1
        # is extracted via the correct column mapping (EmbeddingComposite may
        # reorder columns relative to logical qubit indices).
        P = compute_marginal(
            np.asarray(samples),
            list(range(sub_size)),
            variables=list(variables),
        )
        confusion[label] = P
        target_key = tuple(target_spins := [1 if b == 0 else -1 for b in bits])
        p_correct = P.get(target_key, 0.0)
        print(f"  {label}: fidelity={p_correct:.4f}")

    out_json = out / "confusion_matrix.json"
    with open(out_json, "w") as f:
        json.dump(
            {k: {str(kk): vv for kk, vv in v.items()} for k, v in confusion.items()},
            f, indent=2,
        )
    print(f"Readout calibration complete. Saved to {out_json}")

    if not dry_run:
        try:
            base_sampler.client.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Gibbs-fit campaign driver")
    parser.add_argument("--config", default="configs/phase6_gibbs.yaml")
    parser.add_argument("--output", default="data/raw")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("calibrate_beta_eff", "Beta_eff calibration via single-qubit probes"),
        ("calibrate_readout", "Readout confusion-matrix calibration"),
        ("run_phase2_exact", "Phase 2 exact small-system QPU runs"),
        ("analyze_phase2_exact", "Phase 2 theoretical comparison"),
        ("run_phase3a_scaled", "Phase 3A scaled-Hamiltonian sweep"),
        ("analyze_phase3a_scaled", "Phase 3A theoretical comparison"),
        ("run_phase3b_bath_ensemble", "Phase 3B |S|=6 bath-ensemble QPU runs"),
        ("analyze_phase3b_bath_ensemble", "Phase 3B bath-ensemble analysis + effective H_S fit"),
        ("run_phase3c_disorder_sweep", "Phase 3C W-disorder sweep with 10 seeds"),
        ("analyze_phase3c_disorder_sweep", "Phase 3C disorder-averaged analysis"),
        ("run_phase6_robustness", "Phase 6 robustness panel (ramp + gauge)"),
        ("analyze_phase6_robustness", "Phase 6 robustness analysis"),
        ("run_phase6_frustrated", "Phase 6 frustrated (EA bond disorder) pilot"),
        ("analyze_phase6_frustrated", "Phase 6 frustrated analysis"),
    ]:
        cmd_parser = sub.add_parser(name, help=help_text)
        if name.startswith("run_") or name.startswith("calibrate_"):
            cmd_parser.add_argument("--dry-run", action="store_true",
                                    help="Skip QPU; use classical sampling instead.")
        if name == "run_phase3b_bath_ensemble":
            cmd_parser.add_argument("--scale", type=float, default=1.0,
                                    help="Global Hamiltonian scaling factor (1.0 = unscaled).")
        if name == "analyze_phase3b_bath_ensemble":
            cmd_parser.add_argument("--subdir", type=str, default="phase3b_bath_unscaled",
                                    help="Subdir under data/raw/phase6_gibbs/ to analyze.")
        if name == "run_phase3a_scaled":
            cmd_parser.add_argument("--scale", type=float, default=None,
                                    help="Override the hamiltonian_scale from config.")
        if name == "analyze_phase3a_scaled":
            cmd_parser.add_argument("--subdir", type=str, default="phase3a_scaled",
                                    help="Subdir under data/raw/phase6_gibbs/ to analyze.")
        if name in ("calibrate_beta_eff", "run_phase3c_disorder_sweep"):
            cmd_parser.add_argument("--qpu", type=str, default=None,
                                    help="Override the QPU name from config (e.g. advantage_system64).")
        if name == "analyze_phase3c_disorder_sweep":
            cmd_parser.add_argument("--subdir", type=str, default="phase3c_disorder_sweep",
                                    help="Subdir under data/raw/phase6_gibbs/ to analyze.")

    args = parser.parse_args()
    config = _load_config(args.config)
    output_dir = Path(args.output)

    if args.command == "calibrate_beta_eff":
        calibrate_beta_eff(config, output_dir, dry_run=args.dry_run, qpu_override=args.qpu)
    elif args.command == "calibrate_readout":
        calibrate_readout(config, output_dir, dry_run=args.dry_run)
    elif args.command == "run_phase2_exact":
        run_phase2_exact(config, output_dir, dry_run=args.dry_run)
    elif args.command == "analyze_phase2_exact":
        analyze_phase2_exact(config, output_dir)
    elif args.command == "run_phase3a_scaled":
        run_phase3a_scaled(config, output_dir, dry_run=args.dry_run, scale_override=args.scale)
    elif args.command == "analyze_phase3a_scaled":
        analyze_phase3a_scaled(config, output_dir, subdir=args.subdir)
    elif args.command == "run_phase3b_bath_ensemble":
        run_phase3b_bath_ensemble(config, output_dir, dry_run=args.dry_run, scale=args.scale)
    elif args.command == "analyze_phase3b_bath_ensemble":
        analyze_phase3b_bath_ensemble(config, output_dir, subdir_tag=args.subdir)
    elif args.command == "run_phase3c_disorder_sweep":
        run_phase3c_disorder_sweep(config, output_dir, dry_run=args.dry_run, qpu_override=args.qpu)
    elif args.command == "analyze_phase3c_disorder_sweep":
        analyze_phase3c_disorder_sweep(config, output_dir, subdir=args.subdir)
    elif args.command == "run_phase6_robustness":
        run_phase6_robustness(config, output_dir, dry_run=args.dry_run)
    elif args.command == "analyze_phase6_robustness":
        analyze_phase6_robustness(config, output_dir)
    elif args.command == "run_phase6_frustrated":
        run_phase6_frustrated(config, output_dir, dry_run=args.dry_run)
    elif args.command == "analyze_phase6_frustrated":
        analyze_phase6_frustrated(config, output_dir)
    elif args.command == "run_phase3_bath":
        run_phase3_bath(config, output_dir, dry_run=args.dry_run)
    elif args.command == "analyze_phase3_bath":
        analyze_phase3_bath(config, output_dir)


if __name__ == "__main__":
    main()
