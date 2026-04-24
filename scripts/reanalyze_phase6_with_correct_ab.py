#!/usr/bin/env python3
"""Re-analyze every Phase 6 summary.json with the corrected per-QPU A(s_p)/B(s_p).

Background
----------
The original Phase 6 campaign was run with an approximate A(s_p)/B(s_p) = 1.333
fed to ``quantum_reduced_gibbs_marginal`` / ``quantum_conditional_marginal``.
After extracting the exact schedule values from the D-Wave spreadsheets, the
true ratios at s_p=0.4 are

    Advantage2_system1     A/B = 0.2603
    Advantage_system6.4    A/B = 0.2589

(see ``data/raw/phase6_gibbs/phase1_calibration/schedules/schedule_at_sp0.4.json``).

This script:

1. Walks every ``data/raw/phase6_gibbs/*/summary.json``.
2. For each condition, recomputes ``vs_classical_conditional``,
   ``vs_classical_unconditional``, ``vs_quantum_conditional``, and
   ``vs_quantum_unconditional`` using the patched A/B field.
3. Writes a fresh ``comparison.json`` alongside each summary, preserving the
   original as ``comparison.json.bak_approx_ab1p33``.
4. Prints before/after D_TV summaries for the relaxed conditions so the
   impact of the correction is visible.

Only the "flat conditions list" summaries (phase2_exact style) are re-analyzed
here; phase3b (effective-H fit) and phase6_robustness have bespoke
post-processing that is handled by the campaign script's own
analyze_phase3b_bath_ensemble / analyze_phase6_robustness subcommands and are
skipped here.
"""
from __future__ import annotations

import ast
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from locth1.gibbs_comparison import (  # noqa: E402  # type: ignore[import-not-found]
    _compute_full_system_rho_diag,
    _trace_out_environment_diagonal,
    classical_conditional_marginal,
    classical_diagonal_marginal,
    compute_gibbs_fit_metrics,
    one_point_magnetization,
)


def _quantum_marginals_from_rho_diag(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    S_indices: list[int],
    E_state: dict[int, int],
    A_over_B: float,
    beta: float,
) -> tuple[dict, dict]:
    """Compute both conditional and unconditional quantum subsystem marginals
    from a single eigh of H(s_p)."""
    # Ensure every E qubit appears in h so _compute_full_system_rho_diag
    # infers the correct system size.
    h_full = dict(h)
    for q in E_state.keys():
        h_full.setdefault(q, 0.0)
    rho_diag, n = _compute_full_system_rho_diag(h_full, J, A_over_B, beta, S_indices)
    if n == 0:
        return {(): 1.0}, {(): 1.0}

    # Unconditional: trace out E.
    P_uncond = _trace_out_environment_diagonal(rho_diag, S_indices, n)

    # Conditional: pick indices where E qubits match E_state, then renormalize.
    n_S = len(S_indices)
    P_cond: dict[tuple[int, ...], float] = {}
    total = 0.0
    for s_int in range(2 ** n_S):
        s_bits = [(s_int >> (n_S - 1 - k)) & 1 for k in range(n_S)]
        full_bits = [0] * n
        for k, idx in enumerate(S_indices):
            full_bits[idx] = s_bits[k]
        # Fill in E qubits from E_state (spin +1 -> bit 0, spin -1 -> bit 1).
        for q, spin in E_state.items():
            full_bits[q] = 0 if spin == 1 else 1
        basis_idx = sum(b << (n - 1 - i) for i, b in enumerate(full_bits))
        p = float(rho_diag[basis_idx])
        cfg = tuple(1 - 2 * b for b in s_bits)
        P_cond[cfg] = p
        total += p
    if total > 0:
        P_cond = {k: v / total for k, v in P_cond.items()}
    return P_cond, P_uncond


PHASE6_ROOT = Path("data/raw/phase6_gibbs")
ED_CAP_N = 12
MEMORY_THRESHOLD = 0.05

# Summaries with the "conditions: [ {N,W,seed,...per_initial_state} ]" schema.
FLAT_SUMMARIES = [
    "phase2_exact",
    "phase2_exact_randomreg",
    "phase3a_scale0.35",
    "phase3a_scale0.5",
    "phase3a_scale0.75",
    "phase6_frustrated",
]

# Summaries that need their own bespoke analyzer (skip here).
SKIP = {
    "phase3b_bath_scale0.5",
    "phase3b_bath_unscaled",
    "phase3c_disorder_sweep",
    "phase3c_disorder_sweep_advantage_system64",
    "phase6_robustness",
}


def _parse_J(raw: dict[str, float]) -> dict[tuple[int, int], float]:
    J: dict[tuple[int, int], float] = {}
    for k, v in raw.items():
        k_clean = k.strip("()").replace(" ", "")
        parts = k_clean.split(",")
        J[(int(parts[0]), int(parts[1]))] = float(v)
    return J


def _shannon_entropy(P: dict[tuple[int, ...], float]) -> float:
    total = 0.0
    for p in P.values():
        if p > 1e-15:
            total -= p * math.log(p)
    return total


def _pool_measured(cond: dict[str, Any]) -> tuple[dict, dict]:
    per_init: dict[str, dict[tuple[int, ...], float]] = {}
    for res in cond["per_initial_state"]:
        P: dict[tuple[int, ...], float] = {}
        for k_str, v in res["P_S"].items():
            P[ast.literal_eval(k_str)] = float(v)
        per_init[res["initial_state"]] = P

    pooled: dict[tuple[int, ...], float] = {}
    n = len(per_init)
    for P in per_init.values():
        for k, v in P.items():
            pooled[k] = pooled.get(k, 0.0) + v
    if n > 0:
        pooled = {k: v / n for k, v in pooled.items()}
    return per_init, pooled


def _memory_order_param(per_init: dict[str, dict]) -> float:
    labels = sorted(per_init.keys())
    best = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
            tvd = 0.5 * sum(
                abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                for k in keys
            )
            if tvd > best:
                best = tvd
    return best


def reanalyze_flat(dirname: str) -> dict[str, Any] | None:
    summary_path = PHASE6_ROOT / dirname / "summary.json"
    if not summary_path.exists():
        print(f"[{dirname}] no summary.json — skipping")
        return None
    with open(summary_path) as f:
        summary = json.load(f)

    conditions = summary.get("conditions", [])
    if not conditions:
        print(f"[{dirname}] empty conditions list — skipping")
        return None

    top_ab = summary.get("A_over_B")
    legacy_ab = summary.get("A_over_B_approx_legacy")
    print(f"\n=== {dirname} ===")
    print(f"  solver={summary.get('solver')}  A/B={top_ab}  (legacy={legacy_ab})  "
          f"n_conditions={len(conditions)}")

    comparison: list[dict[str, Any]] = []
    relaxed_before: list[dict[str, Any]] = []
    for condition in conditions:
        n_total = int(condition["N"])
        W = float(condition["W"])
        seed_val = int(condition["seed"])
        lam = float(condition.get("coupling_lambda", 0.5))
        S_indices = list(condition["S_qubits"])
        E_indices_meta = list(condition.get("E_qubits", []))
        beta = float(condition["beta_eff"])
        A_over_B = float(condition["A_over_B"])  # already patched

        h = {int(k): float(v) for k, v in condition["h"].items()}
        J = _parse_J(condition["J"])

        per_init, pooled = _pool_measured(condition)
        M = _memory_order_param(per_init)
        relaxed = M <= MEMORY_THRESHOLD
        pooled_entropy = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}

        P_cl_cond = classical_conditional_marginal(h, J, S_indices, E_state, beta)
        P_cl_uncond = classical_diagonal_marginal(h, J, S_indices, beta)
        m_cl_cond = compute_gibbs_fit_metrics(pooled, P_cl_cond)
        m_cl_uncond = compute_gibbs_fit_metrics(pooled, P_cl_uncond)

        m_q_cond = None
        m_q_uncond = None
        if n_total <= ED_CAP_N:
            P_q_cond, P_q_uncond = _quantum_marginals_from_rho_diag(
                h, J, S_indices, E_state, A_over_B, beta,
            )
            m_q_cond = compute_gibbs_fit_metrics(pooled, P_q_cond)
            m_q_uncond = compute_gibbs_fit_metrics(pooled, P_q_uncond)

        entry = {
            "N": n_total,
            "W": W,
            "seed": seed_val,
            "coupling_lambda": lam,
            "beta_eff": beta,
            "A_over_B": A_over_B,
            "memory_order_param": M,
            "relaxed": relaxed,
            "pooled_entropy_nats": pooled_entropy,
            "classical_cond_entropy_nats": _shannon_entropy(P_cl_cond),
            "vs_classical_conditional": m_cl_cond,
            "vs_classical_unconditional": m_cl_uncond,
            "vs_quantum_conditional": m_q_cond,
            "vs_quantum_unconditional": m_q_uncond,
            "measured_magnetization": one_point_magnetization(pooled).tolist(),
            "classical_cond_magnetization": one_point_magnetization(P_cl_cond).tolist(),
        }
        comparison.append(entry)
        if relaxed:
            relaxed_before.append(entry)

    out_struct = {
        "beta_eff": summary.get("beta_eff"),
        "A_over_B": summary.get("A_over_B"),
        "A_over_B_approx_legacy": summary.get("A_over_B_approx_legacy"),
        "s_p": summary.get("s_p"),
        "solver": summary.get("solver"),
        "memory_threshold": MEMORY_THRESHOLD,
        "conditions": comparison,
    }
    out_path = summary_path.parent / "comparison.json"
    if out_path.exists():
        bak = out_path.with_suffix(".json.bak_approx_ab1p33")
        if not bak.exists():
            shutil.copy(out_path, bak)
    with open(out_path, "w") as f:
        json.dump(out_struct, f, indent=2)

    # Report
    n_relaxed = sum(1 for c in comparison if c["relaxed"])
    n_total_conds = len(comparison)
    print(f"  {n_relaxed}/{n_total_conds} relaxed")
    if relaxed_before:
        dtv_cl = [c["vs_classical_conditional"]["d_tv"] for c in relaxed_before]
        dtv_q_cond = [
            c["vs_quantum_conditional"]["d_tv"]
            for c in relaxed_before
            if c["vs_quantum_conditional"] is not None
        ]
        dtv_q_unc = [
            c["vs_quantum_unconditional"]["d_tv"]
            for c in relaxed_before
            if c["vs_quantum_unconditional"] is not None
        ]
        print(f"  relaxed D_TV (classical cond):   "
              f"mean={np.mean(dtv_cl):.4f}  median={np.median(dtv_cl):.4f}  max={np.max(dtv_cl):.4f}")
        if dtv_q_cond:
            print(f"  relaxed D_TV (quantum cond NEW): "
                  f"mean={np.mean(dtv_q_cond):.4f}  median={np.median(dtv_q_cond):.4f}  max={np.max(dtv_q_cond):.4f}  (n={len(dtv_q_cond)})")
        if dtv_q_unc:
            print(f"  relaxed D_TV (quantum uncond NEW):"
                  f" mean={np.mean(dtv_q_unc):.4f}  median={np.median(dtv_q_unc):.4f}  max={np.max(dtv_q_unc):.4f}  (n={len(dtv_q_unc)})")

    return out_struct


def reanalyze_phase3c(dirname: str) -> None:
    """Phase 3C has a per_seed_conditions / per_w_stats schema; reproduce it."""
    from collections import defaultdict

    summary_path = PHASE6_ROOT / dirname / "summary.json"
    if not summary_path.exists():
        print(f"[{dirname}] no summary.json — skipping")
        return
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\n=== {dirname} ===")
    conditions = summary.get("conditions", [])
    print(f"  solver={summary.get('solver')}  A/B={summary.get('A_over_B')}  n_conds={len(conditions)}")

    per_seed: list[dict[str, Any]] = []
    for condition in conditions:
        N = int(condition["N"])
        W = float(condition["W"])
        seed = int(condition["seed"])
        lam = float(condition["coupling_lambda"])
        beta = float(condition["beta_eff"])
        A_over_B = float(condition["A_over_B"])
        S_indices = list(condition["S_qubits"])
        E_indices_meta = list(condition.get("E_qubits", []))
        h = {int(k): float(v) for k, v in condition["h"].items()}
        J = _parse_J(condition["J"])

        per_init, pooled = _pool_measured(condition)
        M = _memory_order_param(per_init)
        relaxed = M <= MEMORY_THRESHOLD
        H_nats = _shannon_entropy(pooled)

        E_state = {q: 1 for q in E_indices_meta}

        P_cl = classical_conditional_marginal(h, J, S_indices, E_state, beta)
        m_cl = compute_gibbs_fit_metrics(pooled, P_cl)

        m_q_cond = None
        m_q_uncond = None
        if N <= ED_CAP_N:
            P_q, P_q_unc = _quantum_marginals_from_rho_diag(
                h, J, S_indices, E_state, A_over_B, beta,
            )
            m_q_cond = compute_gibbs_fit_metrics(pooled, P_q)
            m_q_uncond = compute_gibbs_fit_metrics(pooled, P_q_unc)

        per_seed.append({
            "N": N,
            "W": W,
            "seed": seed,
            "coupling_lambda": lam,
            "beta_eff": beta,
            "A_over_B": A_over_B,
            "memory_order_param": M,
            "relaxed": relaxed,
            "pooled_entropy_nats": H_nats,
            "vs_classical_conditional": m_cl,
            "vs_quantum_conditional": m_q_cond,
            "vs_quantum_unconditional": m_q_uncond,
        })

    # Per-W aggregate stats (include the legacy keys expected by
    # scripts/figures/fig6_thermal_marginal.py)
    by_w: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for c in per_seed:
        by_w[c["W"]].append(c)
    per_w_stats: list[dict[str, Any]] = []
    for W, group in sorted(by_w.items()):
        relaxed = [g for g in group if g["relaxed"]]
        dtvs_all = [g["vs_classical_conditional"]["d_tv"] for g in group]
        dtvs_cl = [g["vs_classical_conditional"]["d_tv"] for g in relaxed]
        dtvs_q_c = [g["vs_quantum_conditional"]["d_tv"] for g in relaxed if g["vs_quantum_conditional"] is not None]
        dtvs_q_u = [g["vs_quantum_unconditional"]["d_tv"] for g in relaxed if g["vs_quantum_unconditional"] is not None]
        Ms = [g["memory_order_param"] for g in group]
        per_w_stats.append({
            "W": W,
            "n_seeds": len(group),
            "n_total": len(group),
            "n_relaxed": len(relaxed),
            "relax_rate": float(len(relaxed) / len(group)) if group else 0.0,
            "M_mean": float(np.mean(Ms)) if Ms else None,
            "M_std": float(np.std(Ms)) if Ms else None,
            "d_tv_all_mean": float(np.mean(dtvs_all)) if dtvs_all else None,
            "d_tv_all_std": float(np.std(dtvs_all)) if dtvs_all else None,
            "d_tv_relaxed_mean": float(np.mean(dtvs_cl)) if dtvs_cl else None,
            "d_tv_relaxed_std": float(np.std(dtvs_cl)) if dtvs_cl else None,
            "mean_d_tv_classical_cond": float(np.mean(dtvs_cl)) if dtvs_cl else None,
            "median_d_tv_classical_cond": float(np.median(dtvs_cl)) if dtvs_cl else None,
            "mean_d_tv_quantum_cond": float(np.mean(dtvs_q_c)) if dtvs_q_c else None,
            "median_d_tv_quantum_cond": float(np.median(dtvs_q_c)) if dtvs_q_c else None,
            "mean_d_tv_quantum_uncond": float(np.mean(dtvs_q_u)) if dtvs_q_u else None,
            "median_d_tv_quantum_uncond": float(np.median(dtvs_q_u)) if dtvs_q_u else None,
        })

    out_struct = {
        "beta_eff": summary.get("beta_eff"),
        "A_over_B": summary.get("A_over_B"),
        "A_over_B_approx_legacy": summary.get("A_over_B_approx_legacy"),
        "s_p": summary.get("s_p"),
        "solver": summary.get("solver"),
        "memory_threshold": MEMORY_THRESHOLD,
        "per_seed_conditions": per_seed,
        "per_w_stats": per_w_stats,
    }
    out_path = summary_path.parent / "comparison.json"
    if out_path.exists():
        bak = out_path.with_suffix(".json.bak_approx_ab1p33")
        if not bak.exists():
            shutil.copy(out_path, bak)
    with open(out_path, "w") as f:
        json.dump(out_struct, f, indent=2)

    # Quick report: relaxed counts and D_TV stats per W
    for row in per_w_stats:
        print(
            f"  W={row['W']:.2f}: {row['n_relaxed']:2d}/{row['n_total']:2d} relaxed, "
            f"cl median={row['median_d_tv_classical_cond']!r}, "
            f"q_cond median={row['median_d_tv_quantum_cond']!r}, "
            f"q_unc median={row['median_d_tv_quantum_uncond']!r}"
        )


def main() -> None:
    for dirname in FLAT_SUMMARIES:
        reanalyze_flat(dirname)
    for dirname in ("phase3c_disorder_sweep", "phase3c_disorder_sweep_advantage_system64"):
        reanalyze_phase3c(dirname)


if __name__ == "__main__":
    main()
