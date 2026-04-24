#!/usr/bin/env python3
"""Summarize the impact of the A(s_p)/B(s_p) correction on Phase 6 D_TV metrics.

The original Phase 6 quantum-Gibbs comparisons were computed with an
approximate ratio A(s_p)/B(s_p) = 1.333, while the exact ratios at s_p = 0.4
extracted from the D-Wave schedule spreadsheets are

    Advantage2_system1     0.2603
    Advantage_system6.4    0.2589

This script compares every comparison.json against its
comparison.json.bak_approx_ab1p33 twin and prints before/after D_TV statistics
for the quantum conditional and quantum unconditional predictions (the
classical prediction is A/B-independent and is included only as a reference).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("data/raw/phase6_gibbs")
TRUE_AB = {
    "Advantage2_system1": 0.2602641898790776,
    "Advantage_system6.4": 0.258917042998926,
}


def _fmt(values: list[float]) -> str:
    if not values:
        return "(none)"
    a = np.array(values, dtype=float)
    return (
        f"n={len(a):2d}  median={np.median(a):7.4f}  "
        f"mean={np.mean(a):7.4f}  max={np.max(a):7.4f}"
    )


def _extract(cmp: dict) -> list[dict]:
    return cmp.get("conditions") or cmp.get("per_seed_conditions") or []


def _dtv(metric_dict: dict | None) -> float | None:
    if metric_dict is None:
        return None
    return metric_dict.get("d_tv")


def summarize(dirname: str, label: str) -> None:
    p_new = ROOT / dirname / "comparison.json"
    p_old = ROOT / dirname / "comparison.json.bak_approx_ab1p33"
    if not p_new.exists():
        return
    with open(p_new) as f:
        new = json.load(f)
    old = None
    if p_old.exists():
        with open(p_old) as f:
            old = json.load(f)

    solver = new.get("solver", "?")
    new_ab = new.get("A_over_B")

    new_conds = _extract(new)
    old_conds = _extract(old or {}) if old else []

    # Index old by (N, W, seed, lambda) if N is present, else (W, seed)
    def key_of(c: dict) -> tuple:
        if "N" in c:
            return (c["N"], c["W"], c["seed"], c.get("coupling_lambda"))
        return (c["W"], c["seed"])

    old_by_key = {key_of(c): c for c in old_conds}

    cl_new: list[float] = []
    q_cond_old: list[float] = []
    q_cond_new: list[float] = []
    q_unc_old: list[float] = []
    q_unc_new: list[float] = []
    for c in new_conds:
        if not c.get("relaxed"):
            continue
        cl_d = _dtv(c.get("vs_classical_conditional"))
        q_cond_d = _dtv(c.get("vs_quantum_conditional"))
        q_unc_d = _dtv(c.get("vs_quantum_unconditional"))
        if cl_d is not None:
            cl_new.append(cl_d)
        if q_cond_d is not None:
            q_cond_new.append(q_cond_d)
        if q_unc_d is not None:
            q_unc_new.append(q_unc_d)

        c_old = old_by_key.get(key_of(c))
        if c_old:
            q_cond_old_d = _dtv(c_old.get("vs_quantum_conditional"))
            q_unc_old_d = _dtv(c_old.get("vs_quantum_unconditional"))
            if q_cond_old_d is not None:
                q_cond_old.append(q_cond_old_d)
            if q_unc_old_d is not None:
                q_unc_old.append(q_unc_old_d)

    n_relaxed = sum(1 for c in new_conds if c.get("relaxed"))
    print(f"\n### {label}  ({dirname})")
    print(f"    solver={solver}  A/B(new)={new_ab:.4f}  relaxed={n_relaxed}/{len(new_conds)}")
    print(f"    classical cond (A/B-indep): {_fmt(cl_new)}")
    if q_cond_old:
        print(f"    quantum cond  OLD A/B=1.33: {_fmt(q_cond_old)}")
    print(f"    quantum cond  NEW A/B={new_ab:.2f}: {_fmt(q_cond_new)}")
    if q_unc_old:
        print(f"    quantum uncond OLD A/B=1.33: {_fmt(q_unc_old)}")
    print(f"    quantum uncond NEW A/B={new_ab:.2f}: {_fmt(q_unc_new)}")


def main() -> None:
    sources = [
        ("phase2_exact", "Phase 2 native Zephyr  |S|=4, 84 conds"),
        ("phase2_exact_randomreg", "Phase 2 random-regular  |S|=4, 84 conds"),
        ("phase3a_scale0.35", "Phase 3A scale=0.35"),
        ("phase3a_scale0.5", "Phase 3A scale=0.5"),
        ("phase3a_scale0.75", "Phase 3A scale=0.75"),
        ("phase6_frustrated", "Phase 6 frustrated (EA bond disorder)"),
        ("phase3c_disorder_sweep", "Phase 3C disorder sweep (Advantage2)"),
        ("phase3c_disorder_sweep_advantage_system64", "Phase 3C disorder sweep (System6.4)"),
    ]
    print("=" * 80)
    print("Phase 6 Gibbs-fit  A(s_p)/B(s_p) correction — D_TV before/after")
    print("=" * 80)
    for dirname, label in sources:
        summarize(dirname, label)


if __name__ == "__main__":
    main()
