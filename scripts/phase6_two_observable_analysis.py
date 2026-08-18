"""Two-observable confusion analysis, trap beta-sensitivity, and threshold robustness.

Three CPU-only analyses on the deposited phase-6 artifacts (no QPU access needed):

S1  2x2 confusion table over the 494-condition Advantage2 campaign
    (memory pass/fail x pooled-conditional-D_TV pass/fail), plus a
    per-initial-state re-analysis of the memory-retaining conditions whose
    POOLED marginal falls below the distance threshold (the "accidental
    resemblance" cell): for each, the maximum per-initial-state conditional
    D_TV, showing the pooled estimator is what aliases them.
S2  Beta-sensitivity of the wrong-basin trap verdicts: conditional D_TV of the
    two trap conditions (lam0.2 W1.0 seed42 N8, scales 0.35/0.5) against
    references at beta multipliers in [1/3, 3] -- companion to the deposited
    relaxed-subset sweep (beta_sensitivity.json), establishing that the FAIL
    verdict, like the PASS verdict, does not depend on accurate thermometry.
S4  Threshold robustness: the 2x2 classification counts as both thresholds
    (M<=0.05, D_TV<0.05) vary over a grid, showing stability away from the
    single disclosed borderline condition.

Deterministic (fixed inputs, no timing fields); output JSON is byte-identical
on re-run. Writes data/analysis/phase6/two_observable_analysis.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from locth1.analysis import compute_marginal  # noqa: E402
from locth1.gibbs_comparison import classical_conditional_marginal  # noqa: E402
from locth1.hamiltonians import build_logical_graph, build_native_graph  # noqa: E402
from locth1.observables import total_variation_distance  # noqa: E402

ROOT = Path("data/raw/phase6_gibbs")
OUT = Path("data/analysis/phase6/two_observable_analysis.json")

# Advantage2 494-condition scope (matches the main-text campaign inventory).
SCOPE = [
    "phase2_exact",
    "phase2_exact_randomreg",
    "phase3a_scale0.35",
    "phase3a_scale0.5",
    "phase3a_scale0.75",
    "phase3b_bath_scale0.5",
    "phase3b_bath_unscaled",
    "phase3c_disorder_sweep",
]
M_THRESH = 0.05
D_THRESH = 0.05
BETA_EFF = 7.219186155246177
MULTIPLIERS = [1 / 3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
INIT_LABELS = ["S_up", "S_down", "S_random"]


def load_conditions():
    rows = []
    for sub in SCOPE:
        d = json.load(open(ROOT / sub / "comparison.json"))
        for c in d.get("per_seed_conditions") or d.get("conditions") or []:
            m = c.get("memory_order_param")
            dtv = c.get("vs_classical_conditional", {}).get("d_tv")
            if m is None or dtv is None:
                continue
            rows.append({"sub": sub, "c": c, "M": m, "D": dtv})
    return rows


def confusion(rows, m_thresh, d_thresh):
    cells = {"relaxed_pass": 0, "relaxed_fail": 0, "retain_pass": 0, "retain_fail": 0}
    for r in rows:
        relaxed = r["M"] <= m_thresh
        close = r["D"] < d_thresh
        key = ("relaxed_" if relaxed else "retain_") + ("pass" if close else "fail")
        cells[key] += 1
    return cells


def h5_stem(sub, c):
    # File naming differs between sub-campaigns.
    if sub.startswith("phase3c"):
        return f"W{c['W']}_seed{c['seed']}"
    return f"lam{c['coupling_lambda']}_W{c['W']}_seed{c['seed']}_N{c['N']}"


def per_state_reanalysis(rows):
    """For memory-retaining conditions with pooled D_TV < D_THRESH: max per-state D_TV."""
    out = []
    for r in rows:
        if r["M"] <= M_THRESH or r["D"] >= D_THRESH:
            continue
        sub, c = r["sub"], r["c"]
        stem = h5_stem(sub, c)
        files = {}
        for lab in INIT_LABELS:
            hits = sorted(glob.glob(str(ROOT / sub / f"{stem}_{lab}.h5")))
            if hits:
                files[lab] = hits[0]
        if len(files) != 3:
            out.append({"sub": sub, "stem": stem, "M": round(r["M"], 4),
                        "pooled_dtv": round(r["D"], 6), "per_state_max_dtv": None,
                        "note": "raw h5 not found for all init states"})
            continue
        n = c["N"]
        scale = {"phase3a_scale0.35": 0.35, "phase3a_scale0.5": 0.5,
                 "phase3a_scale0.75": 0.75}.get(sub, 1.0)
        native = sub in ("phase2_exact", "phase3a_scale0.35",
                         "phase3a_scale0.5", "phase3a_scale0.75",
                         "phase3b_bath_scale0.5", "phase3b_bath_unscaled")
        builder = build_native_graph if native else build_logical_graph
        args = ("zephyr",) if native else ("random_regular",)
        prob = builder(*args, n, 4, c["coupling_lambda"], c["W"], c["seed"])
        h_raw, J_raw = prob["h"], prob["J"]
        S_raw, E_raw = prob["S_qubits"], prob["E_qubits"]
        allq = list(S_raw) + list(E_raw)
        mp = {q: i for i, q in enumerate(allq)}
        hh = {mp[q]: scale * v for q, v in h_raw.items() if q in mp}
        JJ = {}
        for (a, b), v in J_raw.items():
            if a in mp and b in mp:
                u, w = mp[a], mp[b]
                JJ[(min(u, w), max(u, w))] = scale * v
        P_ref = classical_conditional_marginal(
            hh, JJ, [mp[q] for q in S_raw], {mp[q]: 1 for q in E_raw}, BETA_EFF)
        dmax = 0.0
        for lab, f in files.items():
            with h5py.File(f, "r") as h:
                samples = h["samples"][:]
            P = compute_marginal(samples, [0, 1, 2, 3])
            dmax = max(dmax, total_variation_distance(P, P_ref))
        out.append({"sub": sub, "stem": stem, "M": round(r["M"], 4),
                    "pooled_dtv": round(r["D"], 6),
                    "per_state_max_dtv": round(float(dmax), 6)})
    return out


def trap_beta_sweep():
    """Conditional D_TV of the two trap conditions vs reference beta multiplier."""
    results = []
    for scale, sub in [(0.35, "phase3a_scale0.35"), (0.5, "phase3a_scale0.5")]:
        files = sorted(glob.glob(str(ROOT / sub / "lam0.2_W1.0_seed42_N8_*.h5")))
        samps = []
        for f in files:
            with h5py.File(f, "r") as h:
                samps.append(h["samples"][:])
        pooled = np.concatenate(samps, 0)
        P_meas = compute_marginal(pooled, [0, 1, 2, 3])
        prob = build_native_graph("zephyr", 8, 4, 0.2, 1.0, 42)
        h_raw, J_raw = prob["h"], prob["J"]
        S_raw, E_raw = prob["S_qubits"], prob["E_qubits"]
        allq = list(S_raw) + list(E_raw)
        mp = {q: i for i, q in enumerate(allq)}
        hh = {mp[q]: scale * v for q, v in h_raw.items() if q in mp}
        JJ = {}
        for (a, b), v in J_raw.items():
            if a in mp and b in mp:
                u, w = mp[a], mp[b]
                JJ[(min(u, w), max(u, w))] = scale * v
        row = {"scale": scale, "sweep": []}
        for mult in MULTIPLIERS:
            P_ref = classical_conditional_marginal(
                hh, JJ, [mp[q] for q in S_raw], {mp[q]: 1 for q in E_raw},
                BETA_EFF * mult)
            row["sweep"].append({
                "beta_multiplier": round(mult, 4),
                "d_tv": round(float(total_variation_distance(P_meas, P_ref)), 6),
            })
        results.append(row)
    return results


def threshold_robustness(rows):
    grid_m = [0.02, 0.05, 0.10]
    grid_d = [0.02, 0.05, 0.10]
    table = []
    for tm in grid_m:
        for td in grid_d:
            cells = confusion(rows, tm, td)
            table.append({"M_thresh": tm, "D_thresh": td, **cells})
    return table


def main():
    rows = load_conditions()
    assert len(rows) == 494, f"expected 494 in-scope conditions, got {len(rows)}"
    cells = confusion(rows, M_THRESH, D_THRESH)
    per_state = per_state_reanalysis(rows)
    reanalyzed = [r for r in per_state if r["per_state_max_dtv"] is not None]
    out = {
        "scope": "Advantage2 494-condition thermal-marginal campaign",
        "thresholds": {"M": M_THRESH, "D_TV": D_THRESH},
        "confusion_2x2": cells,
        "retain_pass_per_state_reanalysis": per_state,
        "retain_pass_caught_by_per_state": sum(
            1 for r in reanalyzed if r["per_state_max_dtv"] >= D_THRESH),
        "trap_beta_sweep": trap_beta_sweep(),
        "threshold_robustness": threshold_robustness(rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("confusion_2x2",
          "retain_pass_caught_by_per_state")}, indent=1))
    print(f"trap beta sweep: {out['trap_beta_sweep']}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
