"""beta-sensitivity of the conditional-Boltzmann thermal-marginal claim.

Decisive due-diligence test (Zdeborova review): is the relaxed-subset
agreement D_TV(P_exp, P_th(beta_eff)) < shot-floor a real temperature
measurement, or delta-vs-delta (flat in beta)?  For every relaxed condition
in the deposited disorder-sweep campaigns we recompute the classical
conditional-Boltzmann D_TV over beta in [beta_eff/3, 3*beta_eff] (swept as a
multiplier of each condition's own beta_eff so Advantage2 and System6.4
conditions are commensurable) and aggregate the median across relaxed
conditions.

CPU-only, deposited samplesets only. Flat curve (D_TV stays below the shot
floor across a 3x beta change) => the readout does not measure temperature,
"calibrated thermal readout" must soften to "discrepancy detector".
Structured curve (clear minimum near beta_eff, rising away) => the thermal
claim has real power.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from locth1.gibbs_comparison import (  # noqa: E402
    classical_conditional_marginal,
    compute_gibbs_fit_metrics,
)

PHASE6_ROOT = Path("data/raw/phase6_gibbs")
DIRS = ["phase3c_disorder_sweep", "phase3c_disorder_sweep_advantage_system64"]
MEMORY_THRESHOLD = 0.05
# beta multipliers spanning a 3x change either side of the calibrated beta_eff
MULTIPLIERS = [1/3, 1/2.5, 1/2, 1/1.5, 1/1.25, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def _parse_J(raw: dict) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for k, v in raw.items():
        a, b = ast.literal_eval(k) if isinstance(k, str) else k
        out[(int(a), int(b))] = float(v)
    return out


def _pool_measured(cond: dict):
    per_init: dict[str, dict[tuple[int, ...], float]] = {}
    for res in cond["per_initial_state"]:
        P = {ast.literal_eval(ks): float(v) for ks, v in res["P_S"].items()}
        per_init[res["initial_state"]] = P
    pooled: dict[tuple[int, ...], float] = {}
    n = len(per_init)
    for P in per_init.values():
        for k, v in P.items():
            pooled[k] = pooled.get(k, 0.0) + v
    if n:
        pooled = {k: v / n for k, v in pooled.items()}
    return per_init, pooled


def _memory_order_param(per_init: dict) -> float:
    labels = sorted(per_init)
    best = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            keys = set(per_init[labels[i]]) | set(per_init[labels[j]])
            tvd = 0.5 * sum(
                abs(per_init[labels[i]].get(k, 0.0) - per_init[labels[j]].get(k, 0.0))
                for k in keys
            )
            best = max(best, tvd)
    return best


def main():
    # median D_TV across relaxed conditions, at each beta multiplier
    by_mult: dict[float, list[float]] = {m: [] for m in MULTIPLIERS}
    n_relaxed = 0
    n_reads_min = None

    for dirname in DIRS:
        sp = PHASE6_ROOT / dirname / "summary.json"
        if not sp.exists():
            print(f"[{dirname}] no summary.json — skip", flush=True)
            continue
        summary = json.load(open(sp))
        for cond in summary.get("conditions", []):
            h = {int(k): float(v) for k, v in cond["h"].items()}
            J = _parse_J(cond["J"])
            S_idx = list(cond["S_qubits"])
            E_state = {q: 1 for q in cond.get("E_qubits", [])}
            beta_eff = float(cond["beta_eff"])
            per_init, pooled = _pool_measured(cond)
            if _memory_order_param(per_init) > MEMORY_THRESHOLD:
                continue
            n_relaxed += 1
            reads = sum(
                sum(r.get("counts", {}).values()) if "counts" in r else 0
                for r in cond["per_initial_state"]
            )
            if reads:
                n_reads_min = reads if n_reads_min is None else min(n_reads_min, reads)
            for m in MULTIPLIERS:
                P_th = classical_conditional_marginal(h, J, S_idx, E_state,
                                                      beta_eff * m)
                dtv = compute_gibbs_fit_metrics(pooled, P_th)["d_tv"]
                by_mult[m].append(dtv)

    K = 2 ** 4  # |S|=4 disorder sweep
    nreads = n_reads_min or 6000
    shot_floor = 0.5 * np.sqrt(K / nreads)

    curve = []
    for m in MULTIPLIERS:
        v = np.array(by_mult[m])
        curve.append({
            "beta_over_beta_eff": round(m, 4),
            "median_dtv": float(np.median(v)),
            "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)),
            "max_dtv": float(np.max(v)),
            "n": int(v.size),
        })

    d_at_1 = next(c["median_dtv"] for c in curve if c["beta_over_beta_eff"] == 1.0)
    d_lo = curve[0]["median_dtv"]   # beta_eff/3
    d_hi = curve[-1]["median_dtv"]  # 3*beta_eff
    # "structured" = D_TV rises clearly (>=2x shot floor) away from beta_eff
    structured = (d_lo > 2 * shot_floor) or (d_hi > 2 * shot_floor)
    verdict = ("STRUCTURED: D_TV rises away from beta_eff -> thermal claim has "
               "real power" if structured else
               "FLAT: D_TV stays at/below shot floor across a 3x beta change "
               "-> delta-vs-delta; soften 'calibrated thermal readout' to a "
               "discrepancy-detector framing")

    out = {
        "n_relaxed_conditions": n_relaxed,
        "n_reads_per_condition_min": nreads,
        "shot_floor_dtv": round(float(shot_floor), 5),
        "curve": curve,
        "median_dtv_at_beta_eff": d_at_1,
        "median_dtv_at_beta_eff_over_3": d_lo,
        "median_dtv_at_3x_beta_eff": d_hi,
        "verdict": verdict,
    }
    outp = PHASE6_ROOT / "beta_sensitivity.json"
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\nWrote {outp}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ms = [c["beta_over_beta_eff"] for c in curve]
        md = [c["median_dtv"] for c in curve]
        q1 = [c["q25"] for c in curve]
        q3 = [c["q75"] for c in curve]
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.fill_between(ms, q1, q3, color="#9ecae1", alpha=0.5, label="IQR")
        ax.plot(ms, md, "o-", color="#2166ac", lw=1.5, ms=4,
                label="median $D_\\mathrm{TV}$")
        ax.axhline(shot_floor, ls="--", color="#888", lw=1,
                   label=f"shot floor ({shot_floor:.3f})")
        ax.axvline(1.0, ls=":", color="#b2182b", lw=1, alpha=0.7)
        ax.set_xlabel(r"$\beta / \beta_\mathrm{eff}$")
        ax.set_ylabel(r"classical conditional $D_\mathrm{TV}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=6)
        ax.set_title("Thermal-marginal $\\beta$-sensitivity (relaxed subset)",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig("manuscript/figures/fig_beta_sensitivity.pdf")
        fig.savefig("manuscript/figures/fig_beta_sensitivity.png", dpi=300)
        plt.close(fig)
        print("Wrote manuscript/figures/fig_beta_sensitivity.{pdf,png}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[plot skipped: {e}]", flush=True)


if __name__ == "__main__":
    main()
