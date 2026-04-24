"""Figure for the Phase 6 Gibbs-fit campaign (thermal marginal bridge).

Main text figure showing that in the relaxed regime, the measured subsystem
distribution matches the classical conditional Gibbs at the externally
calibrated beta_eff, independently of disorder strength.

Panel A: relaxation rate vs W (seed-averaged, 10 seeds per W)
Panel B: D_TV(measured, classical conditional Gibbs) vs W, split into
         "all seeds" (hollow) and "relaxed subset only" (filled with error bars)
Panel C: scatter of D_TV vs M across all Phase 6 conditions — shows the
         "fit fails where relaxation fails" negative control
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "phase6_gibbs"
FIG_OUT = REPO_ROOT / "manuscript" / "figures" / "fig6_thermal_marginal.pdf"
FIG_OUT_PNG = REPO_ROOT / "manuscript" / "figures" / "fig6_thermal_marginal.png"


def _load(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _collect_phase3c() -> dict:
    """Load the 10-seed disorder sweep and return per-W aggregates."""
    data = _load(DATA_ROOT / "phase3c_disorder_sweep" / "comparison.json")
    return data["per_w_stats"]


def _collect_all_conditions() -> list[dict]:
    """Aggregate all Phase 2 + Phase 3 conditions for the scatter panel."""
    conds = []
    for subdir in [
        "phase2_exact_randomreg",
        "phase2_exact",
        "phase3a_scale0.5",
        "phase3a_scale0.35",
        "phase3a_scale0.75",
        "phase3b_bath_unscaled",
        "phase3b_bath_scale0.5",
        "phase3c_disorder_sweep",
    ]:
        path = DATA_ROOT / subdir / "comparison.json"
        if not path.exists():
            continue
        try:
            data = _load(path)
        except Exception:
            continue
        entries = data.get("conditions") or data.get("per_seed_conditions") or []
        for entry in entries:
            conds.append({
                "source": subdir,
                "M": float(entry.get("memory_order_param", np.nan)),
                "H": float(entry.get("pooled_entropy_nats", np.nan)),
                "d_tv": float(entry["vs_classical_conditional"]["d_tv"])
                if entry.get("vs_classical_conditional") else np.nan,
                "relaxed": bool(entry.get("relaxed", False)),
            })
    return conds


def main() -> None:
    phase3c = _collect_phase3c()
    all_conds = _collect_all_conditions()

    Ws = np.array([s["W"] for s in phase3c])
    relax_rate = np.array([s["relax_rate"] for s in phase3c])
    relax_err = np.sqrt(relax_rate * (1 - relax_rate) / np.array([s["n_seeds"] for s in phase3c]))

    dtv_relaxed_mean = np.array([s["d_tv_relaxed_mean"] if s["d_tv_relaxed_mean"] is not None else np.nan for s in phase3c])
    dtv_relaxed_std = np.array([s["d_tv_relaxed_std"] if s["d_tv_relaxed_std"] is not None else np.nan for s in phase3c])
    dtv_all_mean = np.array([s["d_tv_all_mean"] for s in phase3c])
    dtv_all_std = np.array([s["d_tv_all_std"] for s in phase3c])

    # 3-panel figure (constrained_layout handles ylabel clipping better than tight_layout)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)

    def _panel_label(ax, letter: str, title: str) -> None:
        ax.text(-0.18, 1.04, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")
        ax.text(-0.10, 1.04, title, transform=ax.transAxes,
                fontsize=11, va="bottom", ha="left")

    # Panel A: relaxation rate vs W
    ax = axes[0]
    ax.errorbar(Ws, relax_rate, yerr=relax_err, marker="o", color="C0",
                markersize=7, capsize=3, linewidth=2)
    ax.set_xlabel("Disorder strength  $W$", fontsize=12)
    ax.set_ylabel("Relaxation rate  (fraction relaxed)", fontsize=12)
    _panel_label(ax, "a", "Disorder arrests relaxation")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.05, 1.35)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.3)

    # Panel B: D_TV vs W (relaxed subset in color, all seeds in gray)
    ax = axes[1]
    ax.errorbar(Ws, dtv_all_mean, yerr=dtv_all_std, marker="s", color="grey",
                markersize=5, capsize=2, linewidth=1, alpha=0.5,
                label="All seeds (mean ± s.d.)")
    ax.errorbar(Ws, dtv_relaxed_mean, yerr=dtv_relaxed_std, marker="o",
                color="C3", markersize=7, capsize=3, linewidth=2,
                label="Relaxed subset")
    ax.axhline(0.01, color="black", linestyle=":", linewidth=1,
               alpha=0.7, label="Shot-noise floor")
    ax.set_xlabel("Disorder strength  $W$", fontsize=12)
    ax.set_ylabel(r"$D_\mathrm{TV}$(measured, classical Gibbs at $\beta_\mathrm{eff}$)", fontsize=10)
    _panel_label(ax, "b", "Thermal-marginal agreement is W-independent")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlim(-0.05, 1.35)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3, which="both")

    # Panel C: D_TV vs M scatter across all Phase 6 conditions
    ax = axes[2]
    Ms = np.array([c["M"] for c in all_conds])
    Ds = np.array([c["d_tv"] for c in all_conds])
    rel = np.array([c["relaxed"] for c in all_conds])
    Ds_plot = np.maximum(Ds, 1e-5)
    ax.scatter(Ms[~rel], Ds_plot[~rel], s=12, alpha=0.4, color="C7",
               label=f"Memory retained ({(~rel).sum()})")
    ax.scatter(Ms[rel], Ds_plot[rel], s=16, alpha=0.75, color="C3",
               label=f"Relaxed ({rel.sum()})")
    ax.axhline(0.01, color="black", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(0.05, color="black", linestyle="--", linewidth=0.8,
               alpha=0.6, label=r"$M = 0.05$ threshold")
    ax.set_xlabel("Memory order parameter  $M$", fontsize=12)
    ax.set_ylabel(r"$D_\mathrm{TV}$  (vs classical Gibbs)", fontsize=11)
    _panel_label(ax, "c", "Fit holds iff system relaxes")
    ax.set_yscale("log")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1e-5, 1.2)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3, which="both")
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, bbox_inches="tight", dpi=200)
    plt.savefig(FIG_OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"Saved {FIG_OUT}")
    print(f"Saved {FIG_OUT_PNG}")

    # Also print key numbers
    print()
    print("=== Phase 3C summary ===")
    for s in phase3c:
        print(f"  W={s['W']:.2f}: {s['n_relaxed']}/{s['n_seeds']} relaxed, "
              f"D_TV_rel={s['d_tv_relaxed_mean']:.5f}±{s['d_tv_relaxed_std']:.5f}")
    rel_conds = [c for c in all_conds if c["relaxed"]]
    mem_conds = [c for c in all_conds if not c["relaxed"]]
    print()
    print(f"All Phase 6 conditions: {len(all_conds)}")
    print(f"  Relaxed: {len(rel_conds)}, mean D_TV = {np.mean([c['d_tv'] for c in rel_conds]):.4f}, max = {np.max([c['d_tv'] for c in rel_conds]):.4f}")
    print(f"  Memory:  {len(mem_conds)}, mean D_TV = {np.mean([c['d_tv'] for c in mem_conds]):.4f}")


if __name__ == "__main__":
    main()
