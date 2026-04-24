"""Generate Phase 7 figures: barrier crossing, schedule engineering, cross-QPU.

Run: PYTHONPATH=src .venv/bin/python scripts/figures/generate_phase7_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("manuscript/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 9,
    "axes.titlesize": 10, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.bbox": "tight",
})

C_A2 = "#2166ac"
C_S64 = "#b2182b"
C_GLAUBER = "#999999"
C_QUANTUM = "#7570b3"
C_CLASSICAL = "#d95f02"


def fig7_barrier_crossing():
    """Figure 7: Quantum-assisted barrier crossing — the headline result."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) Glauber temperature sweep vs QPU — Advantage2
    ax = axes[0]
    betas_a2 = [7.22, 6.0, 5.0, 4.0, 3.6, 3.0, 2.5, 2.0, 1.5, 1.0]
    glauber_frac_a2 = [0.10, 0.10, 0.15, 0.30, 0.25, 0.45, 0.50, 0.55, 0.75, 0.60]
    T_a2 = [1/b for b in betas_a2]

    ax.plot(T_a2, glauber_frac_a2, "o-", color=C_GLAUBER, ms=4, lw=1.2, label="Glauber")
    ax.axhline(0.70, ls="--", color=C_A2, lw=1.5, label="QPU (Adv2, 70%)")
    ax.axvline(1/7.22, ls=":", color=C_A2, lw=0.8, alpha=0.5)
    ax.annotate("$T_{\\mathrm{device}}$", xy=(1/7.22, 0.02), fontsize=6, color=C_A2, ha="center")
    ax.annotate("", xy=(1/7.22, 0.70), xytext=(1/1.5, 0.70),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(0.35, 0.73, "4.8×", fontsize=8, ha="center", fontweight="bold")
    ax.set_xlabel("Classical temperature $T$")
    ax.set_ylabel("Relaxation fraction")
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("a", fontweight="bold", loc="left")

    # (b) Same for System 6.4
    ax = axes[1]
    betas_s64 = [4.29, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]
    glauber_frac_s64 = [0.25, 0.30, 0.45, 0.50, 0.55, 0.75, 0.60]
    T_s64 = [1/b for b in betas_s64]

    ax.plot(T_s64, glauber_frac_s64, "s-", color=C_GLAUBER, ms=4, lw=1.2, label="Glauber")
    ax.axhline(0.50, ls="--", color=C_S64, lw=1.5, label="QPU (Sys6.4, 50%)")
    ax.axvline(1/4.29, ls=":", color=C_S64, lw=0.8, alpha=0.5)
    ax.annotate("$T_{\\mathrm{device}}$", xy=(1/4.29, 0.02), fontsize=6, color=C_S64, ha="center")
    ax.annotate("", xy=(1/4.29, 0.50), xytext=(1/2.5, 0.50),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(0.32, 0.53, "1.7×", fontsize=8, ha="center", fontweight="bold")
    ax.set_xlabel("Classical temperature $T$")
    ax.set_ylabel("Relaxation fraction")
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("b", fontweight="bold", loc="left")

    # (c) Landscape: barrier distribution
    ax = axes[2]
    barriers = [6.30, 1.18, 1.66, 0.66, 0.50, 0.24, 4.23, 0.06, 0.89, 3.56]
    n_minima = [11, 8, 10, 9, 9, 9, 10, 8, 12, 6]

    ax.bar(range(len(barriers)), sorted(barriers, reverse=True),
           color=C_QUANTUM, alpha=0.7, edgecolor="black", lw=0.3)
    ax.axhline(1.93, ls="--", color="black", lw=1, label=f"Mean $\\Delta$={1.93:.2f}")
    ax.set_xlabel("Instance (sorted)")
    ax.set_ylabel("Barrier height $\\Delta$")
    ax.legend(fontsize=6)
    ax.set_title("c", fontweight="bold", loc="left")

    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig7_barrier_crossing.pdf")
    fig.savefig(OUT / "fig7_barrier_crossing.png", dpi=300)
    print("Fig 7 (barrier crossing) saved.")
    plt.close(fig)


def fig8_schedule_engineering():
    """Figure 8: Schedule engineering controls quantum advantage."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # (a) Quantum inversion rate by schedule
    ax = axes[0]
    schedules = ["Baseline\n(s=0.4)", "Long-800\n(s=0.4)", "Deep\n(s=0.35)",
                 "Multi\n(.4→.3)", "Staged\nreturn", "Slow\napproach", "Oscillat."]
    inversion_rates = [17.4, 20.0, 44.4, 44.4, 12.5, 15.0, 20.8]
    relaxed_rates = [57.5, 50.0, 67.5, 67.5, 60.0, 50.0, 60.0]

    x = np.arange(len(schedules))
    bars = ax.bar(x, inversion_rates, color=[C_CLASSICAL]*2 + [C_QUANTUM]*2 + [C_CLASSICAL]*3,
                  alpha=0.8, edgecolor="black", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(schedules, fontsize=5.5)
    ax.set_ylabel("Quantum inversion rate (%)\namong relaxed instances")
    ax.axhline(17.4, ls=":", color=C_CLASSICAL, lw=0.8, alpha=0.5)
    ax.text(6.5, 19, "baseline", fontsize=5, color=C_CLASSICAL, ha="right")
    ax.set_ylim(0, 55)
    ax.set_title("a", fontweight="bold", loc="left")

    # (b) Mixed frustration: relaxation rate vs p_S
    ax = axes[1]
    pS_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    relax_rates = [53, 71, 69, 71, 64, 60]  # combined from both runs
    ax.plot(pS_vals, relax_rates, "o-", color=C_A2, ms=6, lw=1.5)
    ax.fill_between(pS_vals, relax_rates, alpha=0.1, color=C_A2)
    ax.axhline(0, ls=":", color=C_GLAUBER, lw=0.8)
    ax.text(0.25, 3, "Glauber (frozen)", fontsize=6, color=C_GLAUBER)
    ax.set_xlabel("Frustration density $p_S$")
    ax.set_ylabel("QPU relaxation rate (%)")
    ax.set_ylim(-5, 100)
    ax.set_title("b", fontweight="bold", loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig8_schedule_engineering.pdf")
    fig.savefig(OUT / "fig8_schedule_engineering.png", dpi=300)
    print("Fig 8 (schedule engineering) saved.")
    plt.close(fig)


if __name__ == "__main__":
    fig7_barrier_crossing()
    fig8_schedule_engineering()
    print(f"\nAll Phase 7 figures saved to {OUT}/")
