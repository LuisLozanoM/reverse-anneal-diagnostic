"""Generate all 5 paper figures from experimental data.

Run: PYTHONPATH=src .venv/bin/python scripts/figures/generate_all_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT = Path("manuscript/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Nature style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_A2 = "#2166ac"    # blue for Advantage2
C_S64 = "#b2182b"   # red for system6.4
C_GLAUBER = "#999999"  # gray for classical
C_ED = "#4daf4a"    # green for ED


# ============================================================
# FIGURE 1: Protocol + concept (schematic)
# ============================================================
def fig1_protocol():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))

    # (a) Anneal schedule
    ax = axes[0]
    s_p, t_r, t_h = 0.4, 5, 100
    t = [0, t_r, t_r + t_h, 2 * t_r + t_h]
    s = [1.0, s_p, s_p, 1.0]
    ax.plot(t, s, "k-", lw=1.5)
    ax.fill_between(t, s, alpha=0.08, color="steelblue")
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Anneal parameter $s$")
    ax.set_ylim(0, 1.1)
    ax.axhline(s_p, ls=":", color="gray", lw=0.5)
    ax.text(t_r + t_h / 2, s_p + 0.04, "$s_p$", ha="center", fontsize=8)
    ax.text(t_r + t_h / 2, s_p - 0.08, "$t_p$", ha="center", fontsize=8, color=C_A2)
    ax.set_title("a", fontweight="bold", loc="left")

    # (b) S-E partition schematic
    ax = axes[1]
    rng = np.random.default_rng(42)
    n_s, n_e = 6, 24
    s_pos = rng.normal(0, 0.25, (n_s, 2))
    angles = np.linspace(0, 2 * np.pi, n_e, endpoint=False)
    e_pos = np.column_stack([1.3 * np.cos(angles), 1.3 * np.sin(angles)])
    e_pos += rng.normal(0, 0.15, e_pos.shape)
    ax.scatter(*e_pos.T, c="lightsteelblue", s=25, edgecolors=C_A2, lw=0.4, zorder=2)
    ax.scatter(*s_pos.T, c="salmon", s=50, edgecolors="darkred", lw=0.6, zorder=3)
    for i in range(n_s):
        for j in range(min(3, n_e)):
            idx = rng.integers(n_e)
            ax.plot([s_pos[i, 0], e_pos[idx, 0]], [s_pos[i, 1], e_pos[idx, 1]],
                    color="purple", alpha=0.15, lw=0.5, ls="--", zorder=1)
    ax.text(0, 0, "$S$", ha="center", va="center", fontsize=11, fontweight="bold", color="darkred")
    ax.text(1.1, 1.1, "$E$", fontsize=11, fontweight="bold", color=C_A2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("b", fontweight="bold", loc="left")

    # (c) Convergence cartoon
    ax = axes[2]
    rng2 = np.random.default_rng(7)
    x = np.arange(8)
    gibbs = np.exp(-np.arange(8) * 0.6)
    gibbs /= gibbs.sum()
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["$|{\\uparrow}\\rangle$", "$|{\\downarrow}\\rangle$", "Random"]
    w = 0.18
    for i, (c, lab) in enumerate(zip(colors, labels)):
        d = rng2.dirichlet(np.ones(8) * 0.4)
        ax.bar(x + (i - 1) * w, d, w, color=c, alpha=0.35, label=f"{lab} (before)")
    ax.bar(x + 1.5 * w, gibbs, w * 1.2, color="black", alpha=0.6, label="Converged $P_S$")
    ax.set_xlabel("Configuration $\\sigma_S$")
    ax.set_ylabel("$P_S$")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_title("c", fontweight="bold", loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig1_protocol.pdf")
    fig.savefig(OUT / "fig1_protocol.png", dpi=300)
    print("Fig 1 saved.")
    plt.close(fig)


# ============================================================
# FIGURE 2: Discovery — global memory + ED baseline
# ============================================================
def fig2_discovery():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))

    # (a) ED: memory vs t_p at different s_p
    ax = axes[0]
    ed_04 = {0.1: 0.5558, 0.5: 0.4676, 1.0: 0.4902, 5.0: 0.4544}
    ed_05 = {0.1: 0.9752, 0.5: 0.9802, 1.0: 0.9676, 5.0: 0.9456}
    tp = list(ed_04.keys())
    ax.semilogx(tp, [ed_04[t] for t in tp], "o-", color=C_A2, ms=4, lw=1.2, label="$s_p=0.4$ (strong $H_D$)")
    ax.semilogx(tp, [ed_05[t] for t in tp], "s-", color=C_S64, ms=4, lw=1.2, label="$s_p=0.5$ (weak $H_D$)")
    ax.set_xlabel("Pause time $t_p$ (μs)")
    ax.set_ylabel("Memory order parameter")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=6)
    ax.set_title("a", fontweight="bold", loc="left")

    # (b) ED: memory vs system size
    ax = axes[1]
    N_vals = [8, 10, 12]
    mem_04_05 = [0.4676, 0.3130, 0.1342]  # t_p=0.5us, s_p=0.4
    mem_05_05 = [0.9802, 0.9866, 0.9863]  # t_p=0.5us, s_p=0.5
    ax.plot(N_vals, mem_04_05, "o-", color=C_A2, ms=5, lw=1.2, label="$s_p=0.4$")
    ax.plot(N_vals, mem_05_05, "s-", color=C_S64, ms=5, lw=1.2, label="$s_p=0.5$")
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Memory order parameter")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=6)
    ax.set_title("b", fontweight="bold", loc="left")

    # (c) Glauber baseline
    ax = axes[2]
    N_glauber = [8, 16, 32, 50]
    mem_glauber = [1.0, 1.0, 1.0, 1.0]
    ax.plot(N_glauber, mem_glauber, "D-", color=C_GLAUBER, ms=5, lw=1.2, label="Glauber ($T_\\mathrm{device}$)")
    ax.plot(N_vals, mem_04_05, "o-", color=C_ED, ms=5, lw=1.2, label="ED ($s_p=0.4$)")
    ax.axhline(0, ls=":", color="gray", lw=0.5)
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Memory order parameter")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=6)
    ax.set_title("c", fontweight="bold", loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig2_discovery.pdf")
    fig.savefig(OUT / "fig2_discovery.png", dpi=300)
    print("Fig 2 saved.")
    plt.close(fig)


# ============================================================
# FIGURE 3: Subsystem thermalization (THE NATURE FIGURE)
# ============================================================
def fig3_subsystem():
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # (a) Memory vs |E| — both QPUs
    ax = fig.add_subplot(gs[0, 0])
    E_vals = [4, 8, 16, 32, 50]
    mem_a2 = [1.0, 0.005, 0.001, 0.0, 0.0]
    mem_s64 = [0.995, 0.170, 0.007, 0.004, 0.005]
    ax.plot(E_vals, mem_a2, "o-", color=C_A2, ms=5, lw=1.5, label="Advantage2")
    ax.plot(E_vals, mem_s64, "s--", color=C_S64, ms=5, lw=1.5, label="System 6.4")
    ax.axhline(1.0, ls=":", color=C_GLAUBER, lw=0.8, label="Glauber (classical)")
    ax.set_xlabel("Environment size $|E|$")
    ax.set_ylabel("Memory order parameter")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=6)
    ax.set_title("a", fontweight="bold", loc="left")

    # (b) Memory vs lambda — both QPUs
    # Legacy: embedded submission, uniform_torque_compensation, auto_scale=True (solid).
    # Overlay: regenerated native-qubit submission with auto_scale=False (dashed open markers).
    ax = fig.add_subplot(gs[0, 1])
    lam = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    mem_lam_a2 = [0.014, 0.939, 0.800, 0.532, 0.001, 0.0, 0.0, 0.0]
    mem_lam_s64 = [0.129, 0.370, 0.015, 0.011, 0.002, 0.004, 0.007, 0.001]
    # Regenerated native-qubit, auto_scale=False
    lam_ref = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    mem_lam_a2_ref = [0.998, 0.934, 0.101, 0.000, 0.000, 0.000, 0.000, 0.000]
    mem_lam_s64_ref = [0.562, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000]
    ax.plot(lam, mem_lam_a2, "o-", color=C_A2, ms=5, lw=1.5, label="Advantage2 (legacy)")
    ax.plot(lam, mem_lam_s64, "s-", color=C_S64, ms=5, lw=1.5, label="System 6.4 (legacy)")
    ax.plot(lam_ref, mem_lam_a2_ref, "o--", color=C_A2, ms=5, lw=1.2,
            mfc="white", mew=1.2, label="Adv2 ($a_s{=}$F)")
    ax.plot(lam_ref, mem_lam_s64_ref, "s--", color=C_S64, ms=5, lw=1.2,
            mfc="white", mew=1.2, label="Sys6.4 ($a_s{=}$F)")
    ax.axhline(1.0, ls=":", color=C_GLAUBER, lw=0.8)
    ax.set_xlabel("Coupling strength $\\lambda$")
    ax.set_ylabel("Memory order parameter")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_title("b", fontweight="bold", loc="left")

    # (c) s_p sweep — crossover shifts with pause depth
    ax = fig.add_subplot(gs[1, 0])
    test_lam = [0.1, 0.15, 0.2, 0.3, 0.5]
    a2_sweep = {
        0.35: [0.004, 0.001, 0.002, 0.001, 0.000],
        0.40: [0.404, 0.005, 0.009, 0.000, 0.000],
        0.45: [0.940, 0.810, 0.347, 0.000, 0.000],
        0.50: [1.000, 0.995, 0.966, 0.024, 0.002],
    }
    cmap = plt.cm.Blues
    for i, (sp, mem) in enumerate(sorted(a2_sweep.items())):
        c = cmap(0.4 + 0.15 * i)
        ax.plot(test_lam, mem, "o-", color=c, ms=4, lw=1.2, label=f"$s_p={sp}$")
    ax.set_xlabel("Coupling strength $\\lambda$")
    ax.set_ylabel("Memory (Advantage2)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=5.5, ncol=2)
    ax.set_title("c", fontweight="bold", loc="left")

    # (d) s_p sweep — system6.4 matches same shape
    ax = fig.add_subplot(gs[1, 1])
    s64_sweep = {
        0.30: [0.084, 0.031, 0.007, 0.018, 0.003],
        0.35: [0.164, 0.015, 0.009, 0.007, 0.005],
        0.40: [0.115, 0.011, 0.007, 0.008, 0.004],
        0.45: [0.630, 0.601, 0.193, 0.029, 0.004],
        0.50: [0.931, 0.977, 0.885, 0.647, 0.000],
    }
    cmap_r = plt.cm.Reds
    for i, (sp, mem) in enumerate(sorted(s64_sweep.items())):
        c = cmap_r(0.3 + 0.14 * i)
        ax.plot(test_lam, mem, "s-", color=c, ms=4, lw=1.2, label=f"$s_p={sp}$")
    ax.set_xlabel("Coupling strength $\\lambda$")
    ax.set_ylabel("Memory (System 6.4)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=5.5, ncol=2)
    ax.set_title("d", fontweight="bold", loc="left")

    fig.savefig(OUT / "fig3_subsystem.pdf")
    fig.savefig(OUT / "fig3_subsystem.png", dpi=300)
    print("Fig 3 saved.")
    plt.close(fig)


# ============================================================
# FIGURE 4: Disorder arrest + crossover diagram
# ============================================================
def fig4_disorder():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) Memory vs W — both QPUs (legacy solid).  Overlay: single-seed
    # reproduction at uniform_torque_compensation chain strength on Sys6.4.
    ax = axes[0]
    W = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    mem_W_a2 = [0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.002, 0.743]
    mem_W_s64 = [0.005, 0.003, 0.006, 0.004, 0.003, 0.011, 0.498, 0.046]
    W_utc = [0.0, 0.5, 1.0, 1.5, 2.0]
    mem_W_s64_utc = [0.002, 0.002, 0.006, 0.541, 0.010]  # single-seed UTC reproduction
    ax.plot(W, mem_W_a2, "o-", color=C_A2, ms=5, lw=1.5, label="Advantage2 (legacy)")
    ax.plot(W, mem_W_s64, "s-", color=C_S64, ms=5, lw=1.5, label="System 6.4 (legacy)")
    ax.plot(W_utc, mem_W_s64_utc, "^:", color=C_S64, ms=5, lw=1.2,
            mfc="white", mew=1.2, label="Sys6.4 (UTC reproduction, single seed)")
    ax.set_xlabel("Disorder strength $W$")
    ax.set_ylabel("Memory order parameter")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(fontsize=5.5, loc="upper left")
    ax.set_title("a", fontweight="bold", loc="left")

    # (b) Crossover heatmap — Advantage2
    ax = axes[1]
    lam_g = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    W_g = [0.0, 0.5, 1.0, 1.5, 1.7, 1.9, 2.0, 2.5, 3.0]
    grid_data = [
        [0.534, 0.828, 0.369, 0.178, 0.088, 0.190, 0.050, 0.183, 0.077],
        [0.842, 0.530, 0.078, 0.880, 0.025, 0.129, 0.305, 0.091, 0.033],
        [0.011, 0.945, 0.004, 0.013, 0.151, 0.366, 0.096, 0.229, 0.048],
        [0.009, 0.001, 0.000, 0.012, 0.049, 0.122, 0.756, 0.279, 0.064],
        [0.000, 0.001, 0.001, 0.012, 0.002, 0.010, 0.524, 0.190, 0.294],
        [0.000, 0.001, 0.001, 0.001, 0.006, 0.051, 0.053, 0.080, 0.024],
        [0.000, 0.000, 0.000, 0.002, 0.002, 0.027, 0.025, 0.173, 0.070],
        [0.000, 0.000, 0.000, 0.000, 0.004, 0.007, 0.165, 0.041, 0.057],
    ]
    grid = np.array(grid_data)
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[W_g[0], W_g[-1], lam_g[0], lam_g[-1]],
                   cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="bilinear")
    cs = ax.contour(W_g, lam_g, grid, levels=[0.1, 0.5], colors=["white", "black"], linewidths=[0.8, 1.2])
    ax.clabel(cs, fmt="%.1f", fontsize=6)
    ax.set_xlabel("Disorder $W$")
    ax.set_ylabel("Coupling $\\lambda$")
    plt.colorbar(im, ax=ax, label="Memory", shrink=0.9)
    ax.set_title("b", fontweight="bold", loc="left")

    # (c) β_eff comparison
    ax = axes[2]
    beta_a2, beta_s64 = 6.927, 4.461
    ax.bar(["Adv2\n$s_p{=}0.4$", "Sys6.4\n$s_p{=}0.4$"], [beta_a2, beta_s64],
           color=[C_A2, C_S64], width=0.5, edgecolor="black", lw=0.5)
    ax.set_ylabel("$\\beta_{\\mathrm{eff}} = B(s_p)R / 2k_BT$")
    ax.set_title("c", fontweight="bold", loc="left")

    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig4_disorder.pdf")
    fig.savefig(OUT / "fig4_disorder.png", dpi=300)
    print("Fig 4 saved.")
    plt.close(fig)


# ============================================================
# FIGURE 5: Classical baselines fail
# ============================================================
def fig5_baselines():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) QPU vs Glauber vs ED — memory across system sizes
    ax = axes[0]
    N_glauber = [8, 16, 32, 50]
    N_ed = [8, 10, 12]
    mem_glauber = [1.0, 1.0, 1.0, 1.0]
    mem_ed = [0.4676, 0.3130, 0.1342]
    # QPU subsystem at |E|=50 shows memory~0
    ax.plot(N_glauber, mem_glauber, "D-", color=C_GLAUBER, ms=6, lw=1.5, label="Glauber dynamics")
    ax.plot(N_ed, mem_ed, "^-", color=C_ED, ms=6, lw=1.5, label="Exact diag. ($|S|{=}4$)")
    ax.axhspan(-0.05, 0.05, color=C_A2, alpha=0.15, label="QPU ($|E|{\\geq}16$)")
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Memory order parameter")
    ax.set_ylim(-0.1, 1.15)
    ax.legend(fontsize=6, loc="center right")
    ax.set_title("a", fontweight="bold", loc="left")

    # (b) ED memory vs s_p — confirms transverse field drives thermalization
    ax = axes[1]
    sp_vals = [0.4, 0.5]
    mem_N8 = [0.4676, 0.9802]
    mem_N10 = [0.3130, 0.9866]
    mem_N12 = [0.1342, 0.9863]
    ax.plot(sp_vals, mem_N8, "o-", color="#66c2a5", ms=5, lw=1.2, label="$N=8$")
    ax.plot(sp_vals, mem_N10, "s-", color="#fc8d62", ms=5, lw=1.2, label="$N=10$")
    ax.plot(sp_vals, mem_N12, "^-", color="#8da0cb", ms=5, lw=1.2, label="$N=12$")
    ax.set_xlabel("Pause point $s_p$")
    ax.set_ylabel("Memory (ED, $t_p=0.5\\,\\mu$s)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([0.4, 0.5])
    ax.legend(fontsize=6)
    ax.set_title("b", fontweight="bold", loc="left")

    # (c) Two-parameter universality: β_eff · λ_c for both QPUs at different s_p
    ax = axes[2]
    # From s_p sweep: approximate crossover lambda at each s_p
    a2_sp = [0.40, 0.45, 0.50]
    a2_lam_c = [0.10, 0.20, 0.30]  # approximate from data
    s64_sp = [0.45, 0.50]
    s64_lam_c = [0.18, 0.35]  # approximate from data

    ax.plot(a2_sp, a2_lam_c, "o-", color=C_A2, ms=6, lw=1.5, label="Advantage2")
    ax.plot(s64_sp, s64_lam_c, "s--", color=C_S64, ms=6, lw=1.5, label="System 6.4")
    ax.set_xlabel("Pause point $s_p$")
    ax.set_ylabel("Crossover $\\lambda_c$")
    ax.legend(fontsize=6)
    ax.set_title("c", fontweight="bold", loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig5_baselines.pdf")
    fig.savefig(OUT / "fig5_baselines.png", dpi=300)
    print("Fig 5 saved.")
    plt.close(fig)


# ============================================================
# Generate all
# ============================================================
if __name__ == "__main__":
    fig1_protocol()
    fig2_discovery()
    fig3_subsystem()
    fig4_disorder()
    fig5_baselines()
    print(f"\nAll figures saved to {OUT}/")
