"""Figure 1: Protocol schematic and subsystem-environment concept.

Panels:
  (a) Reverse-anneal schedule: s=1 → s_p → hold t_p → s=1
  (b) QPU topology with S highlighted, E surrounding, boundary couplers
  (c) Cartoon: different initial states converge to same P_S
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_anneal_schedule(ax, s_p=0.4, t_ramp=5.0, t_hold=100.0):
    """Panel (a): reverse-anneal schedule diagram."""
    t = [0, t_ramp, t_ramp + t_hold, 2 * t_ramp + t_hold]
    s = [1.0, s_p, s_p, 1.0]
    ax.plot(t, s, "k-", lw=2)
    ax.fill_between(t, s, alpha=0.1, color="blue")
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Anneal parameter $s$")
    ax.set_ylim(0, 1.1)
    ax.axhline(s_p, ls="--", color="gray", alpha=0.5)
    ax.annotate("$s_p$", xy=(t_ramp + t_hold / 2, s_p), fontsize=12,
                ha="center", va="bottom")
    ax.annotate("$t_p$", xy=(t_ramp + t_hold / 2, s_p - 0.05),
                fontsize=11, ha="center", va="top", color="blue")
    ax.set_title("(a) Reverse-anneal protocol")


def plot_subsystem_partition(ax):
    """Panel (b): schematic S-E partition on QPU topology."""
    # Simple graph layout
    rng = np.random.default_rng(42)
    n_s, n_e = 6, 30
    # S qubits in a tight cluster
    s_pos = rng.normal(0, 0.3, size=(n_s, 2))
    # E qubits spread around
    angles = np.linspace(0, 2 * np.pi, n_e, endpoint=False)
    e_pos = np.column_stack([1.5 * np.cos(angles), 1.5 * np.sin(angles)])
    e_pos += rng.normal(0, 0.2, size=e_pos.shape)

    ax.scatter(*e_pos.T, c="lightblue", s=40, edgecolors="steelblue", zorder=2, label="$E$")
    ax.scatter(*s_pos.T, c="tomato", s=80, edgecolors="darkred", zorder=3, label="$S$")

    # Draw some edges
    for i in range(n_s):
        for j in range(i + 1, n_s):
            if rng.random() < 0.5:
                ax.plot([s_pos[i, 0], s_pos[j, 0]], [s_pos[i, 1], s_pos[j, 1]],
                        "darkred", alpha=0.3, lw=0.8, zorder=1)
    # S-E boundary edges
    for i in range(n_s):
        for j in range(min(4, n_e)):
            idx = rng.integers(n_e)
            ax.plot([s_pos[i, 0], e_pos[idx, 0]], [s_pos[i, 1], e_pos[idx, 1]],
                    "purple", alpha=0.2, lw=0.6, ls="--", zorder=1)

    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) Subsystem $S$ + environment $E$")


def plot_convergence_cartoon(ax):
    """Panel (c): cartoon showing P_S convergence across initial states."""
    x = np.arange(8)
    # Different "initial" distributions
    rng = np.random.default_rng(0)
    dists = [rng.dirichlet(np.ones(8) * 0.5) for _ in range(3)]
    # "Converged" distribution
    gibbs = np.exp(-np.arange(8) * 0.5)
    gibbs /= gibbs.sum()

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels_init = ["Init 1", "Init 2", "Init 3"]

    width = 0.2
    for i, (d, c, lab) in enumerate(zip(dists, colors, labels_init)):
        ax.bar(x + i * width - 0.3, d, width, color=c, alpha=0.4, label=f"{lab} (before)")
    ax.bar(x + 0.5, gibbs, 0.3, color="black", alpha=0.7, label="Converged $P_S$")

    ax.set_xlabel("Configuration $\\sigma_S$")
    ax.set_ylabel("$P_S(\\sigma_S)$")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("(c) Initial-state independence")


def make_figure(save_path: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_anneal_schedule(axes[0])
    plot_subsystem_partition(axes[1])
    plot_convergence_cartoon(axes[2])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    make_figure("manuscript/figures/fig1_protocol.pdf")
