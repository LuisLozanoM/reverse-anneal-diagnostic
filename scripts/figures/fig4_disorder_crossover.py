"""Figure 4: Disorder arrests thermalization + crossover diagram (Phase 4 data).

Panels:
  (a) TVD vs W at fixed |E| and lambda
  (b) Crossover diagram in (lambda, W) plane
  (c) Same crossover on both QPUs — universality
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_scan(data_dir: str, scan_name: str, qpu: str = "primary") -> dict:
    path = Path(data_dir) / "phase4" / scan_name / qpu / "summary.json"
    with open(path) as f:
        return json.load(f)


def plot_tvd_vs_W(ax, data_dir: str):
    """Panel (a): TVD vs disorder strength."""
    data = load_scan(data_dir, "scan_W_subsystem")
    W_values = [r["disorder_W"] for r in data["results"]]
    memory_values = [r["memory_order_param"] for r in data["results"]]
    ax.plot(W_values, memory_values, "o-", color="purple", lw=2, markersize=8)
    ax.set_xlabel("Disorder strength $W$")
    ax.set_ylabel("Memory order parameter")
    ax.set_title("(a) Disorder preserves memory")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)


def plot_crossover_diagram(ax, data_dir: str, qpu: str = "primary"):
    """Panel (b/c): 2D crossover heatmap."""
    data = load_scan(data_dir, "scan_crossover", qpu)
    lam_vals = data["lambda_values"]
    W_vals = data["W_values"]
    n_lam = len(lam_vals)
    n_W = len(W_vals)

    memory_grid = np.zeros((n_W, n_lam))
    for r in data["results"]:
        i_lam = lam_vals.index(r["coupling_lambda"])
        i_W = W_vals.index(r["disorder_W"])
        memory_grid[i_W, i_lam] = r["memory_order_param"]

    im = ax.imshow(
        memory_grid, origin="lower", aspect="auto",
        extent=[min(lam_vals), max(lam_vals), min(W_vals), max(W_vals)],
        cmap="RdYlBu_r", vmin=0, vmax=1,
    )
    ax.contour(
        lam_vals, W_vals, memory_grid, levels=[0.5],
        colors="black", linewidths=2,
    )
    ax.set_xlabel("Coupling $\\lambda$")
    ax.set_ylabel("Disorder $W$")
    plt.colorbar(im, ax=ax, label="Memory order parameter")


def make_figure(data_dir: str, save_path: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_tvd_vs_W(axes[0], data_dir)

    plot_crossover_diagram(axes[1], data_dir, qpu="primary")
    axes[1].set_title("(b) Crossover — Advantage2")

    plot_crossover_diagram(axes[2], data_dir, qpu="secondary")
    axes[2].set_title("(c) Crossover — Advantage_system6.4")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    make_figure("data/raw", "manuscript/figures/fig4_crossover.pdf")
