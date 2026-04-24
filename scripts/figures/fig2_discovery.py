"""Figure 2: Global memory loss in clean systems (Phase 3 data).

Panels:
  (a) TVD vs t_p at several s_p values
  (b) Gibbs-fit quality vs t_p
  (c) Memory order parameter vs system size N
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scan(scan_dir: str, scan_name: str) -> dict:
    path = Path(scan_dir) / "phase3" / scan_name / "primary" / "summary.json"
    with open(path) as f:
        return json.load(f)


def plot_tvd_vs_tp(ax, data_dir: str):
    """Panel (a): TVD between all-up and all-down vs pause time."""
    data = load_scan(data_dir, "scan_tp")
    t_values = [r["t_hold"] for r in data["results"]]
    tvd_values = [r["tvd_pairs"].get("all_up_vs_all_down", 0) for r in data["results"]]
    ax.semilogx(t_values, tvd_values, "o-", color="steelblue", lw=2)
    ax.set_xlabel("Pause time $t_p$ (μs)")
    ax.set_ylabel("TVD (all-up vs all-down)")
    ax.set_title("(a) Memory decay with pause time")
    ax.set_ylim(0, 1)


def plot_memory_vs_N(ax, data_dir: str):
    """Panel (c): memory order parameter vs system size."""
    data = load_scan(data_dir, "scan_N")
    N_values = [r["n_qubits"] for r in data["results"]]
    memory_values = [r["memory_order_param"] for r in data["results"]]
    ax.plot(N_values, memory_values, "s-", color="darkred", lw=2)
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Memory order parameter")
    ax.set_title("(c) Size dependence")
    ax.set_ylim(0, 1)


def make_figure(data_dir: str, save_path: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_tvd_vs_tp(axes[0], data_dir)
    # Panel (b): Gibbs fit quality — placeholder until data exists
    axes[1].text(0.5, 0.5, "Gibbs-fit quality\nvs $t_p$\n(awaiting data)",
                 transform=axes[1].transAxes, ha="center", va="center", fontsize=12)
    axes[1].set_title("(b) Gibbs-fit convergence")
    plot_memory_vs_N(axes[2], data_dir)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    make_figure("data/raw", "manuscript/figures/fig2_discovery.pdf")
