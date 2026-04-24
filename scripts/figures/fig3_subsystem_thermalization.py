"""Figure 3: Subsystem thermalization — the Nature figure (Phase 4 data).

Panels:
  (a) P_S convergence at several |E| values
  (b) TVD vs |E| at fixed lambda
  (c) TVD vs lambda at fixed |E|
  (d) Effective temperature beta_eff consistency
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scan(data_dir: str, scan_name: str) -> dict:
    path = Path(data_dir) / "phase4" / scan_name / "primary" / "summary.json"
    with open(path) as f:
        return json.load(f)


def plot_tvd_vs_E(ax, data_dir: str):
    """Panel (b): TVD vs environment size."""
    data = load_scan(data_dir, "scan_E_size")
    E_values = [r["n_environment"] for r in data["results"]]
    tvd_values = [r["tvd_pairs"].get("S_up_vs_S_down", 0) for r in data["results"]]
    ax.plot(E_values, tvd_values, "o-", color="steelblue", lw=2, markersize=8)
    ax.set_xlabel("Environment size $|E|$")
    ax.set_ylabel("TVD($P_S^{\\uparrow}$, $P_S^{\\downarrow}$)")
    ax.set_title("(b) TVD vs $|E|$")
    ax.set_ylim(0, 1)


def plot_tvd_vs_lambda(ax, data_dir: str):
    """Panel (c): TVD vs coupling strength."""
    data = load_scan(data_dir, "scan_lambda")
    lam_values = [r["coupling_lambda"] for r in data["results"]]
    tvd_values = [r["tvd_pairs"].get("S_up_vs_S_down", 0) for r in data["results"]]
    ax.plot(lam_values, tvd_values, "s-", color="darkred", lw=2, markersize=8)
    ax.set_xlabel("Coupling strength $\\lambda$")
    ax.set_ylabel("TVD($P_S^{\\uparrow}$, $P_S^{\\downarrow}$)")
    ax.set_title("(c) TVD vs $\\lambda$")
    ax.set_ylim(0, 1)


def make_figure(data_dir: str, save_path: str | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel (a): P_S distributions — placeholder
    axes[0, 0].text(0.5, 0.5, "$P_S$ distributions\nat several $|E|$\n(awaiting data)",
                    transform=axes[0, 0].transAxes, ha="center", va="center", fontsize=12)
    axes[0, 0].set_title("(a) $P_S$ convergence")

    plot_tvd_vs_E(axes[0, 1], data_dir)
    plot_tvd_vs_lambda(axes[1, 0], data_dir)

    # Panel (d): beta_eff consistency — placeholder
    axes[1, 1].text(0.5, 0.5, "$\\beta_{\\mathrm{eff}}$ consistency\nacross initial states\n(awaiting data)",
                    transform=axes[1, 1].transAxes, ha="center", va="center", fontsize=12)
    axes[1, 1].set_title("(d) $\\beta_{\\mathrm{eff}}$ consistency")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    make_figure("data/raw", "manuscript/figures/fig3_subsystem.pdf")
