"""Figure 5: Classical baselines fail to reproduce the quantum annealer data.

Panels:
  (a) Annealer vs Glauber dynamics
  (b) Annealer vs spin-vector MC
  (c) ED agreement with annealer (small system)
  (d) Optional: Lindblad comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


def plot_annealer_vs_glauber(ax, qpu_dir: str, classical_dir: str):
    """Panel (a): memory order parameter comparison."""
    # These will be populated once both QPU and classical data exist
    ax.text(0.5, 0.5, "Annealer vs Glauber\n(awaiting data)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Memory order parameter")
    ax.set_title("(a) Annealer vs Glauber")


def plot_annealer_vs_spinvector(ax, qpu_dir: str, classical_dir: str):
    """Panel (b): crossover comparison."""
    ax.text(0.5, 0.5, "Annealer vs spin-vector MC\n(awaiting data)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_xlabel("Control parameter")
    ax.set_ylabel("Memory order parameter")
    ax.set_title("(b) Annealer vs spin-vector MC")


def plot_ed_agreement(ax, qpu_dir: str, classical_dir: str):
    """Panel (c): ED and annealer agree for small systems."""
    ax.text(0.5, 0.5, "ED vs annealer\n(small system)\n(awaiting data)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("TVD (ED vs annealer)")
    ax.set_title("(c) ED agreement")


def plot_lindblad_comparison(ax, classical_dir: str):
    """Panel (d): Lindblad Markovian bath is insufficient."""
    ax.text(0.5, 0.5, "Lindblad comparison\n(optional)\n(awaiting data)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Memory order parameter")
    ax.set_title("(d) Lindblad comparison")


def make_figure(qpu_dir: str, classical_dir: str, save_path: str | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_annealer_vs_glauber(axes[0, 0], qpu_dir, classical_dir)
    plot_annealer_vs_spinvector(axes[0, 1], qpu_dir, classical_dir)
    plot_ed_agreement(axes[1, 0], qpu_dir, classical_dir)
    plot_lindblad_comparison(axes[1, 1], classical_dir)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    make_figure("data/raw", "data/classical", "manuscript/figures/fig5_baselines.pdf")
