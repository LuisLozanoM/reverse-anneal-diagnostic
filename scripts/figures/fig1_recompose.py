"""Compose Fig. 1 from the three existing sub-panels with correct panel labels.

The hand-made composite fig1_mockup_combined.png had panel labels that
disagreed with the main.tex caption (caption: a=landscape, b=schedule,
c=topology; old mockup rendered: a=schedule, b=topology, landscape
unlabeled).  This script re-composes the three sub-panel PDFs in the
correct order with LaTeX-rendered `a`, `b`, `c` labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread

FIG_DIR = Path(__file__).resolve().parents[2] / "manuscript" / "figures"
PANELS = [
    ("a", FIG_DIR / "fig1a_header_energy_landscape.png", "Energy landscape"),
    ("b", FIG_DIR / "fig1b_reverse_anneal_schedule.png", "Reverse-anneal schedule"),
    ("c", FIG_DIR / "fig1c_qpu_topology_SE_partition.png", "QPU topology"),
]


def make_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.5), constrained_layout=True)
    for ax, (label, img_path, title) in zip(axes, PANELS):
        img = imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            -0.02, 1.02, label,
            transform=ax.transAxes,
            fontsize=18, fontweight="bold",
            ha="right", va="bottom",
        )
        ax.set_title(title, fontsize=11, loc="left", pad=2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    make_figure(FIG_DIR / "fig1_combined.png")
