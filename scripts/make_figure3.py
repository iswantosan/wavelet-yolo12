"""Generate Figure 3 (WaveDown block diagram) for the wavelet-YOLO paper.

Horizontal (landscape) layout — fits single-column page width without large
vertical real-estate. Black & white styling matches Figure 1/2 conventions.

Usage:
    python scripts/make_figure3.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "figure3_wavedown_block.png"

# Three-shade black & white styling for hierarchy
COL_DWT = "#ffffff"
COL_PROJ_LL = "#ffffff"
COL_PROJ_HF = "#e8e8e8"
COL_FUSE = "#d0d0d0"
COL_IO = "#ffffff"
EDGE = "#000"


def box(ax, x, y, w, h, text, fc, fontsize=10, weight="normal",
        linestyle="-", linewidth=1.2):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=linewidth, edgecolor=EDGE,
        facecolor=fc, linestyle=linestyle,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight)


def arrow(ax, x1, y1, x2, y2):
    ar = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color=EDGE,
    )
    ax.add_patch(ar)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 5), dpi=200)
    ax.set_xlim(-0.2, 18.0)
    ax.set_ylim(-0.3, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")

    Y_MID = 2.5      # main flow line
    Y_TOP = 4.0      # LL branch
    Y_BOT = 1.0      # HF branch

    # ---- Input ----
    box(ax, 0.0, Y_MID - 0.4, 1.5, 0.8, "Input  $X$", COL_IO,
        fontsize=11, weight="bold", linewidth=1.6)
    arrow(ax, 1.5, Y_MID, 2.1, Y_MID)

    # ---- DWT ----
    box(ax, 2.1, Y_MID - 0.55, 2.3, 1.1,
        "Haar 2D DWT\n(depthwise\nstride-2 conv)",
        COL_DWT, fontsize=9.5)

    # ---- split arrows ----
    # to top (LL)
    arrow(ax, 4.4, Y_MID + 0.3, 5.4, Y_TOP)
    ax.text(4.7, Y_MID + 0.95, "$S_{LL}$",
            fontsize=10, style="italic")
    # to bottom (HF group)
    arrow(ax, 4.4, Y_MID - 0.3, 5.0, Y_BOT + 0.55)

    # ---- Top branch: W_LL ----
    box(ax, 5.4, Y_TOP - 0.45, 2.4, 0.9,
        "1×1 conv  $W_{LL}$\n$C \\to c_2$",
        COL_PROJ_LL, fontsize=10)
    arrow(ax, 7.8, Y_TOP, 10.5, Y_MID + 0.2)
    ax.text(9.0, Y_TOP - 0.3, "$Y_{LL}$",
            fontsize=10, style="italic")

    # ---- Bottom branch: S_HF concat box + W_HF (zero-init, dashed) ----
    box(ax, 5.0, Y_BOT - 0.05, 2.0, 0.9,
        "$S_{LH}, S_{HL}, S_{HH}$\nconcat $\\to$ $S_{HF}$",
        COL_PROJ_HF, fontsize=9)
    arrow(ax, 7.0, Y_BOT + 0.4, 7.2, Y_BOT + 0.4)
    box(ax, 7.2, Y_BOT - 0.2, 2.5, 1.1,
        "1×1 conv  $W_{HF}$\n$3C \\to c_2$\n(zero-initialised)",
        COL_PROJ_HF, fontsize=9, linestyle="--")
    arrow(ax, 9.7, Y_BOT + 0.35, 10.5, Y_MID - 0.2)
    ax.text(9.85, Y_BOT - 0.5, "$Y_{HF}$",
            fontsize=10, style="italic")

    # ---- Fusion: concat (2c2) ----
    box(ax, 10.5, Y_MID - 0.45, 1.7, 0.9,
        "concat\n($2c_2$ ch.)",
        COL_FUSE, fontsize=9.5)
    arrow(ax, 12.2, Y_MID, 12.6, Y_MID)

    # ---- W_F ----
    box(ax, 12.6, Y_MID - 0.45, 1.6, 0.9,
        "1×1 conv  $W_F$\n$2c_2 \\to c_2$",
        COL_FUSE, fontsize=9.5)
    arrow(ax, 14.2, Y_MID, 14.6, Y_MID)

    # ---- BN + SiLU ----
    box(ax, 14.6, Y_MID - 0.4, 1.1, 0.8,
        "BN\n$\\to$ SiLU",
        COL_FUSE, fontsize=9.5)
    arrow(ax, 15.7, Y_MID, 16.1, Y_MID)

    # ---- Output ----
    box(ax, 16.1, Y_MID - 0.4, 1.5, 0.8,
        "Output  $Y$", COL_IO,
        fontsize=11, weight="bold", linewidth=1.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
