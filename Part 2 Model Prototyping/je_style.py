"""Shared visual style for the JustEat photography experiment design deck."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK    = "#16232e"
MUTED  = "#8a97a3"
GRID   = "#e2e7eb"
PAPER  = "#ffffff"
ORANGE = "#e0632b"
BLUE   = "#2f6f95"
GREEN  = "#3f8f6d"
RED    = "#b03a34"
GOLD   = "#c99a2e"
PURPLE = "#6b5b95"
LIGHT  = "#f4f6f8"

PALETTE = [ORANGE, BLUE, GREEN, GOLD, PURPLE, RED]


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "Carlito",
        "font.size": 10.5,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def strip(ax, x=True, y=False):
    """Remove grid on one axis for cleaner bar charts."""
    ax.grid(axis="x" if not x else "y", visible=True)
    ax.grid(axis="x" if x else "y", visible=False)


def caption(fig, text, y=-0.02):
    fig.text(0.005, y, text, ha="left", va="top", fontsize=9, color=MUTED, wrap=True)
