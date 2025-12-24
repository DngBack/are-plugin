"""
Plotting utilities for L2R Plugin scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_rc_dual(
    r: np.ndarray,
    e_bal: np.ndarray,
    e_wst: np.ndarray,
    aurc_bal: float,
    aurc_wst: float,
    out_path: Path,
):
    """Plot both balanced and worst-group RC curves."""
    plt.figure(figsize=(7, 5))
    plt.plot(r, e_bal, "o-", color="green", label=f"Balanced (AURC={aurc_bal:.4f})")
    plt.plot(r, e_wst, "s-", color="royalblue", label=f"Worst-group (AURC={aurc_wst:.4f})")
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Error")
    plt.title("Balanced and Worst-group Error vs Rejection Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    ymax = 0.0
    if e_bal.size:
        ymax = max(ymax, float(e_bal.max()))
    if e_wst.size:
        ymax = max(ymax, float(e_wst.max()))
    plt.ylim([0, min(1.05, ymax * 1.1 if ymax > 0 else 1.0)])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rc(
    r: np.ndarray,
    ew: np.ndarray,
    eb: np.ndarray,
    aw: float,
    ab: float,
    out_path: Path,
):
    """Plot worst-group and balanced RC curves."""
    plt.figure(figsize=(7, 5))
    plt.plot(r, ew, "o-", label=f"Worst-group (AURC={aw:.4f})", color="royalblue")
    plt.plot(r, eb, "s-", label=f"Balanced (AURC={ab:.4f})", color="green")
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Error")
    plt.title("Worst-group and Balanced Error vs Rejection Rate")
    plt.xlim([0, 0.8])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tail_head_gap(
    r: np.ndarray,
    gap: np.ndarray,
    out_path: Path,
    title: str = "Tail-Head Error Gap vs Rejection Rate",
):
    """Plot tail-head error gap curve."""
    plt.figure(figsize=(7, 5))
    plt.plot(r, gap, "d-", color="crimson", label="Tail - Head error")
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Tail Error - Head Error")
    plt.title(title)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 0.8])
    ymin = float(min(0.0, gap.min() if gap.size else 0.0))
    ymax = float(max(0.0, gap.max() if gap.size else 0.0))
    pad = 0.05 * (ymax - ymin + 1e-8)
    plt.ylim([ymin - pad, ymax + pad])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

