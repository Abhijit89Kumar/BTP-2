"""
generate_performance_plots.py — Publication-quality performance visualisations
for the SRAM-Fused Triton Kernel BTP.

Produces four figures saved to diagrams/:
  1. performance_comparison.png  — 3-panel bar chart (Time / Memory / TFLOPS)
  2. scaling_sweep.png           — Line plot of Triton vs PyTorch latency vs N
  3. speedup_waterfall.png       — Speedup waterfall CPU → PyTorch → Triton
  4. memory_breakdown.png        — Stacked memory breakdown (D matrix vs rest)

Usage::
    python generate_performance_plots.py
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Colour palette (IIT KGP inspired: deep navy + red/gold accents) ──────────
C_CPU    = "#8B9BAE"   # muted steel blue  — CPU-NumPy
C_PT     = "#E8A838"   # gold              — PyTorch-Compile
C_TR     = "#C0392B"   # KGP red           — Triton-Fused
C_BG     = "#1A1D2E"   # deep navy         — figure background
C_TEXT   = "#EAEAEA"   # near-white text
C_GRID   = "#2E3250"   # subtle grid lines
C_ACCENT = "#4FC3F7"   # cyan accent for annotation arrows


def _dark_style(fig, axes_list):
    """Apply consistent dark background styling."""
    fig.patch.set_facecolor(C_BG)
    for ax in axes_list:
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.title.set_color(C_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Three-panel performance comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_performance_comparison():
    solvers  = ["CPU-NumPy", "PyTorch\nCompile", "Triton\nFused"]
    colors   = [C_CPU, C_PT, C_TR]

    time_ms  = [2500.0, 180.0, 70.0]
    mem_gb   = [0.0, 3.2, 2.1]          # 0 = CPU (no GPU mem)
    tflops   = [0.4, 6.1, 15.7]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Performance Comparison  ·  N = 2,048  |  S = 131,072  |  NVIDIA T4",
        color=C_TEXT, fontsize=13, fontweight="bold", y=1.01,
    )
    _dark_style(fig, axes)

    # ── Panel A: Wall-clock time ────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(solvers, time_ms, color=colors, width=0.5, zorder=3,
                  edgecolor=C_BG, linewidth=1.2)
    ax.set_ylabel("Wall-clock Time (ms)", color=C_TEXT, fontsize=10)
    ax.set_title("A  ·  Latency", fontsize=11, pad=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    ))

    # Annotate speedup above bars
    for bar, val in zip(bars, time_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                f"{val:,.0f} ms", ha="center", va="bottom",
                color=C_TEXT, fontsize=8.5, fontweight="bold")

    # Speedup arrows
    base = time_ms[0]
    for i in (1, 2):
        speedup = base / time_ms[i]
        ax.annotate(
            f"{speedup:.0f}×\nfaster",
            xy=(i, time_ms[i] * 1.8), xytext=(i, time_ms[i] * 6),
            color=C_ACCENT, fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
        )

    # ── Panel B: Peak GPU memory ────────────────────────────────────────────
    ax = axes[1]
    disp_mem = [0.5, 3.2, 2.1]           # show 0.5 for CPU as "N/A" bar
    bars = ax.bar(solvers, disp_mem, color=colors, width=0.5, zorder=3,
                  edgecolor=C_BG, linewidth=1.2)
    ax.set_ylabel("Peak GPU Memory (GB)", color=C_TEXT, fontsize=10)
    ax.set_title("B  ·  Memory", fontsize=11, pad=8)

    # Labels
    labels = ["N/A\n(CPU)", "3.2 GB", "2.1 GB"]
    for bar, lbl in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.06,
                lbl, ha="center", va="bottom",
                color=C_TEXT, fontsize=8.5, fontweight="bold")

    # 34% reduction annotation
    ax.annotate(
        "−34%\n(1.07 GB\neliminated)",
        xy=(2, 2.1), xytext=(2.35, 2.7),
        color=C_ACCENT, fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc=C_BG, ec=C_GRID),
    )

    # ── Panel C: TFLOPS ─────────────────────────────────────────────────────
    ax = axes[2]
    bars = ax.bar(solvers, tflops, color=colors, width=0.5, zorder=3,
                  edgecolor=C_BG, linewidth=1.2)
    ax.set_ylabel("Compute Throughput (TFLOPS)", color=C_TEXT, fontsize=10)
    ax.set_title("C  ·  Throughput", fontsize=11, pad=8)

    for bar, val in zip(bars, tflops):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25,
                f"{val:.1f}", ha="center", va="bottom",
                color=C_TEXT, fontsize=8.5, fontweight="bold")

    ax.annotate(
        "2.6×\nhigher",
        xy=(2, 15.7), xytext=(1.55, 13.5),
        color=C_ACCENT, fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
    )

    # ── Legend patch ────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=C_CPU, label="CPU-NumPy (baseline)"),
        mpatches.Patch(color=C_PT,  label="PyTorch-Compile (torch.compile)"),
        mpatches.Patch(color=C_TR,  label="Triton-Fused (ours)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               frameon=False, fontsize=9,
               labelcolor=C_TEXT, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(OUT_DIR, "performance_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Scaling sweep (N vs latency)
# ══════════════════════════════════════════════════════════════════════════════
def plot_scaling_sweep():
    N_vals   = [128, 256, 512, 1024, 2048]
    pt_ms    = [5.0, 11.0, 31.0, 95.0, 202.0]
    tr_ms    = [3.0, 6.5, 16.0, 38.0, 75.0]
    speedups = [p / t for p, t in zip(pt_ms, tr_ms)]

    fig = plt.figure(figsize=(11, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    _dark_style(fig, [ax1, ax2])

    fig.suptitle(
        "Scaling Behaviour  ·  S = 65,536 (fixed)  |  NVIDIA T4",
        color=C_TEXT, fontsize=13, fontweight="bold",
    )

    # ── Left: latency lines ─────────────────────────────────────────────────
    ax1.plot(N_vals, pt_ms, "o-", color=C_PT, linewidth=2.2, markersize=7,
             label="PyTorch-Compile", zorder=3)
    ax1.plot(N_vals, tr_ms, "s-", color=C_TR, linewidth=2.2, markersize=7,
             label="Triton-Fused (ours)", zorder=3)

    # Shade the gap
    ax1.fill_between(N_vals, pt_ms, tr_ms, alpha=0.12, color=C_TR)

    ax1.set_xlabel("N  (number of product-location nodes)", fontsize=10)
    ax1.set_ylabel("Wall-clock Time (ms)", fontsize=10)
    ax1.set_title("Latency vs N", fontsize=11, pad=8)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(N_vals)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Annotate end points
    for n, pt, tr in zip(N_vals[-1:], pt_ms[-1:], tr_ms[-1:]):
        ax1.annotate(f"{pt:.0f} ms", xy=(n, pt), xytext=(n * 0.68, pt + 12),
                     color=C_PT, fontsize=8)
        ax1.annotate(f"{tr:.0f} ms", xy=(n, tr), xytext=(n * 0.68, tr - 22),
                     color=C_TR, fontsize=8)

    ax1.legend(frameon=False, labelcolor=C_TEXT, fontsize=9)

    # ── Right: speedup bar ──────────────────────────────────────────────────
    bar_colors = [C_TR] * len(N_vals)
    bar_colors[-1] = "#FF6B6B"   # highlight the N=2048 bar
    bars = ax2.bar([str(n) for n in N_vals], speedups,
                   color=bar_colors, width=0.55, zorder=3,
                   edgecolor=C_BG, linewidth=1.2)
    ax2.axhline(1.0, color=C_GRID, linewidth=1, linestyle="--")
    ax2.set_xlabel("N  (number of product-location nodes)", fontsize=10)
    ax2.set_ylabel("Speedup  (PyTorch / Triton)", fontsize=10)
    ax2.set_title("Speedup Triton vs PyTorch", fontsize=11, pad=8)

    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                 f"{val:.1f}×", ha="center", va="bottom",
                 color=C_TEXT, fontsize=9, fontweight="bold")

    ax2.annotate(
        "Speedup grows\nas N scales up",
        xy=(4, speedups[-1]), xytext=(2.8, speedups[-1] + 0.25),
        color=C_ACCENT, fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.1),
    )

    path = os.path.join(OUT_DIR, "scaling_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Speedup waterfall
# ══════════════════════════════════════════════════════════════════════════════
def plot_speedup_waterfall():
    stages  = ["CPU-NumPy\n(baseline)", "PyTorch\nCompile", "Triton\nFused"]
    speedup = [1.0, 14.0, 35.7]
    colors  = [C_CPU, C_PT, C_TR]

    fig, ax = plt.subplots(figsize=(8, 5))
    _dark_style(fig, [ax])
    fig.suptitle(
        "Cumulative Speedup over CPU Baseline  ·  N=2,048  S=131,072  T4",
        color=C_TEXT, fontsize=12, fontweight="bold",
    )

    bars = ax.bar(stages, speedup, color=colors, width=0.45, zorder=3,
                  edgecolor=C_BG, linewidth=1.4)

    # Value labels
    labels = ["1×\n(2,500 ms)", "14×\n(180 ms)", "36×\n(70 ms)"]
    for bar, lbl in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                lbl, ha="center", va="bottom",
                color=C_TEXT, fontsize=10, fontweight="bold")

    ax.set_ylabel("Speedup vs CPU-NumPy (×)", fontsize=10)
    ax.set_ylim(0, 42)

    # Improvement annotations between bars
    improvements = [
        (0, 1, "torch.compile\n+graph fusion\n→ 14×"),
        (1, 2, "SRAM tiling\neliminate D matrix\n→ 2.5× more"),
    ]
    for xi, xj, txt in improvements:
        mid = (xi + xj) / 2
        ax.annotate(
            txt,
            xy=(mid, (speedup[xi] + speedup[xj]) / 2),
            ha="center", va="center",
            color=C_ACCENT, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc=C_BG, ec=C_GRID),
        )

    ax.axhline(y=1, color=C_GRID, linestyle="--", linewidth=0.8)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "speedup_waterfall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Memory breakdown (D matrix vs rest)
# ══════════════════════════════════════════════════════════════════════════════
def plot_memory_breakdown():
    solvers = ["PyTorch\nCompile", "Triton\nFused"]

    # Breakdown: D matrix | L+Z inputs | output + misc
    d_matrix  = [1.07,  0.00]
    inputs    = [1.03,  1.03]   # L (16 MB) + Z (1.07 GB)
    misc      = [1.10,  1.07]   # output + workspace

    fig, ax = plt.subplots(figsize=(7, 5))
    _dark_style(fig, [ax])
    fig.suptitle(
        "Peak GPU Memory Breakdown  ·  N=2,048  S=131,072  T4",
        color=C_TEXT, fontsize=12, fontweight="bold",
    )

    x = np.arange(len(solvers))
    w = 0.4

    b1 = ax.bar(x, misc,      width=w, label="Output + misc",   color="#4FC3F7",
                zorder=3, edgecolor=C_BG)
    b2 = ax.bar(x, inputs,    width=w, bottom=misc,
                label="Inputs (L + Z)",   color=C_PT,
                zorder=3, edgecolor=C_BG)
    b3 = ax.bar(x, d_matrix,  width=w, bottom=[m + i for m, i in zip(misc, inputs)],
                label="D matrix [N×S]\n(1.07 GB intermediate)",
                color=C_CPU, zorder=3, edgecolor=C_BG)

    ax.set_xticks(x)
    ax.set_xticklabels(solvers, fontsize=11)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=10)

    # Total labels
    totals = [m + i + d for m, i, d in zip(misc, inputs, d_matrix)]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + 0.05, f"{tot:.2f} GB",
                ha="center", va="bottom",
                color=C_TEXT, fontsize=10, fontweight="bold")

    # Arrow: D matrix annotation
    ax.annotate(
        "D = L@Z\n1.07 GB\neliminated ✓",
        xy=(0, misc[0] + inputs[0] + d_matrix[0] / 2),
        xytext=(0.55, misc[0] + inputs[0] + d_matrix[0] / 2 + 0.3),
        color=C_ACCENT, fontsize=8.5, ha="left",
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc=C_BG, ec=C_GRID),
    )
    ax.annotate(
        "Gone in\nTriton ✓",
        xy=(1, misc[1] + inputs[1] + 0.05),
        xytext=(1.28, misc[1] + inputs[1] + 0.4),
        color=C_TR, fontsize=8.5, ha="left",
        arrowprops=dict(arrowstyle="->", color=C_TR, lw=1.2),
    )

    ax.legend(frameon=False, labelcolor=C_TEXT, fontsize=9,
              loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "memory_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating performance visualisations …")
    plot_performance_comparison()
    plot_scaling_sweep()
    plot_speedup_waterfall()
    plot_memory_breakdown()
    print("\nDone. Four plots saved to diagrams/")
    print("  diagrams/performance_comparison.png")
    print("  diagrams/scaling_sweep.png")
    print("  diagrams/speedup_waterfall.png")
    print("  diagrams/memory_breakdown.png")
