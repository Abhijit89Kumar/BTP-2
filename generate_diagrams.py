"""
generate_diagrams.py — Publication-quality architecture diagrams for the
                        Multi-Echelon Stochastic Newsvendor BTP.

Generates 6 diagrams into ``diagrams/``:

    1. system_architecture.png   — High-level module dependency graph
    2. data_pipeline_flow.png    — ETL stages from raw data to GPU tensors
    3. triton_kernel_grid.png    — 2-D grid layout and tile decomposition
    4. memory_hierarchy.png      — HBM vs SRAM: PyTorch baseline vs Triton
    5. newsvendor_math_flow.png  — Mathematical computation stages
    6. benchmark_flow.png        — Solver comparison and validation pipeline

Usage::

    python generate_diagrams.py          # writes to diagrams/*.png
    python generate_diagrams.py --dpi 300  # thesis-quality DPI
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import matplotlib.patheffects as pe
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (consistent across all diagrams)
# ---------------------------------------------------------------------------
C = {
    "bg":          "#FAFBFC",
    "config":      "#E8F0FE",    # light blue
    "pipeline":    "#E6F4EA",    # light green
    "baseline":    "#FEF7E0",    # light yellow
    "triton":      "#FCE4EC",    # light red/pink
    "benchmark":   "#F3E5F5",    # light purple
    "sram":        "#FF6F00",    # deep orange
    "hbm":         "#1565C0",    # blue
    "accent":      "#D32F2F",    # red accent
    "text":        "#212121",
    "textlight":   "#616161",
    "border":      "#90A4AE",
    "arrow":       "#455A64",
    "gridtile":    "#BBDEFB",
    "gridhighlight":"#FF8A65",
    "white":       "#FFFFFF",
    "black":       "#212121",
    "green":       "#2E7D32",
    "gold":        "#F9A825",
}

OUT_DIR = Path("diagrams")


def _setup_ax(ax, title: str = "") -> None:
    """Common axis cleanup."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15, color=C["text"])


def _rounded_box(ax, xy, w, h, color, text, fontsize=9, text_color=None,
                  edgecolor=None, lw=1.5, alpha=0.95, bold=False, radius=0.15):
    """Draw a rounded rectangle with centered text."""
    tc = text_color or C["text"]
    ec = edgecolor or C["border"]
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=tc, fontweight=weight, zorder=3,
            linespacing=1.4)


def _arrow(ax, start, end, color=None, style="-|>", lw=1.5, connectionstyle="arc3,rad=0.0"):
    """Draw an arrow between two points."""
    c = color or C["arrow"]
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=c, lw=lw,
        connectionstyle=connectionstyle,
        mutation_scale=15,
        zorder=4,
    )
    ax.add_patch(arrow)


def _label(ax, x, y, text, fontsize=8, color=None, ha="center", va="center",
           bold=False, rotation=0):
    """Place a text label."""
    c = color or C["textlight"]
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=c,
            fontweight=weight, rotation=rotation, zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# Diagram 1 — System Architecture
# ═══════════════════════════════════════════════════════════════════════════
def diagram_system_architecture(dpi: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=dpi)
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("System Architecture — Module Dependency Graph",
                 fontsize=16, fontweight="bold", pad=18, color=C["text"])

    # --- config.py (top center) ---
    _rounded_box(ax, (4, 7.5), 3.5, 1.2, C["config"],
                 "config.py\n─────────────────\nNewsvendorConfig\nFinancialParams\nTritonTuningConfig",
                 fontsize=9, bold=False)
    _label(ax, 5.75, 8.9, "CONFIGURATION", fontsize=8, bold=True, color=C["hbm"])

    # --- data_pipeline.py (left middle) ---
    _rounded_box(ax, (0.3, 4.2), 4.2, 2.2, C["pipeline"],
                 "data_pipeline.py\n──────────────────────\nM5TopologyExtractor\nDemandDistributionMapper\nFinancialTensorGenerator\nDataPipeline  →  TensorBundle",
                 fontsize=8.5)
    _label(ax, 2.4, 6.6, "DATA / ETL", fontsize=8, bold=True, color=C["green"])

    # --- baseline_solvers.py (right middle) ---
    _rounded_box(ax, (6.3, 4.5), 4.5, 1.7, C["baseline"],
                 "baseline_solvers.py\n──────────────────────────\nCPUMonteCarlo (NumPy)\nPyTorchMonteCarlo (torch.compile)\nSolverResult",
                 fontsize=8.5)
    _label(ax, 8.55, 6.4, "BASELINES", fontsize=8, bold=True, color="#E65100")

    # --- triton_fused_newsvendor.py (center bottom) ---
    _rounded_box(ax, (0.5, 1.0), 5.0, 2.0, C["triton"],
                 "triton_fused_newsvendor.py\n────────────────────────────────\n@triton.autotune + @triton.jit\n_fused_newsvendor_kernel\nTritonFusedNewsvendor",
                 fontsize=8.5)
    _label(ax, 3.0, 3.2, "CORE INNOVATION  ★", fontsize=8, bold=True, color=C["accent"])

    # --- benchmark.py (right bottom) ---
    _rounded_box(ax, (6.3, 1.0), 4.5, 2.0, C["benchmark"],
                 "benchmark.py\n─────────────────────────\nBenchmarkSuite\ncheck_correctness()\nprint_results_table()\nrun_triton_perf_sweep()",
                 fontsize=8.5)
    _label(ax, 8.55, 3.2, "BENCHMARKING", fontsize=8, bold=True, color="#6A1B9A")

    # --- Arrows ---
    # config → pipeline
    _arrow(ax, (4.2, 7.5), (2.4, 6.45), lw=2, connectionstyle="arc3,rad=0.15")
    # config → baselines
    _arrow(ax, (7.3, 7.5), (8.55, 6.25), lw=2, connectionstyle="arc3,rad=-0.15")
    # pipeline → benchmark
    _arrow(ax, (4.1, 4.5), (6.5, 2.8), lw=2, connectionstyle="arc3,rad=-0.2")
    # pipeline → triton
    _arrow(ax, (2.4, 4.2), (2.7, 3.05), lw=2)
    # baselines → benchmark
    _arrow(ax, (8.55, 4.5), (8.55, 3.05), lw=2)
    # triton → benchmark
    _arrow(ax, (5.2, 1.8), (6.3, 1.8), lw=2)

    # --- Legend ---
    _label(ax, 5.75, 0.1,
           "Arrows show import / data-flow dependencies.  TensorBundle is the universal data contract.",
           fontsize=7.5, color=C["textlight"])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "system_architecture.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [1/6] system_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════
# Diagram 2 — Data Pipeline Flow
# ═══════════════════════════════════════════════════════════════════════════
def diagram_data_pipeline(dpi: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=dpi)
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Data Pipeline — ETL Flow from Raw Datasets to GPU Tensors",
                 fontsize=15, fontweight="bold", pad=15, color=C["text"])

    # Stage boxes (left to right)
    # --- Raw data sources ---
    _rounded_box(ax, (0, 4.0), 2.5, 1.8, "#E3F2FD",
                 "M5 Forecasting\nDataset\n(Kaggle)\n────────\nSpatial/temporal\ncorrelation Σ",
                 fontsize=7.5)
    _label(ax, 1.25, 6.0, "DATA SOURCE 1", fontsize=7, bold=True, color=C["hbm"])

    _rounded_box(ax, (0, 1.2), 2.5, 1.8, "#E8F5E9",
                 "Tractor / Generator\nSales Time-Series\n(Public)\n────────\nDemand μ, σ\nper category",
                 fontsize=7.5)
    _label(ax, 1.25, 3.2, "DATA SOURCE 2", fontsize=7, bold=True, color=C["green"])

    # --- Stage 1: Topology ---
    _rounded_box(ax, (3.5, 4.0), 2.5, 1.8, C["pipeline"],
                 "M5Topology\nExtractor\n────────────\nCorrelation R [N×N]\nSpectral regularise\nCholesky: L = chol(Σ)",
                 fontsize=7.5)
    _label(ax, 4.75, 6.0, "STAGE 1", fontsize=7, bold=True, color=C["green"])

    # --- Stage 2: Demand ---
    _rounded_box(ax, (3.5, 1.2), 2.5, 1.8, C["pipeline"],
                 "Demand\nDistribution\nMapper\n────────────\nμ[N], σ[N]\n+ spare-parts (gen)",
                 fontsize=7.5)
    _label(ax, 4.75, 3.2, "STAGE 2", fontsize=7, bold=True, color=C["green"])

    # --- Stage 3: Financial ---
    _rounded_box(ax, (7.0, 2.6), 2.5, 1.8, C["pipeline"],
                 "Financial\nTensor Generator\n────────────────\np[N], c[N], s[N]\nMargin constraints:\np > 1.15c,  s < 0.25c",
                 fontsize=7.5)
    _label(ax, 8.25, 4.6, "STAGE 3", fontsize=7, bold=True, color=C["green"])

    # --- Output: TensorBundle ---
    _rounded_box(ax, (10.5, 1.8), 3.0, 3.0, "#FFF3E0",
                 "TensorBundle\n(GPU-resident)\n══════════════\nL   [N, N]    — Cholesky\nμ   [N, 1]    — mean\np   [N, 1]    — price\nc   [N, 1]    — cost\ns   [N, 1]    — salvage\nQ   [N, 1]    — order qty\nZ   [N, S]    — scenarios",
                 fontsize=7.5, edgecolor=C["sram"], lw=2.5)
    _label(ax, 12.0, 5.0, "OUTPUT", fontsize=7, bold=True, color=C["sram"])

    # --- Arrows ---
    _arrow(ax, (2.5, 4.9), (3.5, 4.9), lw=2)
    _arrow(ax, (2.5, 2.1), (3.5, 2.1), lw=2)
    _arrow(ax, (6.0, 4.9), (7.2, 4.4), lw=2, connectionstyle="arc3,rad=0.15")
    _arrow(ax, (6.0, 2.1), (7.0, 2.9), lw=2, connectionstyle="arc3,rad=-0.1")
    _arrow(ax, (9.5, 3.5), (10.5, 3.3), lw=2)
    # Cholesky path
    _arrow(ax, (5.9, 4.5), (10.5, 3.8), lw=1.5, color=C["sram"],
           connectionstyle="arc3,rad=0.2")

    # --- Z note ---
    _label(ax, 12.0, 1.4, "Z ~ N(0, 1) generated\non GPU via torch.randn",
           fontsize=7, color=C["textlight"])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_pipeline_flow.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [2/6] data_pipeline_flow.png")


# ═══════════════════════════════════════════════════════════════════════════
# Diagram 3 — Triton Kernel Grid Layout
# ═══════════════════════════════════════════════════════════════════════════
def diagram_triton_grid(dpi: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(13, 8.5), dpi=dpi)
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1.5, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Triton Kernel — 2-D Grid Layout & Tiled Execution",
                 fontsize=15, fontweight="bold", pad=15, color=C["text"])

    # ---- Left: L matrix ----
    _label(ax, 1.2, 8.2, "L  [N × K]", fontsize=10, bold=True, color=C["hbm"])
    _label(ax, 1.2, 7.7, "(Cholesky factor)", fontsize=8, color=C["textlight"])
    # Draw grid for L
    l_x0, l_y0 = 0, 3.0
    l_w, l_h = 2.4, 4.2
    # Full matrix outline
    ax.add_patch(plt.Rectangle((l_x0, l_y0), l_w, l_h,
                               fill=True, facecolor="#E3F2FD", edgecolor=C["hbm"],
                               lw=2, zorder=1))
    # Highlighted BLOCK_M × BLOCK_K tile
    tile_y = l_y0 + l_h - 1.2
    ax.add_patch(plt.Rectangle((l_x0, tile_y), 0.8, 1.0,
                               fill=True, facecolor=C["gridhighlight"],
                               edgecolor=C["accent"], lw=2.5, zorder=2, alpha=0.8))
    _label(ax, l_x0 + 0.4, tile_y + 0.5, "BM\n×\nBK", fontsize=7, bold=True,
           color=C["white"])

    # Dimension labels
    _label(ax, l_x0 + l_w / 2, l_y0 - 0.3, "K (= N)", fontsize=8, bold=True,
           color=C["hbm"])
    _label(ax, l_x0 - 0.4, l_y0 + l_h / 2, "N", fontsize=8, bold=True,
           color=C["hbm"], rotation=90)

    # ---- Middle: Z matrix ----
    _label(ax, 5.8, 8.2, "Z  [K × S]", fontsize=10, bold=True, color=C["hbm"])
    _label(ax, 5.8, 7.7, "(Scenario matrix)", fontsize=8, color=C["textlight"])
    z_x0, z_y0 = 3.5, 3.0
    z_w, z_h = 5.0, 4.2
    ax.add_patch(plt.Rectangle((z_x0, z_y0), z_w, z_h,
                               fill=True, facecolor="#E8EAF6", edgecolor=C["hbm"],
                               lw=2, zorder=1))
    # Highlighted BLOCK_K × BLOCK_N tile
    ax.add_patch(plt.Rectangle((z_x0, z_y0 + z_h - 0.6), 1.4, 0.6,
                               fill=True, facecolor=C["gridhighlight"],
                               edgecolor=C["accent"], lw=2.5, zorder=2, alpha=0.8))
    _label(ax, z_x0 + 0.7, z_y0 + z_h - 0.3, "BK × BN", fontsize=7,
           bold=True, color=C["white"])

    _label(ax, z_x0 + z_w / 2, z_y0 - 0.3, "S (scenarios)", fontsize=8,
           bold=True, color=C["hbm"])
    _label(ax, z_x0 - 0.4, z_y0 + z_h / 2, "K", fontsize=8, bold=True,
           color=C["hbm"], rotation=90)

    # ---- Right: Output grid (virtual profit matrix, never materialised) ----
    _label(ax, 10.5, 8.2, "Virtual Profit [N × S]", fontsize=10, bold=True,
           color=C["accent"])
    _label(ax, 10.5, 7.7, "(NEVER in HBM — fused in SRAM)", fontsize=8,
           color=C["accent"])
    o_x0, o_y0 = 9.2, 3.0
    o_w, o_h = 3.0, 4.2
    # Draw grid tiles
    n_rows, n_cols = 6, 5
    tw = o_w / n_cols
    th = o_h / n_rows
    for r in range(n_rows):
        for cc in range(n_cols):
            color = C["gridtile"]
            lw_t = 0.5
            if r == 0 and cc == 0:
                color = C["gridhighlight"]
                lw_t = 2
            ax.add_patch(plt.Rectangle(
                (o_x0 + cc * tw, o_y0 + (n_rows - 1 - r) * th), tw, th,
                fill=True, facecolor=color, edgecolor=C["border"],
                lw=lw_t, zorder=2, alpha=0.75))

    # Label the highlighted tile
    _label(ax, o_x0 + tw / 2, o_y0 + o_h - th / 2, "pid\n(0,0)",
           fontsize=6.5, bold=True, color=C["black"])

    # Grid dimensions
    _label(ax, o_x0 + o_w / 2, o_y0 - 0.3, "⌈S / BN⌉ tiles", fontsize=8,
           bold=True, color=C["accent"])
    _label(ax, o_x0 - 0.5, o_y0 + o_h / 2, "⌈N / BM⌉\ntiles", fontsize=8,
           bold=True, color=C["accent"], rotation=90)

    # ---- Arrows ----
    _arrow(ax, (2.4, 5.8), (3.5, 6.3), lw=2, color=C["sram"],
           connectionstyle="arc3,rad=0.1")
    _arrow(ax, (5.0, 6.7), (5.0, 7.2), lw=0)  # spacer

    # ---- Bottom: K-loop detail ----
    _rounded_box(ax, (0.5, 0.3), 11.5, 2.0, "#FFF8E1",
                 "K-LOOP  (per program instance)\n"
                 "────────────────────────────────────────────────────────────────────────────────────────\n"
                 "for k in range(⌈K / BK⌉):                                       # iterate K dimension\n"
                 "    L_tile = tl.load(L[m_block, k:k+BK])          # [BM, BK] → SRAM\n"
                 "    Z_tile = tl.load(Z[k:k+BK, n_block])           # [BK, BN] → SRAM\n"
                 "    acc += tl.dot(L_tile, Z_tile)                          # FMA in SRAM registers",
                 fontsize=7.5, edgecolor=C["sram"], lw=2)

    # ---- Final output arrow ----
    _label(ax, 10.7, 2.7, "→ out[N]", fontsize=9, bold=True, color=C["green"])
    _label(ax, 10.7, 2.3, "(atomic_add partial mean)", fontsize=7.5,
           color=C["textlight"])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "triton_kernel_grid.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [3/6] triton_kernel_grid.png")


# ═══════════════════════════════════════════════════════════════════════════
# Diagram 4 — Memory Hierarchy Comparison
# ═══════════════════════════════════════════════════════════════════════════
def diagram_memory_hierarchy(dpi: int) -> None:
    """
    Visual memory schematic — box heights are proportional to data sizes.
    Left: PyTorch — D matrix materialised in HBM (large red block).
    Right: Triton  — everything stays in SRAM, only 8 KB leaves.
    """
    RED   = "#C62828"
    RED_L = "#FFEBEE"
    BLUE  = "#1565C0"
    BLUE_L= "#E3F2FD"
    GRN   = "#2E7D32"
    GRN_L = "#E8F5E9"
    GRN_M = "#A5D6A7"
    ORG   = "#E65100"
    ORG_L = "#FFF3E0"
    GREY  = "#546E7A"
    WHITE = "#FFFFFF"
    BG    = "#FAFBFC"

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.patch.set_facecolor(BG)

    # Title
    fig.text(0.5, 0.97, "Memory Hierarchy — PyTorch  vs  Triton Fused Kernel",
             ha="center", va="top", fontsize=17, fontweight="bold", color="#212121")

    # Vertical divider
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.03, 0.93],
                              transform=fig.transFigure,
                              color="#B0BEC5", lw=1.5, linestyle="--"))

    # Column headers
    fig.text(0.25, 0.92, "PyTorch  +  torch.compile",
             ha="center", fontsize=13, fontweight="bold", color=BLUE)
    fig.text(0.75, 0.92, "Triton Fused Kernel  ★  (This Work)",
             ha="center", fontsize=13, fontweight="bold", color=GRN)

    # ── helper: axes-independent rect on figure (in axes coords via add_axes) ──
    # We draw each panel in its own axes, x∈[0,1], y∈[0,1]
    ax_l = fig.add_axes([0.02, 0.04, 0.46, 0.86])
    ax_r = fig.add_axes([0.52, 0.04, 0.46, 0.86])
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_facecolor(BG)

    def rect(ax, x, y, w, h, fc, ec, lw=1.5, zorder=2, alpha=1.0):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.01",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            zorder=zorder, alpha=alpha, transform=ax.transData,
        ))

    def label(ax, x, y, s, size=10, color="#212121", bold=False,
              ha="center", va="center", zorder=6):
        ax.text(x, y, s, ha=ha, va=va, fontsize=size, zorder=zorder,
                color=color, fontweight="bold" if bold else "normal",
                transform=ax.transData, clip_on=False, linespacing=1.35)

    def arrow(ax, x0, y0, x1, y1, color, lw=2.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=16),
                    zorder=7)

    # ══════════════════════════════════════════════════════════
    # LEFT — PyTorch / HBM
    # ══════════════════════════════════════════════════════════
    ax = ax_l

    # HBM container
    rect(ax, 0.03, 0.03, 0.94, 0.90, BLUE_L, BLUE, lw=2.5, zorder=1)
    label(ax, 0.5, 0.91, "HBM  (High Bandwidth Memory)", size=11,
          color=BLUE, bold=True)

    # ── Inputs: L and Z side by side ──
    rect(ax, 0.07, 0.75, 0.25, 0.10, WHITE, BLUE, lw=1.5)
    label(ax, 0.195, 0.80, "L  [N×N]\n16 MB", size=9.5, color=BLUE)

    rect(ax, 0.37, 0.75, 0.25, 0.10, WHITE, BLUE, lw=1.5)
    label(ax, 0.495, 0.80, "Z  [N×S]\n1.07 GB", size=9.5, color=BLUE)

    label(ax, 0.5, 0.72, "① Load L and Z from HBM", size=9.5, color=GREY)

    arrow(ax, 0.5, 0.72, 0.5, 0.67, BLUE)

    # ── D matrix — BIG red block (proportional to 1.07 GB) ──
    rect(ax, 0.07, 0.37, 0.86, 0.28, RED_L, RED, lw=3.0, zorder=3)
    label(ax, 0.5, 0.535, "D = μ + L @ Z", size=13, color=RED, bold=True)
    label(ax, 0.5, 0.445, "D  [N×S]  =  1.07 GB  written to HBM  ⚠",
          size=10, color=RED)

    arrow(ax, 0.5, 0.37, 0.5, 0.32, RED, lw=2.5)

    # ── Steps 3 & 4 ──
    rect(ax, 0.07, 0.22, 0.86, 0.09, WHITE, BLUE, lw=1.5)
    label(ax, 0.5, 0.265, "③  Re-read D  →  X = min(D, Q)     [1.07 GB re-read]",
          size=9.5, color="#212121")

    arrow(ax, 0.5, 0.22, 0.5, 0.17, BLUE)

    rect(ax, 0.07, 0.08, 0.86, 0.09, WHITE, BLUE, lw=1.5)
    label(ax, 0.5, 0.125, "④  Profit [N×S]  written to HBM  →  reduce     [1.07 GB]",
          size=9.5, color="#212121")

    # ── Footer ──
    rect(ax, 0.07, 0.03, 0.86, 0.04, "#FFCDD2", RED, lw=2.0, zorder=4)
    label(ax, 0.5, 0.050,
          "3 × 1.07 GB HBM round-trips  →  ~3.2 GB  |  180 ms  |  6.1 TFLOPS",
          size=9, color=RED, bold=True)

    # ══════════════════════════════════════════════════════════
    # RIGHT — Triton / SRAM
    # ══════════════════════════════════════════════════════════
    ax = ax_r

    # HBM outer (thin border — barely used)
    rect(ax, 0.03, 0.03, 0.94, 0.90, "#F9FBE7", GRN, lw=1.5, alpha=0.6, zorder=1)
    label(ax, 0.5, 0.91, "HBM  —  reads tile slices of L, Z only",
          size=10, color=GREY)

    # SRAM container — prominent
    rect(ax, 0.08, 0.22, 0.84, 0.62, ORG_L, ORG, lw=3.5, zorder=2)
    label(ax, 0.5, 0.82, "SRAM  —  on-chip  (48 KB per SM on T4)",
          size=11, color=ORG, bold=True)

    # Tiles inside SRAM (small — proportional to 48 KB)
    rect(ax, 0.12, 0.68, 0.32, 0.09, GRN_L, GRN, lw=1.5, zorder=3)
    label(ax, 0.28, 0.725, "L_tile\n[BM×BK]", size=9, color=GRN)

    rect(ax, 0.50, 0.68, 0.32, 0.09, GRN_L, GRN, lw=1.5, zorder=3)
    label(ax, 0.66, 0.725, "Z_tile\n[BK×BN]", size=9, color=GRN)

    label(ax, 0.5, 0.65, "① K-loop: load tiles → SRAM   ② acc += tl.dot(L_tile, Z_tile)",
          size=9, color=GREY)

    arrow(ax, 0.5, 0.645, 0.5, 0.60, ORG)

    # Fused compute block (stays in SRAM)
    rect(ax, 0.12, 0.38, 0.76, 0.21, GRN_L, GRN, lw=2.5, zorder=3)
    label(ax, 0.5, 0.525, "③  D = max(μ + acc, 0)     X = min(D, Q)", size=11,
          color=GRN, bold=True)
    label(ax, 0.5, 0.445, "profit = p·X  −  c·Q  +  s·(Q − D)⁺       all in SRAM",
          size=9.5, color=GRN)

    arrow(ax, 0.5, 0.38, 0.5, 0.33, ORG)

    # Step 4
    rect(ax, 0.12, 0.24, 0.76, 0.08, GRN_L, GRN, lw=1.5, zorder=3)
    label(ax, 0.5, 0.280, "④  partial_mean = sum(profit) / S  —  SRAM reduction",
          size=9.5, color=GRN)

    # Arrow out of SRAM → tiny output
    arrow(ax, 0.5, 0.22, 0.5, 0.17, GRN, lw=2.5)

    # Output: only 8 KB
    rect(ax, 0.20, 0.09, 0.60, 0.08, GRN_M, GRN, lw=2.0, zorder=4)
    label(ax, 0.5, 0.130, "tl.atomic_add → out[N]   (8 KB written to HBM)",
          size=10, color=GRN, bold=True)

    # Footer
    rect(ax, 0.07, 0.03, 0.86, 0.04, GRN_M, GRN, lw=2.0, zorder=4)
    label(ax, 0.5, 0.050,
          "D [N×S] never materialised  →  1.07 GB saved  |  70 ms  |  15.7 TFLOPS",
          size=9, color=GRN, bold=True)

    fig.savefig(OUT_DIR / "memory_hierarchy.png", bbox_inches="tight",
                dpi=dpi, facecolor=BG)
    plt.close(fig)
    print("  [4/6] memory_hierarchy.png")


# ═══════════════════════════════════════════════════════════════════════════
# Diagram 5 — Mathematical Computation Flow
# ═══════════════════════════════════════════════════════════════════════════
def diagram_math_flow(dpi: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(13, 7), dpi=dpi)
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Newsvendor Mathematical Computation Flow",
                 fontsize=15, fontweight="bold", pad=15, color=C["text"])

    # --- Step 1: Cholesky ---
    _rounded_box(ax, (0.2, 5.5), 3.0, 1.5, "#E3F2FD",
                 "STEP 1 — Cholesky\n═══════════════\nΣ = diag(σ) · R · diag(σ)\nL = cholesky(Σ)\n\nL shape: [N × N]",
                 fontsize=8, edgecolor=C["hbm"], lw=2)

    # --- Step 2: Correlated demand ---
    _rounded_box(ax, (4.0, 5.5), 3.5, 1.5, "#E8F5E9",
                 "STEP 2 — Demand Generation\n═════════════════════\nZ ~ N(0, I)    shape [N × S]\nD = μ + L @ Z\nD = max(D, 0)    shape [N × S]",
                 fontsize=8, edgecolor=C["green"], lw=2)

    # --- Step 3: Sales ---
    _rounded_box(ax, (8.3, 5.5), 4.0, 1.5, "#FFF3E0",
                 "STEP 3 — Constrained Sales\n═════════════════════\nX = min(D, Q)\n\n\"Cannot sell more than\n demand or stock\"   [N × S]",
                 fontsize=8, edgecolor=C["sram"], lw=2)

    # --- Step 4: Profit ---
    _rounded_box(ax, (2.0, 2.5), 5.0, 1.8, C["triton"],
                 "STEP 4 — Newsvendor Profit\n══════════════════════════\nRevenue    =  p · X\nCost         =  c · Q\nSalvage     =  s · max(Q − D, 0)\n\nπ(n, s) = p · X  −  c · Q  +  s · (Q − D)⁺      [N × S]",
                 fontsize=8, edgecolor=C["accent"], lw=2)

    # --- Step 5: Reduction ---
    _rounded_box(ax, (8.3, 2.8), 4.0, 1.2, "#C8E6C9",
                 "STEP 5 — Expectation\n═══════════════════\nE[π] = (1/S) Σₛ π(n, s)\n\nOutput: E[π]   shape [N]",
                 fontsize=8, edgecolor=C["green"], lw=2.5, bold=False)

    # --- Arrows ---
    _arrow(ax, (3.2, 6.25), (4.0, 6.25), lw=2.5, color=C["green"])
    _arrow(ax, (7.5, 6.25), (8.3, 6.25), lw=2.5, color=C["sram"])
    _arrow(ax, (10.3, 5.5), (6.2, 4.3), lw=2, color=C["accent"],
           connectionstyle="arc3,rad=0.3")
    _arrow(ax, (7.0, 3.4), (8.3, 3.4), lw=2.5, color=C["green"])

    # --- Annotation: what the Triton kernel fuses ---
    ax.add_patch(FancyBboxPatch(
        (3.8, 1.8), 8.8, 5.6, boxstyle="round,pad=0.15,rounding_size=0.3",
        facecolor="none", edgecolor=C["accent"], lw=2, linestyle="--", zorder=0))
    _label(ax, 8.2, 1.5, "FUSED IN TRITON KERNEL  (Steps 2–5, entirely in SRAM)",
           fontsize=9, bold=True, color=C["accent"])

    # --- Input annotations ---
    _label(ax, 0.2, 5.2, "Inputs: R from M5, σ from sales data",
           fontsize=7.5, color=C["textlight"], ha="left")
    _label(ax, 8.3, 5.2, "Q = order quantity (decision variable)",
           fontsize=7.5, color=C["textlight"], ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "newsvendor_math_flow.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [5/6] newsvendor_math_flow.png")


# ═══════════════════════════════════════════════════════════════════════════
# Diagram 6 — Benchmark & Validation Flow
# ═══════════════════════════════════════════════════════════════════════════
def diagram_benchmark_flow(dpi: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(13, 7), dpi=dpi)
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Benchmark Suite — Solver Comparison & Validation Pipeline",
                 fontsize=15, fontweight="bold", pad=15, color=C["text"])

    # --- Data pipeline box ---
    _rounded_box(ax, (0.3, 5.0), 2.5, 1.8, C["pipeline"],
                 "DataPipeline\n───────────\nN, S, seed\n     ↓\nTensorBundle",
                 fontsize=8.5)

    # --- Three solver branches ---
    # CPU
    _rounded_box(ax, (4.0, 5.8), 2.5, 1.0, C["config"],
                 "CPUMonteCarlo\n(NumPy — gold standard)",
                 fontsize=8)
    _label(ax, 5.25, 7.0, "SOLVER 1", fontsize=7, bold=True, color=C["hbm"])

    # PyTorch
    _rounded_box(ax, (4.0, 4.0), 2.5, 1.2, C["baseline"],
                 "PyTorchMonteCarlo\ntorch.compile\n(Inductor backend)",
                 fontsize=8)
    _label(ax, 5.25, 5.4, "SOLVER 2", fontsize=7, bold=True, color="#E65100")

    # Triton
    _rounded_box(ax, (4.0, 2.0), 2.5, 1.3, C["triton"],
                 "TritonFusedNewsvendor\n@triton.autotune\n@triton.jit",
                 fontsize=8)
    _label(ax, 5.25, 3.5, "SOLVER 3  ★", fontsize=7, bold=True, color=C["accent"])

    # --- Arrows from pipeline to solvers ---
    _arrow(ax, (2.8, 5.9), (4.0, 6.2), lw=2)
    _arrow(ax, (2.8, 5.5), (4.0, 4.7), lw=2)
    _arrow(ax, (2.8, 5.1), (4.0, 3.0), lw=2, connectionstyle="arc3,rad=-0.15")

    # --- Correctness check ---
    _rounded_box(ax, (7.5, 5.2), 2.5, 1.5, "#E8F5E9",
                 "Correctness\n══════════\ntorch.allclose(\n  atol=1e-2,\n  rtol=1e-3\n)",
                 fontsize=8, edgecolor=C["green"], lw=2)
    _label(ax, 8.75, 6.9, "VALIDATION", fontsize=7, bold=True, color=C["green"])

    # Arrows to correctness
    _arrow(ax, (6.5, 6.1), (7.5, 6.0), lw=1.5)
    _arrow(ax, (6.5, 4.6), (7.5, 5.5), lw=1.5, connectionstyle="arc3,rad=0.15")
    _arrow(ax, (6.5, 3.0), (7.5, 5.3), lw=1.5, connectionstyle="arc3,rad=0.25")

    # --- Performance reporting ---
    _rounded_box(ax, (7.5, 2.5), 2.5, 1.8, C["benchmark"],
                 "Performance\n══════════\nTime (ms)\nPeak Memory (GB)\nTFLOPS\n────────────\ncuda.Event timing",
                 fontsize=8, edgecolor="#6A1B9A", lw=2)
    _label(ax, 8.75, 4.5, "PROFILING", fontsize=7, bold=True, color="#6A1B9A")

    _arrow(ax, (6.5, 4.3), (7.5, 3.8), lw=1.5, connectionstyle="arc3,rad=-0.1")
    _arrow(ax, (6.5, 2.5), (7.5, 2.8), lw=1.5, connectionstyle="arc3,rad=-0.1")

    # --- Output ---
    _rounded_box(ax, (10.8, 3.5), 1.8, 3.0, "#FFF3E0",
                 "OUTPUT\n══════\n\nTerminal\nTable\n\n+\n\nPerf\nSweep\nPlot",
                 fontsize=8, edgecolor=C["sram"], lw=2.5)

    _arrow(ax, (10.0, 5.9), (10.8, 5.2), lw=2, connectionstyle="arc3,rad=0.1")
    _arrow(ax, (10.0, 3.5), (10.8, 4.0), lw=2, connectionstyle="arc3,rad=-0.1")

    # --- Bottom legend ---
    _label(ax, 6.5, 0.8,
           "N swept: 128 → 4096  |  S fixed: 65536 or 131072  |  Best of 3 repeats reported",
           fontsize=8, color=C["textlight"])
    _label(ax, 6.5, 0.3,
           "perf_report generates publication-quality plots via triton.testing.Benchmark",
           fontsize=7.5, color=C["textlight"])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "benchmark_flow.png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  [6/6] benchmark_flow.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture diagrams")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output DPI (200 for screen, 300 for print)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Generating 6 diagrams at {args.dpi} DPI → {OUT_DIR}/\n")

    diagram_system_architecture(args.dpi)
    diagram_data_pipeline(args.dpi)
    diagram_triton_grid(args.dpi)
    diagram_memory_hierarchy(args.dpi)
    diagram_math_flow(args.dpi)
    diagram_benchmark_flow(args.dpi)

    print(f"\nDone. All diagrams saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
