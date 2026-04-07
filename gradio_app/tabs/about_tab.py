"""
about_tab.py -- Tab 5: About / Documentation.

Static markdown content explaining the newsvendor problem variants,
Triton kernel architecture, app usage instructions, and credits.
"""

from __future__ import annotations

import gradio as gr


def create_about_tab(state: gr.State):
    """
    Build the About / Documentation tab.

    Parameters
    ----------
    state : gr.State
        Shared application state (unused here, but kept for interface
        consistency with other tabs).
    """
    with gr.TabItem("5. About"):
        gr.Markdown(ABOUT_CONTENT)


# =========================================================================
# Static content
# =========================================================================

ABOUT_CONTENT = r"""
# GPU-Accelerated Multi-Echelon Stochastic Newsvendor Suite

## Overview

This application provides an interactive interface for solving several
variants of the stochastic newsvendor inventory optimisation problem using
GPU-accelerated Monte-Carlo simulation.  It targets Google Colab T4 GPUs
(15 GB VRAM) and benchmarks three solver backends: CPU (NumPy), PyTorch
GPU with `torch.compile`, and custom Triton fused kernels.

---

## Newsvendor Problem Variants

### 1. Base Newsvendor

The classical single-period inventory problem.  For each product $i$:

$$\pi_i = p_i \cdot \min(D_i, Q_i) - c_i \cdot Q_i + s_i \cdot \max(Q_i - D_i, 0)$$

where:
- $D_i$ is the stochastic demand (correlated across products via Cholesky factor $L$)
- $Q_i$ is the order quantity (set to $\mu_i$ as baseline)
- $p_i, c_i, s_i$ are selling price, unit cost, and salvage value

The expected profit is estimated via Monte-Carlo:

$$\mathbb{E}[\pi_i] = \frac{1}{S} \sum_{j=1}^{S} \pi_i^{(j)}$$

where $D^{(j)} = \max(\mu + L \cdot Z^{(j)}, 0)$ and $Z \sim \mathcal{N}(0, I)$.

### 2. Grid Search (Optimal $Q^*$)

Finds the optimal order quantity $Q_i^*$ by evaluating $\mathbb{E}[\pi_i]$
at $K$ grid points:

$$Q_i^{(k)} = \mu_i \cdot r_k, \quad r_k \in [\text{ratio\_min}, \text{ratio\_max}]$$

$$Q_i^* = \arg\max_k \; \mathbb{E}[\pi_i(Q_i^{(k)})]$$

The profit surface $\mathbb{E}[\pi_i](Q)$ reveals the shape of each
product's objective function.

### 3. CVaR (Risk-Averse)

Conditional Value at Risk at level $\alpha$ is the expected profit in the
worst $\alpha$-fraction of scenarios:

$$\text{VaR}_\alpha = \inf\{x : P(\pi \le x) \ge \alpha\}$$

$$\text{CVaR}_\alpha = \mathbb{E}[\pi \mid \pi \le \text{VaR}_\alpha]$$

CVaR is a **coherent risk measure** widely used in operations research for
risk-averse inventory decisions.  A risk-averse manager might optimise for
CVaR instead of expected profit to protect against worst-case demand
realisations.

### 4. Budget-Constrained

Adds a total procurement budget constraint:

$$\sum_{i=1}^{N} c_i \cdot Q_i \le B$$

Solved via **Lagrangian dual decomposition** with bisection on the multiplier
$\lambda \ge 0$:

- At each $\lambda$, solve the grid search with effective cost
  $c_i' = c_i (1 + \lambda)$
- Update $\lambda$ based on budget violation until convergence

This reuses the grid-search kernels -- no new Triton kernel is needed.

### 5. Multi-Product Substitution

When product $i$ stocks out, a fraction $\beta_{ik}$ of unmet demand is
redirected to substitute product $k$:

$$D_i^{\text{eff}} = D_i + \sum_{k \in \text{subs}(i)} \beta_{ik} \cdot \max(D_k - Q_k, 0)$$

Substitutes are chosen within the same category (tractors with tractors,
generators with generators).

---

## Triton Kernel Architecture

### The Key Innovation: SRAM Fusion

Standard PyTorch must materialise the full demand matrix
$D = \mu + L \cdot Z$ in HBM (e.g., $2048 \times 131072 \times 4\text{B} \approx 1\text{ GB}$).

Our custom Triton kernel **fuses the matrix multiplication with the
newsvendor business logic** so that $D$ never leaves SRAM:

```
Phase 1: Tiled matmul    acc = L[tile_m, :] @ Z[:, tile_n]     (SRAM)
Phase 2: Fused logic      D = mu + acc;  X = min(D, Q);  pi    (SRAM)
Phase 3: Partial mean     partial = mean(pi, axis=1)            (SRAM)
Phase 4: Atomic store     out[m] += partial / n_tiles           (HBM)
```

### Memory Hierarchy

| Level | Capacity (T4) | Bandwidth | What lives here |
|-------|---------------|-----------|-----------------|
| Registers | ~256 KB/SM | ~8 TB/s | Accumulators, loop vars |
| Shared (SRAM) | 64 KB/SM | ~2 TB/s | L-tile, Z-tile, acc |
| L2 Cache | 4 MB | ~1 TB/s | Reused tiles |
| HBM (VRAM) | 15 GB | 320 GB/s | L, Z, mu, p, c, s, Q, out |

### Autotuning

The kernel uses `@triton.autotune` with ~9 hand-picked configurations
that all fit within the T4's 48 KB usable SRAM budget per SM.  Tile sizes
range from $32 \times 128$ to $128 \times 128$.

---

## How to Use This App

1. **Tab 1 -- Problem Setup**: Configure $N$ (products), $S$ (scenarios),
   seed, and tractor fraction.  Check the VRAM estimate, then click
   *Generate Data*.

2. **Tab 2 -- Run Solvers**: Select a variant, configure its parameters,
   choose which backends to run (CPU / PyTorch / Triton), and click
   *Run Solvers*.

3. **Tab 3 -- Results Dashboard**: Click *Refresh* to see the performance
   comparison table, bar charts, and variant-specific plots.

4. **Tab 4 -- Per-Product Analysis**: Select a product index to inspect
   its parameters and compare solver outputs.

### Recommended Settings for T4 (15 GB)

| N | S | Est. VRAM | Notes |
|---|---|-----------|-------|
| 128 | 131072 | ~0.13 GB | Fast, good for testing |
| 512 | 32768 | ~0.54 GB | Moderate |
| 1024 | 65536 | ~2.1 GB | Large-scale |
| 2048 | 131072 | ~8.5 GB | Near T4 limit (use Triton!) |

---

## Project Context

This suite was developed as a **B.Tech Project (BTP)** at the
**Indian Institute of Technology Kharagpur (IIT KGP)**, Department of
Industrial and Systems Engineering.

### Technical Stack

- **Triton** -- Custom GPU kernels with SRAM fusion
- **PyTorch** -- Reference GPU implementation with `torch.compile`
- **NumPy** -- Gold-standard CPU reference
- **Gradio** -- Interactive web interface
- **Plotly** -- Interactive visualisations

### Data Sources

- **M5 Forecasting Dataset (Kaggle)** -- Hierarchical correlation structure
- **Tractor / Generator sales data** -- Realistic demand distributions

---

*Built with Triton, PyTorch, and Gradio.*
"""
