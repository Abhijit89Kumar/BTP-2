"""
grid_search_kernel.py — Autotuned Triton kernel for Optimal Q* Grid Search.

╔═══════════════════════════════════════════════════════════════════════════╗
║  KEY INNOVATION                                                          ║
║  The base Triton kernel evaluates E[profit] at a SINGLE Q per product.   ║
║  This kernel evaluates E[profit] at K different Q values per product     ║
║  while performing the expensive matmul ONLY ONCE.  The acc tile          ║
║  [BM, BN] stays in SRAM across all K iterations of the inner loop.      ║
║                                                                          ║
║  Output: profit_surface[N, K] — expected profit at each grid point.      ║
║  Post-processing (on host): argmax over K to find Q*.                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

Kernel layout
─────────────
Grid:  (ceil(N / BLOCK_M),  ceil(S / BLOCK_N))
         ^ product dim       ^ scenario dim

Each program instance:
  Phase 1 — Tiled matmul: acc = L[m_block, :] @ Z[:, n_block]     (ONCE)
  Phase 2 — Loop over K grid points:
            For each k: compute profit at Q = mu_i * Q_grid[k],
            reduce over BN scenarios, atomic_add into profit_surface.
  Phase 3 — Scenario masking + partial mean reduction (inside K loop)
  Phase 4 — Atomic accumulation into profit_surface[i, k]

Memory traffic:
  READ  — tiles of L, Z (matmul), plus mu/p/c/s vectors, Q_grid[K].
  WRITE — partial means into profit_surface[N, K] via atomic add.
  NEVER — the intermediate D[N, S] matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def next_power_of_2(n: int) -> int:
    """Round up to the nearest power of 2."""
    return 1 << (n - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# ═══════════════════════════════════════════════════════════════════════════
# Autotune configuration
# ═══════════════════════════════════════════════════════════════════════════
def _build_autotune_configs() -> list[triton.Config]:
    """
    Hand-picked autotune configurations for the grid search kernel.

    SRAM budget: same 48 KB floor as base kernel (T4-safe).

    The K-loop over grid points runs AFTER matmul completes, reusing the
    same acc tile in SRAM.  Each K iteration only needs a [BLOCK_M]
    accumulator (tiny — 128 * 4 B = 512 B max), so SRAM pressure is
    identical to the base kernel.
    """
    SRAM_BUDGET = 48 * 1024  # 48 KB — safe for T4 and above
    configs: list[triton.Config] = []

    candidates = [
        # (BM, BN, BK, warps, stages)  — SRAM usage noted
        ( 32, 128, 32, 4, 2),   # 20 KB
        ( 64,  64, 32, 4, 2),   # 20 KB
        ( 64, 128, 32, 4, 2),   # 36 KB  — sweet spot on T4
        ( 64, 128, 32, 8, 2),   # 36 KB
        ( 64, 128, 64, 4, 2),   # 40 KB
        ( 64, 128, 64, 8, 2),   # 40 KB
        (128,  64, 32, 4, 2),   # 36 KB
        (128,  64, 32, 8, 2),   # 36 KB
        (128, 128, 32, 8, 2),   # 48 KB
    ]

    for bm, bn, bk, nw, ns in candidates:
        l_tile = bm * bk * 4
        z_tile = bk * bn * 4
        acc    = bm * bn * 4
        total  = l_tile + z_tile + acc
        if total > SRAM_BUDGET:
            continue
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_M": bm,
                 "BLOCK_SIZE_N": bn,
                 "BLOCK_SIZE_K": bk},
                num_warps=nw,
                num_stages=ns,
            )
        )

    assert len(configs) > 0, "No valid autotune configs — relax SRAM budget."
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# The Triton kernel
# ═══════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=_build_autotune_configs(),
    key=["N", "S", "INNER_K", "NUM_Q"],   # re-tune when problem size changes
)
@triton.jit
def _grid_search_kernel(
    # ── Pointers to input tensors ──────────────────────────────────────
    L_ptr,          # float32 [N, INNER_K]  — lower-triangular Cholesky factor
    Z_ptr,          # float32 [INNER_K, S]  — standard-normal scenario matrix
    mu_ptr,         # float32 [N]           — mean demand (flattened from [N,1])
    p_ptr,          # float32 [N]           — selling price
    c_ptr,          # float32 [N]           — unit cost
    s_ptr,          # float32 [N]           — salvage value
    Q_grid_ptr,     # float32 [NUM_Q]       — Q ratio multipliers (grid points)
    # ── Pointer to output tensor ───────────────────────────────────────
    out_ptr,        # float32 [N, NUM_Q]    — profit_surface (atomically accumulated)
    # ── Problem dimensions ─────────────────────────────────────────────
    N: tl.constexpr,               # number of product-location nodes
    S: tl.constexpr,               # number of Monte-Carlo scenarios
    INNER_K: tl.constexpr,         # inner dimension of L @ Z (= N)
    NUM_Q: tl.constexpr,           # number of Q grid points
    # ── Strides (elements, NOT bytes) ──────────────────────────────────
    stride_L_m: tl.constexpr,      # L.stride(0)  — typically INNER_K
    stride_L_k: tl.constexpr,      # L.stride(1)  — typically 1
    stride_Z_k: tl.constexpr,      # Z.stride(0)  — typically S
    stride_Z_n: tl.constexpr,      # Z.stride(1)  — typically 1
    stride_out_m: tl.constexpr,    # out.stride(0) — typically NUM_Q
    stride_out_q: tl.constexpr,    # out.stride(1) — typically 1
    # ── Compile-time tile dimensions (set by autotune) ─────────────────
    BLOCK_SIZE_M: tl.constexpr,    # tile height  (product axis)
    BLOCK_SIZE_N: tl.constexpr,    # tile width   (scenario axis)
    BLOCK_SIZE_K: tl.constexpr,    # tile depth   (reduction axis)
):
    """
    Fused grid-search kernel: matmul ONCE, then loop K times over Q values.

    Each program instance handles a [BLOCK_M, BLOCK_N] tile of the virtual
    profit matrix, evaluated at all K grid points.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  PHASE 1 — Tiled matmul: acc = L[m_block, :] @ Z[:, n_block]      │
    │             (runs ONCE — acc stays in SRAM for all K iterations)    │
    │  PHASE 2 — For k = 0..K-1:                                         │
    │              Q_val = mu * Q_grid[k]                                 │
    │              D = max(mu + acc, 0)    [reuse acc — never recomputed] │
    │              X = min(D, Q_val)                                      │
    │              profit = p*X - c*Q_val + s*max(Q_val - D, 0)          │
    │              partial_sum = sum(profit, axis=1) / S                  │
    │              atomic_add → out[m_block, k]                           │
    └──────────────────────────────────────────────────────────────────────┘
    """

    # ══════════════════════════════════════════════════════════════════════
    # 0.  Identify this program's tile position in the 2-D grid.
    # ══════════════════════════════════════════════════════════════════════
    pid_m = tl.program_id(0)   # which block of products   [0 .. ceil(N/BLOCK_M))
    pid_n = tl.program_id(1)   # which block of scenarios  [0 .. ceil(S/BLOCK_N))

    # Row indices within the M tile (product axis)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Column indices within the N tile (scenario axis)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # ══════════════════════════════════════════════════════════════════════
    # 1.  PHASE 1 — Tiled matrix multiplication (runs ONCE).
    #     acc = L[tile_m, :] @ Z[:, tile_n]   shape [BLOCK_M, BLOCK_N]
    # ══════════════════════════════════════════════════════════════════════

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(INNER_K, BLOCK_SIZE_K)

    for k_idx in range(0, num_k_tiles):
        # Compute K-offsets for this tile
        offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load L tile [BLOCK_M, BLOCK_K]
        L_ptrs = (L_ptr
                  + offs_m[:, None] * stride_L_m
                  + offs_k[None, :] * stride_L_k)
        L_mask = (offs_m[:, None] < N) & (offs_k[None, :] < INNER_K)
        L_tile = tl.load(L_ptrs, mask=L_mask, other=0.0)

        # Load Z tile [BLOCK_K, BLOCK_N]
        Z_ptrs = (Z_ptr
                  + offs_k[:, None] * stride_Z_k
                  + offs_n[None, :] * stride_Z_n)
        Z_mask = (offs_k[:, None] < INNER_K) & (offs_n[None, :] < S)
        Z_tile = tl.load(Z_ptrs, mask=Z_mask, other=0.0)

        # Accumulate dot product in SRAM — NO HBM traffic
        acc += tl.dot(L_tile, Z_tile)

    # ══════════════════════════════════════════════════════════════════════
    # 2.  Load per-product parameters (once, reused across all K iters).
    # ══════════════════════════════════════════════════════════════════════

    m_mask = offs_m < N   # [BLOCK_SIZE_M]

    mu_vals = tl.load(mu_ptr + offs_m, mask=m_mask, other=0.0)  # [BLOCK_M]
    p_vals  = tl.load(p_ptr  + offs_m, mask=m_mask, other=0.0)
    c_vals  = tl.load(c_ptr  + offs_m, mask=m_mask, other=0.0)
    s_vals  = tl.load(s_ptr  + offs_m, mask=m_mask, other=0.0)

    # Broadcast to [BLOCK_M, BLOCK_N]
    mu_bc = mu_vals[:, None]
    p_bc  = p_vals[:, None]
    c_bc  = c_vals[:, None]
    s_bc  = s_vals[:, None]

    # Compute demand ONCE — reused for every Q grid point
    # D = max(mu + acc, 0)   [BLOCK_M, BLOCK_N] — stays in SRAM
    D = tl.maximum(mu_bc + acc, 0.0)

    # Scenario mask for the last tile (may be partial)
    n_mask = offs_n < S   # [BLOCK_SIZE_N]

    # ══════════════════════════════════════════════════════════════════════
    # 3.  PHASE 2 — Loop over K grid points, compute profit at each Q.
    # ══════════════════════════════════════════════════════════════════════

    for q_idx in range(NUM_Q):
        # Load the Q ratio for this grid point
        Q_ratio = tl.load(Q_grid_ptr + q_idx)   # scalar

        # Q_val = mu_i * Q_ratio  →  [BLOCK_M, 1] broadcast to [BLOCK_M, BLOCK_N]
        Q_bc = mu_vals[:, None] * Q_ratio

        # Sales = min(demand, order quantity)
        X = tl.minimum(D, Q_bc)

        # Overage inventory (unsold units)
        overage = tl.maximum(Q_bc - D, 0.0)

        # Per-scenario profit: pi = p*X - c*Q + s*(Q - D)+
        profit = (p_bc * X) - (c_bc * Q_bc) + (s_bc * overage)

        # Mask out-of-bounds scenarios before summing
        profit = tl.where(n_mask[None, :], profit, 0.0)

        # Sum over scenarios → [BLOCK_M]
        partial_sum = tl.sum(profit, axis=1)

        # Normalise by S to get partial contribution to the mean
        partial_mean = partial_sum / S

        # Atomic accumulation into profit_surface[i, k]
        # Output layout: out[i * NUM_Q + k] = out[i, k]  (row-major)
        out_ptrs = out_ptr + offs_m * stride_out_m + q_idx * stride_out_q
        tl.atomic_add(out_ptrs, partial_mean, mask=m_mask)
