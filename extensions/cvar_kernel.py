"""
cvar_kernel.py -- Autotuned Triton kernel for the CVaR (Conditional Value at
                  Risk) extension of the Multi-Echelon Stochastic Newsvendor.

This kernel computes BOTH the expected profit AND a per-product profit
histogram in a single fused pass, avoiding any materialisation of the
intermediate D[N, S] or profit[N, S] matrices in HBM.

Kernel layout
-------------
Grid:  (ceil(N / BLOCK_M),  ceil(S / BLOCK_N))

Each program instance handles a [BLOCK_M, BLOCK_N] tile of the *virtual*
profit matrix (which is never fully materialised).

Phases 1-2 are identical to the base fused kernel:
  Phase 1 -- Tiled matmul:  acc = L[m_block, :] @ Z[:, n_block]
  Phase 2 -- Fused logic:   D = max(mu + acc, 0); X = min(D, Q);
                             profit = p*X - c*Q + s*max(Q-D, 0)

Phase 3a -- Partial mean reduction -> atomic_add into out_mean[N]
Phase 3b -- Histogram accumulation: for each profit value, compute
            bin_idx = clamp(floor((profit - hist_min) / bin_width), 0, num_bins-1)
            and atomic_add 1.0 into hist[product_idx * num_bins + bin_idx]

Memory traffic:
  READ  -- tiles of L, Z, plus mu, p, c, s, Q vectors, plus hist_min/hist_max.
  WRITE -- partial mean [BLOCK_M] via atomic_add into out_mean[N].
           histogram counts via atomic_add into hist[N, num_bins].
  NEVER -- the intermediate D[N, S] or profit[N, S] matrices.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# =========================================================================
# Helpers
# =========================================================================
def next_power_of_2(n: int) -> int:
    """Round up to the nearest power of 2."""
    return 1 << (n - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# =========================================================================
# Autotune configuration (identical budget to base kernel -- T4 safe)
# =========================================================================
def _build_cvar_autotune_configs() -> list[triton.Config]:
    """
    Hand-picked autotune configurations for the CVaR fused kernel.

    We use the same SRAM budget as the base kernel (48 KB floor for T4).
    The histogram accumulation adds no SRAM pressure because it uses
    atomic adds directly to HBM -- no extra tile is held in SRAM.
    """
    SRAM_BUDGET = 48 * 1024  # 48 KB
    configs: list[triton.Config] = []

    candidates = [
        # (BM, BN, BK, warps, stages)
        ( 32, 128, 32, 4, 2),   # 20 KB
        ( 64,  64, 32, 4, 2),   # 20 KB
        ( 64, 128, 32, 4, 2),   # 36 KB
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

    assert len(configs) > 0, "No valid autotune configs -- relax SRAM budget."
    return configs


# =========================================================================
# The Triton kernel
# =========================================================================
@triton.autotune(
    configs=_build_cvar_autotune_configs(),
    key=["N", "S", "K"],
)
@triton.jit
def _fused_cvar_kernel(
    # -- Pointers to input tensors --
    L_ptr,              # float32 [N, K]
    Z_ptr,              # float32 [K, S]
    mu_ptr,             # float32 [N]
    p_ptr,              # float32 [N]
    c_ptr,              # float32 [N]
    s_ptr,              # float32 [N]
    Q_ptr,              # float32 [N]
    # -- Pointers to output tensors --
    out_mean_ptr,       # float32 [N]     -- expected profit (atomic accum)
    hist_ptr,           # float32 [N, NUM_BINS] -- histogram counts (atomic accum)
    hist_min_ptr,       # float32 [N]     -- per-product histogram lower bound
    hist_max_ptr,       # float32 [N]     -- per-product histogram upper bound
    # -- Problem dimensions --
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    # -- Strides (elements, NOT bytes) --
    stride_L_m: tl.constexpr,
    stride_L_k: tl.constexpr,
    stride_Z_k: tl.constexpr,
    stride_Z_n: tl.constexpr,
    # -- Histogram config --
    NUM_BINS: tl.constexpr,
    # -- Tile dimensions (set by autotune) --
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused Triton kernel: matmul + Newsvendor profit + mean reduction + histogram.

    Each program instance handles a [BLOCK_M, BLOCK_N] tile of the virtual
    profit matrix.

    Phase 1 -- Tiled matmul: acc = L[tile_m, :] @ Z[:, tile_n]
    Phase 2 -- Fused logic:  D, X, profit in SRAM
    Phase 3a -- Partial mean: atomic_add into out_mean
    Phase 3b -- Histogram:   atomic_add counts into hist[product, bin]
    """

    # ==================================================================
    # 0. Tile position
    # ==================================================================
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # ==================================================================
    # 1. PHASE 1 -- Tiled matmul: acc = L[tile_m, :] @ Z[:, tile_n]
    # ==================================================================
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for k_idx in range(0, num_k_tiles):
        offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load L tile [BLOCK_M, BLOCK_K]
        L_ptrs = (L_ptr
                  + offs_m[:, None] * stride_L_m
                  + offs_k[None, :] * stride_L_k)
        L_mask = (offs_m[:, None] < N) & (offs_k[None, :] < K)
        L_tile = tl.load(L_ptrs, mask=L_mask, other=0.0)

        # Load Z tile [BLOCK_K, BLOCK_N]
        Z_ptrs = (Z_ptr
                  + offs_k[:, None] * stride_Z_k
                  + offs_n[None, :] * stride_Z_n)
        Z_mask = (offs_k[:, None] < K) & (offs_n[None, :] < S)
        Z_tile = tl.load(Z_ptrs, mask=Z_mask, other=0.0)

        acc += tl.dot(L_tile, Z_tile)

    # ==================================================================
    # 2. PHASE 2 -- Fused Newsvendor business logic (all in SRAM)
    # ==================================================================
    m_mask = offs_m < N

    mu_vals = tl.load(mu_ptr + offs_m, mask=m_mask, other=0.0)
    p_vals  = tl.load(p_ptr  + offs_m, mask=m_mask, other=0.0)
    c_vals  = tl.load(c_ptr  + offs_m, mask=m_mask, other=0.0)
    s_vals  = tl.load(s_ptr  + offs_m, mask=m_mask, other=0.0)
    Q_vals  = tl.load(Q_ptr  + offs_m, mask=m_mask, other=0.0)

    mu_bc = mu_vals[:, None]
    p_bc  = p_vals[:, None]
    c_bc  = c_vals[:, None]
    s_bc  = s_vals[:, None]
    Q_bc  = Q_vals[:, None]

    # Demand (clamped >= 0)
    D = tl.maximum(mu_bc + acc, 0.0)

    # Sales = min(demand, order quantity)
    X = tl.minimum(D, Q_bc)

    # Overage
    overage = tl.maximum(Q_bc - D, 0.0)

    # Per-scenario profit
    profit = (p_bc * X) - (c_bc * Q_bc) + (s_bc * overage)

    # ==================================================================
    # 3a. PHASE 3a -- Partial mean reduction over scenario axis
    # ==================================================================
    n_mask = offs_n < S
    profit_masked = tl.where(n_mask[None, :], profit, 0.0)

    partial_sum = tl.sum(profit_masked, axis=1)   # [BLOCK_M]
    partial_mean = partial_sum / S

    # Atomic accumulate into out_mean[N]
    out_mean_ptrs = out_mean_ptr + offs_m
    tl.atomic_add(out_mean_ptrs, partial_mean, mask=m_mask)

    # ==================================================================
    # 3b. PHASE 3b -- Histogram accumulation
    #
    #     For each (product i, scenario j) in this tile, compute the
    #     histogram bin index and atomically increment the count.
    #
    #     bin_idx = clamp(floor((profit - hist_min) / bin_width), 0, NUM_BINS-1)
    #     hist[i * NUM_BINS + bin_idx] += 1.0
    #
    #     We iterate over the M dimension (products in this tile) and
    #     process all N scenarios for each product at once. This is more
    #     efficient than element-wise because it keeps the histogram
    #     parameters in registers.
    # ==================================================================

    # Load histogram bounds for this product block
    hist_min_vals = tl.load(hist_min_ptr + offs_m, mask=m_mask, other=0.0)  # [BLOCK_M]
    hist_max_vals = tl.load(hist_max_ptr + offs_m, mask=m_mask, other=0.0)  # [BLOCK_M]

    # Bin width per product: (hist_max - hist_min) / NUM_BINS
    bin_width = (hist_max_vals - hist_min_vals) / NUM_BINS  # [BLOCK_M]

    # Avoid division by zero (if hist_min == hist_max, all goes to bin 0)
    safe_bin_width = tl.maximum(bin_width, 1e-10)

    # Broadcast to [BLOCK_M, BLOCK_N]
    hist_min_bc = hist_min_vals[:, None]
    safe_bw_bc  = safe_bin_width[:, None]

    # Compute bin indices: [BLOCK_M, BLOCK_N]
    bin_float = (profit - hist_min_bc) / safe_bw_bc
    bin_idx = tl.math.floor(bin_float).to(tl.int32)

    # Clamp to [0, NUM_BINS - 1]
    bin_idx = tl.maximum(bin_idx, 0)
    bin_idx = tl.minimum(bin_idx, NUM_BINS - 1)

    # Combined mask: valid product AND valid scenario
    combined_mask = m_mask[:, None] & n_mask[None, :]  # [BLOCK_M, BLOCK_N]

    # Compute pointers into the histogram buffer:
    #   hist_ptr + offs_m[:, None] * NUM_BINS + bin_idx
    hist_ptrs = hist_ptr + offs_m[:, None] * NUM_BINS + bin_idx  # [BLOCK_M, BLOCK_N]

    # Atomic add 1.0 for each valid (product, scenario) element
    tl.atomic_add(hist_ptrs, tl.where(combined_mask, 1.0, 0.0), mask=combined_mask)


# =========================================================================
# Python wrapper
# =========================================================================
class TritonCVaRKernel:
    """
    High-level interface to the fused CVaR Triton kernel.

    Returns both expected profit [N] and per-product histogram [N, num_bins].
    The caller (TritonCVaR solver) post-processes the histogram to extract
    VaR and CVaR.
    """

    def __init__(self, num_bins: int = 256) -> None:
        self.num_bins = num_bins

    def launch(
        self,
        L: torch.Tensor,       # [N, K] contiguous
        Z: torch.Tensor,       # [K, S] contiguous
        mu: torch.Tensor,      # [N] contiguous
        p: torch.Tensor,       # [N] contiguous
        c: torch.Tensor,       # [N] contiguous
        s: torch.Tensor,       # [N] contiguous
        Q: torch.Tensor,       # [N] contiguous
        hist_min: torch.Tensor, # [N] contiguous
        hist_max: torch.Tensor, # [N] contiguous
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Launch the fused CVaR kernel.

        Parameters
        ----------
        L, Z, mu, p, c, s, Q : standard Newsvendor inputs (1-D for mu..Q)
        hist_min, hist_max    : per-product histogram bounds [N]

        Returns
        -------
        out_mean : float32 [N]  -- expected profit
        hist     : float32 [N, num_bins] -- profit histogram counts
        """
        device = L.device
        N = L.shape[0]
        S = Z.shape[1]
        K = L.shape[1]
        num_bins = self.num_bins

        # Output buffers (zero-initialised for atomic_add)
        out_mean = torch.zeros(N, dtype=torch.float32, device=device)
        hist = torch.zeros(N, num_bins, dtype=torch.float32, device=device)

        grid = lambda META: (
            cdiv(N, META["BLOCK_SIZE_M"]),
            cdiv(S, META["BLOCK_SIZE_N"]),
        )

        _fused_cvar_kernel[grid](
            L, Z, mu, p, c, s, Q,
            out_mean, hist,
            hist_min, hist_max,
            N, S, K,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            num_bins,
        )

        return out_mean, hist


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from config import NewsvendorConfig
    from data_pipeline import DataPipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = NewsvendorConfig(N=256, S=8192, device="cuda")
    bundle = DataPipeline(cfg=cfg).run()

    N, S = bundle.N, bundle.S
    mu = bundle.mu.squeeze(1).contiguous()
    p  = bundle.p.squeeze(1).contiguous()
    c  = bundle.c.squeeze(1).contiguous()
    s  = bundle.s.squeeze(1).contiguous()
    Q  = bundle.Q.squeeze(1).contiguous()

    # Analytical profit bounds
    hist_min = -c * Q  # worst case: zero demand
    hist_max = (p - c) * Q  # best case: demand >> Q

    # Add margin
    margin = 0.1 * (hist_max - hist_min).clamp(min=1.0)
    hist_min = hist_min - margin
    hist_max = hist_max + margin

    kernel = TritonCVaRKernel(num_bins=256)
    out_mean, hist = kernel.launch(
        bundle.L, bundle.Z, mu, p, c, s, Q,
        hist_min, hist_max,
    )
    torch.cuda.synchronize()

    print(f"Expected profit range: [{out_mean.min().item():.2f}, {out_mean.max().item():.2f}]")
    print(f"Histogram shape: {hist.shape}")
    print(f"Histogram sum per product (should ~ {S}): "
          f"[{hist.sum(dim=1).min().item():.0f}, {hist.sum(dim=1).max().item():.0f}]")
