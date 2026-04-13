"""
substitution_kernel.py -- Two-pass Triton kernels for the Multi-Product
                          Substitution Newsvendor extension.

When a product stocks out, a fraction of its unmet demand is redirected to
substitute products.  This cross-product coupling means the full stockout
matrix ``max(D - Q, 0)`` of shape ``[N, S]`` must be materialised in HBM
so that each product can read its neighbours' stockouts.

Kernel architecture (two-pass)
------------------------------

**Pass 1 -- ``_stockout_kernel``**
    Identical Phase 1 (tiled matmul) and Phase 2 (demand generation) as the
    base fused kernel, but instead of computing profit it writes the per-
    scenario stockout to HBM:

        stockout[i, j] = max(D[i, j] - Q[i], 0)

    This is the *only* write to HBM from Pass 1.

**Pass 2 -- ``_substitution_profit_kernel``**
    For each product tile [BLOCK_M]:
      1. Recompute D[BLOCK_M, BLOCK_N] via the same tiled matmul (Phase 1).
      2. For each substitute k (up to max_subs):
            redirected += sub_frac[j, k] * stockout[sub_idx[j, k], scenario_tile]
      3. Effective demand:  D_eff = D + redirected
      4. Profit with D_eff, reduce to partial mean, atomic_add to output.
      5. Also reduce redirected demand and effective profit for diagnostics.

    Recomputing the matmul in Pass 2 costs an extra O(N^2 S) FLOPs but
    avoids storing the full D[N, S] matrix (1.07 GB at N=2048, S=131072),
    cutting HBM footprint by half compared to a store-and-load approach.

Both kernels share the same autotune configuration space, targeting the T4's
48 KB usable SRAM budget.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# =========================================================================
# Helpers (same as base kernel)
# =========================================================================
def next_power_of_2(n: int) -> int:
    """Round up to the nearest power of 2."""
    return 1 << (n - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# =========================================================================
# Autotune configuration (shared by both passes)
# =========================================================================
def _build_substitution_autotune_configs() -> list[triton.Config]:
    """
    Hand-picked autotune configurations for substitution kernels.

    Same SRAM budget and tile-size logic as the base fused kernel -- both
    passes perform a tiled matmul with the same L/Z tile shapes.

    SRAM per K-loop iteration:
        L_tile  = BLOCK_M x BLOCK_K x 4 B
        Z_tile  = BLOCK_K x BLOCK_N x 4 B
        acc     = BLOCK_M x BLOCK_N x 4 B   (lives across all K iters)
        Total <= 48 KB = 49 152 B
    """
    SRAM_BUDGET = 48 * 1024  # 48 KB -- safe for T4 and above
    configs: list[triton.Config] = []

    candidates = [
        # (BM, BN, BK, warps, stages)  -- SRAM usage noted
        ( 32, 128, 32, 4, 2),   # 20 KB  -- wide scenario tile
        ( 64,  64, 32, 4, 2),   # 20 KB  -- balanced
        ( 64, 128, 32, 4, 2),   # 36 KB  -- sweet spot on T4
        ( 64, 128, 32, 8, 2),   # 36 KB  -- more warps variant
        ( 64, 128, 64, 4, 2),   # 40 KB  -- deeper K-tile
        ( 64, 128, 64, 8, 2),   # 40 KB  -- deeper K, more warps
        (128,  64, 32, 4, 2),   # 36 KB  -- tall M-tile
        (128,  64, 32, 8, 2),   # 36 KB  -- tall M, more warps
        (128, 128, 32, 8, 2),   # 48 KB  -- large tile, high throughput
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
# Pass 1 -- Stockout kernel
# =========================================================================
@triton.autotune(
    configs=_build_substitution_autotune_configs(),
    key=["N", "S", "K"],
)
@triton.jit
def _stockout_kernel(
    # -- Pointers to input tensors ------------------------------------------
    L_ptr,          # float32 [N, K]  -- lower-triangular Cholesky factor
    Z_ptr,          # float32 [K, S]  -- standard-normal scenario matrix
    mu_ptr,         # float32 [N]     -- mean demand
    Q_ptr,          # float32 [N]     -- order quantity
    # -- Pointer to output tensor -------------------------------------------
    stockout_ptr,   # float32 [N, S]  -- output: max(D - Q, 0)
    # -- Problem dimensions -------------------------------------------------
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    # -- Strides (elements, NOT bytes) --------------------------------------
    stride_L_m: tl.constexpr,
    stride_L_k: tl.constexpr,
    stride_Z_k: tl.constexpr,
    stride_Z_n: tl.constexpr,
    stride_so_m: tl.constexpr,     # stockout.stride(0) -- typically S
    stride_so_n: tl.constexpr,     # stockout.stride(1) -- typically 1
    # -- Tile dimensions (set by autotune) ----------------------------------
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Pass 1: Fused matmul -> demand -> stockout, written to HBM.

    Grid: (ceil(N / BLOCK_M), ceil(S / BLOCK_N))

    For each [BLOCK_M, BLOCK_N] tile, computes:
        acc      = L[tile_m, :] @ Z[:, tile_n]
        D        = max(mu + acc, 0)
        stockout = max(D - Q, 0)
    and writes stockout to HBM.
    """

    # ====================================================================
    # 0. Tile position
    # ====================================================================
    pid_m = tl.program_id(0)   # product axis
    pid_n = tl.program_id(1)   # scenario axis

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # ====================================================================
    # 1. Phase 1 -- Tiled matmul: acc = L[tile_m, :] @ Z[:, tile_n]
    # ====================================================================
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

        # Accumulate in SRAM
        acc += tl.dot(L_tile, Z_tile)

    # ====================================================================
    # 2. Phase 2 -- Demand and stockout computation (all in SRAM)
    # ====================================================================
    m_mask = offs_m < N
    n_mask = offs_n < S

    mu_vals = tl.load(mu_ptr + offs_m, mask=m_mask, other=0.0)
    Q_vals  = tl.load(Q_ptr  + offs_m, mask=m_mask, other=0.0)

    # Broadcast [BLOCK_M] -> [BLOCK_M, BLOCK_N]
    mu_bc = mu_vals[:, None]
    Q_bc  = Q_vals[:, None]

    # Demand (clamped to >= 0)
    D = tl.maximum(mu_bc + acc, 0.0)

    # Stockout = max(D - Q, 0)
    stockout = tl.maximum(D - Q_bc, 0.0)

    # ====================================================================
    # 3. Write stockout to HBM
    # ====================================================================
    so_ptrs = (stockout_ptr
               + offs_m[:, None] * stride_so_m
               + offs_n[None, :] * stride_so_n)
    so_mask = (offs_m[:, None] < N) & (offs_n[None, :] < S)
    tl.store(so_ptrs, stockout, mask=so_mask)


# =========================================================================
# Pass 2 -- Substitution profit kernel
# =========================================================================
@triton.autotune(
    configs=_build_substitution_autotune_configs(),
    key=["N", "S", "K"],
)
@triton.jit
def _substitution_profit_kernel(
    # -- Pointers to input tensors ------------------------------------------
    L_ptr,          # float32 [N, K]  -- lower-triangular Cholesky factor
    Z_ptr,          # float32 [K, S]  -- standard-normal scenario matrix
    mu_ptr,         # float32 [N]     -- mean demand
    p_ptr,          # float32 [N]     -- selling price
    c_ptr,          # float32 [N]     -- unit cost
    s_ptr,          # float32 [N]     -- salvage value
    Q_ptr,          # float32 [N]     -- order quantity
    stockout_ptr,   # float32 [N, S]  -- stockout from Pass 1
    sub_idx_ptr,    # int64   [N, MAX_SUBS]  -- substitute indices (-1 = none)
    sub_frac_ptr,   # float32 [N, MAX_SUBS]  -- substitution fractions beta
    # -- Pointers to output tensors -----------------------------------------
    out_ptr,        # float32 [N]     -- expected profit (atomic accumulation)
    sub_demand_ptr, # float32 [N]     -- avg redirected demand (atomic accum.)
    eff_profit_ptr, # float32 [N]     -- effective profit (atomic accumulation)
    # -- Problem dimensions -------------------------------------------------
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    MAX_SUBS: tl.constexpr,
    # -- Strides (elements, NOT bytes) --------------------------------------
    stride_L_m: tl.constexpr,
    stride_L_k: tl.constexpr,
    stride_Z_k: tl.constexpr,
    stride_Z_n: tl.constexpr,
    stride_so_m: tl.constexpr,     # stockout.stride(0)
    stride_so_n: tl.constexpr,     # stockout.stride(1)
    stride_si_m: tl.constexpr,     # sub_idx.stride(0) -- typically MAX_SUBS
    stride_si_k: tl.constexpr,     # sub_idx.stride(1) -- typically 1
    stride_sf_m: tl.constexpr,     # sub_frac.stride(0)
    stride_sf_k: tl.constexpr,     # sub_frac.stride(1)
    # -- Tile dimensions (set by autotune) ----------------------------------
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Pass 2: Recompute demand via matmul, load substitute stockouts,
    compute effective demand and profit, reduce, and atomic_add.

    Grid: (ceil(N / BLOCK_M), ceil(S / BLOCK_N))

    For each [BLOCK_M, BLOCK_N] tile:
      1. Tiled matmul: acc = L[tile_m, :] @ Z[:, tile_n]
      2. D = max(mu + acc, 0)
      3. For each substitute k of product j:
            redirected += sub_frac[j,k] * stockout[sub_idx[j,k], tile_n]
      4. D_eff = D + redirected
      5. X = min(D_eff, Q);  profit = p*X - c*Q + s*max(Q - D_eff, 0)
      6. Partial mean reduction -> atomic_add to out, sub_demand, eff_profit
    """

    # ====================================================================
    # 0. Tile position
    # ====================================================================
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # ====================================================================
    # 1. Phase 1 -- Tiled matmul (recompute demand)
    # ====================================================================
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for k_idx in range(0, num_k_tiles):
        offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        L_ptrs = (L_ptr
                  + offs_m[:, None] * stride_L_m
                  + offs_k[None, :] * stride_L_k)
        L_mask = (offs_m[:, None] < N) & (offs_k[None, :] < K)
        L_tile = tl.load(L_ptrs, mask=L_mask, other=0.0)

        Z_ptrs = (Z_ptr
                  + offs_k[:, None] * stride_Z_k
                  + offs_n[None, :] * stride_Z_n)
        Z_mask = (offs_k[:, None] < K) & (offs_n[None, :] < S)
        Z_tile = tl.load(Z_ptrs, mask=Z_mask, other=0.0)

        acc += tl.dot(L_tile, Z_tile)

    # ====================================================================
    # 2. Phase 2 -- Demand + substitution + profit (all in SRAM)
    # ====================================================================
    m_mask = offs_m < N
    n_mask = offs_n < S

    mu_vals = tl.load(mu_ptr + offs_m, mask=m_mask, other=0.0)
    p_vals  = tl.load(p_ptr  + offs_m, mask=m_mask, other=0.0)
    c_vals  = tl.load(c_ptr  + offs_m, mask=m_mask, other=0.0)
    s_vals  = tl.load(s_ptr  + offs_m, mask=m_mask, other=0.0)
    Q_vals  = tl.load(Q_ptr  + offs_m, mask=m_mask, other=0.0)

    # Broadcast [BLOCK_M] -> [BLOCK_M, BLOCK_N]
    mu_bc = mu_vals[:, None]
    p_bc  = p_vals[:, None]
    c_bc  = c_vals[:, None]
    s_bc  = s_vals[:, None]
    Q_bc  = Q_vals[:, None]

    # Original demand
    D = tl.maximum(mu_bc + acc, 0.0)   # [BLOCK_M, BLOCK_N]

    # ----------------------------------------------------------------
    # Substitution: accumulate redirected demand from substitutes
    # ----------------------------------------------------------------
    # redirected[i, j] = sum_k sub_frac[i,k] * stockout[sub_idx[i,k], j]
    redirected = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_sub in range(0, MAX_SUBS):
        # Load sub_idx[offs_m, k_sub] -- shape [BLOCK_M]
        si_ptrs = sub_idx_ptr + offs_m * stride_si_m + k_sub * stride_si_k
        sub_indices = tl.load(si_ptrs, mask=m_mask, other=-1)
        # sub_indices: [BLOCK_M] int64 -- index of k-th substitute (-1 = none)

        # Load sub_frac[offs_m, k_sub] -- shape [BLOCK_M]
        sf_ptrs = sub_frac_ptr + offs_m * stride_sf_m + k_sub * stride_sf_k
        sub_fracs = tl.load(sf_ptrs, mask=m_mask, other=0.0)
        # sub_fracs: [BLOCK_M] float32

        # Broadcast sub_fracs to [BLOCK_M, BLOCK_N]
        sf_bc = sub_fracs[:, None]

        # For each substitute index, load stockout[sub_idx, offs_n]
        # sub_indices is [BLOCK_M], offs_n is [BLOCK_N]
        # We need stockout_ptr[sub_indices[i], offs_n[j]] for all i,j
        # Pointer: stockout_ptr + sub_indices[:, None] * stride_so_m
        #                       + offs_n[None, :] * stride_so_n
        #
        # Mask: sub_indices >= 0 (valid substitute) AND offs_n < S
        valid_sub = sub_indices >= 0  # [BLOCK_M]
        combined_mask = (valid_sub[:, None]) & (n_mask[None, :])

        # Clamp negative indices to 0 for safe pointer arithmetic
        # (masked loads will return 0.0 for invalid entries anyway)
        safe_indices = tl.maximum(sub_indices, 0)

        so_ptrs = (stockout_ptr
                   + safe_indices[:, None] * stride_so_m
                   + offs_n[None, :] * stride_so_n)
        sub_stockout = tl.load(so_ptrs, mask=combined_mask, other=0.0)
        # sub_stockout: [BLOCK_M, BLOCK_N]

        redirected += sf_bc * sub_stockout

    # ----------------------------------------------------------------
    # Effective demand and profit
    # ----------------------------------------------------------------
    D_eff = D + redirected                          # [BLOCK_M, BLOCK_N]
    X = tl.minimum(D_eff, Q_bc)                     # sales
    overage = tl.maximum(Q_bc - D_eff, 0.0)         # unsold inventory
    profit = (p_bc * X) - (c_bc * Q_bc) + (s_bc * overage)

    # ====================================================================
    # 3. Phase 3 -- Partial mean reduction over scenario axis
    # ====================================================================
    # Mask out-of-bounds scenarios
    profit     = tl.where(n_mask[None, :], profit, 0.0)
    redirected = tl.where(n_mask[None, :], redirected, 0.0)

    # Sum over scenarios: [BLOCK_M, BLOCK_N] -> [BLOCK_M]
    partial_profit = tl.sum(profit, axis=1) / S
    partial_redir  = tl.sum(redirected, axis=1) / S

    # Also compute effective profit = profit under substitution
    # (same as partial_profit but kept as a separate output for diagnostics)
    partial_eff    = partial_profit  # same quantity by definition

    # ====================================================================
    # 4. Phase 4 -- Atomic accumulation into output arrays
    # ====================================================================
    out_ptrs = out_ptr + offs_m
    tl.atomic_add(out_ptrs, partial_profit, mask=m_mask)

    sd_ptrs = sub_demand_ptr + offs_m
    tl.atomic_add(sd_ptrs, partial_redir, mask=m_mask)

    ep_ptrs = eff_profit_ptr + offs_m
    tl.atomic_add(ep_ptrs, partial_eff, mask=m_mask)


# =========================================================================
# Python wrapper for the two-pass substitution kernel
# =========================================================================
class TritonSubstitutionKernels:
    """
    Manages the two-pass Triton kernel launch for the substitution extension.

    Usage::

        kernels = TritonSubstitutionKernels()
        out, sub_demand, eff_profit, stockout = kernels.launch(
            bundle, sub_idx, sub_frac, max_subs=4
        )
    """

    @staticmethod
    def launch(
        bundle,
        sub_idx: torch.Tensor,
        sub_frac: torch.Tensor,
        max_subs: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute both passes and return (expected_profit, sub_demand,
        effective_profit, stockout).

        Parameters
        ----------
        bundle : TensorBundle
            Standard input bundle with L, Z, mu, p, c, s, Q tensors.
        sub_idx : torch.Tensor
            Shape [N, max_subs], dtype int64.  -1 for unused slots.
        sub_frac : torch.Tensor
            Shape [N, max_subs], dtype float32.  Substitution fractions.
        max_subs : int
            Maximum substitutes per product (compile-time constant).

        Returns
        -------
        expected_profit : torch.Tensor [N]
        sub_demand      : torch.Tensor [N]  -- avg redirected demand
        eff_profit      : torch.Tensor [N]  -- profit under substitution
        stockout        : torch.Tensor [N, S] -- intermediate (for debugging)
        """
        device = bundle.L.device
        assert device.type == "cuda", "Triton kernels require CUDA device."

        N, S = bundle.N, bundle.S
        K = N  # inner dimension of L @ Z

        # Flatten column vectors to 1-D
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.squeeze(1).contiguous()
        p  = bundle.p.squeeze(1).contiguous()
        c  = bundle.c.squeeze(1).contiguous()
        s  = bundle.s.squeeze(1).contiguous()
        Q  = bundle.Q.squeeze(1).contiguous()

        # Ensure substitution tensors are contiguous and on device
        sub_idx  = sub_idx.contiguous().to(device=device, dtype=torch.int64)
        sub_frac = sub_frac.contiguous().to(device=device, dtype=torch.float32)

        # Allocate intermediate stockout buffer [N, S]
        stockout = torch.empty(N, S, dtype=torch.float32, device=device)

        # Allocate output buffers -- zero-initialised for atomic_add
        out        = torch.zeros(N, dtype=torch.float32, device=device)
        sub_demand = torch.zeros(N, dtype=torch.float32, device=device)
        eff_profit = torch.zeros(N, dtype=torch.float32, device=device)

        # Grid: (ceil(N / BLOCK_M), ceil(S / BLOCK_N))
        grid = lambda META: (
            cdiv(N, META["BLOCK_SIZE_M"]),
            cdiv(S, META["BLOCK_SIZE_N"]),
        )

        # ---- Pass 1: Compute stockout[N, S] ----
        _stockout_kernel[grid](
            L, Z, mu, Q,
            stockout,
            N, S, K,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            stockout.stride(0), stockout.stride(1),
        )

        # ---- Pass 2: Compute profit with substitution ----
        _substitution_profit_kernel[grid](
            L, Z, mu, p, c, s, Q,
            stockout,
            sub_idx, sub_frac,
            out, sub_demand, eff_profit,
            N, S, K, max_subs,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            stockout.stride(0), stockout.stride(1),
            sub_idx.stride(0), sub_idx.stride(1),
            sub_frac.stride(0), sub_frac.stride(1),
        )

        torch.cuda.synchronize()
        return out, sub_demand, eff_profit, stockout
