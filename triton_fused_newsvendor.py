"""
triton_fused_newsvendor.py — Autotuned, SRAM-fused Triton kernel for the
                              Multi-Echelon Stochastic Newsvendor problem.

╔═══════════════════════════════════════════════════════════════════════════╗
║  KEY INNOVATION                                                          ║
║  Standard PyTorch must materialise the full D = μ + L @ Z matrix in      ║
║  HBM (e.g. 2048 × 131 072 × 4 B ≈ 1 GB).  This kernel FUSES the        ║
║  matrix multiplication with the Newsvendor business logic so that D      ║
║  never leaves SRAM.  Only the final Expected_Profit[N] vector is         ║
║  written to HBM.                                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

Kernel layout
─────────────
Grid:  (⌈N / BLOCK_M⌉,  ⌈S / BLOCK_N⌉)
         ↑ product dim     ↑ scenario dim

Each program instance computes a [BLOCK_M, BLOCK_N] tile of the *fused*
profit matrix and reduces it to a partial mean of shape [BLOCK_M].
The partial means are atomically accumulated into the output array
Expected_Profit[N].

Memory traffic:
  READ  — tiles of L [BLOCK_M, BLOCK_K], Z [BLOCK_K, BLOCK_N],
           plus μ, p, c, s, Q vectors (each [BLOCK_M] — negligible).
  WRITE — partial mean [BLOCK_M] into Expected_Profit via atomic add.
  NEVER — the intermediate D[N, S] matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from data_pipeline import TensorBundle


# ═══════════════════════════════════════════════════════════════════════════
# Helper
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
    Hand-picked autotune configurations for the fused Newsvendor kernel.

    SRAM budget considerations per GPU architecture:
        T4  (Turing, sm_75):  64 KB shared memory per SM
        A100 (Ampere, sm_80): 164 KB shared memory per SM
        H100 (Hopper, sm_90): 228 KB shared memory per SM

    We target the **T4's 48 KB usable budget** (64 KB minus driver/Triton
    overhead) as the floor, so every config works on every GPU.

    Tile memory per K-loop iteration:
        L_tile  = BLOCK_M × BLOCK_K × 4 B
        Z_tile  = BLOCK_K × BLOCK_N × 4 B
        acc     = BLOCK_M × BLOCK_N × 4 B   (lives across all K iters)
        Total ≤ 48 KB = 49 152 B

    We keep the config count small (~10) to avoid multi-minute autotune
    on large problem sizes.
    """
    SRAM_BUDGET = 48 * 1024  # 48 KB — safe for T4 and above
    configs: list[triton.Config] = []

    # Curated (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) combos.
    # With FP16 tensor-core dot, larger tiles amortise the K-loop overhead
    # and maximise arithmetic intensity.
    candidates = [
        # (BM, BN, BK, warps, stages)  — SRAM usage noted
        ( 32, 128, 32, 4, 2),   # 20 KB  — wide scenario tile
        ( 64,  64, 32, 4, 2),   # 20 KB  — balanced
        ( 64, 128, 32, 4, 2),   # 36 KB  — sweet spot on T4
        ( 64, 128, 32, 8, 2),   # 36 KB  — more warps variant
        ( 64, 128, 64, 4, 2),   # 40 KB  — deeper K-tile
        ( 64, 128, 64, 8, 2),   # 40 KB  — deeper K, more warps
        (128,  64, 32, 4, 2),   # 36 KB  — tall M-tile
        (128,  64, 32, 8, 2),   # 36 KB  — tall M, more warps
        (128, 128, 32, 8, 2),   # 48 KB  — large tile, high throughput
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
    key=["N", "S", "K"],            # re-tune when problem size changes
)
@triton.jit
def _fused_newsvendor_kernel(
    # ── Pointers to input tensors ──────────────────────────────────────
    L_ptr,          # float32 [N, K]  — lower-triangular Cholesky factor
    Z_ptr,          # float32 [K, S]  — standard-normal scenario matrix
    mu_ptr,         # float32 [N, 1]  — mean demand
    p_ptr,          # float32 [N, 1]  — selling price
    c_ptr,          # float32 [N, 1]  — unit cost
    s_ptr,          # float32 [N, 1]  — salvage value
    Q_ptr,          # float32 [N, 1]  — order quantity
    # ── Pointer to output tensor ───────────────────────────────────────
    out_ptr,        # float32 [N]     — expected profit (atomically accumulated)
    # ── Problem dimensions ─────────────────────────────────────────────
    N: tl.constexpr,               # number of product-location nodes
    S: tl.constexpr,               # number of Monte-Carlo scenarios
    K: tl.constexpr,               # inner dimension of L @ Z (= N)
    # ── Strides (elements, NOT bytes) ──────────────────────────────────
    stride_L_m: tl.constexpr,      # L.stride(0)  — typically K (row-major)
    stride_L_k: tl.constexpr,      # L.stride(1)  — typically 1
    stride_Z_k: tl.constexpr,      # Z.stride(0)  — typically S
    stride_Z_n: tl.constexpr,      # Z.stride(1)  — typically 1
    # ── Compile-time tile dimensions (set by autotune) ─────────────────
    BLOCK_SIZE_M: tl.constexpr,    # tile height  (product axis)
    BLOCK_SIZE_N: tl.constexpr,    # tile width   (scenario axis)
    BLOCK_SIZE_K: tl.constexpr,    # tile depth   (reduction axis)
):
    """
    Fused Triton kernel: demand generation + Newsvendor profit + reduction.

    Each program instance handles a [BLOCK_M, BLOCK_N] tile of the *virtual*
    profit matrix (which is never fully materialised).

    ┌──────────────────────────────────────────────────────────────────────┐
    │  PHASE 1 — Tiled matmul:  acc = L[m_block, :] @ Z[:, n_block]      │
    │  PHASE 2 — Fused logic:   D = μ + acc;  X = min(D, Q);  π = …      │
    │  PHASE 3 — Partial mean:  partial = mean(π, axis=1) over BLOCK_N    │
    │  PHASE 4 — Atomic store:  out[m_block] += partial / n_tiles_N       │
    └──────────────────────────────────────────────────────────────────────┘
    """

    # ══════════════════════════════════════════════════════════════════════
    # 0.  Identify this program's tile position in the 2-D grid.
    # ══════════════════════════════════════════════════════════════════════
    #     pid_m indexes the product (M) axis;
    #     pid_n indexes the scenario (N) axis.
    pid_m = tl.program_id(0)   # which block of products   [0 .. ⌈N/BLOCK_M⌉)
    pid_n = tl.program_id(1)   # which block of scenarios  [0 .. ⌈S/BLOCK_N⌉)

    # Row indices within the M tile (product axis)
    # offs_m : [BLOCK_SIZE_M]  — e.g. [pid_m*64, pid_m*64+1, …, pid_m*64+63]
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Column indices within the N tile (scenario axis)
    # offs_n : [BLOCK_SIZE_N]  — e.g. [pid_n*128, …, pid_n*128+127]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # ══════════════════════════════════════════════════════════════════════
    # 1.  PHASE 1 — Tiled matrix multiplication:  acc = L[tile_m,:] @ Z[:,tile_n]
    #
    #     We iterate over the K dimension in chunks of BLOCK_SIZE_K.
    #     On each iteration we load:
    #       L_tile : [BLOCK_M, BLOCK_K]  from L[offs_m, offs_k]
    #       Z_tile : [BLOCK_K, BLOCK_N]  from Z[offs_k, offs_n]
    #     and accumulate:
    #       acc += L_tile @ Z_tile        (in SRAM, fp32)
    #
    #     After all K iterations, acc[i, j] = Σ_k L[m+i, k] · Z[k, n+j].
    # ══════════════════════════════════════════════════════════════════════

    # Accumulator lives in registers / SRAM for the entire K-loop.
    # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_N], initialised to zero.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Number of full K-tiles we need to iterate over.
    # (K is always a multiple of BLOCK_SIZE_K because N is a power of 2.)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for k_idx in range(0, num_k_tiles):
        # ── Compute K-offsets for this tile ────────────────────────────
        offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # ── Load L tile [BLOCK_M, BLOCK_K] ────────────────────────────
        # Pointer arithmetic:
        #   L_ptr + offs_m[:, None] * stride_L_m + offs_k[None, :] * stride_L_k
        # This gives a [BLOCK_M, BLOCK_K] grid of pointers into L.
        L_ptrs = (L_ptr
                  + offs_m[:, None] * stride_L_m
                  + offs_k[None, :] * stride_L_k)

        # Mask: do not read out-of-bounds elements.
        # offs_m < N guards the product axis; offs_k < K guards reduction.
        L_mask = (offs_m[:, None] < N) & (offs_k[None, :] < K)

        # tl.load: read from HBM into SRAM.  Out-of-bounds → 0.0.
        L_tile = tl.load(L_ptrs, mask=L_mask, other=0.0)
        #          ↑ shape [BLOCK_SIZE_M, BLOCK_SIZE_K], lives in SRAM

        # ── Load Z tile [BLOCK_K, BLOCK_N] ────────────────────────────
        Z_ptrs = (Z_ptr
                  + offs_k[:, None] * stride_Z_k
                  + offs_n[None, :] * stride_Z_n)
        Z_mask = (offs_k[:, None] < K) & (offs_n[None, :] < S)
        Z_tile = tl.load(Z_ptrs, mask=Z_mask, other=0.0)
        #          ↑ shape [BLOCK_SIZE_K, BLOCK_SIZE_N], lives in SRAM

        # ── Accumulate dot product in SRAM ────────────────────────────
        # ── Accumulate dot product in SRAM ────────────────────────────
        # tl.dot computes a matrix multiply of two tiles, both in SRAM.
        # Result is accumulated into `acc` (also SRAM / registers).
        # NO HBM traffic occurs here — this is the critical optimisation.
        acc += tl.dot(L_tile, Z_tile)

    # ══════════════════════════════════════════════════════════════════════
    # 2.  PHASE 2 — Fused Newsvendor business logic (all in SRAM).
    #
    #     At this point, acc[i, j] holds the j-th correlated demand
    #     component for product i, *before* adding the mean μ.
    #
    #     We now compute, entirely in SRAM / registers:
    #       D = max(μ + acc, 0)          — correlated demand (non-negative)
    #       X = min(D, Q)                — constrained sales
    #       π = p · X − c · Q + s · max(Q − D, 0)   — per-scenario profit
    # ══════════════════════════════════════════════════════════════════════

    # Load μ, p, c, s, Q for this product-block.
    # These are [N, 1] vectors; we load [BLOCK_M] elements and broadcast
    # across the scenario dimension via [:, None].
    m_mask = offs_m < N   # [BLOCK_SIZE_M] — guard out-of-bounds products

    mu_vals = tl.load(mu_ptr + offs_m, mask=m_mask, other=0.0)  # [BLOCK_M]
    p_vals  = tl.load(p_ptr  + offs_m, mask=m_mask, other=0.0)
    c_vals  = tl.load(c_ptr  + offs_m, mask=m_mask, other=0.0)
    s_vals  = tl.load(s_ptr  + offs_m, mask=m_mask, other=0.0)
    Q_vals  = tl.load(Q_ptr  + offs_m, mask=m_mask, other=0.0)

    # Broadcast from [BLOCK_M] → [BLOCK_M, BLOCK_N] by adding a new axis.
    mu_bc = mu_vals[:, None]   # [BLOCK_M, 1] — Triton broadcasts automatically
    p_bc  = p_vals[:, None]
    c_bc  = c_vals[:, None]
    s_bc  = s_vals[:, None]
    Q_bc  = Q_vals[:, None]

    # Demand (clamped to ≥ 0)
    D = mu_bc + acc                              # [BLOCK_M, BLOCK_N] — SRAM
    D = tl.maximum(D, 0.0)

    # Sales = min(demand, order quantity)
    X = tl.minimum(D, Q_bc)                      # [BLOCK_M, BLOCK_N] — SRAM

    # Overage inventory (unsold units → salvaged)
    overage = tl.maximum(Q_bc - D, 0.0)          # [BLOCK_M, BLOCK_N] — SRAM

    # Per-scenario profit  π = p·X − c·Q + s·(Q − D)⁺
    profit = (p_bc * X) - (c_bc * Q_bc) + (s_bc * overage)
    #          ↑ [BLOCK_M, BLOCK_N] — still entirely in SRAM

    # ══════════════════════════════════════════════════════════════════════
    # 3.  PHASE 3 — Partial mean reduction over the scenario axis.
    #
    #     We sum profit across the BLOCK_N (scenario) columns to get a
    #     [BLOCK_M] partial sum.  The full expectation is:
    #
    #         E[π_i] = (1/S) Σ_j π(i, j)
    #
    #     Since we tile the S axis into ⌈S/BLOCK_N⌉ tiles, each tile
    #     contributes  partial_sum / S  to the final answer.
    #
    #     We also need to handle the mask for the last tile where
    #     offs_n may exceed S.
    # ══════════════════════════════════════════════════════════════════════

    # Mask out-of-bounds scenarios (last tile may be partial).
    n_mask = offs_n < S                           # [BLOCK_SIZE_N]
    # Zero out profits for out-of-bounds scenarios before summing.
    profit = tl.where(n_mask[None, :], profit, 0.0)

    # Sum over scenarios:  [BLOCK_M, BLOCK_N] → [BLOCK_M]
    partial_sum = tl.sum(profit, axis=1)          # [BLOCK_M]

    # Normalise by S (total number of scenarios) to get the partial
    # contribution to the mean.
    partial_mean = partial_sum / S                # [BLOCK_M]

    # ══════════════════════════════════════════════════════════════════════
    # 4.  PHASE 4 — Atomic accumulation into the output array.
    #
    #     Multiple program instances along the scenario axis (pid_n = 0, 1, …)
    #     each produce a partial_mean for the same product indices.  We use
    #     ``tl.atomic_add`` to safely accumulate these partials.
    #
    #     After ALL programs complete:
    #       out[i] = Σ_{pid_n}  partial_mean(pid_n)[i]  =  E[π_i]
    #
    #     This is the ONLY write to HBM from this kernel.
    # ══════════════════════════════════════════════════════════════════════

    out_ptrs = out_ptr + offs_m                   # [BLOCK_M] pointers
    tl.atomic_add(out_ptrs, partial_mean, mask=m_mask)
    #   ↑ Atomic because multiple pid_n tiles write to the same product slots.


# ═══════════════════════════════════════════════════════════════════════════
# Python wrapper
# ═══════════════════════════════════════════════════════════════════════════
class TritonFusedNewsvendor:
    """
    High-level Python interface to the fused Triton kernel.

    Usage::

        solver = TritonFusedNewsvendor()
        result = solver.solve(bundle)
        print(result.expected_profit)     # shape [N]
    """

    def __init__(self) -> None:
        self.label = "Triton-Fused"

    # ------------------------------------------------------------------
    def solve(self, bundle: TensorBundle) -> "SolverResult":
        """
        Launch the fused kernel and return a ``SolverResult``.

        Steps
        -----
        1. Flatten μ, p, c, s, Q from [N, 1] → [N] for simpler pointer math.
        2. Allocate output buffer (zeroed) of shape [N].
        3. Compute the 2-D grid and launch the kernel.
        4. Wrap the result.
        """
        from baseline_solvers import SolverResult

        device = bundle.L.device
        assert device.type == "cuda", "Triton kernel requires CUDA device."

        N, S = bundle.N, bundle.S
        K = N  # inner dimension of L @ Z

        # Flatten column vectors to 1-D for the kernel
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.squeeze(1).contiguous()
        p  = bundle.p.squeeze(1).contiguous()
        c  = bundle.c.squeeze(1).contiguous()
        s  = bundle.s.squeeze(1).contiguous()
        Q  = bundle.Q.squeeze(1).contiguous()

        # Output buffer — zero-initialised for atomic_add
        out = torch.zeros(N, dtype=torch.float32, device=device)

        # Grid shape:  (⌈N / BLOCK_M⌉,  ⌈S / BLOCK_N⌉)
        # BLOCK_M, BLOCK_N are injected by autotune from the winning config.
        grid = lambda META: (
            cdiv(N, META["BLOCK_SIZE_M"]),
            cdiv(S, META["BLOCK_SIZE_N"]),
        )

        # ── Warm-up (triggers autotune on first call; cached after) ──
        # Measure peak memory during warmup (same as PyTorch solver).
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        _fused_newsvendor_kernel[grid](
            L, Z, mu, p, c, s, Q, out,
            N, S, K,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
        )
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)

        # ── Timed launch ─────────────────────────────────────────────
        out.zero_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Reuse the same grid lambda — autotune is already cached.
        _fused_newsvendor_kernel[grid](
            L, Z, mu, p, c, s, Q, out,
            N, S, K,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
        )

        end_event.record()
        torch.cuda.synchronize()

        wall_ms = start_event.elapsed_time(end_event)

        return SolverResult(
            expected_profit=out,
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
        )


# ---------------------------------------------------------------------------
# Quick self-test (requires CUDA)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from config import NewsvendorConfig
    from data_pipeline import DataPipeline
    from baseline_solvers import PyTorchMonteCarlo

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = NewsvendorConfig(N=256, S=8192, device="cuda")
    bundle = DataPipeline(cfg=cfg).run()

    # Reference
    pt = PyTorchMonteCarlo(use_compile=False)
    ref = pt.solve(bundle)

    # Triton
    tr = TritonFusedNewsvendor()
    tri = tr.solve(bundle)

    diff = (ref.expected_profit - tri.expected_profit).abs()
    print(f"Max diff vs PyTorch: {diff.max().item():.6f}")
    print(f"Mean diff:           {diff.mean().item():.6f}")
    print(f"PyTorch time:  {ref.wall_time_ms:.2f} ms")
    print(f"Triton  time:  {tri.wall_time_ms:.2f} ms")
