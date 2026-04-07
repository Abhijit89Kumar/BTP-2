"""
grid_search_solvers.py — Three solver implementations for Optimal Q* Grid Search.

1. CPUGridSearch     — Pure NumPy on CPU (gold-standard correctness reference).
2. PyTorchGridSearch — GPU solver using native torch ops + torch.compile.
3. TritonGridSearch  — GPU solver wrapping the fused Triton grid-search kernel.

All three return ``GridSearchResult`` with:
  - profit_surface[N, K]  — expected profit at each grid point
  - Q_star[N]             — optimal order quantity per product
  - best_profit[N]        — E[profit] at Q*
  - Q_grid[K]             — the ratio multipliers used

Usage::

    from config import GridSearchConfig
    from data_pipeline import DataPipeline, TensorBundle
    from extensions.grid_search_solvers import TritonGridSearch

    bundle = DataPipeline(cfg=...).run()
    grid_cfg = GridSearchConfig(K=64, q_ratio_min=0.3, q_ratio_max=2.5)

    solver = TritonGridSearch()
    result = solver.solve(bundle, grid_cfg)
    print(result.Q_star, result.best_profit)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from data_pipeline import TensorBundle
from config import GridSearchConfig
from extensions.common import GridSearchResult


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════
def _build_q_ratios(grid_cfg: GridSearchConfig) -> np.ndarray:
    """
    Return [K] uniformly-spaced ratio multipliers in [q_ratio_min, q_ratio_max].

    Q_grid[k] is a *ratio* — actual order quantity for product i is mu_i * Q_grid[k].
    """
    return np.linspace(
        grid_cfg.q_ratio_min,
        grid_cfg.q_ratio_max,
        grid_cfg.K,
        dtype=np.float64,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Solver 1 — CPU (NumPy)
# ═══════════════════════════════════════════════════════════════════════════
class CPUGridSearch:
    """
    Pure NumPy CPU grid search over K order-quantity levels.

    Steps:
      1. Compute D[N, S] = max(mu + L @ Z, 0)
      2. For each k in range(K):
           Q_val = mu * ratio[k]
           profit_k = p * min(D, Q_val) - c * Q_val + s * max(Q_val - D, 0)
           profit_surface[:, k] = mean(profit_k, axis=1)
      3. Q_star = mu * ratio[argmax(profit_surface, axis=1)]
    """

    def __init__(self) -> None:
        self.label = "CPU-GridSearch"

    def solve(
        self, bundle: TensorBundle, grid_cfg: GridSearchConfig
    ) -> GridSearchResult:
        # Move to CPU NumPy
        L  = bundle.L.cpu().numpy()
        Z  = bundle.Z.cpu().numpy()
        mu = bundle.mu.cpu().numpy()    # [N, 1]
        p  = bundle.p.cpu().numpy()     # [N, 1]
        c  = bundle.c.cpu().numpy()     # [N, 1]
        s  = bundle.s.cpu().numpy()     # [N, 1]

        K = grid_cfg.K
        N, S = bundle.N, bundle.S
        q_ratios = _build_q_ratios(grid_cfg)  # [K]

        t0 = time.perf_counter()

        # 1. Correlated demand [N, S]
        D = mu + (L @ Z)
        del L, Z
        np.maximum(D, 0.0, out=D)

        # 2. Evaluate profit at each grid point
        profit_surface = np.empty((N, K), dtype=np.float64)

        for k in range(K):
            Q_val = mu * q_ratios[k]                          # [N, 1]
            X = np.minimum(D, Q_val)                          # [N, S]
            overage = np.maximum(Q_val - D, 0.0)              # [N, S]
            profit_k = (p * X) - (c * Q_val) + (s * overage)  # [N, S]
            profit_surface[:, k] = profit_k.mean(axis=1)      # [N]

        # 3. Find optimal Q* per product
        best_k = profit_surface.argmax(axis=1)                 # [N]
        best_ratios = q_ratios[best_k]                         # [N]
        Q_star = mu.squeeze(1) * best_ratios                   # [N]
        best_profit = profit_surface[np.arange(N), best_k]     # [N]

        wall_ms = (time.perf_counter() - t0) * 1e3

        return GridSearchResult(
            expected_profit=torch.tensor(best_profit, dtype=torch.float32),
            wall_time_ms=wall_ms,
            peak_memory_bytes=0,
            label=self.label,
            Q_star=torch.tensor(Q_star, dtype=torch.float32),
            best_profit=torch.tensor(best_profit, dtype=torch.float32),
            profit_surface=torch.tensor(profit_surface, dtype=torch.float32),
            Q_grid=torch.tensor(q_ratios, dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Solver 2 — PyTorch GPU (torch.compile)
# ═══════════════════════════════════════════════════════════════════════════
class PyTorchGridSearch:
    """
    PyTorch GPU grid search using torch.compile on the inner profit function.

    Strategy: compute D[N, S] once, then loop over K grid points on GPU.
    Each iteration is a fast elementwise + reduction — avoids the OOM risk
    of broadcasting Q_all[N, K, 1] vs D[N, 1, S] for large K * S.
    """

    def __init__(self, use_compile: bool = True) -> None:
        self.label = "PyTorch-GridSearch-Compile" if use_compile else "PyTorch-GridSearch-Eager"
        self.use_compile = use_compile
        self._compiled_fn: Optional[object] = None

    @staticmethod
    def _profit_at_q(
        D: torch.Tensor,     # [N, S]
        mu: torch.Tensor,    # [N, 1]
        p: torch.Tensor,     # [N, 1]
        c: torch.Tensor,     # [N, 1]
        s: torch.Tensor,     # [N, 1]
        q_ratio: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        """Compute E[profit] at Q = mu * q_ratio. Returns [N]."""
        Q_val = mu * q_ratio                                  # [N, 1]
        X = torch.minimum(D, Q_val)                           # [N, S]
        overage = torch.clamp(Q_val - D, min=0.0)             # [N, S]
        profit = (p * X) - (c * Q_val) + (s * overage)        # [N, S]
        return profit.mean(dim=1)                              # [N]

    def solve(
        self, bundle: TensorBundle, grid_cfg: GridSearchConfig
    ) -> GridSearchResult:
        device = bundle.L.device
        K = grid_cfg.K
        N, S = bundle.N, bundle.S

        # Prepare tensors
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.contiguous()
        p  = bundle.p.contiguous()
        c  = bundle.c.contiguous()
        s  = bundle.s.contiguous()

        q_ratios_np = _build_q_ratios(grid_cfg)
        q_ratios = torch.tensor(q_ratios_np, dtype=torch.float32, device=device)

        # Optionally compile the inner function
        fn = self._profit_at_q
        if self.use_compile and self._compiled_fn is None:
            self._compiled_fn = torch.compile(fn, backend="inductor")
        if self.use_compile:
            fn = self._compiled_fn

        # ── Warm-up ──────────────────────────────────────────────────────
        if device.type == "cuda":
            D_warmup = torch.clamp(mu + torch.mm(L, Z), min=0.0)
            _ = fn(D_warmup, mu, p, c, s, q_ratios[0])
            torch.cuda.synchronize()
            del D_warmup

        # ── Memory measurement ───────────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            D_mem = torch.clamp(mu + torch.mm(L, Z), min=0.0)
            profit_surface_mem = torch.empty(N, K, dtype=torch.float32, device=device)
            for k in range(K):
                profit_surface_mem[:, k] = fn(D_mem, mu, p, c, s, q_ratios[k])
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device)
            del D_mem, profit_surface_mem
        else:
            peak_mem = 0

        # ── Timed run ────────────────────────────────────────────────────
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()

        t0 = time.perf_counter()

        # Compute demand ONCE
        D = torch.clamp(mu + torch.mm(L, Z), min=0.0)   # [N, S]

        # Evaluate profit at each grid point
        profit_surface = torch.empty(N, K, dtype=torch.float32, device=device)
        for k in range(K):
            profit_surface[:, k] = fn(D, mu, p, c, s, q_ratios[k])

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            wall_ms = start_event.elapsed_time(end_event)
        else:
            wall_ms = (time.perf_counter() - t0) * 1e3

        # Post-process: find optimal Q* per product
        best_k = profit_surface.argmax(dim=1)                  # [N]
        best_ratios = q_ratios[best_k]                         # [N]
        Q_star = mu.squeeze(1) * best_ratios                   # [N]
        best_profit = profit_surface[torch.arange(N, device=device), best_k]

        return GridSearchResult(
            expected_profit=best_profit.detach(),
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
            Q_star=Q_star.detach(),
            best_profit=best_profit.detach(),
            profit_surface=profit_surface.detach(),
            Q_grid=q_ratios.detach(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Solver 3 — Triton Fused Grid Search
# ═══════════════════════════════════════════════════════════════════════════
class TritonGridSearch:
    """
    Wrapper around the fused Triton grid-search kernel.

    The kernel performs the matmul ONCE in SRAM, then loops over K grid
    points — writing profit_surface[N, K] via atomic_add.  Post-processing
    (argmax to find Q*) is done on the host after the kernel completes.

    Follows the same warm-up / memory measurement / timed run pattern as
    the base ``TritonFusedNewsvendor.solve()``.
    """

    def __init__(self) -> None:
        self.label = "Triton-GridSearch"

    def solve(
        self, bundle: TensorBundle, grid_cfg: GridSearchConfig
    ) -> GridSearchResult:
        from extensions.grid_search_kernel import (
            _grid_search_kernel,
            cdiv,
        )

        device = bundle.L.device
        assert device.type == "cuda", "Triton kernel requires CUDA device."

        N, S = bundle.N, bundle.S
        INNER_K = N   # inner dimension of L @ Z
        NUM_Q = grid_cfg.K

        # Flatten column vectors to 1-D for the kernel
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.squeeze(1).contiguous()
        p  = bundle.p.squeeze(1).contiguous()
        c  = bundle.c.squeeze(1).contiguous()
        s  = bundle.s.squeeze(1).contiguous()

        # Build Q grid on device
        q_ratios_np = _build_q_ratios(grid_cfg).astype(np.float32)
        Q_grid = torch.tensor(q_ratios_np, dtype=torch.float32, device=device).contiguous()

        # Output buffer — zero-initialised for atomic_add
        out = torch.zeros(N, NUM_Q, dtype=torch.float32, device=device)

        # Grid shape: (ceil(N / BLOCK_M), ceil(S / BLOCK_N))
        grid = lambda META: (
            cdiv(N, META["BLOCK_SIZE_M"]),
            cdiv(S, META["BLOCK_SIZE_N"]),
        )

        # ── Warm-up (triggers autotune on first call; cached after) ──
        _grid_search_kernel[grid](
            L, Z, mu, p, c, s, Q_grid,
            out,
            N, S, INNER_K, NUM_Q,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            out.stride(0), out.stride(1),
        )
        torch.cuda.synchronize()

        # ── Memory measurement (autotune already cached) ─────────────
        out.zero_()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        _grid_search_kernel[grid](
            L, Z, mu, p, c, s, Q_grid,
            out,
            N, S, INNER_K, NUM_Q,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            out.stride(0), out.stride(1),
        )
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)

        # ── Timed launch ─────────────────────────────────────────────
        out.zero_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        start_event.record()

        _grid_search_kernel[grid](
            L, Z, mu, p, c, s, Q_grid,
            out,
            N, S, INNER_K, NUM_Q,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            out.stride(0), out.stride(1),
        )

        end_event.record()
        torch.cuda.synchronize()

        wall_ms = start_event.elapsed_time(end_event)

        # ── Post-process: argmax to find Q* per product ──────────────
        profit_surface = out   # [N, NUM_Q]
        best_k = profit_surface.argmax(dim=1)                            # [N]
        best_ratios = Q_grid[best_k]                                     # [N]
        Q_star = mu * best_ratios                                        # [N]
        best_profit = profit_surface[torch.arange(N, device=device), best_k]  # [N]

        return GridSearchResult(
            expected_profit=best_profit,
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
            Q_star=Q_star,
            best_profit=best_profit,
            profit_surface=profit_surface,
            Q_grid=Q_grid,
        )


# ---------------------------------------------------------------------------
# Quick self-test (requires CUDA)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from config import NewsvendorConfig, GridSearchConfig
    from data_pipeline import DataPipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = NewsvendorConfig(N=256, S=8192, device="cuda")
    grid_cfg = GridSearchConfig(K=32, q_ratio_min=0.3, q_ratio_max=2.5)
    bundle = DataPipeline(cfg=cfg).run()

    # CPU reference
    cpu = CPUGridSearch()
    res_cpu = cpu.solve(bundle, grid_cfg)
    print(f"[{res_cpu.label}]  time={res_cpu.wall_time_ms:.1f} ms  "
          f"best_profit range=[{res_cpu.best_profit.min():.2f}, "
          f"{res_cpu.best_profit.max():.2f}]")

    # PyTorch GPU
    pt = PyTorchGridSearch(use_compile=False)
    res_pt = pt.solve(bundle, grid_cfg)
    print(f"[{res_pt.label}]  time={res_pt.wall_time_ms:.1f} ms  "
          f"best_profit range=[{res_pt.best_profit.min():.2f}, "
          f"{res_pt.best_profit.max():.2f}]")

    # Triton
    tr = TritonGridSearch()
    res_tr = tr.solve(bundle, grid_cfg)
    print(f"[{res_tr.label}]  time={res_tr.wall_time_ms:.1f} ms  "
          f"best_profit range=[{res_tr.best_profit.min():.2f}, "
          f"{res_tr.best_profit.max():.2f}]")

    # Numerical comparison
    diff_pt = (res_cpu.best_profit - res_pt.best_profit).abs().max().item()
    diff_tr = (res_cpu.best_profit - res_tr.best_profit).abs().max().item()
    print(f"\nMax |best_profit| diff CPU vs PyTorch: {diff_pt:.6f}")
    print(f"Max |best_profit| diff CPU vs Triton:  {diff_tr:.6f}")

    diff_qs = (res_cpu.Q_star - res_tr.Q_star).abs().max().item()
    print(f"Max |Q_star| diff CPU vs Triton:       {diff_qs:.6f}")
