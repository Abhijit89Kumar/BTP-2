"""
substitution_solvers.py -- Three solver implementations for the Multi-Product
                           Substitution Newsvendor extension.

Solvers
-------
1. ``CPUSubstitution``     -- Pure NumPy on CPU (gold-standard correctness).
2. ``PyTorchSubstitution`` -- Native ``torch`` ops on GPU with optional
                              ``torch.compile`` (Inductor backend).
3. ``TritonSubstitution``  -- Two-pass fused Triton kernel (see
                              ``substitution_kernel.py`` for architecture).

All three return a ``SubstitutionResult`` (defined in ``extensions/common.py``)
with identical semantics so that the benchmark can compare them uniformly.

Substitution logic (shared by all solvers)
------------------------------------------
1. Correlated demand:  D = max(mu + L @ Z, 0)            [N, S]
2. Stockout:           stockout = max(D - Q, 0)           [N, S]
3. Redirected demand:  redirected[j] = sum_k beta[j,k] * stockout[sub_idx[j,k]]
4. Effective demand:   D_eff = D + redirected             [N, S]
5. Sales:              X = min(D_eff, Q)                  [N, S]
6. Profit:             pi = p*X - c*Q + s*max(Q - D_eff, 0)
7. Expected profit:    E[pi] = mean(pi, axis=1)           [N]
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from data_pipeline import TensorBundle
from extensions.common import SubstitutionResult


# =========================================================================
# Solver 1 -- CPU (NumPy)
# =========================================================================
class CPUSubstitution:
    """
    Pure NumPy Monte-Carlo Newsvendor with multi-product substitution.

    Intentionally un-optimised for readability and correctness validation.
    """

    def __init__(self) -> None:
        self.label = "CPU-Substitution"

    def solve(
        self,
        bundle: TensorBundle,
        sub_idx: np.ndarray,
        sub_frac: np.ndarray,
    ) -> SubstitutionResult:
        """
        Execute substitution newsvendor on CPU.

        Parameters
        ----------
        bundle : TensorBundle
            Standard input tensors (may be on GPU -- will be moved to CPU).
        sub_idx : np.ndarray
            Shape [N, max_subs], dtype int64.  -1 for unused slots.
        sub_frac : np.ndarray
            Shape [N, max_subs], dtype float64.  Substitution fractions.

        Returns
        -------
        SubstitutionResult
        """
        # Move to CPU NumPy
        L  = bundle.L.cpu().numpy()
        Z  = bundle.Z.cpu().numpy()
        mu = bundle.mu.cpu().numpy()     # [N, 1]
        p  = bundle.p.cpu().numpy()      # [N, 1]
        c  = bundle.c.cpu().numpy()      # [N, 1]
        s  = bundle.s.cpu().numpy()      # [N, 1]
        Q  = bundle.Q.cpu().numpy()      # [N, 1]

        # Ensure sub arrays are NumPy
        if isinstance(sub_idx, torch.Tensor):
            sub_idx = sub_idx.cpu().numpy()
        if isinstance(sub_frac, torch.Tensor):
            sub_frac = sub_frac.cpu().numpy()

        N = bundle.N
        max_subs = sub_idx.shape[1]

        t0 = time.perf_counter()

        # 1. Correlated demand [N, S]
        D = mu + (L @ Z)
        del L, Z
        np.maximum(D, 0.0, out=D)

        # 2. Stockout [N, S]
        stockout = np.maximum(D - Q, 0.0)

        # 3. Redirected demand [N, S]
        redirected = np.zeros_like(D)
        for k in range(max_subs):
            idx_k = sub_idx[:, k]           # [N]
            frac_k = sub_frac[:, k]         # [N]
            valid = idx_k >= 0              # [N] bool
            if not np.any(valid):
                continue
            # Gather stockout for substitute products
            # For invalid indices, clamp to 0 and mask later
            safe_idx = np.where(valid, idx_k, 0).astype(np.intp)
            sub_stockout = stockout[safe_idx]   # [N, S]
            # Zero out invalid entries and weight by fraction
            redirected += np.where(
                valid[:, None],
                frac_k[:, None] * sub_stockout,
                0.0,
            )

        # 4. Effective demand
        D_eff = D + redirected

        # 5. Sales
        X = np.minimum(D_eff, Q)

        # 6. Per-scenario profit
        overage = np.maximum(Q - D_eff, 0.0)
        profit = (p * X) - (c * Q) + (s * overage)

        # 7. Reduce over scenarios
        expected_profit = profit.mean(axis=1).squeeze()   # [N]
        avg_redirected  = redirected.mean(axis=1).squeeze()  # [N]
        eff_prof        = expected_profit.copy()

        wall_ms = (time.perf_counter() - t0) * 1e3

        return SubstitutionResult(
            expected_profit=torch.tensor(expected_profit, dtype=torch.float32),
            wall_time_ms=wall_ms,
            peak_memory_bytes=0,
            label=self.label,
            substitution_demand=torch.tensor(avg_redirected, dtype=torch.float32),
            effective_profit=torch.tensor(eff_prof, dtype=torch.float32),
        )


# =========================================================================
# Solver 2 -- PyTorch GPU
# =========================================================================
class PyTorchSubstitution:
    """
    Standard PyTorch implementation of substitution newsvendor on GPU,
    with optional ``torch.compile`` (Inductor backend).

    Uses advanced indexing for the substitution gather, which is efficient
    on GPU but still materialises the full D[N,S] and stockout[N,S] tensors
    in HBM.
    """

    def __init__(self, use_compile: bool = True) -> None:
        self.label = "PyTorch-Substitution-Compile" if use_compile else "PyTorch-Substitution-Eager"
        self.use_compile = use_compile
        self._compiled_fn: Optional[object] = None

    @staticmethod
    def _substitution_forward(
        L: torch.Tensor,
        Z: torch.Tensor,
        mu: torch.Tensor,
        p: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        Q: torch.Tensor,
        sub_idx: torch.Tensor,
        sub_frac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute expected profit under substitution using standard torch ops.

        Parameters
        ----------
        L, Z, mu, p, c, s, Q : standard newsvendor tensors
        sub_idx  : int64  [N, max_subs]  -- substitute indices (-1 = none)
        sub_frac : float32 [N, max_subs] -- substitution fractions

        Returns
        -------
        expected_profit : [N]
        avg_redirected  : [N]
        effective_profit: [N]
        """
        N, S_dim = Z.shape

        # 1. Correlated demand [N, S]
        D = mu + torch.mm(L, Z)
        D = torch.clamp(D, min=0.0)

        # 2. Stockout [N, S]
        stockout = torch.clamp(D - Q, min=0.0)

        # 3. Redirected demand via advanced indexing
        # sub_idx may contain -1; clamp to 0 for safe indexing, then mask
        valid_mask = sub_idx >= 0                         # [N, max_subs] bool
        safe_idx = sub_idx.clamp(min=0)                   # [N, max_subs]

        # Gather: stockout[safe_idx] -> [N, max_subs, S]
        sub_stockout = stockout[safe_idx]                 # [N, max_subs, S]

        # Zero out invalid substitutes
        sub_stockout = sub_stockout * valid_mask.unsqueeze(-1).float()

        # Weight by substitution fractions and sum over substitutes
        # sub_frac: [N, max_subs] -> [N, max_subs, 1] for broadcasting
        redirected = (sub_frac.unsqueeze(-1) * sub_stockout).sum(dim=1)
        # redirected: [N, S]

        # 4. Effective demand
        D_eff = D + redirected

        # 5. Sales
        X = torch.minimum(D_eff, Q)

        # 6. Profit
        overage = torch.clamp(Q - D_eff, min=0.0)
        profit = (p * X) - (c * Q) + (s * overage)

        # 7. Reduce
        expected_profit = profit.mean(dim=1)              # [N]
        avg_redirected  = redirected.mean(dim=1)          # [N]

        return expected_profit, avg_redirected, expected_profit.clone()

    def solve(
        self,
        bundle: TensorBundle,
        sub_idx: torch.Tensor,
        sub_frac: torch.Tensor,
    ) -> SubstitutionResult:
        """
        Execute substitution newsvendor on GPU.

        Parameters
        ----------
        bundle : TensorBundle
        sub_idx : torch.Tensor  [N, max_subs] int64
        sub_frac : torch.Tensor [N, max_subs] float32
        """
        device = bundle.L.device
        dtype = bundle.L.dtype

        # Ensure contiguous and on correct device
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.contiguous()
        p  = bundle.p.contiguous()
        c  = bundle.c.contiguous()
        s  = bundle.s.contiguous()
        Q  = bundle.Q.contiguous()

        sub_idx  = sub_idx.contiguous().to(device=device, dtype=torch.int64)
        sub_frac = sub_frac.contiguous().to(device=device, dtype=torch.float32)

        # Optionally compile
        fn = self._substitution_forward
        if self.use_compile and self._compiled_fn is None:
            self._compiled_fn = torch.compile(fn, backend="inductor")
        if self.use_compile:
            fn = self._compiled_fn

        # Warm-up
        if device.type == "cuda":
            _ = fn(L, Z, mu, p, c, s, Q, sub_idx, sub_frac)
            torch.cuda.synchronize()

        # Memory measurement
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _ = fn(L, Z, mu, p, c, s, Q, sub_idx, sub_frac)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device)
        else:
            peak_mem = 0

        # Timed run
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()

        t0 = time.perf_counter()
        expected_profit, avg_redirected, eff_profit = fn(
            L, Z, mu, p, c, s, Q, sub_idx, sub_frac
        )

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            wall_ms = start_event.elapsed_time(end_event)
        else:
            wall_ms = (time.perf_counter() - t0) * 1e3
            peak_mem = 0

        return SubstitutionResult(
            expected_profit=expected_profit.detach(),
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
            substitution_demand=avg_redirected.detach(),
            effective_profit=eff_profit.detach(),
        )


# =========================================================================
# Solver 3 -- Triton (two-pass fused kernel)
# =========================================================================
class TritonSubstitution:
    """
    Two-pass Triton kernel for substitution newsvendor.

    Pass 1: Fused matmul -> stockout written to HBM.
    Pass 2: Fused matmul (recomputed) + substitute gather + profit + reduce.

    Memory advantage over PyTorch: only stockout[N,S] is materialised (1.07 GB
    at N=2048, S=131072) instead of D + stockout + sub_stockout.
    """

    def __init__(self) -> None:
        self.label = "Triton-Substitution"

    def solve(
        self,
        bundle: TensorBundle,
        sub_idx: torch.Tensor,
        sub_frac: torch.Tensor,
        max_subs: int = 4,
    ) -> SubstitutionResult:
        """
        Launch the two-pass Triton kernel and return a SubstitutionResult.

        Parameters
        ----------
        bundle : TensorBundle
        sub_idx : torch.Tensor  [N, max_subs] int64
        sub_frac : torch.Tensor [N, max_subs] float32
        max_subs : int
            Maximum substitutes per product (must match sub_idx.shape[1]).
        """
        from extensions.substitution_kernel import TritonSubstitutionKernels, cdiv

        device = bundle.L.device
        assert device.type == "cuda", "Triton kernel requires CUDA device."

        N, S = bundle.N, bundle.S
        K = N

        # Flatten column vectors to 1-D
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.squeeze(1).contiguous()
        p  = bundle.p.squeeze(1).contiguous()
        c  = bundle.c.squeeze(1).contiguous()
        s  = bundle.s.squeeze(1).contiguous()
        Q  = bundle.Q.squeeze(1).contiguous()

        sub_idx  = sub_idx.contiguous().to(device=device, dtype=torch.int64)
        sub_frac = sub_frac.contiguous().to(device=device, dtype=torch.float32)

        # Import kernel module
        from extensions.substitution_kernel import (
            _stockout_kernel,
            _substitution_profit_kernel,
        )

        # Grid lambda (shared by both passes)
        grid = lambda META: (
            cdiv(N, META["BLOCK_SIZE_M"]),
            cdiv(S, META["BLOCK_SIZE_N"]),
        )

        # ---- Warm-up (triggers autotune on first call; cached after) ----
        stockout   = torch.empty(N, S, dtype=torch.float32, device=device)
        out        = torch.zeros(N, dtype=torch.float32, device=device)
        sub_demand = torch.zeros(N, dtype=torch.float32, device=device)
        eff_profit = torch.zeros(N, dtype=torch.float32, device=device)

        _stockout_kernel[grid](
            L, Z, mu, Q,
            stockout,
            N, S, K,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            stockout.stride(0), stockout.stride(1),
        )

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

        # ---- Memory measurement ----
        out.zero_()
        sub_demand.zero_()
        eff_profit.zero_()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        _stockout_kernel[grid](
            L, Z, mu, Q,
            stockout,
            N, S, K,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            stockout.stride(0), stockout.stride(1),
        )

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
        peak_mem = torch.cuda.max_memory_allocated(device)

        # ---- Timed launch ----
        out.zero_()
        sub_demand.zero_()
        eff_profit.zero_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        start_event.record()

        _stockout_kernel[grid](
            L, Z, mu, Q,
            stockout,
            N, S, K,
            L.stride(0), L.stride(1),
            Z.stride(0), Z.stride(1),
            stockout.stride(0), stockout.stride(1),
        )

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

        end_event.record()
        torch.cuda.synchronize()
        wall_ms = start_event.elapsed_time(end_event)

        return SubstitutionResult(
            expected_profit=out,
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
            substitution_demand=sub_demand,
            effective_profit=eff_profit,
        )


# =========================================================================
# Quick self-test
# =========================================================================
if __name__ == "__main__":
    import logging
    from config import NewsvendorConfig, SubstitutionConfig
    from data_pipeline import DataPipeline, SubstitutionGraphGenerator

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Small problem for quick testing
    cfg = NewsvendorConfig(N=64, S=4096, device="cpu")
    bundle = DataPipeline(cfg=cfg).run()

    # Generate substitution graph
    sub_cfg = SubstitutionConfig()
    sub_gen = SubstitutionGraphGenerator(sub_cfg)
    cat_mask_np = bundle.category_mask.cpu().numpy()
    sub_idx_np, sub_frac_np = sub_gen.generate(cfg.N, cat_mask_np, cfg.seed)

    # --- CPU solver ---
    cpu_solver = CPUSubstitution()
    res_cpu = cpu_solver.solve(bundle, sub_idx_np, sub_frac_np)
    print(f"[{res_cpu.label}]  time={res_cpu.wall_time_ms:.1f} ms  "
          f"E[profit] range=[{res_cpu.expected_profit.min():.2f}, "
          f"{res_cpu.expected_profit.max():.2f}]")
    print(f"  avg redirected demand: [{res_cpu.substitution_demand.min():.2f}, "
          f"{res_cpu.substitution_demand.max():.2f}]")

    # --- PyTorch solver (eager, CPU) ---
    pt_solver = PyTorchSubstitution(use_compile=False)
    sub_idx_t  = torch.tensor(sub_idx_np, dtype=torch.int64)
    sub_frac_t = torch.tensor(sub_frac_np, dtype=torch.float32)
    res_pt = pt_solver.solve(bundle, sub_idx_t, sub_frac_t)
    print(f"[{res_pt.label}]  time={res_pt.wall_time_ms:.1f} ms  "
          f"E[profit] range=[{res_pt.expected_profit.min():.2f}, "
          f"{res_pt.expected_profit.max():.2f}]")

    # Numerical parity
    diff = (res_cpu.expected_profit - res_pt.expected_profit).abs().max().item()
    print(f"Max absolute diff (CPU vs PyTorch-Eager): {diff:.6f}")

    # --- Triton solver (if CUDA available) ---
    if torch.cuda.is_available():
        cfg_gpu = NewsvendorConfig(N=64, S=4096, device="cuda")
        bundle_gpu = DataPipeline(cfg=cfg_gpu).run()

        cat_mask_gpu_np = bundle_gpu.category_mask.cpu().numpy()
        sub_idx_np_g, sub_frac_np_g = sub_gen.generate(
            cfg_gpu.N, cat_mask_gpu_np, cfg_gpu.seed
        )
        sub_idx_g  = torch.tensor(sub_idx_np_g, dtype=torch.int64, device="cuda")
        sub_frac_g = torch.tensor(sub_frac_np_g, dtype=torch.float32, device="cuda")

        tr_solver = TritonSubstitution()
        res_tr = tr_solver.solve(bundle_gpu, sub_idx_g, sub_frac_g)
        print(f"\n[{res_tr.label}]  time={res_tr.wall_time_ms:.2f} ms  "
              f"E[profit] range=[{res_tr.expected_profit.min():.2f}, "
              f"{res_tr.expected_profit.max():.2f}]")

        # Compare Triton vs CPU
        cpu_ep = res_cpu.expected_profit
        tri_ep = res_tr.expected_profit.cpu()
        diff_tr = (cpu_ep - tri_ep).abs()
        print(f"Max diff vs CPU:  {diff_tr.max().item():.6f}")
        print(f"Mean diff vs CPU: {diff_tr.mean().item():.6f}")
    else:
        print("\nCUDA not available -- skipping Triton solver test.")
