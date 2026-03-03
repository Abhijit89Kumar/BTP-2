"""
baseline_solvers.py — Reference implementations of the Monte-Carlo Newsvendor.

Two baselines are provided so the Triton kernel can be validated against them:

1. ``CPUMonteCarlo``     — Pure NumPy on CPU (gold-standard correctness).
2. ``PyTorchMonteCarlo`` — Native ``torch`` ops on GPU, wrapped with
                           ``torch.compile`` (inductor backend) to show the
                           ceiling of standard graph-level compilation.

Both return a ``SolverResult`` so that the benchmark can compare them
uniformly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from data_pipeline import TensorBundle


# ---------------------------------------------------------------------------
# Uniform result container
# ---------------------------------------------------------------------------
@dataclass
class SolverResult:
    """Holds the output of any solver for uniform comparison."""
    expected_profit: torch.Tensor   # shape [N]
    wall_time_ms: float             # end-to-end wall-clock time (ms)
    peak_memory_bytes: int          # peak GPU memory allocated (0 for CPU)
    label: str                      # human-readable solver name


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 1 — CPU (NumPy)
# ═══════════════════════════════════════════════════════════════════════════
class CPUMonteCarlo:
    """
    Pure NumPy Monte-Carlo Newsvendor on CPU.

    This is intentionally *un-optimised* to serve as a readable reference.
    """

    def __init__(self) -> None:
        self.label = "CPU-NumPy"

    # ------------------------------------------------------------------
    def solve(self, bundle: TensorBundle) -> SolverResult:
        """
        Execute the full Newsvendor simulation on CPU.

        Steps
        -----
        1. Correlated demand: D = μ + L @ Z          [N, S]
        2. Sales:             X = min(D, Q)           [N, S]
        3. Profit:            π = p·X − c·Q + s·max(Q−D, 0)
        4. Expected profit:   E[π] = mean(π, axis=1)  [N]
        """
        # Move everything to CPU NumPy (stay in float32 to fit in RAM)
        L  = bundle.L.cpu().numpy()
        Z  = bundle.Z.cpu().numpy()
        mu = bundle.mu.cpu().numpy()    # [N,1]
        p  = bundle.p.cpu().numpy()
        c  = bundle.c.cpu().numpy()
        s  = bundle.s.cpu().numpy()
        Q  = bundle.Q.cpu().numpy()

        t0 = time.perf_counter()

        # 1. Correlated demand  [N, S]
        #    Peak RAM here: L(16 MB) + Z(1 GB) + D(1 GB) ≈ 2 GB
        D = mu + (L @ Z)
        del L, Z                         # free inputs immediately
        np.maximum(D, 0.0, out=D)        # demand ≥ 0 (in-place)

        # 2. Sales
        X = np.minimum(D, Q)             # [N, S]  — 1 GB

        # 3. Per-scenario profit
        #    Compute overage in-place into D (no longer needed as D)
        np.subtract(Q, D, out=D)         # D now holds Q - D
        np.maximum(D, 0.0, out=D)        # D now holds max(Q - D, 0)
        # profit = p*X - c*Q + s*overage — reuse D for overage
        profit = (p * X) - (c * Q) + (s * D)   # [N, S]
        del X, D                          # free before reduction

        # 4. Reduce over scenarios
        expected_profit = profit.mean(axis=1).squeeze()  # [N]
        del profit

        wall_ms = (time.perf_counter() - t0) * 1e3

        return SolverResult(
            expected_profit=torch.tensor(expected_profit, dtype=torch.float32),
            wall_time_ms=wall_ms,
            peak_memory_bytes=0,
            label=self.label,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 2 — PyTorch GPU (torch.compile)
# ═══════════════════════════════════════════════════════════════════════════
class PyTorchMonteCarlo:
    """
    Standard PyTorch implementation on GPU, using ``torch.compile`` with the
    Inductor backend.

    **Why this is sub-optimal (and why we need Triton):**

    ``torch.compile`` will lower the operations to Triton *under the hood*,
    but it cannot fuse across the matmul boundary.  As a result, the full
    intermediate ``D`` tensor of shape ``[N, S]`` (e.g., 2048 × 131072 =
    1 GB in fp32) must be materialised in HBM between the matmul and the
    element-wise business logic.  Our custom Triton kernel eliminates this
    HBM round-trip entirely.
    """

    def __init__(self, use_compile: bool = True) -> None:
        self.label = "PyTorch-Compile" if use_compile else "PyTorch-Eager"
        self.use_compile = use_compile
        self._compiled_fn: Optional[object] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _newsvendor_forward(
        L: torch.Tensor,
        Z: torch.Tensor,
        mu: torch.Tensor,
        p: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expected profit using standard torch ops.

        Returns shape [N].
        """
        # Correlated demand  [N, S]
        D = mu + torch.mm(L, Z)
        D = torch.clamp(D, min=0.0)

        # Sales
        X = torch.minimum(D, Q)                         # [N, S]

        # Profit per scenario
        overage = torch.clamp(Q - D, min=0.0)           # [N, S]
        profit = (p * X) - (c * Q) + (s * overage)      # [N, S]

        # Reduce
        return profit.mean(dim=1)                        # [N]

    # ------------------------------------------------------------------
    def solve(self, bundle: TensorBundle) -> SolverResult:
        device = bundle.L.device
        dtype = bundle.L.dtype

        # Ensure tensors are contiguous and on GPU
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.contiguous()
        p  = bundle.p.contiguous()
        c  = bundle.c.contiguous()
        s  = bundle.s.contiguous()
        Q  = bundle.Q.contiguous()

        # Optionally compile
        fn = self._newsvendor_forward
        if self.use_compile and self._compiled_fn is None:
            self._compiled_fn = torch.compile(
                fn, mode="max-autotune", backend="inductor"
            )
        if self.use_compile:
            fn = self._compiled_fn

        # Warm-up (torch.compile JIT's on first call)
        if device.type == "cuda":
            _ = fn(L, Z, mu, p, c, s, Q)
            torch.cuda.synchronize()

        # Clear cached blocks so the timed run's allocations (including
        # the D=[N,S] intermediate) show up as fresh peaks, giving an
        # accurate comparison against the Triton kernel which never
        # allocates D.
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # Timed run
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        import time as _time
        t0 = _time.perf_counter()
        result = fn(L, Z, mu, p, c, s, Q)
        # Clone immediately — torch.compile with max-autotune uses CUDA
        # graphs whose output buffers are overwritten on subsequent runs.
        result = result.clone()

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            wall_ms = start_event.elapsed_time(end_event)
            peak_mem = torch.cuda.max_memory_allocated(device)
        else:
            wall_ms = (_time.perf_counter() - t0) * 1e3
            peak_mem = 0

        return SolverResult(
            expected_profit=result.detach(),
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import NewsvendorConfig
    from data_pipeline import DataPipeline

    cfg = NewsvendorConfig(N=64, S=4096, device="cpu")
    bundle = DataPipeline(cfg=cfg).run()

    cpu_solver = CPUMonteCarlo()
    res_cpu = cpu_solver.solve(bundle)
    print(f"[{res_cpu.label}]  time={res_cpu.wall_time_ms:.1f} ms  "
          f"E[profit] range=[{res_cpu.expected_profit.min():.2f}, "
          f"{res_cpu.expected_profit.max():.2f}]")

    pt_solver = PyTorchMonteCarlo(use_compile=False)
    res_pt = pt_solver.solve(bundle)
    print(f"[{res_pt.label}]  time={res_pt.wall_time_ms:.1f} ms  "
          f"E[profit] range=[{res_pt.expected_profit.min():.2f}, "
          f"{res_pt.expected_profit.max():.2f}]")

    # Numerical parity
    diff = (res_cpu.expected_profit - res_pt.expected_profit).abs().max().item()
    print(f"Max absolute diff (CPU vs PyTorch-Eager): {diff:.6f}")
