"""
cvar_solvers.py -- Three CVaR (Conditional Value at Risk) solver implementations
                   for the Multi-Echelon Stochastic Newsvendor.

Provides:
  1. CPUCVaR        -- Pure NumPy on CPU (gold-standard correctness reference).
  2. PyTorchCVaR    -- GPU solver using standard torch ops + torch.compile.
  3. TritonCVaR     -- GPU solver using the fused Triton CVaR kernel.

All three return a ``CVaRResult`` with expected_profit, VaR, CVaR, and timing.

CVaR (Conditional Value at Risk) at level alpha is the expected profit in the
worst alpha-fraction of scenarios.  It is a coherent risk measure widely used
in operations research for risk-averse inventory decisions.

For a random profit Pi:
  VaR_alpha  = inf{ x : P(Pi <= x) >= alpha }          (alpha-quantile)
  CVaR_alpha = E[ Pi | Pi <= VaR_alpha ]                (tail conditional mean)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from data_pipeline import TensorBundle
from extensions.common import CVaRResult
from config import CVaRConfig, DEFAULT_CVAR


# =========================================================================
# Utility: compute CVaR from a histogram (used by TritonCVaR)
# =========================================================================
def compute_cvar_from_histogram(
    hist: torch.Tensor,        # [N, num_bins] -- counts per bin
    hist_min: torch.Tensor,    # [N] -- lower bound per product
    hist_max: torch.Tensor,    # [N] -- upper bound per product
    alpha: float,
    N: int,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract VaR and CVaR from per-product profit histograms.

    Parameters
    ----------
    hist     : float32 [N, num_bins] -- raw bin counts (sum ~ S per product)
    hist_min : float32 [N] -- left edge of histogram per product
    hist_max : float32 [N] -- right edge of histogram per product
    alpha    : float -- risk level (e.g. 0.05 for worst 5%)
    N        : int -- number of products
    num_bins : int -- number of histogram bins

    Returns
    -------
    VaR  : float32 [N] -- Value at Risk per product
    CVaR : float32 [N] -- Conditional Value at Risk per product
    """
    device = hist.device

    # Bin geometry
    bin_width = (hist_max - hist_min) / num_bins                      # [N]
    bin_edges_left = hist_min[:, None] + torch.arange(
        num_bins, device=device, dtype=torch.float32
    )[None, :] * bin_width[:, None]                                   # [N, num_bins]
    bin_centers = bin_edges_left + 0.5 * bin_width[:, None]           # [N, num_bins]

    # Normalise counts to probabilities
    total_counts = hist.sum(dim=1, keepdim=True).clamp(min=1.0)      # [N, 1]
    probs = hist / total_counts                                       # [N, num_bins]

    # Cumulative distribution
    cum_probs = probs.cumsum(dim=1)                                   # [N, num_bins]

    # VaR: first bin where cumulative probability >= alpha
    # argmax on a boolean tensor returns the index of the first True
    var_bin = (cum_probs >= alpha).float().argmax(dim=1)              # [N]
    VaR = bin_centers[torch.arange(N, device=device), var_bin]        # [N]

    # CVaR: weighted mean of all bins up to and including the VaR bin
    bin_indices = torch.arange(num_bins, device=device)[None, :]      # [1, num_bins]
    mask = bin_indices <= var_bin[:, None]                             # [N, num_bins]
    masked_probs = probs * mask.float()                               # [N, num_bins]

    # Re-normalise the masked probabilities so they sum to 1
    masked_total = masked_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
    masked_probs = masked_probs / masked_total

    CVaR = (masked_probs * bin_centers).sum(dim=1)                    # [N]

    return VaR, CVaR


# =========================================================================
# Solver 1 -- CPU (NumPy)
# =========================================================================
class CPUCVaR:
    """
    Pure NumPy CVaR solver on CPU.

    Computes the full profit matrix [N, S], then for each product:
      - E[profit] = mean over scenarios
      - VaR       = alpha-quantile of the profit distribution
      - CVaR      = mean of profits at or below VaR
    """

    def __init__(self, cvar_cfg: CVaRConfig = DEFAULT_CVAR) -> None:
        self.alpha = cvar_cfg.alpha
        self.label = "CPU-CVaR"

    def solve(self, bundle: TensorBundle) -> CVaRResult:
        L  = bundle.L.cpu().numpy()
        Z  = bundle.Z.cpu().numpy()
        mu = bundle.mu.cpu().numpy()       # [N, 1]
        p  = bundle.p.cpu().numpy()
        c  = bundle.c.cpu().numpy()
        s  = bundle.s.cpu().numpy()
        Q  = bundle.Q.cpu().numpy()

        alpha = self.alpha

        t0 = time.perf_counter()

        # 1. Correlated demand [N, S]
        D = mu + (L @ Z)
        del L, Z
        np.maximum(D, 0.0, out=D)

        # 2. Sales
        X = np.minimum(D, Q)

        # 3. Per-scenario profit
        overage = np.maximum(Q - D, 0.0)
        profit = (p * X) - (c * Q) + (s * overage)
        del X, D, overage

        # 4. Expected profit
        expected_profit = profit.mean(axis=1).squeeze()               # [N]

        # 5. VaR and CVaR
        # Sort profits along scenario axis for each product
        N_prod = profit.shape[0]
        VaR_np = np.percentile(profit, alpha * 100.0, axis=1)        # [N]

        # CVaR: mean of profits <= VaR (per product)
        # Use broadcasting: mask[i, j] = (profit[i, j] <= VaR[i])
        VaR_bc = VaR_np.reshape(-1, 1)                               # [N, 1]
        mask = profit <= VaR_bc                                       # [N, S]

        # Compute conditional mean safely (handle case where mask is all False)
        CVaR_np = np.zeros(N_prod, dtype=np.float64)
        for i in range(N_prod):
            below = profit[i, mask[i]]
            if len(below) > 0:
                CVaR_np[i] = below.mean()
            else:
                CVaR_np[i] = VaR_np[i]

        del profit, mask

        wall_ms = (time.perf_counter() - t0) * 1e3

        return CVaRResult(
            expected_profit=torch.tensor(expected_profit, dtype=torch.float32),
            wall_time_ms=wall_ms,
            peak_memory_bytes=0,
            label=self.label,
            VaR=torch.tensor(VaR_np, dtype=torch.float32),
            CVaR=torch.tensor(CVaR_np, dtype=torch.float32),
            alpha=alpha,
            histogram=None,
        )


# =========================================================================
# Solver 2 -- PyTorch GPU (torch.compile)
# =========================================================================
class PyTorchCVaR:
    """
    GPU CVaR solver using standard torch ops.

    Uses ``torch.quantile`` for VaR and masked mean for CVaR.
    Optionally wraps the forward pass in ``torch.compile`` for Inductor
    fusion of pointwise ops (the matmul boundary is still not fused).
    """

    def __init__(
        self,
        cvar_cfg: CVaRConfig = DEFAULT_CVAR,
        use_compile: bool = True,
    ) -> None:
        self.alpha = cvar_cfg.alpha
        self.use_compile = use_compile
        self.label = "PyTorch-CVaR-Compile" if use_compile else "PyTorch-CVaR-Eager"
        self._compiled_fn: Optional[object] = None

    @staticmethod
    def _profit_forward(
        L: torch.Tensor,
        Z: torch.Tensor,
        mu: torch.Tensor,
        p: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the per-scenario profit matrix using standard torch ops.

        Returns profit [N, S].

        NOTE: torch.quantile is NOT compile-safe (Dynamo cannot handle
        symbolic sizes in quantile), so VaR/CVaR are computed outside
        the compiled region.
        """
        # Correlated demand [N, S]
        D = mu + torch.mm(L, Z)
        D = torch.clamp(D, min=0.0)

        # Sales
        X = torch.minimum(D, Q)

        # Profit per scenario
        overage = torch.clamp(Q - D, min=0.0)
        profit = (p * X) - (c * Q) + (s * overage)

        return profit

    @staticmethod
    def _compute_cvar(profit: torch.Tensor, alpha: float):
        """Compute E[profit], VaR, CVaR from the profit matrix (NOT compiled)."""
        expected_profit = profit.mean(dim=1)                          # [N]

        # VaR: alpha-quantile (smallest alpha fraction)
        VaR = torch.quantile(profit, alpha, dim=1)                    # [N]

        # CVaR: mean of profits <= VaR
        mask = profit <= VaR[:, None]                                 # [N, S]
        masked_profit = torch.where(mask, profit, torch.zeros_like(profit))
        count = mask.float().sum(dim=1).clamp(min=1.0)               # [N]
        CVaR = masked_profit.sum(dim=1) / count                      # [N]

        return expected_profit, VaR, CVaR

    def solve(self, bundle: TensorBundle) -> CVaRResult:
        device = bundle.L.device

        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.contiguous()
        p  = bundle.p.contiguous()
        c  = bundle.c.contiguous()
        s  = bundle.s.contiguous()
        Q  = bundle.Q.contiguous()

        alpha = self.alpha

        # Only compile the profit computation (quantile is NOT compile-safe)
        fn = self._profit_forward
        if self.use_compile and self._compiled_fn is None:
            self._compiled_fn = torch.compile(fn, backend="inductor")
        if self.use_compile:
            fn = self._compiled_fn

        # Warm-up (JIT compile on first call)
        if device.type == "cuda":
            profit_warmup = fn(L, Z, mu, p, c, s, Q)
            _ = self._compute_cvar(profit_warmup, alpha)
            torch.cuda.synchronize()
            del profit_warmup

        # Memory measurement
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            profit_mem = fn(L, Z, mu, p, c, s, Q)
            _ = self._compute_cvar(profit_mem, alpha)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device)
            del profit_mem
        else:
            peak_mem = 0

        # Timed run
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()

        t0 = time.perf_counter()
        profit = fn(L, Z, mu, p, c, s, Q)
        expected_profit, VaR, CVaR = self._compute_cvar(profit, alpha)
        del profit

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            wall_ms = start_event.elapsed_time(end_event)
        else:
            wall_ms = (time.perf_counter() - t0) * 1e3

        return CVaRResult(
            expected_profit=expected_profit.detach(),
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
            VaR=VaR.detach(),
            CVaR=CVaR.detach(),
            alpha=alpha,
            histogram=None,
        )


# =========================================================================
# Solver 3 -- Triton fused kernel
# =========================================================================
class TritonCVaR:
    """
    CVaR solver using the fused Triton kernel from ``cvar_kernel.py``.

    Strategy:
      1. Estimate per-product profit bounds analytically for histogram range.
      2. Launch the fused kernel (matmul + profit + mean + histogram in one pass).
      3. Post-process histogram on GPU to extract VaR and CVaR.

    The kernel never materialises D[N, S] or profit[N, S] in HBM.
    """

    def __init__(self, cvar_cfg: CVaRConfig = DEFAULT_CVAR) -> None:
        self.alpha = cvar_cfg.alpha
        self.num_bins = cvar_cfg.num_bins
        self.label = "Triton-CVaR"

    @staticmethod
    def _estimate_profit_bounds(
        mu: torch.Tensor,   # [N]
        p: torch.Tensor,    # [N]
        c: torch.Tensor,    # [N]
        s: torch.Tensor,    # [N]
        Q: torch.Tensor,    # [N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conservative analytical bounds on per-product profit.

        Worst case (demand = 0):
            profit_min = -c * Q + s * Q = (s - c) * Q

        Best case (demand >> Q, all units sold):
            profit_max = (p - c) * Q

        We add a 10% margin on each side for safety.
        """
        profit_min = (s - c) * Q     # worst: zero demand, all salvaged
        profit_max = (p - c) * Q     # best: demand exceeds Q, all sold

        # Add safety margin so extreme values are not clipped
        margin = 0.1 * (profit_max - profit_min).clamp(min=1.0)
        profit_min = profit_min - margin
        profit_max = profit_max + margin

        return profit_min, profit_max

    def solve(self, bundle: TensorBundle) -> CVaRResult:
        from extensions.cvar_kernel import TritonCVaRKernel

        device = bundle.L.device
        assert device.type == "cuda", "Triton kernel requires CUDA device."

        N, S = bundle.N, bundle.S
        num_bins = self.num_bins
        alpha = self.alpha

        # Flatten column vectors to 1-D
        L  = bundle.L.contiguous()
        Z  = bundle.Z.contiguous()
        mu = bundle.mu.squeeze(1).contiguous()
        p  = bundle.p.squeeze(1).contiguous()
        c  = bundle.c.squeeze(1).contiguous()
        s  = bundle.s.squeeze(1).contiguous()
        Q  = bundle.Q.squeeze(1).contiguous()

        # Step 1: Estimate histogram bounds analytically
        hist_min, hist_max = self._estimate_profit_bounds(mu, p, c, s, Q)
        hist_min = hist_min.contiguous()
        hist_max = hist_max.contiguous()

        kernel = TritonCVaRKernel(num_bins=num_bins)

        # -- Warm-up (triggers autotune on first call; cached after) --
        out_mean, hist = kernel.launch(L, Z, mu, p, c, s, Q, hist_min, hist_max)
        torch.cuda.synchronize()

        # -- Memory measurement (autotune already cached) --
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        out_mean, hist = kernel.launch(L, Z, mu, p, c, s, Q, hist_min, hist_max)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)

        # -- Timed launch --
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        start_event.record()

        out_mean, hist = kernel.launch(L, Z, mu, p, c, s, Q, hist_min, hist_max)

        end_event.record()
        torch.cuda.synchronize()
        wall_ms = start_event.elapsed_time(end_event)

        # Step 3: Post-process histogram to extract VaR and CVaR
        VaR, CVaR = compute_cvar_from_histogram(
            hist, hist_min, hist_max, alpha, N, num_bins,
        )

        return CVaRResult(
            expected_profit=out_mean,
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
            VaR=VaR,
            CVaR=CVaR,
            alpha=alpha,
            histogram=hist,
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from config import NewsvendorConfig, CVaRConfig
    from data_pipeline import DataPipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = NewsvendorConfig(N=256, S=8192, device="cuda")
    cvar_cfg = CVaRConfig(alpha=0.05, num_bins=256)
    bundle = DataPipeline(cfg=cfg).run()

    # --- CPU reference ---
    cpu = CPUCVaR(cvar_cfg)
    res_cpu = cpu.solve(bundle)
    print(f"[{res_cpu.label}]  time={res_cpu.wall_time_ms:.1f} ms  "
          f"E[profit]=[{res_cpu.expected_profit.min():.2f}, "
          f"{res_cpu.expected_profit.max():.2f}]  "
          f"VaR=[{res_cpu.VaR.min():.2f}, {res_cpu.VaR.max():.2f}]  "
          f"CVaR=[{res_cpu.CVaR.min():.2f}, {res_cpu.CVaR.max():.2f}]")

    # --- PyTorch GPU ---
    pt = PyTorchCVaR(cvar_cfg, use_compile=False)
    res_pt = pt.solve(bundle)
    print(f"[{res_pt.label}]  time={res_pt.wall_time_ms:.2f} ms  "
          f"E[profit]=[{res_pt.expected_profit.min():.2f}, "
          f"{res_pt.expected_profit.max():.2f}]  "
          f"VaR=[{res_pt.VaR.min():.2f}, {res_pt.VaR.max():.2f}]  "
          f"CVaR=[{res_pt.CVaR.min():.2f}, {res_pt.CVaR.max():.2f}]")

    # --- Triton fused ---
    tr = TritonCVaR(cvar_cfg)
    res_tr = tr.solve(bundle)
    print(f"[{res_tr.label}]  time={res_tr.wall_time_ms:.2f} ms  "
          f"E[profit]=[{res_tr.expected_profit.min():.2f}, "
          f"{res_tr.expected_profit.max():.2f}]  "
          f"VaR=[{res_tr.VaR.min():.2f}, {res_tr.VaR.max():.2f}]  "
          f"CVaR=[{res_tr.CVaR.min():.2f}, {res_tr.CVaR.max():.2f}]")

    # --- Numerical comparison ---
    print("\n--- Validation ---")
    diff_ep = (res_cpu.expected_profit - res_tr.expected_profit.cpu()).abs()
    print(f"E[profit] max diff (CPU vs Triton): {diff_ep.max().item():.6f}")

    diff_var = (res_cpu.VaR - res_tr.VaR.cpu()).abs()
    print(f"VaR max diff (CPU vs Triton):       {diff_var.max().item():.4f}")

    diff_cvar = (res_cpu.CVaR - res_tr.CVaR.cpu()).abs()
    print(f"CVaR max diff (CPU vs Triton):      {diff_cvar.max().item():.4f}")

    print(f"\nHistogram sum (should ~ {cfg.S}): "
          f"{res_tr.histogram.sum(dim=1).mean().item():.0f}")
