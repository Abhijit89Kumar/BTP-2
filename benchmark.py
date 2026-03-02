"""
benchmark.py — End-to-end benchmarking suite for the Newsvendor solvers.

Executes all three solver backends (CPU-NumPy, PyTorch-Compile, Triton-Fused),
verifies numerical parity, and reports a formatted comparison table.

Usage::

    python benchmark.py                    # default N=2048, S=131072
    python benchmark.py --N 512 --S 32768  # scaled-down test

Output
------
1. Correctness assertion (torch.allclose with atol=1e-2).
2. Terminal table:  Algorithm | Time (ms) | Peak Memory (GB) | TFLOPS.
3. Optional: Triton ``perf_report`` sweep over multiple problem sizes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import torch

from config import NewsvendorConfig
from data_pipeline import DataPipeline, TensorBundle
from baseline_solvers import CPUMonteCarlo, PyTorchMonteCarlo, SolverResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# FLOP estimation
# ═══════════════════════════════════════════════════════════════════════════
def estimate_flops(N: int, S: int) -> float:
    """
    Estimate total floating-point operations for one Newsvendor solve.

    Dominant cost:
        MatMul  L @ Z :  2 · N · N · S  FLOPs  (FMA counted as 2).
        Elementwise    :  ~6 · N · S  (add μ, clamp, min, sub, mul, add).
        Reduction      :  N · S.

    Total ≈ 2·N²·S + 7·N·S.
    """
    return 2.0 * N * N * S + 7.0 * N * S


# ═══════════════════════════════════════════════════════════════════════════
# Correctness checker
# ═══════════════════════════════════════════════════════════════════════════
def check_correctness(
    ref: SolverResult,
    test: SolverResult,
    atol: float = 1e-2,
    rtol: float = 1e-3,
) -> None:
    """
    Assert numerical parity between a reference and test solver.

    Uses ``torch.allclose`` and also reports the maximum absolute
    deviation for the thesis write-up.
    """
    ref_ep  = ref.expected_profit.cpu().float()
    test_ep = test.expected_profit.cpu().float()

    max_diff  = (ref_ep - test_ep).abs().max().item()
    mean_diff = (ref_ep - test_ep).abs().mean().item()

    close = torch.allclose(ref_ep, test_ep, atol=atol, rtol=rtol)
    status = "PASS" if close else "FAIL"

    print(f"  [{status}] {ref.label} vs {test.label}  "
          f"| max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
          f"(atol={atol}, rtol={rtol})")

    if not close:
        # Print first few diverging products for debugging
        diffs = (ref_ep - test_ep).abs()
        top_idx = diffs.topk(min(5, len(diffs))).indices
        for idx in top_idx:
            i = idx.item()
            print(f"    product {i}: ref={ref_ep[i]:.4f}  "
                  f"test={test_ep[i]:.4f}  diff={diffs[i]:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Pretty table
# ═══════════════════════════════════════════════════════════════════════════
def print_results_table(results: list[SolverResult], N: int, S: int) -> None:
    """Print a formatted comparison table to stdout."""
    flops = estimate_flops(N, S)

    hdr = (f"{'Algorithm':<22} | {'Time (ms)':>12} | "
           f"{'Peak Mem (GB)':>14} | {'TFLOPS':>10}")
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f"  Benchmark  —  N = {N:,}   S = {S:,}")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    for r in results:
        mem_gb = r.peak_memory_bytes / (1024 ** 3) if r.peak_memory_bytes else 0.0
        tflops = (flops / (r.wall_time_ms * 1e-3)) / 1e12 if r.wall_time_ms > 0 else 0.0
        print(f"{r.label:<22} | {r.wall_time_ms:>12.2f} | "
              f"{mem_gb:>14.3f} | {tflops:>10.2f}")

    print(sep)

    # Speedup summary
    if len(results) >= 2:
        base = results[0].wall_time_ms
        for r in results[1:]:
            if r.wall_time_ms > 0 and base > 0:
                speedup = base / r.wall_time_ms
                print(f"  Speedup  {results[0].label} → {r.label}: {speedup:.1f}x")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark suite
# ═══════════════════════════════════════════════════════════════════════════
class BenchmarkSuite:
    """Orchestrates data generation, solver execution, and reporting."""

    def __init__(self, cfg: NewsvendorConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def run(self, skip_cpu: bool = False, num_repeats: int = 3) -> None:
        """
        Full benchmark run.

        Parameters
        ----------
        skip_cpu : bool
            Skip the CPU solver for large N (it can take minutes).
        num_repeats : int
            Number of timed repeats for GPU solvers (report best time).
        """
        N, S = self.cfg.N, self.cfg.S
        device = self.cfg.device

        # ── Build data ────────────────────────────────────────────────
        print(f"[Pipeline] Building tensors  N={N}, S={S}, device={device} …")
        bundle = DataPipeline(cfg=self.cfg).run()

        results: list[SolverResult] = []

        # ── CPU baseline ──────────────────────────────────────────────
        if not skip_cpu:
            print("[CPU-NumPy]  Running …")
            cpu_solver = CPUMonteCarlo()
            res_cpu = cpu_solver.solve(bundle)
            results.append(res_cpu)
            print(f"  done in {res_cpu.wall_time_ms:.1f} ms")
        else:
            print("[CPU-NumPy]  Skipped (--skip-cpu)")

        # ── PyTorch baseline ──────────────────────────────────────────
        if device == "cuda":
            print("[PyTorch-Compile]  Running …")
            pt_solver = PyTorchMonteCarlo(use_compile=True)
            best_pt: Optional[SolverResult] = None
            for rep in range(num_repeats):
                r = pt_solver.solve(bundle)
                if best_pt is None or r.wall_time_ms < best_pt.wall_time_ms:
                    best_pt = r
            results.append(best_pt)
            print(f"  best of {num_repeats}: {best_pt.wall_time_ms:.2f} ms")

        # ── Triton fused kernel ───────────────────────────────────────
        if device == "cuda":
            print("[Triton-Fused]  Running …")
            from triton_fused_newsvendor import TritonFusedNewsvendor

            tr_solver = TritonFusedNewsvendor()
            best_tr: Optional[SolverResult] = None
            for rep in range(num_repeats):
                r = tr_solver.solve(bundle)
                if best_tr is None or r.wall_time_ms < best_tr.wall_time_ms:
                    best_tr = r
            results.append(best_tr)
            print(f"  best of {num_repeats}: {best_tr.wall_time_ms:.2f} ms")

        # ── Correctness ───────────────────────────────────────────────
        print("\n── Correctness Checks ──")
        ref = results[0]
        for test in results[1:]:
            check_correctness(ref, test)

        # Cross-check GPU solvers if both present
        if device == "cuda" and len(results) >= 3:
            check_correctness(results[1], results[2])

        # ── Results table ─────────────────────────────────────────────
        print_results_table(results, N, S)


# ═══════════════════════════════════════════════════════════════════════════
# Triton perf_report sweep (optional)
# ═══════════════════════════════════════════════════════════════════════════
def run_triton_perf_sweep() -> None:
    """
    Use ``triton.testing.perf_report`` to sweep N across powers of 2
    and produce a publication-quality plot of kernel throughput.
    """
    import triton
    from triton_fused_newsvendor import (
        TritonFusedNewsvendor,
        _fused_newsvendor_kernel,
        cdiv,
    )

    S_FIXED = 65_536  # fix scenarios for the sweep

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2 ** i for i in range(7, 13)],   # 128 … 4096
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["PyTorch-Compile", "Triton-Fused"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="ms",
            plot_name="newsvendor-kernel-time",
            args={"S": S_FIXED},
        )
    )
    def bench_fn(N: int, S: int, provider: str) -> float:
        cfg = NewsvendorConfig(N=N, S=S, device="cuda")
        bundle = DataPipeline(cfg=cfg).run()

        if provider == "torch":
            solver = PyTorchMonteCarlo(use_compile=True)
        else:
            solver = TritonFusedNewsvendor()

        # Warm-up
        solver.solve(bundle)

        # Timed
        res = solver.solve(bundle)
        return res.wall_time_ms

    bench_fn.run(print_data=True, save_path="./benchmark_results/")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark: Newsvendor Monte-Carlo Solvers"
    )
    parser.add_argument("--N", type=int, default=2048,
                        help="Number of product-location nodes (power of 2)")
    parser.add_argument("--S", type=int, default=131_072,
                        help="Number of Monte-Carlo scenarios")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip CPU solver (slow at large N)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of timed repeats per GPU solver")
    parser.add_argument("--sweep", action="store_true",
                        help="Run triton.testing.perf_report sweep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.sweep:
        run_triton_perf_sweep()
        return

    cfg = NewsvendorConfig(
        N=args.N, S=args.S, device=args.device, seed=args.seed,
    )
    suite = BenchmarkSuite(cfg)
    suite.run(skip_cpu=args.skip_cpu, num_repeats=args.repeats)


if __name__ == "__main__":
    main()
