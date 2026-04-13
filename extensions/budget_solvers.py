"""
budget_solvers.py — Three solver implementations for the Budget-Constrained
                    Newsvendor extension via Lagrangian Dual decomposition.

The budget constraint is:
    Σ c_i · Q_i  ≤  B       (total procurement budget)

Solved via bisection on the Lagrange multiplier λ ≥ 0:
  - At each λ, solve the grid-search kernel with effective cost c' = c · (1 + λ)
  - Update λ based on budget violation

This extension reuses the Grid Search extension's kernels — no new Triton
kernel is needed.

Solvers
-------
1. ``CPUBudget``     — CPU NumPy Lagrangian loop using CPUGridSearch.
2. ``PyTorchBudget`` — GPU PyTorch Lagrangian loop using PyTorchGridSearch.
3. ``TritonBudget``  — GPU Triton Lagrangian loop using TritonGridSearch.
"""

from __future__ import annotations

import time
from copy import copy
from typing import Optional

import numpy as np
import torch

from data_pipeline import TensorBundle
from config import GridSearchConfig, BudgetConfig
from extensions.common import BudgetResult
from extensions.grid_search_solvers import (
    CPUGridSearch,
    PyTorchGridSearch,
    TritonGridSearch,
    _build_q_ratios,
)


def _compute_total_cost(
    Q_star: torch.Tensor,
    c: torch.Tensor,
) -> float:
    """Compute total procurement cost Σ c_i · Q_i*."""
    c_flat = c.squeeze() if c.dim() > 1 else c
    return (c_flat.cpu().float() * Q_star.cpu().float()).sum().item()


def _compute_budget_default(bundle: TensorBundle, budget_cfg: BudgetConfig) -> float:
    """
    Compute the default budget B as a fraction of the unconstrained cost.

    B = budget_fraction × Σ c_i · μ_i
    """
    c_flat = bundle.c.squeeze().cpu().float()
    mu_flat = bundle.mu.squeeze().cpu().float()
    unconstrained_cost = (c_flat * mu_flat).sum().item()
    return budget_cfg.budget_fraction * unconstrained_cost


class _LagrangianSolver:
    """
    Base class implementing the Lagrangian bisection loop.

    Subclasses provide the inner grid-search solver (CPU, PyTorch, or Triton).
    """

    def __init__(self, inner_solver, label: str) -> None:
        self._inner_solver = inner_solver
        self.label = label

    def solve(
        self,
        bundle: TensorBundle,
        grid_cfg: GridSearchConfig,
        budget_cfg: BudgetConfig,
        budget: Optional[float] = None,
    ) -> BudgetResult:
        B = budget if budget is not None else _compute_budget_default(bundle, budget_cfg)

        N = bundle.N
        c_flat = bundle.c.squeeze()  # [N]
        mu_flat = bundle.mu.squeeze()  # [N]
        device = bundle.L.device

        max_iter = budget_cfg.max_iterations
        tol = budget_cfg.tolerance

        lambda_history: list[float] = []
        cost_history: list[float] = []

        # Bisection bounds for λ
        lambda_lo = 0.0
        lambda_hi = 10.0  # generous upper bound — will be expanded if needed

        t0 = time.perf_counter()

        # First: solve unconstrained (λ=0) to check if budget is already satisfied
        result_0 = self._inner_solver.solve(bundle, grid_cfg)
        cost_0 = _compute_total_cost(result_0.Q_star, bundle.c)

        if cost_0 <= B * (1 + tol):
            # Budget not binding — unconstrained solution is feasible
            wall_ms = (time.perf_counter() - t0) * 1e3
            return BudgetResult(
                expected_profit=result_0.expected_profit,
                wall_time_ms=wall_ms,
                peak_memory_bytes=result_0.peak_memory_bytes,
                label=self.label,
                Q_star=result_0.Q_star,
                best_profit=result_0.best_profit,
                lambda_star=0.0,
                total_cost=cost_0,
                budget=B,
                lambda_history=[0.0],
                cost_history=[cost_0],
            )

        # Bisection: find λ such that Σ c_i · Q_i*(λ) ≈ B
        final_result = None
        final_lambda = 0.0
        peak_mem = result_0.peak_memory_bytes

        for iteration in range(max_iter):
            lam = (lambda_lo + lambda_hi) / 2.0
            lambda_history.append(lam)

            # Modify the bundle's cost: c' = c * (1 + λ)
            modified_bundle = _modify_cost(bundle, lam)

            # Solve grid search with modified cost
            result = self._inner_solver.solve(modified_bundle, grid_cfg)
            peak_mem = max(peak_mem, result.peak_memory_bytes)

            # Compute total cost at this λ (use original c, not modified)
            total_cost = _compute_total_cost(result.Q_star, bundle.c)
            cost_history.append(total_cost)

            final_result = result
            final_lambda = lam

            # Bisection update
            if total_cost > B * (1 + tol):
                # Still over budget → increase λ (raise effective cost)
                lambda_lo = lam
                # Expand upper bound if needed
                if lam >= lambda_hi * 0.95:
                    lambda_hi *= 2.0
            elif total_cost < B * (1 - tol):
                # Under budget → decrease λ
                lambda_hi = lam
            else:
                # Within tolerance — converged
                break

        wall_ms = (time.perf_counter() - t0) * 1e3

        final_cost = _compute_total_cost(final_result.Q_star, bundle.c)

        return BudgetResult(
            expected_profit=final_result.expected_profit,
            wall_time_ms=wall_ms,
            peak_memory_bytes=peak_mem,
            label=self.label,
            Q_star=final_result.Q_star,
            best_profit=final_result.best_profit,
            lambda_star=final_lambda,
            total_cost=final_cost,
            budget=B,
            lambda_history=lambda_history,
            cost_history=cost_history,
        )


def _modify_cost(bundle: TensorBundle, lam: float) -> TensorBundle:
    """
    Create a modified TensorBundle with effective cost c' = c * (1 + λ).

    We create a new TensorBundle with the modified cost — all other tensors
    are shared (no copy needed since they are read-only).
    """
    c_modified = bundle.c * (1.0 + lam)
    # TensorBundle is frozen, so we need to create a new one
    return TensorBundle(
        L=bundle.L,
        mu=bundle.mu,
        p=bundle.p,
        c=c_modified,
        s=bundle.s,
        Q=bundle.Q,
        Z=bundle.Z,
        N=bundle.N,
        S=bundle.S,
        category_mask=bundle.category_mask,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Solver 1 — CPU (NumPy) with Lagrangian bisection
# ═══════════════════════════════════════════════════════════════════════════
class CPUBudget(_LagrangianSolver):
    """CPU NumPy budget-constrained solver using Lagrangian dual."""

    def __init__(self) -> None:
        super().__init__(
            inner_solver=CPUGridSearch(),
            label="CPU-Budget",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Solver 2 — PyTorch GPU with Lagrangian bisection
# ═══════════════════════════════════════════════════════════════════════════
class PyTorchBudget(_LagrangianSolver):
    """PyTorch GPU budget-constrained solver using Lagrangian dual."""

    def __init__(self, use_compile: bool = True) -> None:
        super().__init__(
            inner_solver=PyTorchGridSearch(use_compile=use_compile),
            label="PyTorch-Budget-Compile" if use_compile else "PyTorch-Budget-Eager",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Solver 3 — Triton Fused with Lagrangian bisection
# ═══════════════════════════════════════════════════════════════════════════
class TritonBudget(_LagrangianSolver):
    """Triton GPU budget-constrained solver using Lagrangian dual."""

    def __init__(self) -> None:
        super().__init__(
            inner_solver=TritonGridSearch(),
            label="Triton-Budget",
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from config import NewsvendorConfig, GridSearchConfig, BudgetConfig
    from data_pipeline import DataPipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = NewsvendorConfig(N=256, S=8192, device="cuda")
    grid_cfg = GridSearchConfig(K=32, q_ratio_min=0.3, q_ratio_max=2.5)
    budget_cfg = BudgetConfig(budget_fraction=0.7, max_iterations=20, tolerance=1e-3)
    bundle = DataPipeline(cfg=cfg).run()

    B = _compute_budget_default(bundle, budget_cfg)
    print(f"Budget B = {B:.2f}")

    for SolverClass in [CPUBudget, TritonBudget]:
        solver = SolverClass()
        res = solver.solve(bundle, grid_cfg, budget_cfg)
        print(f"[{res.label}]  time={res.wall_time_ms:.1f} ms  "
              f"lambda*={res.lambda_star:.4f}  "
              f"cost={res.total_cost:.2f} / {res.budget:.2f}  "
              f"iters={len(res.lambda_history)}")
