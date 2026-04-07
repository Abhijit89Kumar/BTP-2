"""
common.py — Shared dataclasses and utilities for newsvendor extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SolverResult:
    """Uniform result container (mirrors baseline_solvers.SolverResult)."""
    expected_profit: torch.Tensor   # [N]
    wall_time_ms: float
    peak_memory_bytes: int
    label: str


@dataclass
class GridSearchResult(SolverResult):
    """Result from optimal Q* grid search."""
    Q_star: Optional[torch.Tensor] = None            # [N] — optimal order quantities
    best_profit: Optional[torch.Tensor] = None        # [N] — E[π] at Q*
    profit_surface: Optional[torch.Tensor] = None     # [N, K] — profit at each grid point
    Q_grid: Optional[torch.Tensor] = None             # [K] — the Q ratio grid used


@dataclass
class CVaRResult(SolverResult):
    """Result from CVaR risk-averse newsvendor."""
    VaR: Optional[torch.Tensor] = None                # [N] — Value at Risk per product
    CVaR: Optional[torch.Tensor] = None               # [N] — Conditional VaR per product
    alpha: float = 0.05                                # risk level used
    histogram: Optional[torch.Tensor] = None           # [N, B] — profit histograms (Triton)


@dataclass
class BudgetResult(SolverResult):
    """Result from budget-constrained newsvendor."""
    Q_star: Optional[torch.Tensor] = None             # [N] — constrained optimal Q
    best_profit: Optional[torch.Tensor] = None        # [N] — E[π] at constrained Q*
    lambda_star: float = 0.0                           # optimal Lagrange multiplier
    total_cost: float = 0.0                            # Σ c_i · Q_i* at solution
    budget: float = 0.0                                # budget B
    lambda_history: list[float] = field(default_factory=list)
    cost_history: list[float] = field(default_factory=list)


@dataclass
class SubstitutionResult(SolverResult):
    """Result from multi-product substitution newsvendor."""
    substitution_demand: Optional[torch.Tensor] = None  # [N] — avg redirected demand
    effective_profit: Optional[torch.Tensor] = None     # [N] — profit under substitution


def estimate_flops_base(N: int, S: int) -> float:
    """FLOP estimate for base newsvendor: matmul + elementwise + reduction."""
    return 2.0 * N * N * S + 7.0 * N * S


def estimate_flops_grid_search(N: int, S: int, K: int) -> float:
    """FLOP estimate for grid search: matmul once + K × elementwise."""
    return 2.0 * N * N * S + K * 7.0 * N * S


def estimate_flops_cvar(N: int, S: int) -> float:
    """FLOP estimate for CVaR: base + histogram binning."""
    return 2.0 * N * N * S + 10.0 * N * S


def estimate_flops_substitution(N: int, S: int, max_subs: int = 4) -> float:
    """FLOP estimate for substitution: base + stockout + redirect."""
    return 2.0 * N * N * S + (10.0 + max_subs * 3.0) * N * S
