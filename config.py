"""
config.py — Central configuration for the Multi-Echelon Stochastic Newsvendor
              Inventory Optimization Engine.

All hyper-parameters (problem size, financial ranges, Triton tuning knobs) live
here so that every other module imports a single source of truth.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Inventory category enum
# ---------------------------------------------------------------------------
class InventoryCategory(Enum):
    """Two product lines modelled by the dealership network."""
    TRACTOR = "tractor"
    GENERATOR = "generator"


# ---------------------------------------------------------------------------
# Core problem dimensions
# ---------------------------------------------------------------------------
@dataclass
class NewsvendorConfig:
    """
    Governs the scale of the Monte-Carlo simulation.

    Parameters
    ----------
    N : int
        Number of unique product-location nodes in the network.
        Must be a power of 2 for efficient Triton tiling (default 2048).
    S : int
        Number of Monte-Carlo demand scenarios (default 131072 = 2^17).
    seed : int
        Global RNG seed for reproducibility.
    device : str
        PyTorch device string.
    dtype : torch.dtype
        Floating-point precision for GPU tensors.
    tractor_fraction : float
        Fraction of N nodes that are tractor nodes (remainder = generators).
    """
    N: int = 2048
    S: int = 131_072
    seed: int = 42
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    tractor_fraction: float = 0.6  # 60 % tractors, 40 % generators

    # Derived ---
    @property
    def N_tractors(self) -> int:
        return int(self.N * self.tractor_fraction)

    @property
    def N_generators(self) -> int:
        return self.N - self.N_tractors

    def __post_init__(self) -> None:
        assert self.N > 0 and (self.N & (self.N - 1)) == 0, \
            f"N must be a power of 2, got {self.N}"
        assert self.S > 0, f"S must be positive, got {self.S}"
        assert 0.0 < self.tractor_fraction < 1.0


# ---------------------------------------------------------------------------
# Financial parameter ranges (per inventory category)
# ---------------------------------------------------------------------------
@dataclass
class FinancialParams:
    """
    Realistic price / cost / salvage ranges for heavy-machinery dealerships.

    All values are in INR (₹ lakhs) for tractors and INR (₹ thousands) for
    generators.  The pipeline normalises everything to a common unit before
    constructing tensors, so the *ratios* (margin %) are what matter.

    Ranges are (low, high) tuples — the pipeline draws uniformly.
    """
    # --- Tractors (₹ lakhs) ---
    tractor_unit_cost:     tuple[float, float] = (4.5, 9.0)
    tractor_selling_price: tuple[float, float] = (6.0, 13.5)
    tractor_salvage_value: tuple[float, float] = (0.5, 1.5)   # very low resale
    tractor_demand_mu:     tuple[float, float] = (8.0, 40.0)   # units / period
    tractor_demand_sigma:  tuple[float, float] = (3.0, 15.0)

    # --- Generators (₹ thousands) ---
    gen_unit_cost:         tuple[float, float] = (35.0, 80.0)
    gen_selling_price:     tuple[float, float] = (55.0, 120.0)
    gen_salvage_value:     tuple[float, float] = (5.0, 15.0)
    gen_demand_mu:         tuple[float, float] = (15.0, 60.0)
    gen_demand_sigma:      tuple[float, float] = (5.0, 25.0)

    # Spare-parts correlated failure rate (generators only)
    gen_failure_rate_mu:   tuple[float, float] = (0.02, 0.08)
    gen_failure_corr:      float = 0.35   # inter-location failure correlation


# ---------------------------------------------------------------------------
# Triton kernel tuning search space
# ---------------------------------------------------------------------------
@dataclass
class TritonTuningConfig:
    """
    Defines the autotune search grid passed to ``@triton.autotune``.

    Block sizes are chosen so that:
        BLOCK_SIZE_M * BLOCK_SIZE_K  fits a tile of L in SRAM,
        BLOCK_SIZE_K * BLOCK_SIZE_N  fits a tile of Z in SRAM,
    and the fp32 accumulator of shape [BLOCK_SIZE_M, BLOCK_SIZE_N] also fits.

    A100/H100 SM has ~228 KB shared memory; each fp32 element = 4 B.
    Tile budget ≈ 228 KB / 4 B ≈ 57 344 elements split across L-tile,
    Z-tile, and accumulator.
    """
    block_m_options: list[int] = field(default_factory=lambda: [32, 64, 128])
    block_n_options: list[int] = field(default_factory=lambda: [64, 128, 256])
    block_k_options: list[int] = field(default_factory=lambda: [32, 64])
    num_warps_options: list[int] = field(default_factory=lambda: [4, 8])
    num_stages_options: list[int] = field(default_factory=lambda: [2, 3, 4])


# ---------------------------------------------------------------------------
# Convenience: default configs used when running ``python benchmark.py``
# ---------------------------------------------------------------------------
DEFAULT_CONFIG   = NewsvendorConfig()
DEFAULT_FINANCE  = FinancialParams()
DEFAULT_TRITON   = TritonTuningConfig()
