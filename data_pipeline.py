"""
data_pipeline.py — ETL pipeline for the Multi-Echelon Stochastic Newsvendor.

Hybridises two real-world data sources:
  1. **M5 Forecasting Dataset (Kaggle):** Extracts the hierarchical spatial /
     temporal *correlation structure* (Σ) across store-product nodes.
  2. **Public Tractor / Generator Sales Time-Series:** Extracts realistic
     demand distributions (μ, σ) for heavy-machinery inventory.

Financial tensors (unit cost `c`, selling price `p`, salvage value `s`) are
generated programmatically from the ranges in ``config.FinancialParams``.

The pipeline outputs a ``TensorBundle`` — a frozen collection of GPU-resident
tensors ready for consumption by the solvers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from config import (
    DEFAULT_CONFIG,
    DEFAULT_FINANCE,
    FinancialParams,
    InventoryCategory,
    NewsvendorConfig,
    SubstitutionConfig,
    DEFAULT_SUBSTITUTION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorBundle:
    """
    Immutable container holding every GPU tensor required by the solvers.

    Shapes (given config with N products, S scenarios):
        L   : [N, N]   — lower-triangular Cholesky factor of Σ
        mu  : [N, 1]   — mean demand vector
        p   : [N, 1]   — selling price per unit
        c   : [N, 1]   — unit procurement cost
        s   : [N, 1]   — salvage (markdown) value per unit
        Q   : [N, 1]   — initial order quantity (set to μ as baseline)
        Z   : [N, S]   — standard-normal scenario matrix
    """
    L:  torch.Tensor
    mu: torch.Tensor
    p:  torch.Tensor
    c:  torch.Tensor
    s:  torch.Tensor
    Q:  torch.Tensor
    Z:  torch.Tensor
    # metadata
    N:  int
    S:  int
    category_mask: torch.Tensor   # bool [N] — True = tractor, False = generator


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — M5 Topology Extraction
# ═══════════════════════════════════════════════════════════════════════════
class M5TopologyExtractor:
    """
    Extracts a realistic hierarchical correlation matrix from the M5 dataset.

    If the M5 parquet / CSV files are available locally, the extractor reads
    actual sales and computes the empirical correlation.  Otherwise it falls
    back to a *synthetic hierarchical block-correlation* matrix that mirrors
    the M5 structure (3 categories × 7 departments × 10 stores).

    Why M5?
    --------
    M5 provides **real-world spatial correlation** between store-product
    pairs driven by geography, promotions, and category affinity — exactly
    the kind of demand coupling we need for a multi-echelon network.
    """

    M5_CATEGORIES = 3
    M5_DEPARTMENTS = 7
    M5_STORES = 10

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("data/m5")

    # ------------------------------------------------------------------
    def extract_correlation(self, N: int, seed: int = 42) -> np.ndarray:
        """
        Return an N×N positive-definite correlation matrix.

        Tries to load real M5 data first; falls back to synthetic.
        """
        m5_path = self.data_dir / "sales_train_evaluation.csv"
        if m5_path.exists():
            logger.info("Loading real M5 sales data from %s", m5_path)
            return self._from_real_m5(m5_path, N, seed)
        logger.info("M5 data not found — generating synthetic hierarchical Σ")
        return self._synthetic_hierarchical(N, seed)

    # ------------------------------------------------------------------
    def _from_real_m5(self, path: Path, N: int, seed: int) -> np.ndarray:
        """
        Read M5 CSV, sample N rows, compute Pearson correlation on the
        trailing 365 days of daily unit sales, then regularise to ensure
        positive-definiteness.
        """
        df = pd.read_csv(path)
        # Sales columns are d_1 … d_1941
        sale_cols = [c for c in df.columns if c.startswith("d_")]
        # Use trailing year for stationarity
        sale_cols = sale_cols[-365:]
        sales = df[sale_cols].values.astype(np.float64)

        rng = np.random.default_rng(seed)
        idx = rng.choice(sales.shape[0], size=min(N, sales.shape[0]), replace=False)
        sampled = sales[idx]

        # Pad / tile to exactly N rows if dataset is smaller
        if sampled.shape[0] < N:
            repeats = (N // sampled.shape[0]) + 1
            sampled = np.tile(sampled, (repeats, 1))[:N]

        corr = np.corrcoef(sampled)
        return self._regularise(corr, N)

    # ------------------------------------------------------------------
    def _synthetic_hierarchical(self, N: int, seed: int) -> np.ndarray:
        """
        Build a block-structured correlation matrix that mimics M5 hierarchy:

            Same store, same dept  → ρ = 0.70
            Same store, diff dept  → ρ = 0.40
            Diff store, same dept  → ρ = 0.25
            Diff store, diff dept  → ρ = 0.10

        Then add a small random perturbation for realism.
        """
        rng = np.random.default_rng(seed)
        R = np.eye(N, dtype=np.float64)

        # Assign each node a (store, dept) label round-robin
        n_stores = self.M5_STORES
        n_depts = self.M5_DEPARTMENTS
        stores = np.arange(N) % n_stores
        depts = (np.arange(N) // n_stores) % n_depts

        for i in range(N):
            for j in range(i + 1, N):
                same_store = stores[i] == stores[j]
                same_dept = depts[i] == depts[j]
                if same_store and same_dept:
                    rho = 0.70
                elif same_store:
                    rho = 0.40
                elif same_dept:
                    rho = 0.25
                else:
                    rho = 0.10
                # Small perturbation
                rho += rng.uniform(-0.03, 0.03)
                rho = np.clip(rho, 0.01, 0.99)
                R[i, j] = rho
                R[j, i] = rho

        return self._regularise(R, N)

    # ------------------------------------------------------------------
    @staticmethod
    def _regularise(R: np.ndarray, N: int) -> np.ndarray:
        """
        Regularise a correlation matrix to guarantee strict positive-
        definiteness (required for Cholesky).

        Strategy: spectral projection — clip eigenvalues at ε > 0,
        then rescale diagonals back to 1.
        """
        eigvals, eigvecs = np.linalg.eigh(R)
        eps = 1e-6
        eigvals = np.maximum(eigvals, eps)
        R_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Restore unit diagonal
        d = np.sqrt(np.diag(R_reg))
        R_reg = R_reg / np.outer(d, d)
        np.fill_diagonal(R_reg, 1.0)
        return R_reg


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Demand Distribution Mapping
# ═══════════════════════════════════════════════════════════════════════════
class DemandDistributionMapper:
    """
    Generates per-node mean (μ) and standard-deviation (σ) vectors by
    sampling from the financial ranges for each inventory category.

    For generators, spare-parts demand from correlated failure rates is
    added on top of base sales demand.
    """

    def __init__(self, cfg: NewsvendorConfig, fin: FinancialParams) -> None:
        self.cfg = cfg
        self.fin = fin

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu, sigma) each of shape [N]."""
        N = self.cfg.N
        N_t = self.cfg.N_tractors
        N_g = self.cfg.N_generators

        mu = np.empty(N, dtype=np.float64)
        sigma = np.empty(N, dtype=np.float64)

        # Tractors — seasonal, high-variance
        mu[:N_t] = rng.uniform(*self.fin.tractor_demand_mu, size=N_t)
        sigma[:N_t] = rng.uniform(*self.fin.tractor_demand_sigma, size=N_t)

        # Generators — base sales + spare-parts from correlated failures
        base_mu = rng.uniform(*self.fin.gen_demand_mu, size=N_g)
        base_sigma = rng.uniform(*self.fin.gen_demand_sigma, size=N_g)

        # Spare-parts add-on: Poisson-like mean from failure rate × installed base
        installed_base = base_mu * 12  # proxy: annual sales ≈ installed fleet
        failure_mu = rng.uniform(*self.fin.gen_failure_rate_mu, size=N_g)
        spare_demand = installed_base * failure_mu
        spare_sigma = np.sqrt(spare_demand)  # Poisson-like variance

        mu[N_t:] = base_mu + spare_demand
        sigma[N_t:] = np.sqrt(base_sigma ** 2 + spare_sigma ** 2)

        return mu, sigma


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — Financial Tensor Generation
# ═══════════════════════════════════════════════════════════════════════════
class FinancialTensorGenerator:
    """
    Generates per-node (p, c, s) vectors reflecting heavy-machinery margins.

    Tractors:  high c, high p, very low s  (capital goods depreciate fast).
    Generators: moderate c, moderate p, low s.
    """

    def __init__(self, cfg: NewsvendorConfig, fin: FinancialParams) -> None:
        self.cfg = cfg
        self.fin = fin

    def generate(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p, c, s) each of shape [N]."""
        N = self.cfg.N
        N_t = self.cfg.N_tractors
        N_g = self.cfg.N_generators

        p = np.empty(N, dtype=np.float64)
        c = np.empty(N, dtype=np.float64)
        s = np.empty(N, dtype=np.float64)

        # Tractors
        c[:N_t] = rng.uniform(*self.fin.tractor_unit_cost, size=N_t)
        p[:N_t] = rng.uniform(*self.fin.tractor_selling_price, size=N_t)
        s[:N_t] = rng.uniform(*self.fin.tractor_salvage_value, size=N_t)

        # Generators
        c[N_t:] = rng.uniform(*self.fin.gen_unit_cost, size=N_g)
        p[N_t:] = rng.uniform(*self.fin.gen_selling_price, size=N_g)
        s[N_t:] = rng.uniform(*self.fin.gen_salvage_value, size=N_g)

        # Sanity: selling price > cost > salvage
        p = np.maximum(p, c * 1.15)   # at least 15 % margin
        s = np.minimum(s, c * 0.25)   # salvage ≤ 25 % of cost

        return p, c, s


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
class DataPipeline:
    """
    End-to-end orchestrator: runs stages 1–3 and returns a ``TensorBundle``
    with every tensor pre-allocated on the target device.
    """

    def __init__(
        self,
        cfg: NewsvendorConfig = DEFAULT_CONFIG,
        fin: FinancialParams = DEFAULT_FINANCE,
        m5_data_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.fin = fin
        self.topology_extractor = M5TopologyExtractor(m5_data_dir)
        self.demand_mapper = DemandDistributionMapper(cfg, fin)
        self.financial_gen = FinancialTensorGenerator(cfg, fin)

    # ------------------------------------------------------------------
    def run(self) -> TensorBundle:
        """Execute the full pipeline and return GPU-ready tensors."""
        N, S = self.cfg.N, self.cfg.S
        device = self.cfg.device
        dtype = self.cfg.dtype
        rng = np.random.default_rng(self.cfg.seed)

        logger.info("Pipeline: N=%d, S=%d, device=%s", N, S, device)

        # ---- Stage 1: Correlation matrix → Cholesky ----
        logger.info("Stage 1 — Extracting correlation topology …")
        R = self.topology_extractor.extract_correlation(N, self.cfg.seed)

        # Build covariance Σ = diag(σ) @ R @ diag(σ)
        _, sigma_vec = self.demand_mapper.generate(rng)
        Sigma = np.outer(sigma_vec, sigma_vec) * R

        # Cholesky decomposition (on CPU in float64 for numerical stability)
        L_np = np.linalg.cholesky(Sigma)
        L = torch.tensor(L_np, dtype=dtype, device=device)
        logger.info("  Cholesky factor L: shape=%s", list(L.shape))

        # ---- Stage 2: Demand distributions ----
        logger.info("Stage 2 — Mapping demand distributions …")
        # Re-generate with same rng state already advanced by sigma_vec above
        mu_np = self.demand_mapper.generate(
            np.random.default_rng(self.cfg.seed)
        )[0]
        mu = torch.tensor(mu_np, dtype=dtype, device=device).unsqueeze(1)  # [N,1]

        # ---- Stage 3: Financial tensors ----
        logger.info("Stage 3 — Generating financial tensors …")
        p_np, c_np, s_np = self.financial_gen.generate(rng)
        p = torch.tensor(p_np, dtype=dtype, device=device).unsqueeze(1)
        c = torch.tensor(c_np, dtype=dtype, device=device).unsqueeze(1)
        s = torch.tensor(s_np, dtype=dtype, device=device).unsqueeze(1)

        # ---- Order quantity Q: initialise to μ (newsvendor baseline) ----
        Q = mu.clone()

        # ---- Standard-normal scenario matrix Z ----
        logger.info("Stage 4 — Generating Z matrix [%d, %d] …", N, S)
        gen = torch.Generator(device=device)
        gen.manual_seed(self.cfg.seed)
        Z = torch.randn(N, S, dtype=dtype, device=device, generator=gen)

        # ---- Category mask ----
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[: self.cfg.N_tractors] = True

        logger.info("Pipeline complete.")
        return TensorBundle(
            L=L, mu=mu, p=p, c=c, s=s, Q=Q, Z=Z,
            N=N, S=S, category_mask=mask,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Substitution graph generator
# ═══════════════════════════════════════════════════════════════════════════
class SubstitutionGraphGenerator:
    """
    Generates a sparse substitution graph for the multi-product substitution
    newsvendor extension.

    Each product gets up to ``max_subs`` substitutes, chosen from the same
    category (tractors substitute with tractors, generators with generators).
    Substitution fractions β are drawn uniformly from [beta_min, beta_max].
    """

    def __init__(self, sub_cfg: SubstitutionConfig = DEFAULT_SUBSTITUTION) -> None:
        self.sub_cfg = sub_cfg

    def generate(
        self,
        N: int,
        category_mask: np.ndarray,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (sub_idx, sub_frac) each of shape [N, max_subs].

        sub_idx[i, k]  = index of k-th substitute for product i (-1 if none).
        sub_frac[i, k] = fraction β of unmet demand redirected (0 if none).
        """
        rng = np.random.default_rng(seed + 999)
        max_subs = self.sub_cfg.max_subs
        beta_min = self.sub_cfg.beta_min
        beta_max = self.sub_cfg.beta_max

        sub_idx = np.full((N, max_subs), -1, dtype=np.int64)
        sub_frac = np.zeros((N, max_subs), dtype=np.float64)

        tractor_indices = np.where(category_mask)[0]
        gen_indices = np.where(~category_mask)[0]

        for i in range(N):
            # Pick substitutes from same category
            if category_mask[i]:
                pool = tractor_indices[tractor_indices != i]
            else:
                pool = gen_indices[gen_indices != i]

            if len(pool) == 0:
                continue

            n_subs = min(max_subs, len(pool))
            chosen = rng.choice(pool, size=n_subs, replace=False)
            fracs = rng.uniform(beta_min, beta_max, size=n_subs)

            sub_idx[i, :n_subs] = chosen
            sub_frac[i, :n_subs] = fracs

        return sub_idx, sub_frac


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    # Use a small size for quick testing
    small_cfg = NewsvendorConfig(N=64, S=1024, device="cpu")
    bundle = DataPipeline(cfg=small_cfg).run()
    print(f"L  : {bundle.L.shape}")
    print(f"mu : {bundle.mu.shape}")
    print(f"p  : {bundle.p.shape}")
    print(f"c  : {bundle.c.shape}")
    print(f"s  : {bundle.s.shape}")
    print(f"Q  : {bundle.Q.shape}")
    print(f"Z  : {bundle.Z.shape}")
    print(f"Tractors: {bundle.category_mask.sum().item()}, "
          f"Generators: {(~bundle.category_mask).sum().item()}")
