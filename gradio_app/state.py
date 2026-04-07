"""
state.py -- Application state management for the Gradio newsvendor suite.

Wraps all mutable data needed by the Gradio interface into a single
``AppState`` dataclass that is passed between tabs via ``gr.State``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class AppState:
    """Mutable state shared across Gradio tabs via gr.State."""

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    bundle: Optional[Any] = None  # TensorBundle (Any avoids circular imports)
    N: int = 512
    S: int = 32768
    seed: int = 42
    tractor_fraction: float = 0.6
    data_generated: bool = False

    # ------------------------------------------------------------------
    # Substitution graph
    # ------------------------------------------------------------------
    sub_idx: Optional[torch.Tensor] = None   # [N, max_subs]
    sub_frac: Optional[torch.Tensor] = None  # [N, max_subs]

    # ------------------------------------------------------------------
    # Solver results -- keyed by solver label
    # ------------------------------------------------------------------
    results: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Extension-specific data
    # ------------------------------------------------------------------
    current_variant: str = "base"
    extension_data: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Performance metrics for display
    # ------------------------------------------------------------------
    variant_label: str = "Base Newsvendor"

    # ==================================================================
    # Helper methods
    # ==================================================================

    @staticmethod
    def estimate_vram_gb(N: int, S: int) -> float:
        """
        Estimate peak GPU memory (in GB) for the full newsvendor solve.

        Major allocations (fp32, 4 bytes each):
            L       : N x N         -- Cholesky factor (kept resident)
            Z       : N x S         -- scenario matrix (kept resident)
            D       : N x S         -- correlated demand (intermediate)
            mu,p,c,s,Q : 5 x N x 1 -- parameter vectors (negligible)

        The matmul ``L @ Z`` produces D in-place for the fused Triton
        kernel, but PyTorch materialises it as a separate tensor.
        Peak = L + Z + D + small elementwise buffers.

        Returns
        -------
        float
            Estimated peak VRAM in gigabytes.
        """
        bytes_per_element = 4  # fp32

        l_bytes = N * N * bytes_per_element         # Cholesky factor
        z_bytes = N * S * bytes_per_element         # scenario matrix
        d_bytes = N * S * bytes_per_element         # demand intermediate
        param_bytes = 5 * N * bytes_per_element     # mu, p, c, s, Q
        # Elementwise intermediates (X, overage, profit) -- worst case 2 extra
        elem_bytes = 2 * N * S * bytes_per_element

        total_bytes = l_bytes + z_bytes + d_bytes + param_bytes + elem_bytes
        return total_bytes / (1024 ** 3)

    def clear_results(self) -> None:
        """Reset solver results and extension-specific data."""
        self.results.clear()
        self.extension_data.clear()

    def get_solver_labels(self) -> List[str]:
        """Return a sorted list of solver labels available in current results."""
        return sorted(self.results.keys())
