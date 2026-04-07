"""
app.py — Main Gradio application for the Newsvendor Solver Suite.

Assembles all five tabs into a single Gradio Blocks app:
  1. Problem Setup       — configure N, S, generate data
  2. Solver Config & Run — select variant, run CPU/PyTorch/Triton solvers
  3. Results Dashboard    — performance comparison, variant-specific plots
  4. Per-Product Analysis — drill down into individual products
  5. About               — mathematical formulations, architecture, credits

Usage (local):
    python -m gradio_app.app

Usage (Google Colab):
    from gradio_app.app import launch
    launch(share=True)
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
from gradio_app.state import AppState
from gradio_app.tabs import (
    create_setup_tab,
    create_solver_tab,
    create_results_tab,
    create_product_tab,
    create_about_tab,
)


# ---------------------------------------------------------------------------
# App title and description
# ---------------------------------------------------------------------------
APP_TITLE = "Newsvendor Solver Suite"
APP_DESCRIPTION = """\
### GPU-Accelerated Newsvendor Inventory Optimization

**What is this?** A dealership must decide how many units of each product to order
*before* knowing actual customer demand. Order too many and you lose money on unsold
stock; order too few and you miss sales. This app simulates that decision across
hundreds of correlated products and compares three solver implementations:
a plain CPU solver, a PyTorch GPU solver, and a **custom Triton GPU kernel** that
keeps all intermediate data in fast on-chip SRAM (the main technical contribution).

**Quick start:** Go to **Tab 1**, click *Generate Data*, then go to **Tab 2**, pick
a variant, and click *Run Solvers*. Results appear in **Tab 3**.

| Variant | Plain-English Description |
|---------|--------------------------|
| **Base** | Evaluate profit at a fixed order quantity (simplest baseline) |
| **Grid Search** | Try many order quantities, find the best one per product |
| **CVaR** | Focus on the worst-case scenarios (risk-averse manager) |
| **Budget** | Find optimal orders when total spending is capped |
| **Substitution** | If one product stocks out, customers switch to a substitute |

*IIT Kharagpur B.Tech Project -- built with Triton, PyTorch, and Gradio.*
"""


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app (without launching it)."""

    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ),
        css="""
        .main-header { text-align: center; margin-bottom: 0.5em; }
        .tab-content { min-height: 400px; }
        """,
    ) as app:
        # Shared mutable state across all tabs
        state = gr.State(value=AppState())

        # Header
        gr.Markdown(f"# {APP_TITLE}", elem_classes=["main-header"])
        gr.Markdown(APP_DESCRIPTION)

        # Tabs
        with gr.Tabs():
            create_setup_tab(state)
            create_solver_tab(state)
            create_results_tab(state)
            create_product_tab(state)
            create_about_tab(state)

    return app


def launch(share: bool = False, server_port: int = 7860, **kwargs):
    """
    Create and launch the Gradio app.

    Parameters
    ----------
    share : bool
        If True, creates a public Gradio link (useful for Colab).
    server_port : int
        Port to serve on locally.
    **kwargs
        Additional arguments passed to ``app.launch()``.
    """
    app = create_app()
    app.launch(
        share=share,
        server_port=server_port,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Newsvendor Solver Suite — Gradio App")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link (for Colab)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Server port (default: 7860)")
    args = parser.parse_args()

    # Detect Colab environment
    in_colab = "google.colab" in sys.modules
    share = args.share or in_colab

    if in_colab:
        print("Detected Google Colab — enabling share=True for public URL")

    launch(share=share, server_port=args.port)
