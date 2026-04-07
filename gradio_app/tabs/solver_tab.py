"""
solver_tab.py -- Tab 2: Solver Configuration & Run.

Lets the user select a newsvendor variant, configure variant-specific
parameters, choose which solver backends (CPU / PyTorch / Triton) to
execute, and run them with progress feedback.
"""

from __future__ import annotations

import sys
import os
import time
import traceback

import gradio as gr

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Variant labels
# ---------------------------------------------------------------------------
VARIANT_CHOICES = [
    "Base",
    "Grid Search (Q*)",
    "CVaR (Risk-Averse)",
    "Budget-Constrained",
    "Substitution",
]


def create_solver_tab(state: gr.State):
    """
    Build the Solver Configuration & Run tab.

    Parameters
    ----------
    state : gr.State
        Shared application state (``AppState`` instance).
    """
    with gr.TabItem("2. Run Solvers"):
        gr.Markdown(
            "## Solver Configuration & Run\n"
            "Select a newsvendor variant, configure its parameters, pick "
            "solver backends, and run."
        )

        with gr.Row():
            # ---- Left column: variant + parameters ----
            with gr.Column(scale=1):
                variant_radio = gr.Radio(
                    choices=VARIANT_CHOICES,
                    value="Base",
                    label="Newsvendor Variant",
                )

                # -- Grid Search parameters --
                with gr.Group(visible=False) as gs_group:
                    gr.Markdown("### Grid Search Parameters")
                    gs_k = gr.Slider(
                        minimum=16, maximum=128, step=16, value=64,
                        label="K (grid points)",
                    )
                    gs_qmin = gr.Slider(
                        minimum=0.1, maximum=1.0, step=0.1, value=0.3,
                        label="Q ratio min",
                    )
                    gs_qmax = gr.Slider(
                        minimum=1.0, maximum=4.0, step=0.1, value=2.5,
                        label="Q ratio max",
                    )

                # -- CVaR parameters --
                with gr.Group(visible=False) as cvar_group:
                    gr.Markdown("### CVaR Parameters")
                    cvar_alpha = gr.Slider(
                        minimum=0.01, maximum=0.20, step=0.01, value=0.05,
                        label="Alpha (risk level)",
                        info="Fraction of worst-case scenarios (e.g. 0.05 = worst 5%)",
                    )
                    cvar_bins = gr.Slider(
                        minimum=64, maximum=512, step=64, value=256,
                        label="Histogram bins (Triton)",
                    )

                # -- Budget parameters --
                with gr.Group(visible=False) as budget_group:
                    gr.Markdown("### Budget-Constrained Parameters")
                    budget_frac = gr.Slider(
                        minimum=0.3, maximum=1.0, step=0.05, value=0.7,
                        label="Budget fraction",
                        info="B = fraction x total unconstrained cost",
                    )

                # -- Substitution parameters --
                with gr.Group(visible=False) as sub_group:
                    gr.Markdown("### Substitution Parameters")
                    sub_max_subs = gr.Slider(
                        minimum=1, maximum=4, step=1, value=4,
                        label="Max substitutes per product",
                    )
                    sub_beta_min = gr.Slider(
                        minimum=0.01, maximum=0.20, step=0.01, value=0.05,
                        label="Beta min (substitution fraction)",
                    )
                    sub_beta_max = gr.Slider(
                        minimum=0.10, maximum=0.50, step=0.01, value=0.30,
                        label="Beta max (substitution fraction)",
                    )

            # ---- Right column: solver selection + run ----
            with gr.Column(scale=1):
                gr.Markdown("### Solver Backends")
                chk_cpu = gr.Checkbox(value=True, label="CPU (NumPy)")
                chk_pytorch = gr.Checkbox(value=True, label="PyTorch GPU (torch.compile)")
                chk_triton = gr.Checkbox(value=True, label="Triton Fused Kernel")

                run_btn = gr.Button("Run Solvers", variant="primary")

                status_box = gr.Textbox(
                    label="Run Log",
                    interactive=False,
                    lines=16,
                    value="Select a variant and click Run Solvers.",
                )

        # -----------------------------------------------------------------
        # Visibility toggle for variant parameter groups
        # -----------------------------------------------------------------
        def _toggle_variant_groups(variant):
            return (
                gr.update(visible=(variant == "Grid Search (Q*)")),
                gr.update(visible=(variant == "CVaR (Risk-Averse)")),
                gr.update(visible=(variant == "Budget-Constrained")),
                gr.update(visible=(variant == "Substitution")),
            )

        variant_radio.change(
            fn=_toggle_variant_groups,
            inputs=[variant_radio],
            outputs=[gs_group, cvar_group, budget_group, sub_group],
        )

        # -----------------------------------------------------------------
        # Main solver run callback
        # -----------------------------------------------------------------
        def _run_solvers(
            st,
            variant,
            # checkboxes
            run_cpu, run_pytorch, run_triton,
            # grid search params
            k_val, qmin, qmax,
            # cvar params
            alpha, num_bins,
            # budget params
            b_frac,
            # substitution params
            max_subs, beta_min, beta_max,
        ):
            import torch
            from gradio_app.state import AppState

            if st is None or not st.data_generated:
                return st, "ERROR: Please generate data first (Tab 1)."

            bundle = st.bundle
            log_lines: list[str] = []
            st.clear_results()

            # Map variant string to internal key
            variant_map = {
                "Base": "base",
                "Grid Search (Q*)": "grid_search",
                "CVaR (Risk-Averse)": "cvar",
                "Budget-Constrained": "budget",
                "Substitution": "substitution",
            }
            var_key = variant_map.get(variant, "base")
            st.current_variant = var_key
            st.variant_label = variant

            log_lines.append(f"Variant: {variant}")
            log_lines.append(f"N={st.N}, S={st.S}")
            log_lines.append("")

            # Collect solver classes to run
            solvers_to_run = []

            try:
                if var_key == "base":
                    solvers_to_run = _get_base_solvers(run_cpu, run_pytorch, run_triton)
                elif var_key == "grid_search":
                    solvers_to_run = _get_grid_search_solvers(
                        run_cpu, run_pytorch, run_triton,
                        int(k_val), float(qmin), float(qmax),
                    )
                elif var_key == "cvar":
                    solvers_to_run = _get_cvar_solvers(
                        run_cpu, run_pytorch, run_triton,
                        float(alpha), int(num_bins),
                    )
                elif var_key == "budget":
                    solvers_to_run = _get_budget_solvers(
                        run_cpu, run_pytorch, run_triton,
                        float(b_frac),
                    )
                elif var_key == "substitution":
                    solvers_to_run = _get_substitution_solvers(
                        run_cpu, run_pytorch, run_triton,
                    )
            except Exception as e:
                log_lines.append(f"ERROR building solvers: {e}")
                return st, "\n".join(log_lines)

            if not solvers_to_run:
                log_lines.append("No solvers selected. Check at least one backend.")
                return st, "\n".join(log_lines)

            # Execute solvers
            for label, solve_fn in solvers_to_run:
                log_lines.append(f"--- Running: {label} ---")
                try:
                    t0 = time.perf_counter()
                    result = solve_fn(bundle, st)
                    elapsed = (time.perf_counter() - t0) * 1e3

                    st.results[result.label] = result

                    log_lines.append(f"  Wall time: {result.wall_time_ms:.2f} ms")
                    log_lines.append(
                        f"  Peak VRAM: {result.peak_memory_bytes / 1024**2:.1f} MB"
                        if result.peak_memory_bytes > 0
                        else "  Peak VRAM: N/A (CPU)"
                    )
                    ep = result.expected_profit
                    if ep is not None and ep.numel() > 0:
                        ep_cpu = ep.cpu().float()
                        log_lines.append(
                            f"  E[profit] range: [{ep_cpu.min().item():.2f}, "
                            f"{ep_cpu.max().item():.2f}]"
                        )
                    log_lines.append("")

                except torch.cuda.OutOfMemoryError:
                    log_lines.append("  *** CUDA OOM -- skipped ***")
                    log_lines.append("")
                except Exception as e:
                    log_lines.append(f"  ERROR: {e}")
                    log_lines.append(traceback.format_exc())
                    log_lines.append("")

                # Clean up between solvers
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            log_lines.append(f"Completed {len(st.results)} solver(s).")
            return st, "\n".join(log_lines)

        run_btn.click(
            fn=_run_solvers,
            inputs=[
                state, variant_radio,
                chk_cpu, chk_pytorch, chk_triton,
                gs_k, gs_qmin, gs_qmax,
                cvar_alpha, cvar_bins,
                budget_frac,
                sub_max_subs, sub_beta_min, sub_beta_max,
            ],
            outputs=[state, status_box],
        )


# =========================================================================
# Solver factory helpers -- each returns a list of (label, solve_fn) pairs
# =========================================================================

def _get_base_solvers(run_cpu, run_pytorch, run_triton):
    """Return solver (label, callable) pairs for the base newsvendor."""
    solvers = []
    if run_cpu:
        def _solve_cpu(bundle, st):
            from baseline_solvers import CPUMonteCarlo
            return CPUMonteCarlo().solve(bundle)
        solvers.append(("CPU-NumPy", _solve_cpu))
    if run_pytorch:
        def _solve_pt(bundle, st):
            from baseline_solvers import PyTorchMonteCarlo
            return PyTorchMonteCarlo(use_compile=True).solve(bundle)
        solvers.append(("PyTorch-Compile", _solve_pt))
    if run_triton:
        def _solve_tr(bundle, st):
            from triton_fused_newsvendor import TritonFusedNewsvendor
            return TritonFusedNewsvendor().solve(bundle)
        solvers.append(("Triton-Fused", _solve_tr))
    return solvers


def _get_grid_search_solvers(run_cpu, run_pytorch, run_triton, K, qmin, qmax):
    """Return solver pairs for the Grid Search variant."""
    from config import GridSearchConfig
    grid_cfg = GridSearchConfig(K=K, q_ratio_min=qmin, q_ratio_max=qmax)

    solvers = []
    if run_cpu:
        def _solve_cpu(bundle, st):
            from extensions.grid_search_solvers import CPUGridSearch
            return CPUGridSearch().solve(bundle, grid_cfg)
        solvers.append(("CPU-GridSearch", _solve_cpu))
    if run_pytorch:
        def _solve_pt(bundle, st):
            from extensions.grid_search_solvers import PyTorchGridSearch
            return PyTorchGridSearch(use_compile=True).solve(bundle, grid_cfg)
        solvers.append(("PyTorch-GridSearch", _solve_pt))
    if run_triton:
        def _solve_tr(bundle, st):
            from extensions.grid_search_solvers import TritonGridSearch
            return TritonGridSearch().solve(bundle, grid_cfg)
        solvers.append(("Triton-GridSearch", _solve_tr))
    return solvers


def _get_cvar_solvers(run_cpu, run_pytorch, run_triton, alpha, num_bins):
    """Return solver pairs for the CVaR variant."""
    from config import CVaRConfig
    cvar_cfg = CVaRConfig(alpha=alpha, num_bins=int(num_bins))

    solvers = []
    if run_cpu:
        def _solve_cpu(bundle, st):
            from extensions.cvar_solvers import CPUCVaR
            return CPUCVaR(cvar_cfg).solve(bundle)
        solvers.append(("CPU-CVaR", _solve_cpu))
    if run_pytorch:
        def _solve_pt(bundle, st):
            from extensions.cvar_solvers import PyTorchCVaR
            return PyTorchCVaR(cvar_cfg, use_compile=True).solve(bundle)
        solvers.append(("PyTorch-CVaR", _solve_pt))
    if run_triton:
        def _solve_tr(bundle, st):
            from extensions.cvar_solvers import TritonCVaR
            return TritonCVaR(cvar_cfg).solve(bundle)
        solvers.append(("Triton-CVaR", _solve_tr))
    return solvers


def _get_budget_solvers(run_cpu, run_pytorch, run_triton, budget_fraction):
    """Return solver pairs for the Budget-Constrained variant."""
    from config import GridSearchConfig, BudgetConfig
    grid_cfg = GridSearchConfig(K=64, q_ratio_min=0.3, q_ratio_max=2.5)
    budget_cfg = BudgetConfig(budget_fraction=budget_fraction)

    solvers = []
    if run_cpu:
        def _solve_cpu(bundle, st):
            from extensions.budget_solvers import CPUBudget
            return CPUBudget().solve(bundle, grid_cfg, budget_cfg)
        solvers.append(("CPU-Budget", _solve_cpu))
    if run_pytorch:
        def _solve_pt(bundle, st):
            from extensions.budget_solvers import PyTorchBudget
            return PyTorchBudget(use_compile=True).solve(bundle, grid_cfg, budget_cfg)
        solvers.append(("PyTorch-Budget", _solve_pt))
    if run_triton:
        def _solve_tr(bundle, st):
            from extensions.budget_solvers import TritonBudget
            return TritonBudget().solve(bundle, grid_cfg, budget_cfg)
        solvers.append(("Triton-Budget", _solve_tr))
    return solvers


def _get_substitution_solvers(run_cpu, run_pytorch, run_triton):
    """Return solver pairs for the Substitution variant."""
    solvers = []
    if run_cpu:
        def _solve_cpu(bundle, st):
            from extensions.substitution_solvers import CPUSubstitution
            sub_idx_np = st.sub_idx.cpu().numpy()
            sub_frac_np = st.sub_frac.cpu().numpy()
            return CPUSubstitution().solve(bundle, sub_idx_np, sub_frac_np)
        solvers.append(("CPU-Substitution", _solve_cpu))
    if run_pytorch:
        def _solve_pt(bundle, st):
            from extensions.substitution_solvers import PyTorchSubstitution
            return PyTorchSubstitution(use_compile=True).solve(
                bundle, st.sub_idx, st.sub_frac
            )
        solvers.append(("PyTorch-Substitution", _solve_pt))
    if run_triton:
        def _solve_tr(bundle, st):
            from extensions.substitution_solvers import TritonSubstitution
            return TritonSubstitution().solve(
                bundle, st.sub_idx, st.sub_frac,
                max_subs=st.sub_idx.shape[1],
            )
        solvers.append(("Triton-Substitution", _solve_tr))
    return solvers
