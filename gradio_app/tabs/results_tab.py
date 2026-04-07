"""
results_tab.py -- Tab 3: Results Dashboard.

Displays a performance comparison table, bar charts, variant-specific
visualisations, and correctness metrics across solver backends.
"""

from __future__ import annotations

import sys
import os

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def create_results_tab(state: gr.State):
    """
    Build the Results Dashboard tab.

    Parameters
    ----------
    state : gr.State
        Shared application state.
    """
    with gr.TabItem("3. Results Dashboard"):
        gr.Markdown(
            "## Results Dashboard\n"
            "Compare performance across solver backends.  "
            "Click **Refresh** after running solvers."
        )

        refresh_btn = gr.Button("Refresh Results", variant="secondary")

        with gr.Row():
            with gr.Column(scale=1):
                perf_table = gr.Dataframe(
                    label="Performance Comparison",
                    headers=[
                        "Solver", "Wall Time (ms)", "Peak VRAM (MB)",
                        "Speedup vs CPU", "E[profit] Mean",
                    ],
                    interactive=False,
                )
            with gr.Column(scale=1):
                correctness_box = gr.Textbox(
                    label="Correctness Metrics",
                    interactive=False,
                    lines=8,
                    value="Run solvers first.",
                )

        perf_bar_plot = gr.Plot(label="Wall Time Comparison")
        variant_plot = gr.Plot(label="Variant-Specific Visualisation")

        # -----------------------------------------------------------------
        # Refresh callback
        # -----------------------------------------------------------------
        def _refresh(st):
            if st is None or not st.results:
                empty_df = pd.DataFrame(
                    columns=[
                        "Solver", "Wall Time (ms)", "Peak VRAM (MB)",
                        "Speedup vs CPU", "E[profit] Mean",
                    ]
                )
                return (
                    empty_df,
                    "No results yet. Run solvers first (Tab 2).",
                    None,
                    None,
                )

            results = st.results
            labels = sorted(results.keys())

            # ----------------------------------------------------------
            # Build performance table
            # ----------------------------------------------------------
            rows = []
            cpu_time = None
            for lbl in labels:
                r = results[lbl]
                # Detect CPU solver for speedup baseline
                if "cpu" in lbl.lower() or "CPU" in lbl:
                    cpu_time = r.wall_time_ms

            for lbl in labels:
                r = results[lbl]
                ep_mean = r.expected_profit.cpu().float().mean().item()
                peak_mb = r.peak_memory_bytes / (1024 ** 2) if r.peak_memory_bytes > 0 else 0.0
                speedup = (cpu_time / r.wall_time_ms) if (cpu_time and r.wall_time_ms > 0) else 0.0
                rows.append({
                    "Solver": r.label,
                    "Wall Time (ms)": round(r.wall_time_ms, 2),
                    "Peak VRAM (MB)": round(peak_mb, 1),
                    "Speedup vs CPU": round(speedup, 2),
                    "E[profit] Mean": round(ep_mean, 2),
                })

            df = pd.DataFrame(rows)

            # ----------------------------------------------------------
            # Bar chart: wall time
            # ----------------------------------------------------------
            bar_fig = _build_perf_bar_chart(results, labels)

            # ----------------------------------------------------------
            # Correctness metrics
            # ----------------------------------------------------------
            correctness_text = _compute_correctness(results, labels)

            # ----------------------------------------------------------
            # Variant-specific plot
            # ----------------------------------------------------------
            var_fig = _build_variant_plot(st)

            return df, correctness_text, bar_fig, var_fig

        refresh_btn.click(
            fn=_refresh,
            inputs=[state],
            outputs=[perf_table, correctness_box, perf_bar_plot, variant_plot],
        )


# =========================================================================
# Helper plotting functions
# =========================================================================

def _build_perf_bar_chart(results: dict, labels: list) -> go.Figure:
    """Bar chart comparing wall time and memory across solvers."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Wall Time (ms)", "Peak VRAM (MB)"),
    )

    solver_names = []
    times = []
    mems = []

    for lbl in labels:
        r = results[lbl]
        solver_names.append(r.label)
        times.append(r.wall_time_ms)
        mems.append(r.peak_memory_bytes / (1024 ** 2) if r.peak_memory_bytes > 0 else 0.0)

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    fig.add_trace(
        go.Bar(
            x=solver_names,
            y=times,
            marker_color=colors[:len(solver_names)],
            name="Wall Time",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=solver_names,
            y=mems,
            marker_color=colors[:len(solver_names)],
            name="Peak VRAM",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        title_text="Solver Performance Comparison",
    )

    return fig


def _compute_correctness(results: dict, labels: list) -> str:
    """Compute pairwise max/mean diffs between solver expected profits."""
    if len(labels) < 2:
        return "Need at least 2 solver results for correctness comparison."

    lines = ["Pairwise Expected Profit Differences:", ""]

    # Use the first solver as reference
    ref_label = labels[0]
    ref_ep = results[ref_label].expected_profit.cpu().float()

    for lbl in labels[1:]:
        ep = results[lbl].expected_profit.cpu().float()
        diff = (ref_ep - ep).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        lines.append(
            f"  {ref_label} vs {lbl}:"
        )
        lines.append(
            f"    Max  |diff|: {max_diff:.6f}"
        )
        lines.append(
            f"    Mean |diff|: {mean_diff:.6f}"
        )
        lines.append("")

    # If there are 3+ solvers, also compare second vs third, etc.
    for i in range(1, len(labels)):
        for j in range(i + 1, len(labels)):
            ep_i = results[labels[i]].expected_profit.cpu().float()
            ep_j = results[labels[j]].expected_profit.cpu().float()
            diff = (ep_i - ep_j).abs()
            lines.append(
                f"  {labels[i]} vs {labels[j]}:"
            )
            lines.append(
                f"    Max  |diff|: {diff.max().item():.6f}"
            )
            lines.append(
                f"    Mean |diff|: {diff.mean().item():.6f}"
            )
            lines.append("")

    return "\n".join(lines)


def _build_variant_plot(st) -> go.Figure | None:
    """
    Build a variant-specific visualisation depending on which variant was run.
    """
    var_key = st.current_variant
    results = st.results

    if not results:
        return None

    # Pick the first result that has variant-specific data
    any_result = next(iter(results.values()))

    if var_key == "grid_search":
        return _plot_grid_search(results)
    elif var_key == "cvar":
        return _plot_cvar(results)
    elif var_key == "budget":
        return _plot_budget(results)
    elif var_key == "substitution":
        return _plot_substitution(results)
    else:
        return _plot_base(results)


def _plot_base(results: dict) -> go.Figure:
    """Histogram of expected profit across products for each solver."""
    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for idx, (lbl, r) in enumerate(sorted(results.items())):
        ep = r.expected_profit.cpu().float().numpy()
        fig.add_trace(go.Histogram(
            x=ep,
            name=r.label,
            opacity=0.6,
            marker_color=colors[idx % len(colors)],
            nbinsx=40,
        ))

    fig.update_layout(
        title="Expected Profit Distribution (Base)",
        xaxis_title="E[profit] per product",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=400,
    )
    return fig


def _plot_grid_search(results: dict) -> go.Figure:
    """
    Plot profit surface for a few sample products from the first result
    that has profit_surface data.
    """
    fig = go.Figure()
    for lbl, r in sorted(results.items()):
        if hasattr(r, "profit_surface") and r.profit_surface is not None:
            surface = r.profit_surface.cpu().float().numpy()  # [N, K]
            q_grid = r.Q_grid.cpu().float().numpy()            # [K]
            N = surface.shape[0]
            # Show 5 sample products
            sample_indices = np.linspace(0, N - 1, min(5, N), dtype=int)
            for pi in sample_indices:
                fig.add_trace(go.Scatter(
                    x=q_grid,
                    y=surface[pi],
                    mode="lines",
                    name=f"Product {pi} ({r.label})",
                ))
            break  # Only plot from first solver with surface data

    fig.update_layout(
        title="Profit Surface: E[profit](Q) for Sample Products",
        xaxis_title="Q ratio (Q / mu)",
        yaxis_title="E[profit]",
        template="plotly_white",
        height=400,
    )
    return fig


def _plot_cvar(results: dict) -> go.Figure:
    """Plot E[profit], VaR, and CVaR side by side for a subset of products."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("E[profit]", "VaR", "CVaR"),
    )

    for lbl, r in sorted(results.items()):
        if hasattr(r, "VaR") and r.VaR is not None:
            N = r.expected_profit.shape[0]
            x = np.arange(min(N, 50))  # Show first 50 products

            ep = r.expected_profit.cpu().float().numpy()[:len(x)]
            var = r.VaR.cpu().float().numpy()[:len(x)]
            cvar = r.CVaR.cpu().float().numpy()[:len(x)]

            fig.add_trace(
                go.Bar(x=x, y=ep, name=f"E[pi] ({r.label})", showlegend=True),
                row=1, col=1,
            )
            fig.add_trace(
                go.Bar(x=x, y=var, name=f"VaR ({r.label})", showlegend=False),
                row=1, col=2,
            )
            fig.add_trace(
                go.Bar(x=x, y=cvar, name=f"CVaR ({r.label})", showlegend=False),
                row=1, col=3,
            )
            break  # Show one solver's risk measures

    fig.update_layout(
        title="Risk Metrics: E[profit], VaR, CVaR (first 50 products)",
        template="plotly_white",
        height=400,
    )
    return fig


def _plot_budget(results: dict) -> go.Figure:
    """Plot Lagrangian convergence (lambda and cost vs iteration)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Lambda vs Iteration", "Total Cost vs Iteration"),
    )

    for lbl, r in sorted(results.items()):
        if hasattr(r, "lambda_history") and r.lambda_history:
            iters = list(range(len(r.lambda_history)))
            fig.add_trace(
                go.Scatter(
                    x=iters, y=r.lambda_history,
                    mode="lines+markers",
                    name=f"lambda ({r.label})",
                ),
                row=1, col=1,
            )
        if hasattr(r, "cost_history") and r.cost_history:
            iters = list(range(len(r.cost_history)))
            fig.add_trace(
                go.Scatter(
                    x=iters, y=r.cost_history,
                    mode="lines+markers",
                    name=f"cost ({r.label})",
                ),
                row=1, col=2,
            )
            # Add budget line
            if hasattr(r, "budget") and r.budget > 0:
                fig.add_hline(
                    y=r.budget, line_dash="dash", line_color="red",
                    annotation_text="Budget B",
                    row=1, col=2,
                )

    fig.update_layout(
        title="Lagrangian Bisection Convergence",
        template="plotly_white",
        height=400,
    )
    return fig


def _plot_substitution(results: dict) -> go.Figure:
    """Plot average redirected demand and profit comparison."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Avg Redirected Demand", "Effective Profit"),
    )

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for idx, (lbl, r) in enumerate(sorted(results.items())):
        if hasattr(r, "substitution_demand") and r.substitution_demand is not None:
            sub_d = r.substitution_demand.cpu().float().numpy()
            fig.add_trace(
                go.Histogram(
                    x=sub_d, name=f"Redirect ({r.label})",
                    opacity=0.6,
                    marker_color=colors[idx % len(colors)],
                    nbinsx=30,
                ),
                row=1, col=1,
            )
        if hasattr(r, "effective_profit") and r.effective_profit is not None:
            eff = r.effective_profit.cpu().float().numpy()
            fig.add_trace(
                go.Histogram(
                    x=eff, name=f"E[pi] ({r.label})",
                    opacity=0.6,
                    marker_color=colors[idx % len(colors)],
                    nbinsx=30,
                ),
                row=1, col=2,
            )

    fig.update_layout(
        title="Substitution: Redirected Demand & Effective Profit",
        barmode="overlay",
        template="plotly_white",
        height=400,
    )
    return fig
