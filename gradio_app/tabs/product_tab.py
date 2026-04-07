"""
product_tab.py -- Tab 4: Per-Product Analysis.

Allows the user to select an individual product by index, view its
parameters (mu, p, c, s, Q, category), and compare expected profits
across solver backends for that product.  For grid-search variants,
a per-product profit curve is also displayed.
"""

from __future__ import annotations

import sys
import os

import gradio as gr
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def create_product_tab(state: gr.State):
    """
    Build the Per-Product Analysis tab.

    Parameters
    ----------
    state : gr.State
        Shared application state.
    """
    with gr.TabItem("4. Per-Product Analysis"):
        gr.Markdown(
            "## Per-Product Analysis\n"
            "Select a product index to inspect its parameters and compare "
            "solver outputs.  Click **Load Product** after generating data "
            "and running solvers."
        )

        with gr.Row():
            with gr.Column(scale=1):
                product_idx = gr.Slider(
                    minimum=0,
                    maximum=511,
                    step=1,
                    value=0,
                    label="Product Index",
                    info="Select a product (0 to N-1)",
                )
                load_btn = gr.Button("Load Product", variant="secondary")
                params_box = gr.Textbox(
                    label="Product Parameters",
                    interactive=False,
                    lines=12,
                    value="Generate data first.",
                )

            with gr.Column(scale=1):
                profit_comparison_box = gr.Textbox(
                    label="Profit Comparison Across Solvers",
                    interactive=False,
                    lines=10,
                    value="Run solvers first.",
                )

        product_plot = gr.Plot(label="Per-Product Visualisation")

        # -----------------------------------------------------------------
        # Update slider max when data is generated
        # -----------------------------------------------------------------
        def _update_slider_max(st):
            if st is None or not st.data_generated:
                return gr.update(maximum=511)
            return gr.update(maximum=st.N - 1)

        load_btn.click(
            fn=_update_slider_max,
            inputs=[state],
            outputs=[product_idx],
        )

        # -----------------------------------------------------------------
        # Load product callback
        # -----------------------------------------------------------------
        def _load_product(st, idx):
            if st is None or not st.data_generated:
                return (
                    "No data generated yet. Please run Tab 1 first.",
                    "No results yet.",
                    None,
                )

            idx = int(idx)
            bundle = st.bundle
            N = bundle.N

            if idx < 0 or idx >= N:
                return (
                    f"Invalid index {idx}. Must be 0 to {N - 1}.",
                    "Invalid product index.",
                    None,
                )

            # Extract product parameters
            mu_val = bundle.mu[idx, 0].item()
            p_val = bundle.p[idx, 0].item()
            c_val = bundle.c[idx, 0].item()
            s_val = bundle.s[idx, 0].item()
            q_val = bundle.Q[idx, 0].item()
            is_tractor = bundle.category_mask[idx].item()
            category = "Tractor" if is_tractor else "Generator"

            # Margin calculations
            margin_pct = ((p_val - c_val) / c_val) * 100 if c_val > 0 else 0
            salvage_pct = (s_val / c_val) * 100 if c_val > 0 else 0

            params_lines = [
                f"Product Index: {idx}",
                f"Category:      {category}",
                f"",
                f"Mean Demand (mu):   {mu_val:.2f} units/period",
                f"Selling Price (p):  {p_val:.4f}",
                f"Unit Cost (c):      {c_val:.4f}",
                f"Salvage Value (s):  {s_val:.4f}",
                f"Order Quantity (Q): {q_val:.2f}  (= mu)",
                f"",
                f"Gross Margin:       {margin_pct:.1f}%",
                f"Salvage / Cost:     {salvage_pct:.1f}%",
            ]

            # Substitution info
            if st.sub_idx is not None:
                sub_idx_row = st.sub_idx[idx].cpu().numpy()
                sub_frac_row = st.sub_frac[idx].cpu().numpy()
                valid = sub_idx_row >= 0
                n_subs = valid.sum()
                params_lines.append(f"")
                params_lines.append(f"Substitutes: {n_subs}")
                for k in range(len(sub_idx_row)):
                    if sub_idx_row[k] >= 0:
                        params_lines.append(
                            f"  -> Product {sub_idx_row[k]}, "
                            f"beta={sub_frac_row[k]:.3f}"
                        )

            # ----------------------------------------------------------
            # Profit comparison across solvers
            # ----------------------------------------------------------
            profit_lines = []
            if st.results:
                profit_lines.append(
                    f"Expected Profit at product {idx} ({category}):"
                )
                profit_lines.append("")

                for lbl in sorted(st.results.keys()):
                    r = st.results[lbl]
                    ep = r.expected_profit.cpu().float()
                    if idx < ep.shape[0]:
                        val = ep[idx].item()
                        profit_lines.append(f"  {r.label:30s}  E[pi] = {val:.4f}")

                    # Show variant-specific per-product data
                    if hasattr(r, "Q_star") and r.Q_star is not None:
                        qs = r.Q_star.cpu().float()
                        if idx < qs.shape[0]:
                            profit_lines.append(
                                f"  {'':30s}  Q* = {qs[idx].item():.2f}"
                            )
                    if hasattr(r, "VaR") and r.VaR is not None:
                        var = r.VaR.cpu().float()
                        if idx < var.shape[0]:
                            profit_lines.append(
                                f"  {'':30s}  VaR = {var[idx].item():.4f}"
                            )
                    if hasattr(r, "CVaR") and r.CVaR is not None:
                        cvar = r.CVaR.cpu().float()
                        if idx < cvar.shape[0]:
                            profit_lines.append(
                                f"  {'':30s}  CVaR = {cvar[idx].item():.4f}"
                            )
                    if (hasattr(r, "substitution_demand")
                            and r.substitution_demand is not None):
                        sd = r.substitution_demand.cpu().float()
                        if idx < sd.shape[0]:
                            profit_lines.append(
                                f"  {'':30s}  Redirected = {sd[idx].item():.4f}"
                            )
                    profit_lines.append("")
            else:
                profit_lines.append("No solver results yet. Run solvers first (Tab 2).")

            # ----------------------------------------------------------
            # Per-product plot
            # ----------------------------------------------------------
            fig = _build_product_plot(st, idx)

            return (
                "\n".join(params_lines),
                "\n".join(profit_lines),
                fig,
            )

        load_btn.click(
            fn=_load_product,
            inputs=[state, product_idx],
            outputs=[params_box, profit_comparison_box, product_plot],
        )


def _build_product_plot(st, idx: int) -> go.Figure | None:
    """Build a per-product visualisation depending on the variant."""
    if not st.results:
        return None

    var_key = st.current_variant

    if var_key == "grid_search":
        return _plot_product_grid_search(st, idx)
    elif var_key == "cvar":
        return _plot_product_cvar(st, idx)
    elif var_key == "base" or var_key == "budget" or var_key == "substitution":
        return _plot_product_profit_bar(st, idx)
    return None


def _plot_product_profit_bar(st, idx: int) -> go.Figure:
    """Bar chart of expected profit for this product across solvers."""
    fig = go.Figure()
    labels = []
    values = []
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    for i, (lbl, r) in enumerate(sorted(st.results.items())):
        ep = r.expected_profit.cpu().float()
        if idx < ep.shape[0]:
            labels.append(r.label)
            values.append(ep[idx].item())

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors[:len(labels)],
    ))

    fig.update_layout(
        title=f"E[profit] for Product {idx} Across Solvers",
        xaxis_title="Solver",
        yaxis_title="E[profit]",
        template="plotly_white",
        height=400,
    )

    return fig


def _plot_product_grid_search(st, idx: int) -> go.Figure:
    """Plot the profit curve over Q for a specific product."""
    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    for i, (lbl, r) in enumerate(sorted(st.results.items())):
        if hasattr(r, "profit_surface") and r.profit_surface is not None:
            surface = r.profit_surface.cpu().float().numpy()
            q_grid = r.Q_grid.cpu().float().numpy()
            if idx < surface.shape[0]:
                fig.add_trace(go.Scatter(
                    x=q_grid,
                    y=surface[idx],
                    mode="lines+markers",
                    name=r.label,
                    line=dict(color=colors[i % len(colors)]),
                ))

                # Mark the optimal Q*
                if hasattr(r, "Q_star") and r.Q_star is not None:
                    q_star = r.Q_star.cpu().float()[idx].item()
                    best_p = r.best_profit.cpu().float()[idx].item()
                    mu_val = st.bundle.mu[idx, 0].item()
                    # Q* is in absolute units; Q_grid is in ratios
                    q_star_ratio = q_star / mu_val if mu_val > 0 else 0
                    fig.add_trace(go.Scatter(
                        x=[q_star_ratio],
                        y=[best_p],
                        mode="markers",
                        name=f"Q* ({r.label})",
                        marker=dict(
                            size=12, symbol="star",
                            color=colors[i % len(colors)],
                        ),
                        showlegend=True,
                    ))

    fig.update_layout(
        title=f"Profit Curve for Product {idx}: E[profit](Q)",
        xaxis_title="Q / mu ratio",
        yaxis_title="E[profit]",
        template="plotly_white",
        height=400,
    )

    return fig


def _plot_product_cvar(st, idx: int) -> go.Figure:
    """Show E[profit], VaR, CVaR as grouped bars for one product."""
    fig = go.Figure()

    for lbl, r in sorted(st.results.items()):
        if hasattr(r, "VaR") and r.VaR is not None:
            ep = r.expected_profit.cpu().float()
            var = r.VaR.cpu().float()
            cvar = r.CVaR.cpu().float()

            if idx < ep.shape[0]:
                metrics = ["E[profit]", "VaR", "CVaR"]
                vals = [
                    ep[idx].item(),
                    var[idx].item(),
                    cvar[idx].item(),
                ]
                fig.add_trace(go.Bar(
                    x=metrics,
                    y=vals,
                    name=r.label,
                ))

    fig.update_layout(
        title=f"Risk Metrics for Product {idx}",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        template="plotly_white",
        height=400,
    )

    return fig
