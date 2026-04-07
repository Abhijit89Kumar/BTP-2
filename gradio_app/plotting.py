"""
plotting.py -- Plotly figure builders for the Gradio newsvendor results dashboard.

Every public function returns a ``plotly.graph_objects.Figure`` ready for
rendering inside a Gradio ``gr.Plot`` component.  Edge cases (empty data,
None values) are handled gracefully by returning an annotated empty figure.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_TEMPLATE = "plotly_white"
_PALETTE = px.colors.qualitative.Set2
_DEFAULT_WIDTH = 800
_DEFAULT_HEIGHT = 500

# Canonical colours per solver family
_SOLVER_COLORS: Dict[str, str] = {
    "CPU":     _PALETTE[1],  # orange
    "PyTorch": _PALETTE[0],  # blue-green
    "Triton":  _PALETTE[2],  # green
}


def _color_for_label(label: str) -> str:
    """Pick a colour based on whether the label contains a known keyword."""
    for key, color in _SOLVER_COLORS.items():
        if key.lower() in label.lower():
            return color
    # Fallback: cycle through the palette
    return _PALETTE[hash(label) % len(_PALETTE)]


def _empty_figure(message: str = "No data available.") -> go.Figure:
    """Return a blank figure with a centred annotation."""
    fig = go.Figure()
    fig.update_layout(
        template=_TEMPLATE,
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=18, color="grey"),
            )
        ],
    )
    return fig


def _try_cpu_numpy(tensor: Any) -> Optional[np.ndarray]:
    """Safely convert a tensor (torch or numpy) to a numpy array on CPU."""
    if tensor is None:
        return None
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
    except ImportError:
        pass
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.asarray(tensor)


# ===================================================================
# 1. Performance bars  (Wall Time | Peak Memory | TFLOPS)
# ===================================================================
def plot_performance_bars(results_dict: Dict[str, Any]) -> go.Figure:
    """
    Three-panel bar chart comparing solvers on Wall Time, Peak Memory,
    and TFLOPS.  Speedup annotations relative to the slowest solver are
    added on the time panel.

    Parameters
    ----------
    results_dict : dict
        Maps solver label -> SolverResult (or any object with
        ``wall_time_ms``, ``peak_memory_bytes``, ``label`` attrs).
    """
    if not results_dict:
        return _empty_figure("Run solvers first to see performance comparison.")

    labels: List[str] = []
    times: List[float] = []
    mems: List[float] = []
    tflops_list: List[float] = []
    colors: List[str] = []

    for lbl, res in results_dict.items():
        labels.append(lbl)
        wt = getattr(res, "wall_time_ms", 0.0)
        times.append(wt)
        peak = getattr(res, "peak_memory_bytes", 0)
        mems.append(peak / (1024 ** 3) if peak else 0.0)

        # Estimate TFLOPS -- need N and S from expected_profit shape or attr
        ep = getattr(res, "expected_profit", None)
        n_val = ep.shape[0] if ep is not None and hasattr(ep, "shape") else 0
        # Use stored metadata if available
        n_sz = getattr(res, "_N", n_val)
        s_sz = getattr(res, "_S", 0)
        flops = 2.0 * n_sz * n_sz * s_sz + 7.0 * n_sz * s_sz if s_sz > 0 else 0.0
        tf = (flops / (wt * 1e-3)) / 1e12 if wt > 0 and flops > 0 else 0.0
        tflops_list.append(tf)

        colors.append(_color_for_label(lbl))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Wall Time (ms)", "Peak Memory (GB)", "TFLOPS"],
        horizontal_spacing=0.08,
    )

    # -- Panel 1: Wall Time --
    fig.add_trace(
        go.Bar(
            x=labels, y=times, marker_color=colors,
            text=[f"{t:.1f}" for t in times],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f} ms<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    # Speedup annotations relative to the slowest solver
    max_time = max(times) if times else 1.0
    for i, (lbl, t) in enumerate(zip(labels, times)):
        if t > 0 and t < max_time:
            speedup = max_time / t
            fig.add_annotation(
                x=lbl, y=t,
                text=f"{speedup:.1f}x",
                showarrow=False,
                yshift=20,
                font=dict(size=11, color=colors[i]),
                row=1, col=1,
            )

    # -- Panel 2: Peak Memory --
    fig.add_trace(
        go.Bar(
            x=labels, y=mems, marker_color=colors,
            text=[f"{m:.3f}" for m in mems],
            textposition="outside",
            hovertemplate="%{x}: %{y:.3f} GB<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # -- Panel 3: TFLOPS --
    fig.add_trace(
        go.Bar(
            x=labels, y=tflops_list, marker_color=colors,
            text=[f"{t:.2f}" for t in tflops_list],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f} TFLOPS<extra></extra>",
            showlegend=False,
        ),
        row=1, col=3,
    )

    fig.update_layout(
        template=_TEMPLATE,
        width=1100,
        height=_DEFAULT_HEIGHT,
        title_text="Solver Performance Comparison",
        title_x=0.5,
        margin=dict(t=80, b=40),
    )
    return fig


# ===================================================================
# 2. Profit distribution  (overlaid histograms)
# ===================================================================
def plot_profit_distribution(
    expected_profits_dict: Dict[str, Any],
    variant: str = "base",
) -> go.Figure:
    """
    Overlaid semi-transparent histograms of E[profit] across products
    for each solver.

    Parameters
    ----------
    expected_profits_dict : dict
        Maps solver label -> tensor/array of shape [N] (expected profit).
    variant : str
        Problem variant name shown in the title.
    """
    if not expected_profits_dict:
        return _empty_figure("No profit data to display.")

    fig = go.Figure()

    for idx, (lbl, ep) in enumerate(expected_profits_dict.items()):
        arr = _try_cpu_numpy(ep)
        if arr is None or arr.size == 0:
            continue
        color = _color_for_label(lbl)
        fig.add_trace(
            go.Histogram(
                x=arr,
                name=lbl,
                marker_color=color,
                opacity=0.55,
                nbinsx=60,
                hovertemplate="%{x:.2f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
        title_text=f"E[Profit] Distribution ({variant})",
        title_x=0.5,
        xaxis_title="Expected Profit per Product",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ===================================================================
# 3. Grid-search profit surface
# ===================================================================
def plot_grid_search_surface(
    profit_surface: Any,
    Q_grid: Any,
    mu: Any,
    product_indices: Optional[Sequence[int]] = None,
) -> go.Figure:
    """
    Line plots of E[profit] vs Q-ratio for selected products, with the
    optimal Q* marked as a star.

    Parameters
    ----------
    profit_surface : Tensor [N, K]
        Expected profit at each grid point for every product.
    Q_grid : Tensor [K]
        The ratio multipliers used in the grid search.
    mu : Tensor [N] or [N,1]
        Mean demand per product (used for hover info).
    product_indices : list[int], optional
        Indices of products to plot.  Defaults to 5 evenly spaced.
    """
    ps = _try_cpu_numpy(profit_surface)
    qg = _try_cpu_numpy(Q_grid)
    mu_arr = _try_cpu_numpy(mu)

    if ps is None or qg is None:
        return _empty_figure("Run Grid Search first.")

    N, K = ps.shape
    if mu_arr is not None and mu_arr.ndim == 2:
        mu_arr = mu_arr.squeeze(1)

    if product_indices is None:
        product_indices = np.linspace(0, N - 1, min(5, N), dtype=int).tolist()

    fig = go.Figure()

    for rank, idx in enumerate(product_indices):
        if idx < 0 or idx >= N:
            continue
        color = _PALETTE[rank % len(_PALETTE)]
        profits = ps[idx]
        mu_val = mu_arr[idx] if mu_arr is not None else 0.0
        label = f"Product {idx} (mu={mu_val:.1f})"

        fig.add_trace(
            go.Scatter(
                x=qg, y=profits,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate="Q/mu=%{x:.2f}  E[pi]=%{y:.2f}<extra></extra>",
            )
        )

        # Mark optimal Q*
        best_k = int(np.argmax(profits))
        fig.add_trace(
            go.Scatter(
                x=[qg[best_k]], y=[profits[best_k]],
                mode="markers",
                marker=dict(symbol="star", size=14, color=color, line=dict(width=1, color="black")),
                name=f"Q* prod {idx}",
                hovertemplate=f"Q*/mu={qg[best_k]:.2f}  E[pi]={profits[best_k]:.2f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
        title_text="Grid Search: E[Profit] vs Order Quantity Ratio",
        title_x=0.5,
        xaxis_title="Q / mu (order-quantity ratio)",
        yaxis_title="E[Profit]",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ===================================================================
# 4. CVaR distribution  (single product)
# ===================================================================
def plot_cvar_distribution(
    profits_per_scenario: Any,
    VaR: Any,
    CVaR: Any,
    alpha: float,
    product_idx: int,
) -> go.Figure:
    """
    Profit histogram for a single product with VaR and CVaR vertical
    lines and a shaded tail region below VaR.

    Parameters
    ----------
    profits_per_scenario : Tensor [N, S] or [S]
        Per-scenario profits.  If 2-D, ``product_idx`` selects the row.
    VaR : Tensor [N] or scalar
        Value at Risk per product (or single product).
    CVaR : Tensor [N] or scalar
        Conditional VaR per product.
    alpha : float
        Risk level (e.g. 0.05).
    product_idx : int
        Which product to visualise.
    """
    profits = _try_cpu_numpy(profits_per_scenario)
    var_arr = _try_cpu_numpy(VaR)
    cvar_arr = _try_cpu_numpy(CVaR)

    if profits is None:
        return _empty_figure("Run CVaR solver first.")

    # Extract single product
    if profits.ndim == 2:
        if product_idx < 0 or product_idx >= profits.shape[0]:
            return _empty_figure(f"Product index {product_idx} out of range.")
        profits = profits[product_idx]

    var_val = float(var_arr[product_idx]) if var_arr is not None and var_arr.ndim >= 1 else float(var_arr) if var_arr is not None else None
    cvar_val = float(cvar_arr[product_idx]) if cvar_arr is not None and cvar_arr.ndim >= 1 else float(cvar_arr) if cvar_arr is not None else None

    fig = go.Figure()

    # Full histogram
    fig.add_trace(
        go.Histogram(
            x=profits,
            nbinsx=80,
            marker_color=_PALETTE[0],
            opacity=0.7,
            name="Profit",
            hovertemplate="%{x:.2f}<extra></extra>",
        )
    )

    # Shaded tail (profits <= VaR)
    if var_val is not None:
        tail = profits[profits <= var_val]
        if tail.size > 0:
            fig.add_trace(
                go.Histogram(
                    x=tail,
                    nbinsx=40,
                    marker_color="rgba(231, 76, 60, 0.6)",
                    name=f"Tail ({alpha*100:.0f}%)",
                    hovertemplate="%{x:.2f}<extra></extra>",
                )
            )

    # VaR line
    if var_val is not None:
        fig.add_vline(
            x=var_val, line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"VaR = {var_val:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="red", size=12),
        )

    # CVaR line
    if cvar_val is not None:
        fig.add_vline(
            x=cvar_val, line_dash="dot", line_color="darkred", line_width=2,
            annotation_text=f"CVaR = {cvar_val:.2f}",
            annotation_position="top right",
            annotation_font=dict(color="darkred", size=12),
        )

    fig.update_layout(
        template=_TEMPLATE,
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
        title_text=f"CVaR Analysis -- Product {product_idx} (alpha={alpha})",
        title_x=0.5,
        xaxis_title="Per-Scenario Profit",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ===================================================================
# 5. Budget convergence  (dual y-axis: lambda + cost)
# ===================================================================
def plot_budget_convergence(
    lambda_history: Sequence[float],
    cost_history: Sequence[float],
    budget: float,
) -> go.Figure:
    """
    Dual y-axis plot showing Lagrange multiplier (lambda) convergence
    and total procurement cost convergence vs iteration, with a
    horizontal budget line.

    Parameters
    ----------
    lambda_history : list[float]
        Lambda values at each bisection iteration.
    cost_history : list[float]
        Total cost at each bisection iteration.
    budget : float
        Target budget B.
    """
    if not lambda_history or not cost_history:
        return _empty_figure("Run Budget-Constrained solver first.")

    iterations = list(range(1, len(lambda_history) + 1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Lambda trace (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=iterations, y=list(lambda_history),
            mode="lines+markers",
            name="Lambda",
            line=dict(color=_PALETTE[0], width=2),
            marker=dict(size=5),
            hovertemplate="Iter %{x}: lambda=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Cost trace (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=iterations[:len(cost_history)],
            y=list(cost_history),
            mode="lines+markers",
            name="Total Cost",
            line=dict(color=_PALETTE[2], width=2),
            marker=dict(size=5),
            hovertemplate="Iter %{x}: cost=%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Budget line (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=[iterations[0], iterations[-1]],
            y=[budget, budget],
            mode="lines",
            name=f"Budget = {budget:.1f}",
            line=dict(color="red", dash="dash", width=2),
            hovertemplate=f"Budget = {budget:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template=_TEMPLATE,
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
        title_text="Budget Constraint Convergence (Lagrangian Bisection)",
        title_x=0.5,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Lambda (Lagrange multiplier)", secondary_y=False)
    fig.update_yaxes(title_text="Total Procurement Cost", secondary_y=True)

    return fig


# ===================================================================
# 6. Substitution flow  (horizontal bar chart)
# ===================================================================
def plot_substitution_flow(
    sub_idx: Any,
    sub_frac: Any,
    substitution_demand: Any,
    N: int,
    category_mask: Any,
    top_k: int = 10,
) -> go.Figure:
    """
    Horizontal bar chart showing the top-K products by total redirected
    demand received through substitution.

    Parameters
    ----------
    sub_idx : Tensor [N, max_subs]
        Substitute indices per product (-1 = no substitute).
    sub_frac : Tensor [N, max_subs]
        Substitution fractions per product.
    substitution_demand : Tensor [N]
        Average redirected demand received per product.
    N : int
        Number of products.
    category_mask : Tensor [N] (bool)
        True = tractor, False = generator.
    top_k : int
        Number of products to display.
    """
    sub_dem = _try_cpu_numpy(substitution_demand)
    cat_mask = _try_cpu_numpy(category_mask)

    if sub_dem is None or sub_dem.size == 0:
        return _empty_figure("Run Substitution solver first.")

    # Identify top-K products by redirected demand
    k = min(top_k, len(sub_dem))
    top_indices = np.argsort(sub_dem)[-k:][::-1]  # descending

    product_labels = []
    demand_vals = []
    bar_colors = []

    for idx in top_indices:
        cat = "Tractor" if (cat_mask is not None and cat_mask[idx]) else "Generator"
        product_labels.append(f"P{idx} ({cat})")
        demand_vals.append(float(sub_dem[idx]))
        bar_colors.append(_PALETTE[1] if cat == "Tractor" else _PALETTE[2])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=product_labels[::-1],
            x=demand_vals[::-1],
            orientation="h",
            marker_color=bar_colors[::-1],
            text=[f"{v:.1f}" for v in demand_vals[::-1]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f} units redirected<extra></extra>",
        )
    )

    fig.update_layout(
        template=_TEMPLATE,
        width=_DEFAULT_WIDTH,
        height=max(400, k * 40),
        title_text=f"Top-{k} Products by Redirected Demand (Substitution)",
        title_x=0.5,
        xaxis_title="Avg. Redirected Demand (units / scenario)",
        yaxis_title="Product",
        margin=dict(l=120),
    )
    return fig


# ===================================================================
# 7. Product detail  (parameters + profit comparison)
# ===================================================================
def plot_product_detail(
    product_idx: int,
    bundle: Any,
    results_dict: Dict[str, Any],
    variant: str = "base",
) -> go.Figure:
    """
    Per-product detail view: a table of financial parameters and a bar
    chart comparing E[profit] across solvers for the chosen product.

    Parameters
    ----------
    product_idx : int
        Index of the product to inspect.
    bundle : TensorBundle
        Data bundle with mu, p, c, s tensors.
    results_dict : dict
        Maps solver label -> SolverResult.
    variant : str
        Problem variant label.
    """
    if bundle is None or not results_dict:
        return _empty_figure("Generate data and run solvers first.")

    # Extract financial parameters for the product
    mu_val = _try_cpu_numpy(bundle.mu)
    p_val = _try_cpu_numpy(bundle.p)
    c_val = _try_cpu_numpy(bundle.c)
    s_val = _try_cpu_numpy(bundle.s)

    if mu_val is None:
        return _empty_figure("Bundle data not available.")

    # Flatten [N,1] -> [N]
    for arr_name in ["mu_val", "p_val", "c_val", "s_val"]:
        arr = locals()[arr_name]
        if arr is not None and arr.ndim == 2:
            locals()[arr_name] = arr.squeeze(1)

    mu_val = mu_val.squeeze() if mu_val.ndim > 1 else mu_val
    p_val = p_val.squeeze() if p_val.ndim > 1 else p_val
    c_val = c_val.squeeze() if c_val.ndim > 1 else c_val
    s_val = s_val.squeeze() if s_val.ndim > 1 else s_val

    N = len(mu_val)
    if product_idx < 0 or product_idx >= N:
        return _empty_figure(f"Product index {product_idx} out of range (N={N}).")

    # Build subplot: left = parameter table, right = profit bars
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.35, 0.65],
        specs=[[{"type": "table"}, {"type": "bar"}]],
        subplot_titles=["Financial Parameters", "E[Profit] by Solver"],
    )

    # -- Parameter table --
    cat_mask = _try_cpu_numpy(bundle.category_mask)
    cat_str = "Tractor" if (cat_mask is not None and cat_mask[product_idx]) else "Generator"
    margin_pct = ((p_val[product_idx] - c_val[product_idx]) / p_val[product_idx] * 100) if p_val[product_idx] > 0 else 0.0

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Parameter", "Value"],
                fill_color=_PALETTE[7] if len(_PALETTE) > 7 else "lightgrey",
                font=dict(size=13),
                align="left",
            ),
            cells=dict(
                values=[
                    ["Category", "mu (demand)", "p (price)", "c (cost)",
                     "s (salvage)", "Margin %"],
                    [cat_str,
                     f"{mu_val[product_idx]:.2f}",
                     f"{p_val[product_idx]:.2f}",
                     f"{c_val[product_idx]:.2f}",
                     f"{s_val[product_idx]:.2f}",
                     f"{margin_pct:.1f}%"],
                ],
                fill_color="white",
                font=dict(size=12),
                align="left",
            ),
        ),
        row=1, col=1,
    )

    # -- Profit bar chart --
    labels = []
    profits = []
    colors = []

    for lbl, res in results_dict.items():
        ep = _try_cpu_numpy(getattr(res, "expected_profit", None))
        if ep is None:
            continue
        ep = ep.squeeze()
        if product_idx < len(ep):
            labels.append(lbl)
            profits.append(float(ep[product_idx]))
            colors.append(_color_for_label(lbl))

    fig.add_trace(
        go.Bar(
            x=labels, y=profits,
            marker_color=colors,
            text=[f"{v:.2f}" for v in profits],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        template=_TEMPLATE,
        width=1000,
        height=_DEFAULT_HEIGHT,
        title_text=f"Product {product_idx} Detail ({variant})",
        title_x=0.5,
    )
    return fig


# ===================================================================
# 8. Demand distribution  (scatter of mu vs sigma)
# ===================================================================
def plot_demand_distribution(bundle: Any) -> go.Figure:
    """
    Scatter plot of (mu, sigma) for all products, coloured by category
    (tractor / generator) for the data exploration tab.

    sigma is approximated from the Cholesky diagonal: sigma_i = L[i, i].

    Parameters
    ----------
    bundle : TensorBundle
        Data bundle with mu, L, category_mask tensors.
    """
    if bundle is None:
        return _empty_figure("Generate data first.")

    mu_arr = _try_cpu_numpy(bundle.mu)
    L_arr = _try_cpu_numpy(bundle.L)
    cat_mask = _try_cpu_numpy(bundle.category_mask)

    if mu_arr is None or L_arr is None:
        return _empty_figure("Bundle tensors not available.")

    # Flatten
    mu_arr = mu_arr.squeeze()
    sigma_arr = np.diag(L_arr)  # diagonal of Cholesky = marginal std dev

    fig = go.Figure()

    if cat_mask is not None:
        # Tractors
        t_mask = cat_mask.astype(bool)
        fig.add_trace(
            go.Scatter(
                x=mu_arr[t_mask], y=sigma_arr[t_mask],
                mode="markers",
                name="Tractor",
                marker=dict(
                    color=_PALETTE[1], size=6, opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate="mu=%{x:.1f}  sigma=%{y:.1f}<extra>Tractor</extra>",
            )
        )
        # Generators
        g_mask = ~t_mask
        fig.add_trace(
            go.Scatter(
                x=mu_arr[g_mask], y=sigma_arr[g_mask],
                mode="markers",
                name="Generator",
                marker=dict(
                    color=_PALETTE[2], size=6, opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate="mu=%{x:.1f}  sigma=%{y:.1f}<extra>Generator</extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=mu_arr, y=sigma_arr,
                mode="markers",
                name="Products",
                marker=dict(
                    color=_PALETTE[0], size=6, opacity=0.7,
                ),
                hovertemplate="mu=%{x:.1f}  sigma=%{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
        title_text="Demand Distribution: Mean vs Std Dev by Category",
        title_x=0.5,
        xaxis_title="Mean Demand (mu)",
        yaxis_title="Std Dev (sigma)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig
