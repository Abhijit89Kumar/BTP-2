"""
setup_tab.py -- Tab 1: Problem Setup for the Gradio newsvendor app.

Allows the user to configure problem dimensions (N, S, seed, tractor
fraction), previews estimated VRAM usage, generates the TensorBundle
and substitution graph, and shows a demand distribution plot.
"""

from __future__ import annotations

import sys
import os
import traceback

import gradio as gr
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that config / data_pipeline import
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def create_setup_tab(state: gr.State):
    """
    Build the Problem Setup tab inside a ``gr.TabItem``.

    Parameters
    ----------
    state : gr.State
        Shared application state (``AppState`` instance).
    """
    with gr.TabItem("1. Problem Setup"):
        gr.Markdown(
            "## Problem Setup\n"
            "Configure the Monte-Carlo simulation dimensions and generate "
            "input tensors.  The memory estimator targets a Google Colab "
            "T4 GPU (15 GB VRAM)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                n_dropdown = gr.Dropdown(
                    choices=["128", "256", "512", "1024", "2048"],
                    value="512",
                    label="N (products)",
                    info="Number of product-location nodes (power of 2)",
                )
                s_dropdown = gr.Dropdown(
                    choices=[
                        "4096", "8192", "16384", "32768",
                        "65536", "131072",
                    ],
                    value="32768",
                    label="S (scenarios)",
                    info="Number of Monte-Carlo demand scenarios",
                )
                seed_input = gr.Number(
                    value=42,
                    label="Random Seed",
                    precision=0,
                )
                tractor_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    step=0.1,
                    value=0.6,
                    label="Tractor Fraction",
                    info="Fraction of products that are tractors (rest = generators)",
                )

            with gr.Column(scale=1):
                vram_display = gr.Textbox(
                    label="Estimated VRAM",
                    interactive=False,
                    value="Estimated VRAM: 0.54 GB / 15 GB (T4)",
                )
                generate_btn = gr.Button(
                    "Generate Data",
                    variant="primary",
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=8,
                    value="No data generated yet.",
                )

        demand_plot = gr.Plot(label="Demand Distribution (mu by category)")

        # -----------------------------------------------------------------
        # Callback: update VRAM estimate whenever N or S changes
        # -----------------------------------------------------------------
        def _update_vram_estimate(n_str, s_str):
            from gradio_app.state import AppState

            try:
                n_val = int(n_str)
                s_val = int(s_str)
            except (ValueError, TypeError):
                return "Invalid N or S value."

            gb = AppState.estimate_vram_gb(n_val, s_val)
            bar_len = int(min(gb / 15.0, 1.0) * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            warning = "  *** WARNING: may exceed T4 VRAM! ***" if gb > 14.0 else ""
            return f"Estimated VRAM: {gb:.2f} GB / 15 GB (T4)  [{bar}]{warning}"

        n_dropdown.change(
            fn=_update_vram_estimate,
            inputs=[n_dropdown, s_dropdown],
            outputs=[vram_display],
        )
        s_dropdown.change(
            fn=_update_vram_estimate,
            inputs=[n_dropdown, s_dropdown],
            outputs=[vram_display],
        )

        # -----------------------------------------------------------------
        # Callback: generate data
        # -----------------------------------------------------------------
        def _generate_data(st, n_str, s_str, seed, tractor_frac):
            from config import NewsvendorConfig, SubstitutionConfig
            from data_pipeline import DataPipeline, SubstitutionGraphGenerator
            from gradio_app.state import AppState

            if st is None:
                st = AppState()

            try:
                N = int(n_str)
                S = int(s_str)
                seed = int(seed)
            except (ValueError, TypeError):
                return st, "ERROR: N, S, and seed must be integers.", None

            # Determine device
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            status_lines = [f"Generating data: N={N}, S={S}, seed={seed}, device={device}"]
            status_lines.append(f"Tractor fraction: {tractor_frac:.1f}")

            try:
                cfg = NewsvendorConfig(
                    N=N, S=S, seed=seed,
                    device=device,
                    tractor_fraction=tractor_frac,
                )
                pipeline = DataPipeline(cfg=cfg)
                bundle = pipeline.run()

                status_lines.append("")
                status_lines.append("Tensor shapes:")
                status_lines.append(f"  L  (Cholesky):    {list(bundle.L.shape)}")
                status_lines.append(f"  Z  (scenarios):   {list(bundle.Z.shape)}")
                status_lines.append(f"  mu (mean demand): {list(bundle.mu.shape)}")
                status_lines.append(f"  p  (price):       {list(bundle.p.shape)}")
                status_lines.append(f"  c  (cost):        {list(bundle.c.shape)}")
                status_lines.append(f"  s  (salvage):     {list(bundle.s.shape)}")
                status_lines.append(f"  Q  (order qty):   {list(bundle.Q.shape)}")

                # Category summary
                n_tractors = bundle.category_mask.sum().item()
                n_gens = N - n_tractors
                status_lines.append(f"\nProducts: {n_tractors} tractors, {n_gens} generators")

                # Generate substitution graph
                sub_cfg = SubstitutionConfig()
                sub_gen = SubstitutionGraphGenerator(sub_cfg)
                cat_mask_np = bundle.category_mask.cpu().numpy()
                sub_idx_np, sub_frac_np = sub_gen.generate(N, cat_mask_np, seed)

                sub_idx_t = torch.tensor(sub_idx_np, dtype=torch.int64, device=device)
                sub_frac_t = torch.tensor(sub_frac_np, dtype=torch.float32, device=device)

                status_lines.append(
                    f"Substitution graph: max_subs={sub_cfg.max_subs}, "
                    f"beta=[{sub_cfg.beta_min}, {sub_cfg.beta_max}]"
                )

                # Update state
                st.bundle = bundle
                st.N = N
                st.S = S
                st.seed = seed
                st.tractor_fraction = tractor_frac
                st.data_generated = True
                st.sub_idx = sub_idx_t
                st.sub_frac = sub_frac_t
                st.clear_results()

                # Build demand distribution plot
                fig = _build_demand_plot(bundle)

                if device == "cuda":
                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                    status_lines.append(f"\nGPU memory allocated: {mem_alloc:.2f} GB")

                status_lines.append("\nData generation complete.")
                return st, "\n".join(status_lines), fig

            except Exception as e:
                status_lines.append(f"\nERROR: {e}")
                status_lines.append(traceback.format_exc())
                return st, "\n".join(status_lines), None

        generate_btn.click(
            fn=_generate_data,
            inputs=[state, n_dropdown, s_dropdown, seed_input, tractor_slider],
            outputs=[state, status_box, demand_plot],
        )


def _build_demand_plot(bundle) -> go.Figure:
    """
    Create a Plotly figure showing the demand distribution (mu) by category.
    """
    mu_np = bundle.mu.squeeze().cpu().numpy()
    mask_np = bundle.category_mask.cpu().numpy()

    mu_tractors = mu_np[mask_np]
    mu_generators = mu_np[~mask_np]

    fig = go.Figure()

    if len(mu_tractors) > 0:
        fig.add_trace(go.Histogram(
            x=mu_tractors,
            name="Tractors",
            opacity=0.7,
            marker_color="#636EFA",
            nbinsx=30,
        ))

    if len(mu_generators) > 0:
        fig.add_trace(go.Histogram(
            x=mu_generators,
            name="Generators",
            opacity=0.7,
            marker_color="#EF553B",
            nbinsx=30,
        ))

    fig.update_layout(
        title="Mean Demand (mu) Distribution by Category",
        xaxis_title="Mean Demand (units/period)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
        height=400,
    )

    return fig
