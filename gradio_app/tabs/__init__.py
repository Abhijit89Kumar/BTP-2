"""Tab modules for the Gradio newsvendor app."""

from gradio_app.tabs.setup_tab import create_setup_tab
from gradio_app.tabs.solver_tab import create_solver_tab
from gradio_app.tabs.results_tab import create_results_tab
from gradio_app.tabs.product_tab import create_product_tab
from gradio_app.tabs.about_tab import create_about_tab

__all__ = [
    "create_setup_tab",
    "create_solver_tab",
    "create_results_tab",
    "create_product_tab",
    "create_about_tab",
]
