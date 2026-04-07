"""
extensions — Newsvendor problem extensions with GPU-accelerated Triton kernels.

Provides four extensions beyond the base newsvendor:
  1. grid_search  — Optimal Q* via Monte-Carlo grid search
  2. cvar         — CVaR (Conditional Value at Risk) risk-averse newsvendor
  3. budget       — Budget-constrained newsvendor via Lagrangian dual
  4. substitution — Multi-product substitution newsvendor
"""
