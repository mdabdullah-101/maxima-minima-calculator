"""
utils/plotting.py
=================
Plotting module using Matplotlib for rendering functions and critical points.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_function_with_extrema(
    func_np,
    x_range: tuple,
    extrema_points: list,
    title: str = "Function Graph",
    extra_scatter: list = None,
    extra_scatter_label: str = "Data Points",
):
    """Plots a 2D function line along with maxima/minima points."""
    x_min, x_max = x_range
    padding = (x_max - x_min) * 0.15 if x_max != x_min else 2.0
    x_vals = np.linspace(x_min - padding, x_max + padding, 500)

    try:
        y_vals = func_np(x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals)
    except Exception:
        y_vals = np.zeros_like(x_vals)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(x_vals, y_vals, label="f(x)", color="#2563eb", linewidth=2)

    if extra_scatter:
        ex_x, ex_y = zip(*extra_scatter)
        ax.scatter(ex_x, ex_y, color="#64748b", zorder=4, label=extra_scatter_label, s=40)

    max_x, max_y, min_x, min_y = [], [], [], []
    for pt in extrema_points:
        x_p = pt["x_value"]
        y_p = pt.get("fx_value", pt.get("y_value", 0.0))
        pt_type = pt["point_type"]

        if pt_type == "Maximum":
            max_x.append(x_p)
            max_y.append(y_p)
        elif pt_type == "Minimum":
            min_x.append(x_p)
            min_y.append(y_p)

    if max_x:
        ax.scatter(max_x, max_y, color="#dc2626", s=80, zorder=5, label="Maximum", edgecolors="black")
    if min_x:
        ax.scatter(min_x, min_y, color="#16a34a", s=80, zorder=5, label="Minimum", edgecolors="black")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig