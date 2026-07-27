"""
utils/tabular.py
================
Tabular data analysis using Newton's Forward Difference Formula with full step-by-step expansion.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import sympy as sp


@dataclass
class TabularPointResult:
    x_value: float
    y_value: float
    f2_value: float
    point_type: str


@dataclass
class TabularResult:
    steps: List[dict]
    diff_table: np.ndarray
    polynomial: sp.Expr
    critical_points: List[TabularPointResult]
    x_values: List[float]
    y_values: List[float]
    error: Optional[str] = None


def build_forward_diff_table(y_vals: List[float]) -> np.ndarray:
    n = len(y_vals)
    table = np.zeros((n, n))
    table[:, 0] = y_vals
    for col in range(1, n):
        for row in range(n - col):
            table[row, col] = table[row + 1, col - 1] - table[row, col - 1]
    return table


def analyze_table(x_vals: List[float], y_vals: List[float], decimal_places: int = 4) -> TabularResult:
    steps = []
    n = len(x_vals)

    if n < 3:
        return TabularResult([], np.array([]), None, [], x_vals, y_vals, error="At least 3 points required.")

    h_list = [round(x_vals[i+1] - x_vals[i], 8) for i in range(n - 1)]
    if len(set(h_list)) > 1:
        return TabularResult([], np.array([]), None, [], x_vals, y_vals, error="x values must be equally spaced.")

    h = h_list[0]
    x0 = x_vals[0]
    y0 = y_vals[0]

    diff_table = build_forward_diff_table(y_vals)

    # Difference Table Display
    table_md = "| $X$ | $Y=F(X)$ | " + " | ".join([f"$\\Delta^{{{i}}}$" if i > 1 else "$\\Delta$" for i in range(1, n)]) + " |\n"
    table_md += "|---" * (n + 1) + "|\n"
    for i in range(n):
        row_vals = [f"**{x_vals[i]}**", f"**{y_vals[i]}**"]
        for j in range(1, n - i):
            row_vals.append(f"{diff_table[i, j]:.4f}".rstrip('0').rstrip('.'))
        for j in range(n - i, n):
            row_vals.append("")
        table_md += "| " + " | ".join(row_vals) + " |\n"

    steps.append({
        "title": "Solution: The difference table for the given data:",
        "content": table_md
    })

    # Formula Setup
    x = sp.Symbol("x", real=True)
    steps_text = f"Here $x_0 = {x0}$, $y_0 = {y0}$ and $h = {h}$.\n\n"
    steps_text += f"Now, $p = \\frac{{x - {x0}}}{{{h}}}$\n\n"
    steps_text += "We have Newton's forward difference formula as:\n"
    steps_text += "$$y(x) = y_0 + p\\Delta y_0 + \\frac{p(p-1)}{2!}\\Delta^2 y_0 + \\frac{p(p-1)(p-2)}{3!}\\Delta^3 y_0 + \\dots$$\n\n"

    # Algebraic expansion
    u = (x - x0) / h
    poly_u = diff_table[0, 0]
    u_term = 1
    fact = 1
    for k in range(1, n):
        u_term *= (u - (k - 1))
        fact *= k
        poly_u += (diff_table[0, k] / fact) * u_term

    poly_x = sp.expand(poly_u)

    steps_text += f"Putting values from above mentioned table and simplifying, we get:\n"
    steps_text += f"$$y(x) = {sp.latex(poly_x)}$$"

    steps.append({
        "title": "Interpolating Polynomial Derivation",
        "content": steps_text
    })

    # Differentiation & Critical Points
    f1 = sp.diff(poly_x, x)
    f2 = sp.diff(f1, x)

    deriv_text = f"Now differentiating two times w.r.t. $x$, we get:\n"
    deriv_text += f"$$\\frac{{dy}}{{dx}} = {sp.latex(f1)} \\quad \\implies \\quad \\frac{{d^2y}}{{dx^2}} = {sp.latex(f2)}$$\n\n"
    deriv_text += "For maxima and minima, $\\frac{dy}{dx} = 0$\n\n"
    
    raw_roots = sp.solve(f1, x)
    real_roots = []
    for r in raw_roots:
        try:
            val = complex(r.evalf())
            if abs(val.imag) < 1e-9:
                real_roots.append(float(val.real))
        except Exception:
            pass

    real_roots = sorted(list(set([round(r, decimal_places) for r in real_roots])))

    deriv_text += f"$$\\implies {sp.latex(f1)} = 0$$\n\n"
    roots_str = ", ".join([f"{sp.latex(sp.sympify(r))}" for r in real_roots])
    deriv_text += f"$$\\implies x = {roots_str}$$\n\n"

    critical_points = []
    for r_val in real_roots:
        fx_val = float(poly_x.subs(x, r_val).evalf())
        f2_val = float(f2.subs(x, r_val).evalf())

        if f2_val < -1e-9:
            pt_type = "Maximum"
            deriv_text += f"When $x = {r_val}$ then $\\frac{{d^2y}}{{dx^2}} = {f2_val:.{decimal_places}f} < 0$.\n\n"
            deriv_text += f"Therefore, $y$ value is maximum at $x = {r_val}$\n\n"
            deriv_text += f"So, the maximum value, $y_{{max}} = {fx_val:.{decimal_places}f}$\n\n"
        elif f2_val > 1e-9:
            pt_type = "Minimum"
            deriv_text += f"When $x = {r_val}$ then $\\frac{{d^2y}}{{dx^2}} = {f2_val:.{decimal_places}f} > 0$.\n\n"
            deriv_text += f"Therefore, $y$ value is minimum at $x = {r_val}$\n\n"
            deriv_text += f"So, the minimum value, $y_{{min}} = {fx_val:.{decimal_places}f}$\n\n"
        else:
            pt_type = "Point of Inflexion"

        critical_points.append(TabularPointResult(
            x_value=round(r_val, decimal_places),
            y_value=round(fx_val, decimal_places),
            f2_value=round(f2_val, decimal_places),
            point_type=pt_type
        ))

    steps.append({
        "title": "Finding Maxima and Minima",
        "content": deriv_text
    })

    return TabularResult(steps, diff_table, poly_x, critical_points, x_vals, y_vals)