"""
utils/analytical.py
====================
Analytical function differentiation with detailed textbook-style step-by-step output.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import sympy as sp
import numpy as np


@dataclass
class AnalyticalPointResult:
    x_value: float
    fx_value: float
    f2_value: float
    point_type: str


@dataclass
class AnalyticalResult:
    steps: List[dict]
    results: List[AnalyticalPointResult]
    expr: sp.Expr
    first_deriv: sp.Expr
    second_deriv: sp.Expr
    x_symbol: sp.Symbol
    error: Optional[str] = None


def parse_function_string(func_str: str) -> Tuple[sp.Expr, sp.Symbol]:
    clean_str = func_str.replace("^", "**")
    x = sp.Symbol("x", real=True)
    expr = sp.sympify(clean_str, locals={"x": x})
    return expr, x


def analyze_function(func_str: str, decimal_places: int = 4) -> AnalyticalResult:
    steps = []
    try:
        expr, x = parse_function_string(func_str)
    except Exception as e:
        return AnalyticalResult([], [], None, None, None, None, error=f"Invalid expression: {e}")

    # Step 1: Solution Given
    steps.append({
        "title": "Solution:",
        "content": f"Here, $f(x) = {sp.latex(expr)}$"
    })

    # Step 2: First Derivative Detailed Breakdown
    f1 = sp.diff(expr, x)
    terms = expr.as_ordered_terms() if isinstance(expr, sp.Add) else [expr]
    diff_terms_str = " + ".join([f"\\frac{{d}}{{dx}}\\left({sp.latex(t)}\\right)" for t in terms]).replace("+ -", "- ")
    eval_terms_str = " + ".join([f"{sp.latex(sp.diff(t, x))}" for t in terms]).replace("+ -", "- ")

    step1_content = (
        f"$$\\therefore f'(x) = \\frac{{d}}{{dx}}\\left({sp.latex(expr)}\\right)$$\n\n"
        f"$$= {diff_terms_str}$$\n\n"
        f"$$= {eval_terms_str}$$\n\n"
        f"$$= {sp.latex(f1)}$$"
    )
    steps.append({
        "title": "Step-1: Find the derivative of the function",
        "content": step1_content
    })

    # Step 3: Critical Points Breakdown
    steps.append({
        "title": "Step-2: Find the critical points of the derivative function",
        "content": "To find critical points, set $f'(x) = 0$ and then solve for $x$."
    })

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

    # Factorization / Solving Steps Display
    solve_steps_text = f"$$f'(x) = 0$$\n\n"
    solve_steps_text += f"$$\\implies {sp.latex(f1)} = 0$$\n\n"
    
    # Try factoring for textbook representation
    factored_f1 = sp.factor(f1)
    if factored_f1 != f1:
        solve_steps_text += f"$$\\implies {sp.latex(factored_f1)} = 0$$\n\n"

    roots_str_list = [f"x = {sp.latex(sp.sympify(r))}" for r in real_roots]
    solve_steps_text += f"$$\\implies " + " \\text{ or } ".join(roots_str_list) + "$$\n\n"
    solve_steps_text += f"$$\\therefore \\text{{The critical points are }} " + " \\text{ and } ".join(roots_str_list) + "$$"

    steps.append({
        "title": "Solving for x:",
        "content": solve_steps_text
    })

    # Step 4: Second Derivative Test
    f2 = sp.diff(f1, x)
    f2_terms = f1.as_ordered_terms() if isinstance(f1, sp.Add) else [f1]
    f2_diff_terms = " + ".join([f"\\frac{{d}}{{dx}}\\left({sp.latex(t)}\\right)" for t in f2_terms]).replace("+ -", "- ")
    
    step3_content = (
        f"Now, $f''(x) = \\frac{{d}}{{dx}}\\left({sp.latex(f1)}\\right)$\n\n"
        f"$$= {f2_diff_terms}$$\n\n"
        f"$$= {sp.latex(f2)}$$"
    )
    steps.append({
        "title": "Step-3: Apply the second derivative test",
        "content": step3_content
    })

    # Evaluate f''(x) at critical points
    eval_step_text = ""
    results = []
    for r_val in real_roots:
        f2_val = float(f2.subs(x, r_val).evalf())
        fx_val = float(expr.subs(x, r_val).evalf())

        if f2_val < -1e-9:
            pt_type = "Maximum"
            cond_str = f"f''({r_val}) = {sp.latex(sp.sympify(f2_val))} < 0"
            conclusion = f"\\therefore \\text{{At }} x = {r_val} \\text{{ the function is local maximum}}"
        elif f2_val > 1e-9:
            pt_type = "Minimum"
            cond_str = f"f''({r_val}) = {sp.latex(sp.sympify(f2_val))} > 0"
            conclusion = f"\\therefore \\text{{At }} x = {r_val} \\text{{ the function is local minimum}}"
        else:
            pt_type = "Point of Inflexion"
            cond_str = f"f''({r_val}) = 0"
            conclusion = f"\\therefore \\text{{At }} x = {r_val} \\text{{ it is a point of inflexion}}"

        results.append(AnalyticalPointResult(
            x_value=round(r_val, decimal_places),
            fx_value=round(fx_val, decimal_places),
            f2_value=round(f2_val, decimal_places),
            point_type=pt_type
        ))

        eval_step_text += f"**For $x = {r_val}$**\n\n"
        eval_step_text += f"$$f''({r_val}) = {sp.latex(f2.subs(x, r_val))} = {f2_val:.{decimal_places}f}$$\n\n"
        eval_step_text += f"$${cond_str}$$\n\n"
        eval_step_text += f"$${conclusion}$$\n\n---\n\n"

    steps.append({
        "title": "Evaluate f''(x) at the critical points",
        "content": eval_step_text
    })

    # Step 5: Calculate Extrema Values
    step4_text = f"Substitute the $x$ values back into the original function $f(x) = {sp.latex(expr)}$\n\n"
    for r_val in real_roots:
        fx_val = float(expr.subs(x, r_val).evalf())
        pt_kind = "local minimum" if [r for r in results if r.x_value == r_val][0].point_type == "Minimum" else "local maximum"
        step4_text += f"**At $x = {r_val}$**\n\n"
        step4_text += f"$$f({r_val}) = {sp.latex(expr.subs(x, r_val))} = {fx_val:.{decimal_places}f}$$\n\n"
        step4_text += f"$$\\text{{{pt_kind} point}} = ({r_val}, {fx_val:.{decimal_places}f})$$\n\n"

    steps.append({
        "title": "Step-4: Calculate the extrema values",
        "content": step4_text
    })

    return AnalyticalResult(steps, results, expr, f1, f2, x)


def get_numpy_func(expr: sp.Expr, x_sym: sp.Symbol):
    return sp.lambdify(x_sym, expr, modules=["numpy"])