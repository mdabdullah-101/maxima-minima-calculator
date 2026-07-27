"""
utils/analytical.py
====================
Analytical function differentiation with detailed textbook-style step-by-step output
including step-by-step value substitution and Higher-Order Derivative Test.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import sympy as sp
import numpy as np
import re
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)


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
    if not func_str or not func_str.strip():
        raise ValueError("Input string is empty.")

    cleaned = func_str.lower().strip()
    cleaned = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', cleaned)

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    expr = parse_expr(cleaned, transformations=transformations)
    
    free_symbols = list(expr.free_symbols)
    if free_symbols:
        var = free_symbols[0]
    else:
        var = sp.Symbol("x", real=True)
        
    return expr, var


def get_step_by_step_eval(expr: sp.Expr, x_sym: sp.Symbol, val: float) -> str:
    """Helper function to format step-by-step substitution into latex"""
    val_sym = sp.sympify(val)
    
    # Substitute variable with value in symbolic form
    sub_expr = expr.subs(x_sym, sp.Symbol(f"({val_sym})"))
    sub_latex = sp.latex(sub_expr)
    
    # Evaluate final numerical result
    final_val = float(expr.subs(x_sym, val).evalf())
    
    # If the expression is simple or just a constant, show directly
    if expr.is_number or sub_latex == sp.latex(expr):
        return f"{final_val:.4f}"
    
    return f"{sub_latex} = {final_val:.4f}"


def analyze_function(func_str: str, decimal_places: int = 4) -> AnalyticalResult:
    steps = []
    try:
        expr, x = parse_function_string(func_str)
    except Exception as e:
        return AnalyticalResult([], [], None, None, None, None, error=f"Invalid expression: {e}")

    sym_name = x.name

    # Step 1: Solution Given
    steps.append({
        "title": "Solution:",
        "content": f"Here, $f({sym_name}) = {sp.latex(expr)}$"
    })

    # Step 2: First Derivative Detailed Breakdown
    f1 = sp.diff(expr, x)
    terms = expr.as_ordered_terms() if isinstance(expr, sp.Add) else [expr]
    diff_terms_str = " + ".join([f"\\frac{{d}}{{d{sym_name}}}\\left({sp.latex(t)}\\right)" for t in terms]).replace("+ -", "- ")
    eval_terms_str = " + ".join([f"{sp.latex(sp.diff(t, x))}" for t in terms]).replace("+ -", "- ")

    step1_content = (
        f"$$\\therefore f'({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(expr)}\\right)$$\n\n"
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
        "content": f"To find critical points, set $f'({sym_name}) = 0$ and then solve for ${sym_name}$."
    })

    raw_roots = sp.solve(f1, x)
    real_roots = []
    for r in raw_roots:
        try:
            v = complex(r.evalf())
            if abs(v.imag) < 1e-9:
                real_roots.append(float(v.real))
        except Exception:
            pass

    real_roots = sorted(list(set([round(r, decimal_places) for r in real_roots])))

    solve_steps_text = f"$$f'({sym_name}) = 0$$\n\n"
    solve_steps_text += f"$$\\implies {sp.latex(f1)} = 0$$\n\n"
    
    factored_f1 = sp.factor(f1)
    if factored_f1 != f1:
        solve_steps_text += f"$$\\implies {sp.latex(factored_f1)} = 0$$\n\n"

    roots_str_list = [f"{sym_name} = {sp.latex(sp.sympify(r))}" for r in real_roots]
    solve_steps_text += f"$$\\implies " + " \\text{ or } ".join(roots_str_list) + "$$\n\n"
    solve_steps_text += f"$$\\therefore \\text{{The critical points are }} " + " \\text{ and } ".join(roots_str_list) + "$$"

    steps.append({
        "title": f"Solving for {sym_name}:",
        "content": solve_steps_text
    })

    # Step 4: Second Derivative Test & Higher Order Analysis
    f2 = sp.diff(f1, x)
    f2_terms = f1.as_ordered_terms() if isinstance(f1, sp.Add) else [f1]
    f2_diff_terms = " + ".join([f"\\frac{{d}}{{d{sym_name}}}\\left({sp.latex(t)}\\right)" for t in f2_terms]).replace("+ -", "- ")
    
    step3_content = (
        f"Now, $f''({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(f1)}\\right)$\n\n"
        f"$$= {f2_diff_terms}$$\n\n"
        f"$$= {sp.latex(f2)}$$"
    )
    steps.append({
        "title": "Step-3: Apply the second derivative test",
        "content": step3_content
    })

    eval_step_text = ""
    results = []
    
    for r_val in real_roots:
        f2_val = float(f2.subs(x, r_val).evalf())
        fx_val = float(expr.subs(x, r_val).evalf())

        # Format second derivative substitution step
        f2_sub_step = get_step_by_step_eval(f2, x, r_val)

        if f2_val < -1e-9:
            pt_type = "Maximum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} < 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
            higher_order_text = ""
        elif f2_val > 1e-9:
            pt_type = "Minimum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} > 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"
            higher_order_text = ""
        else:
            # Higher-Order Derivative with Step-by-Step Substitution
            order = 3
            prev_deriv = f2
            current_deriv = sp.diff(prev_deriv, x)
            val = float(current_deriv.subs(x, r_val).evalf())
            
            higher_order_text = f"Since $f''({r_val}) = 0$, the second derivative test is inconclusive. We test higher-order derivatives:\n\n"
            
            # Step 1: Derivative Equation
            higher_order_text += f"$$f^{{({order})}}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
            # Step 2: Explicit Substitution Step
            h_sub_step = get_step_by_step_eval(current_deriv, x, r_val)
            higher_order_text += f"$$f^{{({order})}}({r_val}) = {h_sub_step}$$\n\n"

            while abs(val) < 1e-9 and order <= 10:
                order += 1
                prev_deriv = current_deriv
                current_deriv = sp.diff(prev_deriv, x)
                val = float(current_deriv.subs(x, r_val).evalf())
                higher_order_text += f"$$f^{{({order})}}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
                h_sub_step = get_step_by_step_eval(current_deriv, x, r_val)
                higher_order_text += f"$$f^{{({order})}}({r_val}) = {h_sub_step}$$\n\n"

            if order % 2 != 0:
                pt_type = "Point of Inflection"
                cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} \\neq 0 \\quad (\\text{{First non-zero derivative is Odd order}})"
                conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ it is a Point of Inflection (neither max nor min)}}"
            else:
                if val < 0:
                    pt_type = "Maximum"
                    cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} < 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
                else:
                    pt_type = "Minimum"
                    cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} > 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"

        results.append(AnalyticalPointResult(
            x_value=round(r_val, decimal_places),
            fx_value=round(fx_val, decimal_places),
            f2_value=round(f2_val, decimal_places),
            point_type=pt_type
        ))

        eval_step_text += f"**For ${sym_name} = {r_val}$**\n\n"
        eval_step_text += f"$$f''({r_val}) = {f2_sub_step}$$\n\n"
        if higher_order_text:
            eval_step_text += higher_order_text
        eval_step_text += f"$${cond_str}$$\n\n"
        eval_step_text += f"$${conclusion}$$\n\n---\n\n"

    steps.append({
        "title": f"Evaluate derivatives at the critical points",
        "content": eval_step_text
    })

    # Step 5: Calculate Extrema Values with Step-by-Step Substitution
    step4_text = f"Substitute the ${sym_name}$ values back into the original function $f({sym_name}) = {sp.latex(expr)}$\n\n"
    for r_val in real_roots:
        fx_sub_step = get_step_by_step_eval(expr, x, r_val)
        fx_val = float(expr.subs(x, r_val).evalf())
        p_type = [r for r in results if r.x_value == r_val][0].point_type
        
        if p_type == "Minimum":
            pt_kind = "local minimum point"
        elif p_type == "Maximum":
            pt_kind = "local maximum point"
        else:
            pt_kind = "point of inflection"

        step4_text += f"**At ${sym_name} = {r_val}$**\n\n"
        step4_text += f"$$f({r_val}) = {fx_sub_step}$$\n\n"
        step4_text += f"$$\\text{{{pt_kind}}} = ({r_val}, {fx_val:.{decimal_places}f})$$\n\n"

    steps.append({
        "title": "Step-4: Calculate the extrema values",
        "content": step4_text
    })

    return AnalyticalResult(steps, results, expr, f1, f2, x)


def get_numpy_func(expr: sp.Expr, x_sym: sp.Symbol):
    return sp.lambdify(x_sym, expr, modules=["numpy"])
    })

    raw_roots = sp.solve(f1, x)
    real_roots = []
    for r in raw_roots:
        try:
            v = complex(r.evalf())
            if abs(v.imag) < 1e-9:
                real_roots.append(float(v.real))
        except Exception:
            pass

    real_roots = sorted(list(set([round(r, decimal_places) for r in real_roots])))

    solve_steps_text = f"$$f'({sym_name}) = 0$$\n\n"
    solve_steps_text += f"$$\\implies {sp.latex(f1)} = 0$$\n\n"
    
    factored_f1 = sp.factor(f1)
    if factored_f1 != f1:
        solve_steps_text += f"$$\\implies {sp.latex(factored_f1)} = 0$$\n\n"

    roots_str_list = [f"{sym_name} = {sp.latex(sp.sympify(r))}" for r in real_roots]
    solve_steps_text += f"$$\\implies " + " \\text{ or } ".join(roots_str_list) + "$$\n\n"
    solve_steps_text += f"$$\\therefore \\text{{The critical points are }} " + " \\text{ and } ".join(roots_str_list) + "$$"

    steps.append({
        "title": f"Solving for {sym_name}:",
        "content": solve_steps_text
    })

    # Step 4: Second Derivative Test & Higher Order Analysis
    f2 = sp.diff(f1, x)
    f2_terms = f1.as_ordered_terms() if isinstance(f1, sp.Add) else [f1]
    f2_diff_terms = " + ".join([f"\\frac{{d}}{{d{sym_name}}}\\left({sp.latex(t)}\\right)" for t in f2_terms]).replace("+ -", "- ")
    
    step3_content = (
        f"Now, $f''({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(f1)}\\right)$\n\n"
        f"$$= {f2_diff_terms}$$\n\n"
        f"$$= {sp.latex(f2)}$$"
    )
    steps.append({
        "title": "Step-3: Apply the second derivative test",
        "content": step3_content
    })

    eval_step_text = ""
    results = []
    
    for r_val in real_roots:
        f2_val = float(f2.subs(x, r_val).evalf())
        fx_val = float(expr.subs(x, r_val).evalf())

        # Format second derivative substitution step
        f2_sub_step = get_step_by_step_eval(f2, x, r_val)

        if f2_val < -1e-9:
            pt_type = "Maximum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} < 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
            higher_order_text = ""
        elif f2_val > 1e-9:
            pt_type = "Minimum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} > 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"
            higher_order_text = ""
        else:
            # Higher-Order Derivative with Step-by-Step Substitution
            order = 3
            prev_deriv = f2
            current_deriv = sp.diff(prev_deriv, x)
            val = float(current_deriv.subs(x, r_val).evalf())
            
            higher_order_text = f"Since $f''({r_val}) = 0$, the second derivative test is inconclusive. We test higher-order derivatives:\n\n"
            
            # Step 1: Derivative Equation
            higher_order_text += f"$$f^{{({order})}}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
            # Step 2: Explicit Substitution Step
            h_sub_step = get_step_by_step_eval(current_deriv, x, r_val)
            higher_order_text += f"$$f^{{({order})}}({r_val}) = {h_sub_step}$$\n\n"

            while abs(val) < 1e-9 and order <= 10:
                order += 1
                prev_deriv = current_deriv
                current_deriv = sp.diff(prev_deriv, x)
                val = float(current_deriv.subs(x, r_val).evalf())
                higher_order_text += f"$$f^{{({order})}}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
                h_sub_step = get_step_by_step_eval(current_deriv, x, r_val)
                higher_order_text += f"$$f^{{({order})}}({r_val}) = {h_sub_step}$$\n\n"

            if order % 2 != 0:
                pt_type = "Point of Inflection"
                cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} \\neq 0 \\quad (\\text{{First non-zero derivative is Odd order}})"
                conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ it is a Point of Inflection (neither max nor min)}}"
            else:
                if val < 0:
                    pt_type = "Maximum"
                    cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} < 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
                else:
                    pt_type = "Minimum"
                    cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} > 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"

        results.append(AnalyticalPointResult(
            x_value=round(r_val, decimal_places),
            fx_value=round(fx_val, decimal_places),
            f2_value=round(f2_val, decimal_places),
            point_type=pt_type
        ))

        eval_step_text += f"**For ${sym_name} = {r_val}$**\n\n"
        eval_step_text += f"$$f''({r_val}) = {f2_sub_step}$$\n\n"
        if higher_order_text:
            eval_step_text += higher_order_text
        eval_step_text += f"$${cond_str}$$\n\n"
        eval_step_text += f"$${conclusion}$$\n\n---\n\n"

    steps.append({
        "title": f"Evaluate derivatives at the critical points",
        "content": eval_step_text
    })

    # Step 5: Calculate Extrema Values with Step-by-Step Substitution
    step4_text = f"Substitute the ${sym_name}$ values back into the original function $f({sym_name}) = {sp.latex(expr)}$\n\n"
    for r_val in real_roots:
        fx_sub_step = get_step_by_step_eval(expr, x, r_val)
        fx_val = float(expr.subs(x, r_val).evalf())
        p_type = [r for r in results if r.x_value == r_val][0].point_type
        
        if p_type == "Minimum":
            pt_kind = "local minimum point"
        elif p_type == "Maximum":
            pt_kind = "local maximum point"
        else:
            pt_kind = "point of inflection"

        step4_text += f"**At ${sym_name} = {r_val}$**\n\n"
        step4_text += f"$$f({r_val}) = {fx_sub_step}$$\n\n"
        step4_text += f"$$\\text{{{pt_kind}}} = ({r_val}, {fx_val:.{decimal_places}f})$$\n\n"

    steps.append({
        "title": "Step-4: Calculate the extrema values",
        "content": step4_text
    })

    return AnalyticalResult(steps, results, expr, f1, f2, x)


def get_numpy_func(expr: sp.Expr, x_sym: sp.Symbol):
    return sp.lambdify(x_sym, expr, modules=["numpy"])
    })

    raw_roots = sp.solve(f1, x)
    real_roots = []
    for r in raw_roots:
        try:
            v = complex(r.evalf())
            if abs(v.imag) < 1e-9:
                real_roots.append(float(v.real))
        except Exception:
            pass

    real_roots = sorted(list(set([round(r, decimal_places) for r in real_roots])))

    solve_steps_text = f"$$f'({sym_name}) = 0$$\n\n"
    solve_steps_text += f"$$\\implies {sp.latex(f1)} = 0$$\n\n"
    
    factored_f1 = sp.factor(f1)
    if factored_f1 != f1:
        solve_steps_text += f"$$\\implies {sp.latex(factored_f1)} = 0$$\n\n"

    roots_str_list = [f"{sym_name} = {sp.latex(sp.sympify(r))}" for r in real_roots]
    solve_steps_text += f"$$\\implies " + " \\text{ or } ".join(roots_str_list) + "$$\n\n"
    solve_steps_text += f"$$\\therefore \\text{{The critical points are }} " + " \\text{ and } ".join(roots_str_list) + "$$"

    steps.append({
        "title": f"Solving for {sym_name}:",
        "content": solve_steps_text
    })

    # Step 4: Second Derivative Test & Higher Order Analysis
    f2 = sp.diff(f1, x)
    f2_terms = f1.as_ordered_terms() if isinstance(f1, sp.Add) else [f1]
    f2_diff_terms = " + ".join([f"\\frac{{d}}{{d{sym_name}}}\\left({sp.latex(t)}\\right)" for t in f2_terms]).replace("+ -", "- ")
    
    step3_content = (
        f"Now, $f''({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(f1)}\\right)$\n\n"
        f"$$= {f2_diff_terms}$$\n\n"
        f"$$= {sp.latex(f2)}$$"
    )
    steps.append({
        "title": "Step-3: Apply the second derivative test",
        "content": step3_content
    })

    eval_step_text = ""
    results = []
    
    for r_val in real_roots:
        f2_val = float(f2.subs(x, r_val).evalf())
        fx_val = float(expr.subs(x, r_val).evalf())

        # Format second derivative substitution step
        f2_sub_step = get_step_by_step_eval(f2, x, r_val)

        if f2_val < -1e-9:
            pt_type = "Maximum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} < 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
            higher_order_text = ""
        elif f2_val > 1e-9:
            pt_type = "Minimum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} > 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"
            higher_order_text = ""
        else:
            # Higher-Order Derivative with Step-by-Step Substitution
            order = 3
            prev_deriv = f2
            current_deriv = sp.diff(prev_deriv, x)
            val = float(current_deriv.subs(x, r_val).evalf())
            
            higher_order_text = f"Since $f''({r_val}) = 0$, the second derivative test is inconclusive. We test higher-order derivatives:\n\n"
            
            # Step 1: Derivative Equation
            higher_order_text += f"$$f^{{({order})}}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
            # Step 2: Explicit Substitution Step
            h_sub_step = get_step_by_step_eval(current_deriv, x, r_val)
            higher_order_text += f"$$f^{{({order})}}({r_val}) = {h_sub_step}$$\n\n"

            while abs(val) < 1e-9 and order <= 10:
                order += 1
                prev_deriv = current_deriv
                current_deriv = sp.diff(prev_deriv, x)
                val = float(current_deriv.subs(x, r_val).evalf())
                higher_order_text += f"$$f^{{({order})}}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
                h_sub_step = get_step_by_step_eval(current_deriv, x, r_val)
                higher_order_text += f"$$f^{{({order})}}({r_val}) = {h_sub_step}$$\n\n"

            if order % 2 != 0:
                pt_type = "Point of Inflection"
                cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} \\neq 0 \\quad (\\text{{First non-zero derivative is Odd order}})"
                conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ it is a Point of Inflection (neither max nor min)}}"
            else:
                if val < 0:
                    pt_type = "Maximum"
                    cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} < 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
                else:
                    pt_type = "Minimum"
                    cond_str = f"f^{{({order})}}({r_val}) = {val:.{decimal_places}f} > 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"

        results.append(AnalyticalPointResult(
            x_value=round(r_val, decimal_places),
            fx_value=round(fx_val, decimal_places),
            f2_value=round(f2_val, decimal_places),
            point_type=pt_type
        ))

        eval_step_text += f"**For ${sym_name} = {r_val}$**\n\n"
        eval_step_text += f"$$f''({r_val}) = {f2_sub_step}$$\n\n"
        if higher_order_text:
            eval_step_text += higher_order_text
        eval_step_text += f"$${cond_str}$$\n\n"
        eval_step_text += f"$${conclusion}$$\n\n---\n\n"

    steps.append({
        "title": f"Evaluate derivatives at the critical points",
        "content": eval_step_text
    })

    # Step 5: Calculate Extrema Values with Step-by-Step Substitution
    step4_text = f"Substitute the ${sym_name}$ values back into the original function $f({sym_name}) = {sp.latex(expr)}$\n\n"
    for r_val in real_roots:
        fx_sub_step = get_step_by_step_eval(expr, x, r_val)
        fx_val = float(expr.subs(x, r_val).evalf())
        p_type = [r for r in results if r.x_value == r_val][0].point_type
        
        if p_type == "Minimum":
            pt_kind = "local minimum point"
        elif p_type == "Maximum":
            pt_kind = "local maximum point"
        else:
            pt_kind = "point of inflection"

        step4_text += f"**At ${sym_name} = {r_val}$**\n\n"
        step4_text += f"$$f({r_val}) = {fx_sub_step}$$\n\n"
        step4_text += f"$$\\text{{{pt_kind}}} = ({r_val}, {fx_val:.{decimal_places}f})$$\n\n"

    steps.append({
        "title": "Step-4: Calculate the extrema values",
        "content": step4_text
    })

    return AnalyticalResult(steps, results, expr, f1, f2, x)


def get_numpy_func(expr: sp.Expr, x_sym: sp.Symbol):
    return sp.lambdify(x_sym, expr, modules=["numpy"])

    solve_steps_text = f"$$f'({sym_name}) = 0$$\n\n"
    solve_steps_text += f"$$\\implies {sp.latex(f1)} = 0$$\n\n"
    
    factored_f1 = sp.factor(f1)
    if factored_f1 != f1:
        solve_steps_text += f"$$\\implies {sp.latex(factored_f1)} = 0$$\n\n"

    roots_str_list = [f"{sym_name} = {sp.latex(sp.sympify(r))}" for r in real_roots]
    solve_steps_text += f"$$\\implies " + " \\text{ or } ".join(roots_str_list) + "$$\n\n"
    solve_steps_text += f"$$\\therefore \\text{{The critical points are }} " + " \\text{ and } ".join(roots_str_list) + "$$"

    steps.append({
        "title": f"Solving for {sym_name}:",
        "content": solve_steps_text
    })

    # Step 4: Second Derivative Test & Higher Order Analysis
    f2 = sp.diff(f1, x)
    f2_terms = f1.as_ordered_terms() if isinstance(f1, sp.Add) else [f1]
    f2_diff_terms = " + ".join([f"\\frac{{d}}{{d{sym_name}}}\\left({sp.latex(t)}\\right)" for t in f2_terms]).replace("+ -", "- ")
    
    step3_content = (
        f"Now, $f''({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(f1)}\\right)$\n\n"
        f"$$= {f2_diff_terms}$$\n\n"
        f"$$= {sp.latex(f2)}$$"
    )
    steps.append({
        "title": "Step-3: Apply the second derivative test",
        "content": step3_content
    })

    # Evaluate f''(x) and apply Higher-Order Derivative Test if f''(x) == 0
    eval_step_text = ""
    results = []
    
    for r_val in real_roots:
        f2_val = float(f2.subs(x, r_val).evalf())
        fx_val = float(expr.subs(x, r_val).evalf())

        if f2_val < -1e-9:
            pt_type = "Maximum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} < 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
        elif f2_val > 1e-9:
            pt_type = "Minimum"
            cond_str = f"f''({r_val}) = {f2_val:.{decimal_places}f} > 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"
        else:
            # Higher-Order Derivative Test Logic (when f''(x) == 0)
            order = 3
            current_deriv = sp.diff(f2, x)
            val = float(current_deriv.subs(x, r_val).evalf())
            
            # Find the first non-zero higher-order derivative
            while abs(val) < 1e-9 and order <= 10:
                order += 1
                current_deriv = sp.diff(current_deriv, x)
                val = float(current_deriv.subs(x, r_val).evalf())

            if order % 2 != 0:  # Odd order derivative is non-zero
                pt_type = "Point of Inflection"
                cond_str = f"f''({r_val}) = 0, \\text{{ but }} f^{{({order})}}({r_val}) = {val:.{decimal_places}f} \\neq 0 \\text{{ (Odd order)}}"
                conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ it is a Point of Inflection (neither max nor min)}}"
            else:  # Even order derivative is non-zero
                if val < 0:
                    pt_type = "Maximum"
                    cond_str = f"f''({r_val}) = 0, \\text{{ but }} f^{{({order})}}({r_val}) = {val:.{decimal_places}f} < 0 \\text{{ (Even order)}}"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local maximum}}"
                else:
                    pt_type = "Minimum"
                    cond_str = f"f''({r_val}) = 0, \\text{{ but }} f^{{({order})}}({r_val}) = {val:.{decimal_places}f} > 0 \\text{{ (Even order)}}"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {r_val} \\text{{ the function is local minimum}}"

        results.append(AnalyticalPointResult(
            x_value=round(r_val, decimal_places),
            fx_value=round(fx_val, decimal_places),
            f2_value=round(f2_val, decimal_places),
            point_type=pt_type
        ))

        eval_step_text += f"**For ${sym_name} = {r_val}$**\n\n"
        eval_step_text += f"$$f''({r_val}) = {f2_val:.{decimal_places}f}$$\n\n"
        eval_step_text += f"$${cond_str}$$\n\n"
        eval_step_text += f"$${conclusion}$$\n\n---\n\n"

    steps.append({
        "title": f"Evaluate derivatives at the critical points",
        "content": eval_step_text
    })

    # Step 5: Calculate Extrema Values
    step4_text = f"Substitute the ${sym_name}$ values back into the original function $f({sym_name}) = {sp.latex(expr)}$\n\n"
    for r_val in real_roots:
        fx_val = float(expr.subs(x, r_val).evalf())
        p_type = [r for r in results if r.x_value == r_val][0].point_type
        
        if p_type == "Minimum":
            pt_kind = "local minimum point"
        elif p_type == "Maximum":
            pt_kind = "local maximum point"
        else:
            pt_kind = "point of inflection"

        step4_text += f"**At ${sym_name} = {r_val}$**\n\n"
        step4_text += f"$$f({r_val}) = {sp.latex(expr.subs(x, r_val))} = {fx_val:.{decimal_places}f}$$\n\n"
        step4_text += f"$$\\text{{{pt_kind}}} = ({r_val}, {fx_val:.{decimal_places}f})$$\n\n"

    steps.append({
        "title": "Step-4: Calculate the extrema values",
        "content": step4_text
    })

    return AnalyticalResult(steps, results, expr, f1, f2, x)


def get_numpy_func(expr: sp.Expr, x_sym: sp.Symbol):
    return sp.lambdify(x_sym, expr, modules=["numpy"])
