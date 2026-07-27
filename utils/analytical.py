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


def format_num(val: float, decimal_places: int = 4) -> str:
    """Format float cleanly (e.g. 3.0 -> 3, 3.1230 -> 3.123)"""
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.{decimal_places}f}".rstrip('0').rstrip('.')


def get_step_by_step_eval(expr: sp.Expr, x_sym: sp.Symbol, val: float, decimal_places: int = 4) -> str:
    """Helper function to format step-by-step substitution into latex without excessive zeros"""
    clean_val = format_num(val, decimal_places)
    
    val_sym = sp.sympify(clean_val)
    sub_expr = expr.subs(x_sym, sp.Symbol(f"({val_sym})"))
    sub_latex = sp.latex(sub_expr)
    
    final_val = float(expr.subs(x_sym, val).evalf())
    clean_final = format_num(final_val, decimal_places)
    
    if expr.is_number or sub_latex == sp.latex(expr):
        return clean_final
    
    return f"{sub_latex} = {clean_final}"


def get_prime_notation(order: int) -> str:
    """
    Convert order integer into primes or superscript roman numerals:
    1 -> '
    2 -> ''
    3 -> '''
    4 -> ^{iv}
    5 -> ^{v}
    6 -> ^{vi}
    ...
    """
    if order == 1:
        return "'"
    elif order == 2:
        return "''"
    elif order == 3:
        return "'''"
    
    # Roman numeral conversion for 4 and above
    val_map = [
        (10, 'x'), (9, 'ix'), (5, 'v'), (4, 'iv'), (1, 'i')
    ]
    num = order
    roman = ""
    for v, r in val_map:
        while num >= v:
            roman += r
            num -= v
    return f"^{{{roman}}}"


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

    # Step 2: First Derivative Breakdown
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

    roots_str_list = [f"{sym_name} = {format_num(r, decimal_places)}" for r in real_roots]
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
        clean_r = format_num(r_val, decimal_places)
        f2_val = float(f2.subs(x, r_val).evalf())
        fx_val = float(expr.subs(x, r_val).evalf())
        f2_sub_step = get_step_by_step_eval(f2, x, r_val, decimal_places)

        if f2_val < -1e-9:
            pt_type = "Maximum"
            cond_str = f"f''({clean_r}) = {format_num(f2_val, decimal_places)} < 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {clean_r} \\text{{ the function is local maximum}}"
            higher_order_text = ""
        elif f2_val > 1e-9:
            pt_type = "Minimum"
            cond_str = f"f''({clean_r}) = {format_num(f2_val, decimal_places)} > 0"
            conclusion = f"\\therefore \\text{{At }} {sym_name} = {clean_r} \\text{{ the function is local minimum}}"
            higher_order_text = ""
        else:
            order = 3
            prev_deriv = f2
            current_deriv = sp.diff(prev_deriv, x)
            val = float(current_deriv.subs(x, r_val).evalf())
            
            higher_order_text = f"Since $f''({clean_r}) = 0$, the second derivative test is inconclusive. We test higher-order derivatives:\n\n"
            
            prime_symbol = get_prime_notation(order)
            higher_order_text += f"$$f{prime_symbol}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
            h_sub_step = get_step_by_step_eval(current_deriv, x, r_val, decimal_places)
            higher_order_text += f"$$f{prime_symbol}({clean_r}) = {h_sub_step}$$\n\n"

            while abs(val) < 1e-9 and order <= 10:
                order += 1
                prev_deriv = current_deriv
                current_deriv = sp.diff(prev_deriv, x)
                val = float(current_deriv.subs(x, r_val).evalf())
                prime_symbol = get_prime_notation(order)
                higher_order_text += f"$$f{prime_symbol}({sym_name}) = \\frac{{d}}{{d{sym_name}}}\\left({sp.latex(prev_deriv)}\\right) = {sp.latex(current_deriv)}$$\n\n"
                h_sub_step = get_step_by_step_eval(current_deriv, x, r_val, decimal_places)
                higher_order_text += f"$$f{prime_symbol}({clean_r}) = {h_sub_step}$$\n\n"

            prime_symbol = get_prime_notation(order)
            clean_deriv_val = format_num(val, decimal_places)

            if order % 2 != 0:
                pt_type = "Point of Inflection"
                cond_str = f"f{prime_symbol}({clean_r}) = {clean_deriv_val} \\neq 0 \\quad (\\text{{First non-zero derivative is Odd order}})"
                conclusion = f"\\therefore \\text{{At }} {sym_name} = {clean_r} \\text{{ it is a Point of Inflection (neither max nor min)}}"
            else:
                if val < 0:
                    pt_type = "Maximum"
                    cond_str = f"f{prime_symbol}({clean_r}) = {clean_deriv_val} < 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {clean_r} \\text{{ the function is local maximum}}"
                else:
                    pt_type = "Minimum"
                    cond_str = f"f{prime_symbol}({clean_r}) = {clean_deriv_val} > 0 \\quad (\\text{{First non-zero derivative is Even order}})"
                    conclusion = f"\\therefore \\text{{At }} {sym_name} = {clean_r} \\text{{ the function is local minimum}}"

        results.append(AnalyticalPointResult(
            x_value=round(r_val, decimal_places),
            fx_value=round(fx_val, decimal_places),
            f2_value=round(f2_val, decimal_places),
            point_type=pt_type
        ))

        eval_step_text += f"**For ${sym_name} = {clean_r}$**\n\n"
        eval_step_text += f"$$f''({clean_r}) = {f2_sub_step}$$\n\n"
        if higher_order_text:
            eval_step_text += higher_order_text
        eval_step_text += f"$${cond_str}$$\n\n"
        eval_step_text += f"$${conclusion}$$\n\n---\n\n"

    steps.append({
        "title": f"Evaluate derivatives at the critical points",
        "content": eval_step_text
    })

    # Step 5: Extrema Values
    step4_text = f"Substitute the ${sym_name}$ values back into the original function $f({sym_name}) = {sp.latex(expr)}$\n\n"
    for r_val in real_roots:
        clean_r = format_num(r_val, decimal_places)
        fx_sub_step = get_step_by_step_eval(expr, x, r_val, decimal_places)
        fx_val = float(expr.subs(x, r_val).evalf())
        clean_fx = format_num(fx_val, decimal_places)
        
        p_type = [r for r in results if r.x_value == r_val][0].point_type
        pt_kind = "local minimum point" if p_type == "Minimum" else ("local maximum point" if p_type == "Maximum" else "point of inflection")

        step4_text += f"**At ${sym_name} = {clean_r}$**\n\n"
        step4_text += f"$$f({clean_r}) = {fx_sub_step}$$\n\n"
        step4_text += f"$$\\text{{{pt_kind}}} = ({clean_r}, {clean_fx})$$\n\n"

    steps.append({
        "title": "Step-4: Calculate the extrema values",
        "content": step4_text
    })

    return AnalyticalResult(steps, results, expr, f1, f2, x)


def get_numpy_func(expr: sp.Expr, x_sym: sp.Symbol):
    return sp.lambdify(x_sym, expr, modules=["numpy"])
