"""
app.py
======
Streamlit web application: Maxima & Minima Finder with Source Code Viewer & PDF Export.
"""

from __future__ import annotations

import os
import io
import random
import numpy as np
import pandas as pd
import streamlit as st

from utils.analytical import analyze_function, get_numpy_func
from utils.tabular import analyze_table
from utils.plotting import plot_function_with_extrema
from utils.pdf_generator import generate_pdf_report


# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Maxima-Minima Finder | Numerical Differentiation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Custom CSS
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
    .main-title {
        text-align: center;
        padding: 1rem 0 0.2rem 0;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-color);
    }
    .sub-title {
        text-align: center;
        opacity: 0.8;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
        color: var(--text-color);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="main-title">📈 Maxima & Minima Finder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Using Numerical Differentiation — Analytical & Tabular Methods</div>',
    unsafe_allow_html=True,
)

EXAMPLE_FUNCTIONS = {
    "x^3 + 6x^2 - 15x + 7": "x^3 + 6*x^2 - 15*x + 7",
    "4x^3 - 9x^2 + 6x": "4*x^3 - 9*x^2 + 6*x",
    "x^3 - 3x^2 + 2": "x^3 - 3*x^2 + 2",
}

EXAMPLE_TABLE = {
    "x": [0, 2, 4, 6],
    "y": [2, 0, -50, -196],
}


def render_steps(steps):
    """Render a list of step dicts as clean formatted steps."""
    for step in steps:
        st.markdown(f"### {step['title']}")
        st.markdown(step['content'])
        st.markdown("---")


def render_result_table(rows, x_col_label="x", y_col_label="f(x)"):
    """Build and display a results DataFrame + return it for CSV export."""
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "x_value": x_col_label,
        "fx_value": y_col_label,
        "y_value": y_col_label,
        "f2_value": "f''(x)",
        "point_type": "Type",
    })
    if "Iteration" not in df.columns:
        df.insert(0, "Iteration", range(1, len(df) + 1))
    st.dataframe(df, use_container_width=True, hide_index=True)
    return df


def update_func_from_example():
    choice = st.session_state.get("example_choice")
    if choice and choice != "-- pick example --":
        st.session_state["func_input"] = EXAMPLE_FUNCTIONS[choice]

def update_func_random():
    st.session_state["func_input"] = random.choice(list(EXAMPLE_FUNCTIONS.values()))


if "func_input" not in st.session_state:
    st.session_state["func_input"] = "x^3 + 6*x^2 - 15*x + 7"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Navigation",
        ["🔍 Calculator", "💻 View Source Code"],
        help="Switch between Calculator and Project Source Code Viewer."
    )
    
    if mode == "🔍 Calculator":
        method = st.radio(
            "Choose Input Method",
            ["Analytical Function f(x)", "Tabular Data (x, y)"],
            help="Pick whether you want to type a formula or enter numerical data points.",
        )
        decimal_places = st.selectbox("Decimal Places", [2, 3, 4, 5, 6], index=2)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This app finds **Maxima and Minima** using:\n\n"
        "- **Method 1:** Symbolic differentiation (sympy) for f(x)\n"
        "- **Method 2:** Newton's Forward Difference formula for tabular data"
    )


# --------------------------------------------------------------------------- #
# MODE: Source Code Viewer
# --------------------------------------------------------------------------- #
if mode == "💻 View Source Code":
    st.subheader("📁 Project Source Code Viewer")
    st.write("Browse the underlying Python implementation and project files:")

    files_to_show = {
        "app.py": "app.py",
        "utils/analytical.py": "utils/analytical.py",
        "utils/tabular.py": "utils/tabular.py",
        "utils/plotting.py": "utils/plotting.py",
        "utils/pdf_generator.py": "utils/pdf_generator.py",
        "requirements.txt": "requirements.txt",
    }

    selected_file = st.selectbox("📂 Select a file to inspect:", list(files_to_show.keys()))

    file_path = files_to_show[selected_file]
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        lang = "python" if file_path.endswith(".py") else "text"
        st.code(code_content, language=lang, line_numbers=True)
    else:
        st.warning(f"File `{file_path}` not found in project directory.")


# --------------------------------------------------------------------------- #
# MODE: Calculator
# --------------------------------------------------------------------------- #
elif mode == "🔍 Calculator":

    # METHOD 1: Analytical Function
    if method == "Analytical Function f(x)":
        st.subheader("Method 1 — Analytical Function")

        col1, col2 = st.columns([3, 1])
        with col1:
            func_input = st.text_input(
                "f(x) =",
                key="func_input",
                placeholder="e.g. x^3 + 6x^2 - 15x + 7",
            )
        with col2:
            st.write("")
            st.write("")
            st.selectbox(
                "Examples",
                ["-- pick example --"] + list(EXAMPLE_FUNCTIONS.keys()),
                key="example_choice",
                on_change=update_func_from_example,
                label_visibility="collapsed"
            )

        b1, b2, _ = st.columns([1, 1, 4])
        find_clicked = b1.button("🔍 Find", type="primary", use_container_width=True)
        b2.button("🎲 Random", on_click=update_func_random, use_container_width=True)

        if find_clicked or "last_analytical_result" in st.session_state:
            if find_clicked:
                try:
                    result = analyze_function(func_input, decimal_places=decimal_places)
                    st.session_state["last_analytical_result"] = result
                except Exception as exc:
                    st.error(f"❌ Unexpected error while analyzing function: {exc}")
                    result = None
            else:
                result = st.session_state.get("last_analytical_result")

            if result is not None:
                if result.error:
                    st.error(f"⚠️ {result.error}")
                else:
                    st.success("✅ Solution found!")

                    st.markdown("## 📝 Solution")
                    render_steps(result.steps)

                    st.markdown("### 📊 Results Table")
                    rows = [
                        {"x_value": r.x_value, "fx_value": r.fx_value, "f2_value": r.f2_value, "point_type": r.point_type}
                        for r in result.results
                    ]
                    df_res = render_result_table(rows, x_col_label="x", y_col_label="f(x)")

                    fig = None
                    st.markdown("### 📈 Graph")
                    try:
                        numpy_func = get_numpy_func(result.expr, result.x_symbol)
                        x_vals = [r.x_value for r in result.results]
                        x_range = (min(x_vals) - 3, max(x_vals) + 3) if x_vals else (-10, 10)
                        fig = plot_function_with_extrema(
                            numpy_func,
                            x_range,
                            [{"x_value": r.x_value, "fx_value": r.fx_value, "point_type": r.point_type} for r in result.results],
                            title=f"f(x) = {func_input}",
                        )
                        st.pyplot(fig)
                    except Exception as exc:
                        st.warning(f"Could not render graph: {exc}")

                    st.markdown("---")
                    st.markdown("### 💾 Export & Downloads")
                    e_col1, e_col2, e_col3 = st.columns(3)

                    with e_col1:
                        csv_buf = io.StringIO()
                        df_res.to_csv(csv_buf, index=False)
                        st.download_button("📊 Download CSV", data=csv_buf.getvalue(), file_name="results.csv", mime="text/csv", use_container_width=True)

                    with e_col2:
                        if fig is not None:
                            img_buf = io.BytesIO()
                            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=300)
                            st.download_button("🖼️ Download Graph (PNG)", data=img_buf.getvalue(), file_name="graph.png", mime="image/png", use_container_width=True)

                    with e_col3:
                        try:
                            pdf_data = generate_pdf_report(f"f(x) = {func_input}", result.steps, fig, df_res)
                            st.download_button("📄 Download PDF Report", data=pdf_data, file_name="solution_report.pdf", mime="application/pdf", use_container_width=True)
                        except Exception as err:
                            st.error(f"PDF error: {err}")

    # METHOD 2: Tabular Data
    else:
        st.subheader("Method 2 — Tabular Data (Newton's Forward Difference)")

        if "table_rows" not in st.session_state:
            st.session_state["table_rows"] = 4

        top_c1, top_c2, _ = st.columns([1, 1, 2])
        with top_c1:
            n_rows = st.number_input("Rows", min_value=4, max_value=15, value=st.session_state["table_rows"], step=1)
            st.session_state["table_rows"] = n_rows
        with top_c2:
            st.write("")
            st.write("")
            load_example = st.button("📥 Load Example")

        if load_example:
            st.session_state["table_x"] = EXAMPLE_TABLE["x"] + [EXAMPLE_TABLE["x"][-1] + 2 * (i + 1) for i in range(n_rows - 4)] if n_rows > 4 else EXAMPLE_TABLE["x"]
            st.session_state["table_y"] = EXAMPLE_TABLE["y"] + [0.0] * max(0, n_rows - 4)

        default_x = st.session_state.get("table_x", [0, 2, 4, 6] + [0.0] * max(0, n_rows - 4))[:n_rows]
        default_y = st.session_state.get("table_y", [2, 0, -50, -196] + [0.0] * max(0, n_rows - 4))[:n_rows]

        edit_df = pd.DataFrame({"x": default_x, "y = f(x)": default_y})
        edited_df = st.data_editor(edit_df, num_rows="fixed", use_container_width=True, key="data_editor_table")

        find_table_clicked = st.button("🔍 Find Maxima/Minima", type="primary")

        if find_table_clicked or "last_tabular_result" in st.session_state:
            if find_table_clicked:
                try:
                    x_vals = [float(v) for v in edited_df["x"].tolist()]
                    y_vals = [float(v) for v in edited_df["y = f(x)"].tolist()]
                    result = analyze_table(x_vals, y_vals, decimal_places=decimal_places)
                    st.session_state["last_tabular_result"] = result
                except Exception as exc:
                    st.error(f"❌ Error processing table: {exc}")
                    result = None
            else:
                result = st.session_state.get("last_tabular_result")

            if result is not None and not result.error:
                st.success("✅ Solution found!")
                st.markdown("## 📝 Solution")
                render_steps(result.steps)

                df_res = pd.DataFrame()
                if result.critical_points:
                    st.markdown("### 📊 Results Table")
                    rows = [{"x_value": r.x_value, "y_value": r.y_value, "f2_value": r.f2_value, "point_type": r.point_type} for r in result.critical_points]
                    df_res = render_result_table(rows, x_col_label="x", y_col_label="y")

                fig = None
                st.markdown("### 📈 Graph")
                try:
                    import sympy as sp
                    x_sym = sp.symbols("x")
                    numpy_func = sp.lambdify(x_sym, result.polynomial, modules=["numpy"])
                    x_vals_cp = [r.x_value for r in result.critical_points]
                    all_x = result.x_values + x_vals_cp
                    x_range = (min(all_x) - 1, max(all_x) + 1)
                    fig = plot_function_with_extrema(
                        numpy_func, x_range,
                        [{"x_value": r.x_value, "y_value": r.y_value, "point_type": r.point_type} for r in result.critical_points],
                        title="Newton's Forward Difference Fit",
                        extra_scatter=list(zip(result.x_values, result.y_values)),
                        extra_scatter_label="Data Points",
                    )
                    st.pyplot(fig)
                except Exception as exc:
                    st.warning(f"Could not render graph: {exc}")

                st.markdown("---")
                st.markdown("### 💾 Export & Downloads")
                e_col1, e_col2, e_col3 = st.columns(3)

                with e_col1:
                    if not df_res.empty:
                        csv_buf = io.StringIO()
                        df_res.to_csv(csv_buf, index=False)
                        st.download_button("📊 Download CSV", data=csv_buf.getvalue(), file_name="tabular_results.csv", mime="text/csv", use_container_width=True)

                with e_col2:
                    if fig is not None:
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=300)
                        st.download_button("🖼️ Download Graph (PNG)", data=img_buf.getvalue(), file_name="tabular_graph.png", mime="image/png", use_container_width=True)

                with e_col3:
                    try:
                        pdf_data = generate_pdf_report("Tabular Newton Forward Method", result.steps, fig, df_res)
                        st.download_button("📄 Download PDF Report", data=pdf_data, file_name="tabular_solution.pdf", mime="application/pdf", use_container_width=True)
                    except Exception as err:
                        st.error(f"PDF error: {err}")


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown("---")
st.caption("Built with Streamlit • Numerical Methods Presentation")