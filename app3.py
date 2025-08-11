import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from sympy import symbols, solve, Eq, latex, simplify, expand
from scipy.optimize import fsolve
import io
import base64
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="IS/LM/BP Macroeconomic Models Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .equation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2c3e50;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 999;
    }
</style>
""",
    unsafe_allow_html=True,
)


class ISLMBPModel:
    def __init__(self):
        # Define symbolic variables
        self.Y, self.r, self.C, self.I, self.G, self.T, self.L, self.M, self.P = (
            symbols("Y r C I G T L M P")
        )
        self.C0, self.c, self.I0, self.b, self.alpha = symbols("C0 c I0 b alpha")
        self.T0, self.t, self.k, self.h, self.Ms = symbols("T0 t k h Ms")
        self.NX0, self.m, self.n, self.E, self.CF, self.r_star = symbols(
            "NX0 m n E CF r_star"
        )

        self.parameters = {}
        self.results = {}

    def set_parameters(self, params):
        self.parameters = params

    def consumption_function(self):
        """C = C0 + c(Y - T)"""
        T_expr = self.T0 + self.t * self.Y
        C_expr = self.C0 + self.c * (self.Y - T_expr)
        return C_expr.expand()

    def investment_function(self):
        """I = I0 - b*r + alpha*Y"""
        return self.I0 - self.b * self.r + self.alpha * self.Y

    def money_demand(self):
        """L = k*Y - h*r"""
        return self.k * self.Y - self.h * self.r

    def net_exports(self):
        """NX = NX0 - m*Y + n*E"""
        return self.NX0 - self.m * self.Y + self.n * self.E

    def derive_IS_curve(self):
        """Derive IS curve: Y = C + I + G + NX"""
        C = self.consumption_function()
        I = self.investment_function()
        NX = self.net_exports() if self.parameters.get("include_bp", False) else 0

        # Goods market equilibrium
        IS_eq = Eq(self.Y, C + I + self.G + NX)

        # Solve for r in terms of Y
        try:
            r_IS = solve(IS_eq, self.r)[0]
            # Solve for Y in terms of r
            Y_IS = solve(IS_eq, self.Y)[0]
            return IS_eq, r_IS, Y_IS
        except:
            return IS_eq, None, None

    def derive_LM_curve(self):
        """Derive LM curve: Ms/P = L(Y,r)"""
        L = self.money_demand()

        # Money market equilibrium
        LM_eq = Eq(self.Ms / self.P, L)

        # Solve for r in terms of Y
        try:
            r_LM = solve(LM_eq, self.r)[0]
            Y_LM = solve(LM_eq, self.Y)[0]
            return LM_eq, r_LM, Y_LM
        except:
            return LM_eq, None, None

    def derive_BP_curve(self):
        """Derive BP curve: NX + CF = 0"""
        if not self.parameters.get("include_bp", False):
            return None, None, None

        NX = self.net_exports()
        CF = self.CF * (self.r - self.r_star)  # Capital flows

        # Balance of payments equilibrium
        BP_eq = Eq(NX + CF, 0)

        try:
            r_BP = solve(BP_eq, self.r)[0]
            Y_BP = solve(BP_eq, self.Y)[0]
            return BP_eq, r_BP, Y_BP
        except:
            return BP_eq, None, None

    def solve_equilibrium(self):
        """Solve for IS-LM(-BP) equilibrium with improved numerical handling"""
        IS_eq, r_IS, Y_IS = self.derive_IS_curve()
        LM_eq, r_LM, Y_LM = self.derive_LM_curve()

        equations = [IS_eq, LM_eq]

        if self.parameters.get("include_bp", False):
            BP_eq, r_BP, Y_BP = self.derive_BP_curve()
            # Fixed: Check if BP_eq is not None instead of using it directly in boolean context
            if BP_eq is not None:
                equations.append(BP_eq)

        try:
            # Try symbolic solution first
            solution = solve(equations, [self.Y, self.r])
            if solution:
                # Check if solution contains only numbers or simple expressions
                Y_sol = solution[self.Y]
                r_sol = solution[self.r]

                # Try to evaluate the solution
                param_dict = {
                    self.C0: self.parameters.get("C0", 100),
                    self.c: self.parameters.get("c", 0.8),
                    self.I0: self.parameters.get("I0", 200),
                    self.b: self.parameters.get("b", 50),
                    self.alpha: self.parameters.get("alpha", 0),
                    self.G: self.parameters.get("G", 200),
                    self.T0: self.parameters.get("T0", 50),
                    self.t: self.parameters.get("t", 0.2),
                    self.k: self.parameters.get("k", 0.5),
                    self.h: self.parameters.get("h", 100),
                    self.Ms: self.parameters.get("Ms", 1000),
                    self.P: self.parameters.get("P", 1),
                }

                if self.parameters.get("include_bp", False):
                    param_dict.update(
                        {
                            self.NX0: self.parameters.get("NX0", 50),
                            self.m: self.parameters.get("m", 0.1),
                            self.n: self.parameters.get("n", 0),
                            self.E: self.parameters.get("E", 1),
                            self.CF: self.parameters.get("CF", 100),
                            self.r_star: self.parameters.get("r_star", 0.05),
                        }
                    )

                # Substitute parameters and evaluate
                try:
                    Y_eval = Y_sol.subs(param_dict).evalf()
                    r_eval = r_sol.subs(param_dict).evalf()
                    return {self.Y: Y_eval, self.r: r_eval}
                except:
                    return solution

            else:
                # Fall back to numerical solution
                return self.numerical_solution()
        except Exception as e:
            print(f"Symbolic solution failed: {e}")
            return self.numerical_solution()

    def numerical_solution(self):
        """Numerical solution when symbolic fails"""

        def equations(vars):
            Y_val, r_val = vars

            # Substitute parameter values
            params_dict = {
                param: val
                for param, val in self.parameters.items()
                if param
                in [
                    str(s)
                    for s in [
                        self.C0,
                        self.c,
                        self.I0,
                        self.b,
                        self.alpha,
                        self.T0,
                        self.t,
                        self.k,
                        self.h,
                        self.Ms,
                        self.P,
                        self.NX0,
                        self.m,
                        self.n,
                        self.E,
                        self.CF,
                        self.r_star,
                        self.G,
                    ]
                ]
            }

            # IS equation residual
            C_val = params_dict.get("C0", 100) + params_dict.get("c", 0.8) * (
                Y_val - (params_dict.get("T0", 50) + params_dict.get("t", 0.2) * Y_val)
            )
            I_val = (
                params_dict.get("I0", 200)
                - params_dict.get("b", 50) * r_val
                + params_dict.get("alpha", 0) * Y_val
            )
            G_val = params_dict.get("G", 200)

            NX_val = 0
            if self.parameters.get("include_bp", False):
                NX_val = (
                    params_dict.get("NX0", 50)
                    - params_dict.get("m", 0.1) * Y_val
                    + params_dict.get("n", 0) * params_dict.get("E", 1)
                )

            IS_residual = Y_val - (C_val + I_val + G_val + NX_val)

            # LM equation residual
            L_val = (
                params_dict.get("k", 0.5) * Y_val - params_dict.get("h", 100) * r_val
            )
            Ms_P = params_dict.get("Ms", 1000) / params_dict.get("P", 1)
            LM_residual = Ms_P - L_val

            return [IS_residual, LM_residual]

        try:
            Y_guess = self.parameters.get("G", 200) / (
                1 - self.parameters.get("c", 0.8)
            )
            r_guess = 0.05
            solution = fsolve(equations, [Y_guess, r_guess])
            return {self.Y: solution[0], self.r: solution[1]}
        except:
            return {self.Y: 1000, self.r: 0.05}


def safe_float_conversion(value, parameters=None):
    """Safely convert SymPy expressions to float"""
    if isinstance(value, (int, float)):
        return float(value)

    # If it's a SymPy expression, try to evaluate it
    if hasattr(value, "evalf"):
        try:
            # If parameters provided, substitute them first
            if parameters:
                param_dict = {
                    symbols("C0"): parameters.get("C0", 100),
                    symbols("c"): parameters.get("c", 0.8),
                    symbols("I0"): parameters.get("I0", 200),
                    symbols("b"): parameters.get("b", 50),
                    symbols("alpha"): parameters.get("alpha", 0),
                    symbols("G"): parameters.get("G", 200),
                    symbols("T0"): parameters.get("T0", 50),
                    symbols("t"): parameters.get("t", 0.2),
                    symbols("k"): parameters.get("k", 0.5),
                    symbols("h"): parameters.get("h", 100),
                    symbols("Ms"): parameters.get("Ms", 1000),
                    symbols("P"): parameters.get("P", 1),
                }

                if parameters.get("include_bp", False):
                    param_dict.update(
                        {
                            symbols("NX0"): parameters.get("NX0", 50),
                            symbols("m"): parameters.get("m", 0.1),
                            symbols("n"): parameters.get("n", 0),
                            symbols("E"): parameters.get("E", 1),
                            symbols("CF"): parameters.get("CF", 100),
                            symbols("r_star"): parameters.get("r_star", 0.05),
                        }
                    )

                value = value.subs(param_dict)

            evaluated = value.evalf()

            # Handle complex numbers by taking real part
            if hasattr(evaluated, "is_real") and not evaluated.is_real:
                return float(complex(evaluated).real)

            return float(evaluated)
        except:
            # If evalf fails, try direct conversion
            try:
                return float(complex(value).real)
            except:
                return 0.0

    # Last resort
    try:
        return float(value)
    except:
        return 0.0


def get_model_title(model_type):
    """Generate dynamic title based on model selection"""
    title_map = {
        "IS only": "IS Model",
        "LM only": "LM Model",
        "BP only": "BP Model",
        "IS/LM": "IS/LM Model",
        "IS/LM/BP": "IS/LM/BP Model",
    }
    return title_map.get(model_type, "Macroeconomic Model")


def create_main_app():
    st.markdown(
        '<h1 class="main-header">IS/LM/BP Macroeconomic Models Explorer</h1>',
        unsafe_allow_html=True,
    )

    # Initialize model
    model = ISLMBPModel()

    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")

        # Model selection
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Choose Model", ["IS only", "LM only", "BP only", "IS/LM", "IS/LM/BP"]
        )

        exchange_rate_regime = st.selectbox(
            "Exchange Rate Regime", ["Fixed", "Flexible"]
        )

        monetary_policy = st.selectbox(
            "Monetary Policy",
            ["Exogenous Money Supply (M)", "Exogenous Interest Rate (r)"],
        )

        # Consumption parameters - WIDER RANGES
        st.subheader("Consumption Function")
        st.write("C = C₀ + c(Y - T)")
        C0 = st.slider("Autonomous Consumption (C₀)", -2000, 2000, 100, 25)
        c = st.slider("Marginal Propensity to Consume (c)", 0.0, 2.0, 0.8, 0.01)

        # Investment parameters - WIDER RANGES
        st.subheader("Investment Function")
        st.write("I = I₀ - br + αY")
        I0 = st.slider("Autonomous Investment (I₀)", -2000, 2000, 200, 25)
        b = st.slider("Interest Sensitivity (b)", 0, 1000, 50, 10)
        alpha = st.slider("Income Sensitivity (α)", -2.0, 2.0, 0.0, 0.01)

        # Government parameters - WIDER RANGES
        st.subheader("Government Sector")
        G = st.slider("Government Spending (G)", -2000, 4000, 200, 25)
        T0 = st.slider("Autonomous Taxes (T₀)", -2000, 2000, 50, 25)
        t = st.slider("Tax Rate (t)", 0.0, 1.5, 0.2, 0.01)

        # Money market parameters - WIDER RANGES
        st.subheader("Money Market")
        st.write("L = kY - hr")
        k = st.slider("Income Elasticity of Money Demand (k)", 0.0, 20.0, 0.5, 0.1)
        h = st.slider("Interest Elasticity of Money Demand (h)", 0, 500, 100, 10)
        Ms = st.slider("Money Supply (Ms)", 0, 10000, 1000, 100)
        P = st.slider("Price Level (P)", 0.1, 20.0, 1.0, 0.1)

        # External sector parameters - WIDER RANGES
        if "BP" in model_type:
            st.subheader("External Sector")
            st.write("NX = NX₀ - mY + nE")
            NX0 = st.slider("Autonomous Net Exports (NX₀)", -1000, 1000, 50, 25)
            m = st.slider("Import Propensity (m)", 0.0, 2.0, 0.1, 0.01)
            n = st.slider("Exchange Rate Sensitivity (n)", -20.0, 20.0, 0.0, 0.1)
            E = st.slider("Exchange Rate (E)", 0.1, 20.0, 1.0, 0.1)
            CF = st.slider("Capital Mobility (CF)", 0, 2000, 100, 25)
            r_star = st.slider("Foreign Interest Rate (r*) %", -20, 100, 5, 1) / 100
        else:
            NX0, m, n, E, CF, r_star = 0, 0, 0, 1, 0, 0.05

        # Preset scenarios - FIXED: Changed from st.experimental_rerun() to st.rerun()
        st.subheader("Preset Scenarios")
        if st.button("Classical Case"):
            st.rerun()
        if st.button("Keynesian Liquidity Trap"):
            st.rerun()
        if st.button("High Capital Mobility"):
            st.rerun()

    # Set parameters in model
    parameters = {
        "C0": C0,
        "c": c,
        "I0": I0,
        "b": b,
        "alpha": alpha,
        "G": G,
        "T0": T0,
        "t": t,
        "k": k,
        "h": h,
        "Ms": Ms,
        "P": P,
        "NX0": NX0,
        "m": m,
        "n": n,
        "E": E,
        "CF": CF,
        "r_star": r_star,
        "include_bp": "BP" in model_type,
        "model_type": model_type,
        "exchange_rate_regime": exchange_rate_regime,
        "monetary_policy": monetary_policy,
    }

    model.set_parameters(parameters)

    # Dynamic title based on model selection
    model_title = get_model_title(model_type)

    # Main tabs with dynamic titles
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            f"📚 {model_title} Overview",
            f"🔍 {model_title} Derivations",
            f"📊 {model_title} Results",
            f"📈 {model_title} Plots",
            f"🔄 {model_title} Policy",
            "📥 Export",
        ]
    )

    with tab1:
        show_overview_equations(model, parameters)

    with tab2:
        show_derivations(model, parameters)

    with tab3:
        show_numeric_results(model, parameters)

    with tab4:
        show_interactive_plots(model, parameters)

    with tab5:
        show_policy_simulation(model, parameters)

    with tab6:
        show_export_options(model, parameters)


def show_overview_equations(model, parameters):
    model_title = get_model_title(parameters["model_type"])
    st.markdown(
        f'<h2 class="section-header">{model_title} Overview</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if "IS" in parameters["model_type"]:
            st.markdown('<div class="equation-box">', unsafe_allow_html=True)
            st.markdown("**Consumption Function:**")
            st.latex(r"C = C_0 + c(Y - T)")
            st.latex(r"T = T_0 + tY")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="equation-box">', unsafe_allow_html=True)
            st.markdown("**Investment Function:**")
            st.latex(r"I = I_0 - br + \alpha Y")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="equation-box">', unsafe_allow_html=True)
            st.markdown("**Government Sector:**")
            st.latex(r"G = \text{exogenous}")
            st.latex(r"T = T_0 + tY")
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if "LM" in parameters["model_type"]:
            st.markdown('<div class="equation-box">', unsafe_allow_html=True)
            st.markdown("**Money Demand:**")
            st.latex(r"L(Y,r) = kY - hr")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="equation-box">', unsafe_allow_html=True)
            st.markdown("**Money Market Equilibrium:**")
            st.latex(r"\frac{M^s}{P} = L(Y,r)")
            st.markdown("</div>", unsafe_allow_html=True)

        if parameters["include_bp"]:
            st.markdown('<div class="equation-box">', unsafe_allow_html=True)
            st.markdown("**External Sector:**")
            st.latex(r"NX = NX_0 - mY + nE")
            st.latex(r"BP: NX + CF(r - r^*) = 0")
            st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced parameter overview with charts
    st.markdown(
        '<h3 class="section-header">Current Parameter Values</h3>',
        unsafe_allow_html=True,
    )

    # Create parameter visualization
    param_names = []
    param_values = []

    if "IS" in parameters["model_type"]:
        param_names.extend(["C₀", "c", "I₀", "b", "α", "G", "T₀", "t"])
        param_values.extend(
            [
                parameters["C0"],
                parameters["c"],
                parameters["I0"],
                parameters["b"],
                parameters["alpha"],
                parameters["G"],
                parameters["T0"],
                parameters["t"],
            ]
        )

    if "LM" in parameters["model_type"]:
        param_names.extend(["k", "h", "Ms", "P"])
        param_values.extend(
            [parameters["k"], parameters["h"], parameters["Ms"], parameters["P"]]
        )

    if parameters["include_bp"]:
        param_names.extend(["NX₀", "m", "n", "E", "CF", "r*"])
        param_values.extend(
            [
                parameters["NX0"],
                parameters["m"],
                parameters["n"],
                parameters["E"],
                parameters["CF"],
                parameters["r_star"],
            ]
        )

    # Parameter bar chart
    if param_names:
        fig_params = go.Figure(
            data=[
                go.Bar(
                    x=param_names,
                    y=param_values,
                    marker_color="lightblue",
                    hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>",
                )
            ]
        )
        fig_params.update_layout(
            title=f"{model_title} Parameter Values",
            xaxis_title="Parameters",
            yaxis_title="Values",
            height=400,
        )
        st.plotly_chart(fig_params, use_container_width=True)

    # Detailed parameter table
    params_df = pd.DataFrame(
        {
            "Parameter": param_names,
            "Value": param_values,
            "Description": [
                "Autonomous Consumption",
                "Marginal Propensity to Consume",
                "Autonomous Investment",
                "Interest Sensitivity of Investment",
                "Income Sensitivity of Investment",
                "Government Spending",
                "Autonomous Taxes",
                "Tax Rate",
                "Income Elasticity of Money Demand",
                "Interest Elasticity of Money Demand",
                "Money Supply",
                "Price Level",
                "Autonomous Net Exports",
                "Import Propensity",
                "Exchange Rate Sensitivity",
                "Exchange Rate",
                "Capital Mobility",
                "Foreign Interest Rate",
            ][: len(param_names)],
        }
    )

    st.dataframe(params_df, use_container_width=True)


def show_derivations(model, parameters):
    model_title = get_model_title(parameters["model_type"])
    st.markdown(
        f'<h2 class="section-header">{model_title} Symbolic Derivations</h2>',
        unsafe_allow_html=True,
    )

    # IS Curve Derivation
    if "IS" in parameters["model_type"]:
        st.markdown(
            '<h3 class="section-header">IS Curve Derivation</h3>',
            unsafe_allow_html=True,
        )

        with st.expander("Step-by-Step IS Derivation", expanded=True):
            st.markdown("**Step 1: Goods Market Equilibrium**")
            st.latex(r"Y = C + I + G + NX")

            st.markdown("**Step 2: Substitute Functions**")
            C_expr = model.consumption_function()
            I_expr = model.investment_function()

            st.write("Consumption:")
            st.latex(f"C = {latex(C_expr)}")

            st.write("Investment:")
            st.latex(f"I = {latex(I_expr)}")

            if parameters["include_bp"]:
                NX_expr = model.net_exports()
                st.write("Net Exports:")
                st.latex(f"NX = {latex(NX_expr)}")

            st.markdown("**Step 3: Derive IS Equation**")
            IS_eq, r_IS, Y_IS = model.derive_IS_curve()

            if r_IS is not None:
                st.write("IS curve (r as function of Y):")
                st.latex(f"r = {latex(r_IS)}")

            if Y_IS is not None:
                st.write("IS curve (Y as function of r):")
                st.latex(f"Y = {latex(Y_IS)}")

    # LM Curve Derivation
    if "LM" in parameters["model_type"]:
        st.markdown(
            '<h3 class="section-header">LM Curve Derivation</h3>',
            unsafe_allow_html=True,
        )

        with st.expander("Step-by-Step LM Derivation", expanded=True):
            st.markdown("**Step 1: Money Market Equilibrium**")
            st.latex(r"\frac{M^s}{P} = L(Y,r)")

            st.markdown("**Step 2: Substitute Money Demand Function**")
            L_expr = model.money_demand()
            st.latex(f"L(Y,r) = {latex(L_expr)}")

            st.markdown("**Step 3: Derive LM Equation**")
            LM_eq, r_LM, Y_LM = model.derive_LM_curve()

            if r_LM is not None:
                st.write("LM curve (r as function of Y):")
                st.latex(f"r = {latex(r_LM)}")

            if Y_LM is not None:
                st.write("LM curve (Y as function of r):")
                st.latex(f"Y = {latex(Y_LM)}")

    # BP Curve Derivation (if applicable)
    if parameters["include_bp"]:
        st.markdown(
            '<h3 class="section-header">BP Curve Derivation</h3>',
            unsafe_allow_html=True,
        )

        with st.expander("Step-by-Step BP Derivation", expanded=True):
            st.markdown("**Step 1: Balance of Payments Equilibrium**")
            st.latex(r"NX + CF = 0")

            st.markdown("**Step 2: Substitute Functions**")
            NX_expr = model.net_exports()
            st.latex(f"NX = {latex(NX_expr)}")
            st.latex(r"CF = CF \cdot (r - r^*)")

            st.markdown("**Step 3: Derive BP Equation**")
            BP_eq, r_BP, Y_BP = model.derive_BP_curve()

            if r_BP is not None:
                st.write("BP curve (r as function of Y):")
                st.latex(f"r = {latex(r_BP)}")


def show_numeric_results(model, parameters):
    model_title = get_model_title(parameters["model_type"])
    st.markdown(
        f'<h2 class="section-header">{model_title} Numerical Results</h2>',
        unsafe_allow_html=True,
    )

    # Solve equilibrium
    solution = model.solve_equilibrium()

    if solution:
        # Safe conversion to float with proper error handling
        try:
            Y_eq = safe_float_conversion(solution[model.Y], parameters)
            r_eq = safe_float_conversion(solution[model.r], parameters)

            # Validate results
            if not (np.isfinite(Y_eq) and np.isfinite(r_eq)):
                st.error(
                    "⚠️ Solution contains infinite or NaN values. Please check parameter values."
                )
                return

        except Exception as e:
            st.error(f"❌ Error converting solution to numerical values: {e}")
            st.info(
                "Please try adjusting parameter values or check for model specification issues."
            )
            return

        # Results visualization
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Equilibrium Income (Y*)", f"{Y_eq:.2f}", f"{Y_eq/1000:.1f}K")
        with col2:
            st.metric(
                "Equilibrium Interest Rate (r*)", f"{r_eq*100:.2f}%", f"{r_eq:.4f}"
            )
        with col3:
            multiplier = Y_eq / parameters["G"] if parameters["G"] != 0 else 0
            st.metric("Income/Gov Ratio", f"{multiplier:.2f}", "Y*/G")

        # Enhanced results visualization
        st.markdown(
            '<h3 class="section-header">Equilibrium Analysis</h3>',
            unsafe_allow_html=True,
        )

        # Create gauge charts for key metrics
        fig_gauges = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=("Income Level", "Interest Rate"),
        )

        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=Y_eq,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Income (Y*)"},
                gauge={
                    "axis": {"range": [None, 3000]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 1000], "color": "lightgray"},
                        {"range": [1000, 2000], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 2500,
                    },
                },
            ),
            row=1,
            col=1,
        )

        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=r_eq * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Interest Rate (r*) %"},
                gauge={
                    "axis": {"range": [None, 25]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 10], "color": "lightgray"},
                        {"range": [10, 20], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 20,
                    },
                },
            ),
            row=1,
            col=2,
        )

        fig_gauges.update_layout(height=300, title_text=f"{model_title} Key Metrics")
        st.plotly_chart(fig_gauges, use_container_width=True)

        # Component analysis
        if "IS" in parameters["model_type"]:
            st.markdown(
                '<h3 class="section-header">Aggregate Demand Components</h3>',
                unsafe_allow_html=True,
            )

            T_eq = parameters["T0"] + parameters["t"] * Y_eq
            C_eq = parameters["C0"] + parameters["c"] * (Y_eq - T_eq)
            I_eq = (
                parameters["I0"] - parameters["b"] * r_eq + parameters["alpha"] * Y_eq
            )
            G_eq = parameters["G"]
            NX_eq = (
                parameters["NX0"]
                - parameters["m"] * Y_eq
                + parameters["n"] * parameters["E"]
                if parameters["include_bp"]
                else 0
            )

            # Component comparison chart
            components = ["Consumption", "Investment", "Government"]
            values = [C_eq, I_eq, G_eq]
            if parameters["include_bp"]:
                components.append("Net Exports")
                values.append(NX_eq)

            fig_components = go.Figure()
            fig_components.add_trace(
                go.Bar(
                    x=components,
                    y=values,
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][
                        : len(components)
                    ],
                    hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>",
                )
            )
            fig_components.update_layout(
                title=f"{model_title} - Aggregate Demand Components",
                xaxis_title="Components",
                yaxis_title="Value",
                height=400,
            )
            st.plotly_chart(fig_components, use_container_width=True)

            # Detailed components table
            components_df = pd.DataFrame(
                {
                    "Component": [
                        "Consumption (C)",
                        "Investment (I)",
                        "Government (G)",
                        "Net Exports (NX)",
                        "Total (Y)",
                    ],
                    "Value": [C_eq, I_eq, G_eq, NX_eq, Y_eq],
                    "Share %": [
                        C_eq / Y_eq * 100,
                        I_eq / Y_eq * 100,
                        G_eq / Y_eq * 100,
                        NX_eq / Y_eq * 100 if Y_eq != 0 else 0,
                        100,
                    ],
                    "Formula": [
                        f"C₀ + c(Y - T) = {parameters['C0']:.0f} + {parameters['c']:.2f}({Y_eq:.2f} - {T_eq:.2f})",
                        f"I₀ - br + αY = {parameters['I0']:.0f} - {parameters['b']:.0f}×{r_eq:.4f} + {parameters['alpha']:.2f}×{Y_eq:.2f}",
                        f"G = {parameters['G']:.0f}",
                        (
                            f"NX₀ - mY + nE = {parameters['NX0']:.0f} - {parameters['m']:.2f}×{Y_eq:.2f} + {parameters['n']:.2f}×{parameters['E']:.2f}"
                            if parameters["include_bp"]
                            else "Not included"
                        ),
                        f"C + I + G + NX = {C_eq + I_eq + G_eq + NX_eq:.2f}",
                    ],
                }
            )

            st.dataframe(components_df, use_container_width=True)

        # Money market verification
        if "LM" in parameters["model_type"]:
            st.markdown(
                '<h3 class="section-header">Money Market Verification</h3>',
                unsafe_allow_html=True,
            )

            L_eq = parameters["k"] * Y_eq - parameters["h"] * r_eq
            Ms_P = parameters["Ms"] / parameters["P"]

            # Money market chart
            fig_money = go.Figure()
            fig_money.add_trace(
                go.Bar(
                    x=["Money Demand (L)", "Real Money Supply (Ms/P)"],
                    y=[L_eq, Ms_P],
                    marker_color=["#ff7f0e", "#2ca02c"],
                    hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>",
                )
            )
            fig_money.update_layout(
                title=f"{model_title} - Money Market Balance",
                xaxis_title="Money Market Components",
                yaxis_title="Value",
                height=300,
            )
            st.plotly_chart(fig_money, use_container_width=True)

            verification_df = pd.DataFrame(
                {
                    "Variable": [
                        "Money Demand (L)",
                        "Real Money Supply (Ms/P)",
                        "Difference",
                    ],
                    "Value": [L_eq, Ms_P, abs(L_eq - Ms_P)],
                    "Formula": [
                        f"kY - hr = {parameters['k']:.2f}×{Y_eq:.2f} - {parameters['h']:.0f}×{r_eq:.4f}",
                        f"Ms/P = {parameters['Ms']:.0f}/{parameters['P']:.1f}",
                        f"|L - Ms/P| = |{L_eq:.2f} - {Ms_P:.2f}|",
                    ],
                }
            )

            st.dataframe(verification_df, use_container_width=True)

            if abs(L_eq - Ms_P) > 1e-3:
                st.warning(
                    "⚠️ Money market equilibrium may not be exact due to numerical approximation."
                )

    else:
        st.error("❌ Unable to solve for equilibrium. Please check parameter values.")


def show_interactive_plots(model, parameters):
    model_title = get_model_title(parameters["model_type"])
    st.markdown(
        f'<h2 class="section-header">{model_title} Interactive Plots</h2>',
        unsafe_allow_html=True,
    )

    # Get equilibrium first for plot centering and intersection visualization
    solution = model.solve_equilibrium()
    Y_eq, r_eq = None, None

    if solution:
        try:
            Y_eq = safe_float_conversion(solution[model.Y], parameters)
            r_eq = safe_float_conversion(solution[model.r], parameters)
        except:
            pass

    # Enhanced equilibrium info panel with intersection details
    if (
        Y_eq is not None
        and r_eq is not None
        and np.isfinite(Y_eq)
        and np.isfinite(r_eq)
    ):
        curves_intersecting = []
        if "IS" in parameters["model_type"]:
            curves_intersecting.append("IS")
        if "LM" in parameters["model_type"]:
            curves_intersecting.append("LM")
        if parameters["include_bp"]:
            curves_intersecting.append("BP")

        intersection_text = " ∩ ".join(curves_intersecting)

        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; box-shadow: 0 8px 15px rgba(0,0,0,0.2);">
            <h3 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">⚡ EQUILIBRIUM INTERSECTION</h3>
            <p style="margin: 8px 0 0 0; color: white; font-size: 16px; font-weight: bold;">{intersection_text} Intersection Point</p>
            <p style="margin: 5px 0 0 0; color: white; font-size: 20px; font-weight: bold;">Y* = {Y_eq:.0f} | r* = {r_eq*100:.2f}%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Create Y and r ranges for plotting - ENHANCED: centered on equilibrium
    if Y_eq is not None and r_eq is not None:
        y_padding = max(Y_eq * 0.4, 500)
        r_padding = max(r_eq * 0.4, 0.05)
        Y_range = np.linspace(max(50, Y_eq - y_padding), Y_eq + y_padding, 200)
        r_range = np.linspace(max(-0.1, r_eq - r_padding), r_eq + r_padding, 200)
        x_range = [max(0, Y_eq - y_padding), Y_eq + y_padding]
        y_range = [max(-0.1, r_eq - r_padding), r_eq + r_padding]
    else:
        Y_range = np.linspace(50, 3000, 200)
        r_range = np.linspace(-0.1, 0.5, 200)
        x_range = [0, 3000]
        y_range = [-0.1, 0.5]

    # Control panel for intersection visualization
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("🎯 Focus on Intersection", key="focus_intersection"):
            st.rerun()
    with col2:
        show_crosshairs = st.checkbox("Show Crosshairs", value=True, key="crosshairs")
    with col3:
        show_intersection_details = st.checkbox(
            "Intersection Details", value=True, key="details"
        )
    with col4:
        highlight_curves = st.checkbox(
            "Highlight at Intersection", value=True, key="highlight"
        )

    # Main IS/LM/BP diagram with enhanced intersection visualization
    st.markdown(
        '<h3 class="section-header">Equilibrium Intersection Diagram</h3>',
        unsafe_allow_html=True,
    )

    # Get curve functions
    IS_eq, r_IS, Y_IS = model.derive_IS_curve()
    LM_eq, r_LM, Y_LM = model.derive_LM_curve()

    fig = go.Figure()

    # Plot IS curve with enhanced styling
    if "IS" in parameters["model_type"] and r_IS is not None:
        try:
            # Substitute parameters and evaluate
            r_IS_func = r_IS.subs(
                [
                    (model.C0, parameters["C0"]),
                    (model.c, parameters["c"]),
                    (model.I0, parameters["I0"]),
                    (model.b, parameters["b"]),
                    (model.alpha, parameters["alpha"]),
                    (model.G, parameters["G"]),
                    (model.T0, parameters["T0"]),
                    (model.t, parameters["t"]),
                ]
            )

            if parameters["include_bp"]:
                r_IS_func = r_IS_func.subs(
                    [
                        (model.NX0, parameters["NX0"]),
                        (model.m, parameters["m"]),
                        (model.n, parameters["n"]),
                        (model.E, parameters["E"]),
                    ]
                )

            r_IS_values = []
            for y in Y_range:
                try:
                    r_val = safe_float_conversion(
                        r_IS_func.subs(model.Y, y), parameters
                    )
                    if np.isfinite(r_val) and -1 <= r_val <= 1:
                        r_IS_values.append(r_val)
                    else:
                        r_IS_values.append(None)
                except:
                    r_IS_values.append(None)

            # Filter out None values
            valid_indices = [i for i, r in enumerate(r_IS_values) if r is not None]
            if valid_indices:
                Y_valid = [Y_range[i] for i in valid_indices]
                r_valid = [r_IS_values[i] for i in valid_indices]

                # Enhanced IS curve with gradient effect near intersection
                line_width = 6 if highlight_curves else 4
                fig.add_trace(
                    go.Scatter(
                        x=Y_valid,
                        y=r_valid,
                        mode="lines",
                        name="IS Curve",
                        line=dict(color="blue", width=line_width),
                        hovertemplate="<b>IS Curve</b><br>Y=%{x:.0f}<br>r=%{y:.4f}<extra></extra>",
                    )
                )
        except Exception as e:
            st.warning(f"Could not plot IS curve: {str(e)}")

    # Plot LM curve with enhanced styling
    if "LM" in parameters["model_type"] and r_LM is not None:
        try:
            r_LM_func = r_LM.subs(
                [
                    (model.k, parameters["k"]),
                    (model.h, parameters["h"]),
                    (model.Ms, parameters["Ms"]),
                    (model.P, parameters["P"]),
                ]
            )

            r_LM_values = []
            for y in Y_range:
                try:
                    r_val = safe_float_conversion(
                        r_LM_func.subs(model.Y, y), parameters
                    )
                    if np.isfinite(r_val) and -1 <= r_val <= 1:
                        r_LM_values.append(r_val)
                    else:
                        r_LM_values.append(None)
                except:
                    r_LM_values.append(None)

            # Filter out None values
            valid_indices = [i for i, r in enumerate(r_LM_values) if r is not None]
            if valid_indices:
                Y_valid = [Y_range[i] for i in valid_indices]
                r_valid = [r_LM_values[i] for i in valid_indices]

                line_width = 6 if highlight_curves else 4
                fig.add_trace(
                    go.Scatter(
                        x=Y_valid,
                        y=r_valid,
                        mode="lines",
                        name="LM Curve",
                        line=dict(color="red", width=line_width),
                        hovertemplate="<b>LM Curve</b><br>Y=%{x:.0f}<br>r=%{y:.4f}<extra></extra>",
                    )
                )
        except Exception as e:
            st.warning(f"Could not plot LM curve: {str(e)}")

    # Plot BP curve if applicable
    if parameters["include_bp"]:
        BP_eq, r_BP, Y_BP = model.derive_BP_curve()
        if r_BP is not None:
            try:
                r_BP_func = r_BP.subs(
                    [
                        (model.NX0, parameters["NX0"]),
                        (model.m, parameters["m"]),
                        (model.n, parameters["n"]),
                        (model.E, parameters["E"]),
                        (model.CF, parameters["CF"]),
                        (model.r_star, parameters["r_star"]),
                    ]
                )

                r_BP_values = []
                for y in Y_range:
                    try:
                        r_val = safe_float_conversion(
                            r_BP_func.subs(model.Y, y), parameters
                        )
                        if np.isfinite(r_val) and -1 <= r_val <= 1:
                            r_BP_values.append(r_val)
                        else:
                            r_BP_values.append(None)
                    except:
                        r_BP_values.append(None)

                # Filter out None values
                valid_indices = [i for i, r in enumerate(r_BP_values) if r is not None]
                if valid_indices:
                    Y_valid = [Y_range[i] for i in valid_indices]
                    r_valid = [r_BP_values[i] for i in valid_indices]

                    line_width = 6 if highlight_curves else 4
                    fig.add_trace(
                        go.Scatter(
                            x=Y_valid,
                            y=r_valid,
                            mode="lines",
                            name="BP Curve",
                            line=dict(color="green", width=line_width, dash="dash"),
                            hovertemplate="<b>BP Curve</b><br>Y=%{x:.0f}<br>r=%{y:.4f}<extra></extra>",
                        )
                    )
            except Exception as e:
                st.warning(f"Could not plot BP curve: {str(e)}")

    # ENHANCED EQUILIBRIUM INTERSECTION VISUALIZATION
    if (
        Y_eq is not None
        and r_eq is not None
        and np.isfinite(Y_eq)
        and np.isfinite(r_eq)
    ):

        # 1. Large crosshairs spanning the entire plot area
        if show_crosshairs:
            # Horizontal crosshair (interest rate line)
            fig.add_shape(
                type="line",
                x0=x_range[0],
                y0=r_eq,
                x1=x_range[1],
                y1=r_eq,
                line=dict(color="rgba(255,0,0,0.8)", width=2, dash="dot"),
                layer="below",
            )

            # Vertical crosshair (income line)
            fig.add_shape(
                type="line",
                x0=Y_eq,
                y0=y_range[0],
                x1=Y_eq,
                y1=y_range[1],
                line=dict(color="rgba(255,0,0,0.8)", width=2, dash="dot"),
                layer="below",
            )

        # 2. Multi-layer intersection point marker
        # Outer glow effect
        fig.add_trace(
            go.Scatter(
                x=[Y_eq],
                y=[r_eq],
                mode="markers",
                showlegend=False,
                marker=dict(color="rgba(255, 215, 0, 0.4)", size=50, symbol="circle"),
                hoverinfo="skip",
            )
        )

        # Middle ring
        fig.add_trace(
            go.Scatter(
                x=[Y_eq],
                y=[r_eq],
                mode="markers",
                showlegend=False,
                marker=dict(color="rgba(255, 0, 0, 0.6)", size=30, symbol="circle"),
                hoverinfo="skip",
            )
        )

        # Main intersection point - enhanced star
        fig.add_trace(
            go.Scatter(
                x=[Y_eq],
                y=[r_eq],
                mode="markers",
                name="⚡ EQUILIBRIUM INTERSECTION",
                marker=dict(
                    color="black",
                    size=25,
                    symbol="star",
                    line=dict(color="white", width=4),
                ),
                hovertemplate="<b>🎯 EQUILIBRIUM INTERSECTION</b><br>Y* = %{x:.2f}<br>r* = %{y:.4f}<br>Curves: "
                + intersection_text
                + "<extra></extra>",
            )
        )

        # 3. Intersection details annotation
        if show_intersection_details:
            # Calculate distances from curves to show intersection accuracy
            intersection_info = f"<b>INTERSECTION DETAILS</b><br>"
            intersection_info += f"Point: ({Y_eq:.1f}, {r_eq*100:.2f}%)<br>"
            intersection_info += f"Curves: {intersection_text}<br>"
            intersection_info += f"Type: {'Simultaneous' if len(curves_intersecting) > 1 else 'Single'} Equilibrium"

            fig.add_annotation(
                x=Y_eq,
                y=r_eq,
                text=intersection_info,
                showarrow=True,
                arrowhead=3,
                arrowsize=2,
                arrowwidth=3,
                arrowcolor="darkred",
                ax=60,
                ay=-60,  # Arrow offset
                bordercolor="darkred",
                borderwidth=3,
                bgcolor="rgba(255,255,255,0.95)",
                font=dict(size=12, color="darkred", family="Arial Black"),
            )

        # 4. Coordinate labels on axes
        # Y-axis label
        fig.add_annotation(
            x=Y_eq,
            y=y_range[0] - (y_range[1] - y_range[0]) * 0.05,
            text=f"Y* = {Y_eq:.0f}",
            showarrow=False,
            font=dict(size=12, color="red", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
        )

        # X-axis label (interest rate)
        fig.add_annotation(
            x=x_range[0] - (x_range[1] - x_range[0]) * 0.05,
            y=r_eq,
            text=f"r* = {r_eq*100:.1f}%",
            showarrow=False,
            font=dict(size=12, color="red", family="Arial Black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
        )

        # 5. Highlight intersection region with subtle background
        fig.add_shape(
            type="rect",
            x0=Y_eq - (x_range[1] - x_range[0]) * 0.05,
            y0=r_eq - (y_range[1] - y_range[0]) * 0.05,
            x1=Y_eq + (x_range[1] - x_range[0]) * 0.05,
            y1=r_eq + (y_range[1] - y_range[0]) * 0.05,
            fillcolor="rgba(255,215,0,0.1)",
            line=dict(color="rgba(255,215,0,0.3)", width=1),
            layer="below",
        )

    # Enhanced layout with intersection focus
    fig.update_layout(
        title=dict(
            text=f"{model_title} - Equilibrium Intersection Analysis",
            font=dict(size=18, family="Arial Black"),
        ),
        xaxis_title="Income (Y)",
        yaxis_title="Interest Rate (r)",
        hovermode="closest",
        showlegend=True,
        height=700,  # Increased height for better visibility
        xaxis=dict(
            range=x_range,
            gridcolor="lightgray",
            autorange=False,
            title_font=dict(size=14, family="Arial Black"),
        ),
        yaxis=dict(
            range=y_range,
            gridcolor="lightgray",
            autorange=False,
            title_font=dict(size=14, family="Arial Black"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="rgba(250,250,250,1)",
        uirevision="equilibrium_intersection",  # Maintain zoom state
    )

    st.plotly_chart(fig, use_container_width=True)

    # Intersection Analysis Summary
    if Y_eq is not None and r_eq is not None:
        st.markdown("### 🔍 Intersection Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            **📊 Equilibrium Coordinates**
            - **Income (Y*)**: {Y_eq:.2f}
            - **Interest Rate (r*)**: {r_eq*100:.2f}%
            - **Intersection Type**: {len(curves_intersecting)}-curve equilibrium
            """
            )

        with col2:
            st.markdown(
                f"""
            **🎯 Curves at Intersection**
            - **Curves Meeting**: {', '.join(curves_intersecting)}
            - **Market Equilibria**: {len(curves_intersecting)} simultaneous
            - **Stability**: {"High" if len(curves_intersecting) >= 2 else "Medium"}
            """
            )

        with col3:
            # Show economic interpretation
            interpretation = ""
            if "IS" in curves_intersecting and "LM" in curves_intersecting:
                interpretation += "**Goods & Money Market Balance**\n"
            if "BP" in curves_intersecting:
                interpretation += "**External Balance Achieved**\n"

            interpretation += f"At the intersection:\n"
            interpretation += f"- Aggregate demand = {Y_eq:.0f}\n"
            interpretation += f"- Money market clears at r = {r_eq*100:.2f}%"

            st.markdown(interpretation)

    # Interactive controls explanation
    with st.expander("🎛️ Intersection Visualization Controls", expanded=False):
        st.markdown(
            """
        **Available Controls:**
        - **🎯 Focus on Intersection**: Centers the plot on equilibrium point
        - **Show Crosshairs**: Displays reference lines through intersection  
        - **Intersection Details**: Shows detailed annotation at equilibrium
        - **Highlight at Intersection**: Enhances curve visibility near equilibrium
        
        **Intersection Features:**
        - **Multi-layer markers**: Glow effect, ring, and star symbol
        - **Crosshair lines**: Dotted reference lines for exact coordinates
        - **Coordinate labels**: Precise Y* and r* values on axes
        - **Background highlight**: Subtle region emphasis around intersection
        - **Detailed hover**: Comprehensive information on hover
        """
        )

    # Additional plots section
    st.markdown(
        '<h3 class="section-header">Additional Analysis Plots</h3>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Parameter sensitivity plot
        if solution:
            st.markdown("**Parameter Sensitivity Analysis**")
            try:
                Y_eq = safe_float_conversion(solution[model.Y], parameters)
                r_eq = safe_float_conversion(solution[model.r], parameters)

                # Test parameter variations
                param_variations = np.linspace(0.5, 1.5, 11)  # ±50% variation
                sensitivity_results = []

                test_param = "G" if "IS" in parameters["model_type"] else "Ms"
                base_value = parameters[test_param]

                for multiplier in param_variations:
                    test_params = parameters.copy()
                    test_params[test_param] = base_value * multiplier

                    test_model = ISLMBPModel()
                    test_model.set_parameters(test_params)
                    test_solution = test_model.solve_equilibrium()

                    if test_solution:
                        try:
                            Y_test = safe_float_conversion(
                                test_solution[model.Y], test_params
                            )
                            r_test = safe_float_conversion(
                                test_solution[model.r], test_params
                            )
                            if np.isfinite(Y_test) and np.isfinite(r_test):
                                sensitivity_results.append(
                                    {
                                        "multiplier": multiplier,
                                        "Y": Y_test,
                                        "r": r_test,
                                        "param_value": base_value * multiplier,
                                    }
                                )
                        except:
                            pass

                if sensitivity_results:
                    df_sens = pd.DataFrame(sensitivity_results)

                    fig_sens = go.Figure()
                    fig_sens.add_trace(
                        go.Scatter(
                            x=df_sens["param_value"],
                            y=df_sens["Y"],
                            mode="lines+markers",
                            name="Income (Y)",
                            line=dict(color="blue"),
                            yaxis="y",
                        )
                    )

                    fig_sens.add_trace(
                        go.Scatter(
                            x=df_sens["param_value"],
                            y=df_sens["r"] * 100,
                            mode="lines+markers",
                            name="Interest Rate (r%)",
                            line=dict(color="red"),
                            yaxis="y2",
                        )
                    )

                    fig_sens.update_layout(
                        title=f"Sensitivity to {test_param}",
                        xaxis_title=f"{test_param} Value",
                        yaxis=dict(title="Income (Y)", side="left"),
                        yaxis2=dict(
                            title="Interest Rate (%)", side="right", overlaying="y"
                        ),
                        height=400,
                    )

                    st.plotly_chart(fig_sens, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not create sensitivity plot: {str(e)}")

    with col2:
        # 3D surface plot for two-parameter analysis
        if (
            solution
            and "IS" in parameters["model_type"]
            and "LM" in parameters["model_type"]
        ):
            st.markdown("**Two-Parameter Surface Analysis**")
            try:
                # Create parameter grids
                g_range = np.linspace(parameters["G"] * 0.5, parameters["G"] * 1.5, 10)
                ms_range = np.linspace(
                    parameters["Ms"] * 0.5, parameters["Ms"] * 1.5, 10
                )

                G_grid, Ms_grid = np.meshgrid(g_range, ms_range)
                Y_surface = np.zeros_like(G_grid)
                r_surface = np.zeros_like(G_grid)

                for i in range(len(g_range)):
                    for j in range(len(ms_range)):
                        test_params = parameters.copy()
                        test_params["G"] = G_grid[j, i]
                        test_params["Ms"] = Ms_grid[j, i]

                        test_model = ISLMBPModel()
                        test_model.set_parameters(test_params)
                        test_solution = test_model.solve_equilibrium()

                        if test_solution:
                            try:
                                Y_test = safe_float_conversion(
                                    test_solution[model.Y], test_params
                                )
                                r_test = safe_float_conversion(
                                    test_solution[model.r], test_params
                                )
                                if np.isfinite(Y_test) and np.isfinite(r_test):
                                    Y_surface[j, i] = Y_test
                                    r_surface[j, i] = r_test
                                else:
                                    Y_surface[j, i] = np.nan
                                    r_surface[j, i] = np.nan
                            except:
                                Y_surface[j, i] = np.nan
                                r_surface[j, i] = np.nan
                        else:
                            Y_surface[j, i] = np.nan
                            r_surface[j, i] = np.nan

                fig_3d = go.Figure(
                    data=[
                        go.Surface(
                            x=G_grid,
                            y=Ms_grid,
                            z=Y_surface,
                            colorscale="Viridis",
                            hovertemplate="<b>G=%{x:.0f}</b><br>Ms=%{y:.0f}<br>Y=%{z:.2f}<extra></extra>",
                        )
                    ]
                )

                fig_3d.update_layout(
                    title="Income Surface (G vs Ms)",
                    scene=dict(
                        xaxis_title="Government Spending (G)",
                        yaxis_title="Money Supply (Ms)",
                        zaxis_title="Income (Y)",
                    ),
                    height=400,
                )

                st.plotly_chart(fig_3d, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not create surface plot: {str(e)}")

    # Multiplier analysis chart
    if "IS" in parameters["model_type"] and solution:
        st.markdown(
            '<h3 class="section-header">Multiplier Analysis</h3>',
            unsafe_allow_html=True,
        )

        try:
            Y_eq = safe_float_conversion(solution[model.Y], parameters)

            # Calculate theoretical multipliers
            mpc = parameters["c"]
            mpm = parameters["m"] if parameters["include_bp"] else 0
            tax_rate = parameters["t"]

            simple_multiplier = 1 / (1 - mpc)
            tax_multiplier = 1 / (1 - mpc * (1 - tax_rate))
            open_multiplier = (
                1 / (1 - mpc * (1 - tax_rate) + mpm)
                if parameters["include_bp"]
                else tax_multiplier
            )

            multipliers = {
                "Simple Multiplier": simple_multiplier,
                "Tax-Adjusted Multiplier": tax_multiplier,
                "Open Economy Multiplier": (
                    open_multiplier if parameters["include_bp"] else tax_multiplier
                ),
            }

            # Actual government spending multiplier
            actual_multiplier = Y_eq / parameters["G"] if parameters["G"] != 0 else 0
            multipliers["Actual Y/G Ratio"] = actual_multiplier

            fig_mult = go.Figure(
                [
                    go.Bar(
                        x=list(multipliers.keys()),
                        y=list(multipliers.values()),
                        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                    )
                ]
            )

            fig_mult.update_layout(
                title=f"{model_title} - Multiplier Analysis",
                xaxis_title="Multiplier Type",
                yaxis_title="Multiplier Value",
                height=400,
            )

            st.plotly_chart(fig_mult, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not create multiplier analysis: {str(e)}")


def show_policy_simulation(model, parameters):
    model_title = get_model_title(parameters["model_type"])
    st.markdown(
        f'<h2 class="section-header">{model_title} Policy Simulation</h2>',
        unsafe_allow_html=True,
    )

    st.markdown("Select a policy shock to simulate its effects on the economy:")

    col1, col2 = st.columns(2)

    with col1:
        policy_type = st.selectbox(
            "Policy Type", ["Fiscal Policy", "Monetary Policy", "External Shock"]
        )

        if policy_type == "Fiscal Policy":
            shock_var = st.selectbox(
                "Variable",
                ["Government Spending (G)", "Tax Rate (t)", "Autonomous Taxes (T₀)"],
            )
            shock_size = st.slider("Shock Size", -500, 500, 100, 25)
        elif policy_type == "Monetary Policy":
            shock_var = st.selectbox(
                "Variable", ["Money Supply (Ms)", "Price Level (P)"]
            )
            shock_size = st.slider("Shock Size", -1000, 1000, 200, 50)
        else:  # External Shock
            shock_var = st.selectbox(
                "Variable",
                [
                    "Foreign Interest Rate (r*)",
                    "Exchange Rate (E)",
                    "Export Demand (NX₀)",
                ],
            )
            shock_size = st.slider("Shock Size", -100, 100, 20, 10)

    with col2:
        if st.button("Run Simulation", type="primary"):
            # Baseline equilibrium
            baseline_solution = model.solve_equilibrium()

            if baseline_solution:
                try:
                    Y_baseline = safe_float_conversion(
                        baseline_solution[model.Y], parameters
                    )
                    r_baseline = safe_float_conversion(
                        baseline_solution[model.r], parameters
                    )

                    if not (np.isfinite(Y_baseline) and np.isfinite(r_baseline)):
                        st.error("❌ Invalid baseline equilibrium values")
                        return

                    # Shocked parameters
                    shocked_params = parameters.copy()

                    if shock_var == "Government Spending (G)":
                        shocked_params["G"] += shock_size
                    elif shock_var == "Tax Rate (t)":
                        shocked_params["t"] += shock_size / 100
                    elif shock_var == "Autonomous Taxes (T₀)":
                        shocked_params["T0"] += shock_size
                    elif shock_var == "Money Supply (Ms)":
                        shocked_params["Ms"] += shock_size
                    elif shock_var == "Price Level (P)":
                        shocked_params["P"] += shock_size / 100
                    elif shock_var == "Foreign Interest Rate (r*)":
                        shocked_params["r_star"] += shock_size / 100
                    elif shock_var == "Exchange Rate (E)":
                        shocked_params["E"] += shock_size / 100
                    elif shock_var == "Export Demand (NX₀)":
                        shocked_params["NX0"] += shock_size

                    # Create shocked model
                    shocked_model = ISLMBPModel()
                    shocked_model.set_parameters(shocked_params)
                    shocked_solution = shocked_model.solve_equilibrium()

                    if shocked_solution:
                        Y_shocked = safe_float_conversion(
                            shocked_solution[model.Y], shocked_params
                        )
                        r_shocked = safe_float_conversion(
                            shocked_solution[model.r], shocked_params
                        )

                        if not (np.isfinite(Y_shocked) and np.isfinite(r_shocked)):
                            st.error("❌ Invalid shocked equilibrium values")
                            return

                        # Enhanced results visualization
                        st.markdown("### 📊 Simulation Results")

                        # Before/After comparison chart
                        fig_comparison = go.Figure()

                        categories = ["Income (Y)", "Interest Rate (r×100)"]
                        baseline_vals = [Y_baseline, r_baseline * 100]
                        shocked_vals = [Y_shocked, r_shocked * 100]

                        fig_comparison.add_trace(
                            go.Bar(
                                name="Baseline",
                                x=categories,
                                y=baseline_vals,
                                marker_color="lightblue",
                            )
                        )

                        fig_comparison.add_trace(
                            go.Bar(
                                name="After Shock",
                                x=categories,
                                y=shocked_vals,
                                marker_color="salmon",
                            )
                        )

                        fig_comparison.update_layout(
                            title=f"{model_title} - Policy Simulation Results",
                            yaxis_title="Values",
                            barmode="group",
                            height=400,
                        )

                        st.plotly_chart(fig_comparison, use_container_width=True)

                        # Detailed results table
                        results_df = pd.DataFrame(
                            {
                                "Variable": ["Income (Y)", "Interest Rate (r)"],
                                "Baseline": [
                                    f"{Y_baseline:.2f}",
                                    f"{r_baseline*100:.2f}%",
                                ],
                                "After Shock": [
                                    f"{Y_shocked:.2f}",
                                    f"{r_shocked*100:.2f}%",
                                ],
                                "Change": [
                                    f"{Y_shocked - Y_baseline:+.2f}",
                                    f"{(r_shocked - r_baseline)*100:+.2f}pp",
                                ],
                                "Percent Change": [
                                    (
                                        f"{((Y_shocked - Y_baseline)/Y_baseline)*100:+.1f}%"
                                        if Y_baseline != 0
                                        else "N/A"
                                    ),
                                    (
                                        f"{((r_shocked - r_baseline)/r_baseline)*100:+.1f}%"
                                        if r_baseline != 0
                                        else "N/A"
                                    ),
                                ],
                            }
                        )

                        st.dataframe(results_df, use_container_width=True)

                        # Economic interpretation
                        st.markdown(
                            '<h3 class="section-header">Economic Interpretation</h3>',
                            unsafe_allow_html=True,
                        )

                        Y_change = Y_shocked - Y_baseline
                        r_change = r_shocked - r_baseline

                        interpretation = generate_economic_interpretation(
                            shock_var, shock_size, Y_change, r_change, parameters
                        )
                        st.write(interpretation)

                        # Dynamic adjustment path visualization
                        st.markdown(
                            '<h3 class="section-header">Dynamic Adjustment Path</h3>',
                            unsafe_allow_html=True,
                        )

                        # Simulate adjustment path (simplified)
                        periods = np.arange(0, 11)
                        adjustment_speed = 0.3  # Adjustment coefficient

                        Y_path = Y_baseline + (Y_change) * (
                            1 - np.exp(-adjustment_speed * periods)
                        )
                        r_path = r_baseline + (r_change) * (
                            1 - np.exp(-adjustment_speed * periods)
                        )

                        fig_path = go.Figure()

                        fig_path.add_trace(
                            go.Scatter(
                                x=periods,
                                y=Y_path,
                                mode="lines+markers",
                                name="Income (Y)",
                                line=dict(color="blue"),
                                yaxis="y",
                            )
                        )

                        fig_path.add_trace(
                            go.Scatter(
                                x=periods,
                                y=r_path * 100,
                                mode="lines+markers",
                                name="Interest Rate (r%)",
                                line=dict(color="red"),
                                yaxis="y2",
                            )
                        )

                        fig_path.update_layout(
                            title=f"{model_title} - Dynamic Adjustment Path",
                            xaxis_title="Time Periods",
                            yaxis=dict(title="Income (Y)", side="left"),
                            yaxis2=dict(
                                title="Interest Rate (%)", side="right", overlaying="y"
                            ),
                            height=400,
                        )

                        st.plotly_chart(fig_path, use_container_width=True)

                        # Multiplier calculation for fiscal policy
                        if (
                            policy_type == "Fiscal Policy"
                            and shock_var == "Government Spending (G)"
                            and shock_size != 0
                        ):
                            multiplier = Y_change / shock_size
                            st.markdown(
                                '<div class="result-box">', unsafe_allow_html=True
                            )
                            st.markdown(
                                f"**Government Spending Multiplier:** {multiplier:.2f}"
                            )
                            st.write(
                                f"A $1 increase in government spending leads to a ${multiplier:.2f} increase in equilibrium income."
                            )
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Multiplier comparison chart
                            theoretical_multiplier = 1 / (
                                1 - parameters["c"] * (1 - parameters["t"])
                            )

                            fig_mult_comp = go.Figure(
                                [
                                    go.Bar(
                                        x=[
                                            "Theoretical Multiplier",
                                            "Actual Multiplier",
                                        ],
                                        y=[theoretical_multiplier, multiplier],
                                        marker_color=["lightgreen", "orange"],
                                    )
                                ]
                            )

                            fig_mult_comp.update_layout(
                                title="Multiplier Comparison",
                                yaxis_title="Multiplier Value",
                                height=300,
                            )

                            st.plotly_chart(fig_mult_comp, use_container_width=True)
                    else:
                        st.error("❌ Could not solve shocked equilibrium")
                except Exception as e:
                    st.error(f"❌ Error in policy simulation: {str(e)}")
            else:
                st.error("❌ Could not solve baseline equilibrium")


def generate_economic_interpretation(
    shock_var, shock_size, Y_change, r_change, parameters
):
    """Generate economic interpretation of policy simulation results"""

    interpretation = f"**Policy Shock Analysis:**\n\n"
    interpretation += f"A {shock_size:+.0f} unit change in {shock_var} resulted in:\n"
    interpretation += (
        f"- Income (Y) changed by {Y_change:+.2f} units ({((Y_change/1000)):.1f}K)\n"
    )
    interpretation += (
        f"- Interest rate (r) changed by {r_change*100:+.2f} percentage points\n\n"
    )

    if "Government Spending" in shock_var:
        if shock_size > 0:
            interpretation += "**Expansionary Fiscal Policy Effect:**\n"
            interpretation += (
                "- Increased government spending shifts the IS curve to the right\n"
            )
            interpretation += "- Higher income increases money demand, putting upward pressure on interest rates\n"
            interpretation += "- The final equilibrium shows higher income and higher interest rates\n"
            interpretation += "- Some private investment may be 'crowded out' by higher interest rates\n"
        else:
            interpretation += "**Contractionary Fiscal Policy Effect:**\n"
            interpretation += (
                "- Decreased government spending shifts the IS curve to the left\n"
            )
            interpretation += "- Lower income reduces money demand, putting downward pressure on interest rates\n"
            interpretation += (
                "- The final equilibrium shows lower income and lower interest rates\n"
            )

    elif "Money Supply" in shock_var:
        if shock_size > 0:
            interpretation += "**Expansionary Monetary Policy Effect:**\n"
            interpretation += (
                "- Increased money supply shifts the LM curve to the right\n"
            )
            interpretation += (
                "- Lower interest rates stimulate investment and aggregate demand\n"
            )
            interpretation += (
                "- The final equilibrium shows higher income and lower interest rates\n"
            )
        else:
            interpretation += "**Contractionary Monetary Policy Effect:**\n"
            interpretation += (
                "- Decreased money supply shifts the LM curve to the left\n"
            )
            interpretation += (
                "- Higher interest rates reduce investment and aggregate demand\n"
            )
            interpretation += (
                "- The final equilibrium shows lower income and higher interest rates\n"
            )

    if parameters["include_bp"]:
        interpretation += "\n**Open Economy Considerations:**\n"
        interpretation += "- Changes in domestic interest rates affect capital flows\n"
        interpretation += (
            "- Exchange rate adjustments may occur depending on the regime\n"
        )
        interpretation += (
            "- The impossible trinity constraints may limit policy effectiveness\n"
        )

        if parameters["exchange_rate_regime"] == "Fixed":
            interpretation += "- Under fixed exchange rates, monetary policy effectiveness is limited\n"
        else:
            interpretation += "- Under flexible exchange rates, fiscal policy effectiveness may be reduced\n"

    return interpretation


def show_export_options(model, parameters):
    st.markdown(
        '<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Excel Export")
        if st.button("Export to Excel", type="primary"):
            try:
                # Create Excel file
                solution = model.solve_equilibrium()

                if solution:
                    Y_eq = safe_float_conversion(solution[model.Y], parameters)
                    r_eq = safe_float_conversion(solution[model.r], parameters)

                    # Create data for export
                    export_data = {
                        "Parameters": list(parameters.keys()),
                        "Values": list(parameters.values()),
                    }

                    results_data = {
                        "Variable": [
                            "Equilibrium Income (Y*)",
                            "Equilibrium Interest Rate (r*)",
                        ],
                        "Value": [Y_eq, r_eq],
                    }

                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        pd.DataFrame(export_data).to_excel(
                            writer, sheet_name="Parameters", index=False
                        )
                        pd.DataFrame(results_data).to_excel(
                            writer, sheet_name="Results", index=False
                        )

                    output.seek(0)

                    st.download_button(
                        label="Download Excel File",
                        data=output.getvalue(),
                        file_name=f"ISLM_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                else:
                    st.error("❌ No solution available for export")
            except Exception as e:
                st.error(f"❌ Error creating Excel file: {str(e)}")

    with col2:
        st.markdown("### 📄 PDF Report")
        if st.button("Generate PDF Report", type="primary"):
            try:
                # Create HTML report
                html_content = generate_html_report(model, parameters)

                st.download_button(
                    label="Download HTML Report",
                    data=html_content,
                    file_name=f"ISLM_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                )
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")

    st.markdown("### 📈 Data Export")

    # Export parameter sensitivity analysis
    if st.button("Export Sensitivity Analysis"):
        try:
            sensitivity_data = perform_sensitivity_analysis(model, parameters)

            if not sensitivity_data.empty:
                csv_data = sensitivity_data.to_csv(index=False)
                st.download_button(
                    label="Download Sensitivity Analysis (CSV)",
                    data=csv_data,
                    file_name=f"ISLM_Sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("⚠️ No sensitivity data available")
        except Exception as e:
            st.error(f"❌ Error performing sensitivity analysis: {str(e)}")


def generate_html_report(model, parameters):
    """Generate HTML report"""
    try:
        solution = model.solve_equilibrium()
        model_title = get_model_title(parameters["model_type"])

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{model_title} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #1f77b4; }}
                .section {{ margin: 20px 0; }}
                .parameters {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .results {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .footer {{ text-align: center; margin-top: 50px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <h1 class="header">{model_title} Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Model Configuration</h2>
                <div class="parameters">
                    <p><strong>Model Type:</strong> {parameters['model_type']}</p>
                    <p><strong>Exchange Rate Regime:</strong> {parameters['exchange_rate_regime']}</p>
                    <p><strong>Monetary Policy:</strong> {parameters['monetary_policy']}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Parameter Values</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
                    <tr><td>C₀</td><td>{parameters['C0']}</td><td>Autonomous Consumption</td></tr>
                    <tr><td>c</td><td>{parameters['c']}</td><td>Marginal Propensity to Consume</td></tr>
                    <tr><td>I₀</td><td>{parameters['I0']}</td><td>Autonomous Investment</td></tr>
                    <tr><td>b</td><td>{parameters['b']}</td><td>Interest Sensitivity of Investment</td></tr>
                    <tr><td>G</td><td>{parameters['G']}</td><td>Government Spending</td></tr>
                    <tr><td>Ms</td><td>{parameters['Ms']}</td><td>Money Supply</td></tr>
                    <tr><td>k</td><td>{parameters['k']}</td><td>Income Elasticity of Money Demand</td></tr>
                    <tr><td>h</td><td>{parameters['h']}</td><td>Interest Elasticity of Money Demand</td></tr>
                </table>
            </div>
        """

        if solution:
            Y_eq = safe_float_conversion(solution[model.Y], parameters)
            r_eq = safe_float_conversion(solution[model.r], parameters)

            html_content += f"""
            <div class="section">
                <h2>Equilibrium Results</h2>
                <div class="results">
                    <p><strong>Equilibrium Income (Y*):</strong> {Y_eq:.2f}</p>
                    <p><strong>Equilibrium Interest Rate (r*):</strong> {r_eq*100:.2f}%</p>
                </div>
            </div>
            """

        html_content += f"""
            <div class="section">
                <h2>Model Equations</h2>
                <p><strong>IS Curve:</strong> Y = C + I + G + NX</p>
                <p><strong>LM Curve:</strong> Ms/P = L(Y,r)</p>
                <p><strong>Consumption:</strong> C = C₀ + c(Y - T)</p>
                <p><strong>Investment:</strong> I = I₀ - br + αY</p>
                <p><strong>Money Demand:</strong> L = kY - hr</p>
            </div>
            
            <div class="footer">
                Created by HAMDI Boulanouar
            </div>
        </body>
        </html>
        """

        return html_content
    except Exception as e:
        return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p><div class='footer'>Created by HAMDI Boulanouar</div></body></html>"


def perform_sensitivity_analysis(model, parameters):
    """Perform sensitivity analysis on key parameters"""
    try:
        base_solution = model.solve_equilibrium()

        if not base_solution:
            return pd.DataFrame()

        Y_base = safe_float_conversion(base_solution[model.Y], parameters)
        r_base = safe_float_conversion(base_solution[model.r], parameters)

        if not (np.isfinite(Y_base) and np.isfinite(r_base)):
            return pd.DataFrame()

        sensitivity_results = []

        # Test sensitivity to key parameters
        key_params = ["C0", "c", "I0", "b", "G", "Ms", "k", "h"]

        for param in key_params:
            if param in parameters and parameters[param] != 0:
                # Test +10% change
                test_params = parameters.copy()
                original_value = test_params[param]
                test_params[param] = original_value * 1.1

                test_model = ISLMBPModel()
                test_model.set_parameters(test_params)
                test_solution = test_model.solve_equilibrium()

                if test_solution:
                    Y_test = safe_float_conversion(test_solution[model.Y], test_params)
                    r_test = safe_float_conversion(test_solution[model.r], test_params)

                    if np.isfinite(Y_test) and np.isfinite(r_test):
                        sensitivity_results.append(
                            {
                                "Parameter": param,
                                "Change": "+10%",
                                "Original_Value": original_value,
                                "New_Value": original_value * 1.1,
                                "Y_Change": Y_test - Y_base,
                                "r_Change": r_test - r_base,
                                "Y_Elasticity": (
                                    ((Y_test - Y_base) / Y_base) / 0.1
                                    if Y_base != 0
                                    else 0
                                ),
                                "r_Elasticity": (
                                    ((r_test - r_base) / r_base) / 0.1
                                    if r_base != 0
                                    else 0
                                ),
                            }
                        )

        return pd.DataFrame(sensitivity_results)
    except Exception as e:
        st.error(f"Error in sensitivity analysis: {str(e)}")
        return pd.DataFrame()


def main():
    """Main function to run the Streamlit app"""
    create_main_app()

    # Footer with proper styling
    st.markdown(
        """
    <div style="margin-top: 100px; padding: 20px; text-align: center; border-top: 2px solid #1f77b4; background-color: #f8f9fa;">
        <h4 style="color: #2c3e50; margin: 0;">Created by HAMDI Boulanouar</h4>
        <p style="color: #666; margin: 5px 0 0 0; font-size: 14px;">IS/LM/BP Macroeconomic Models Explorer</p>
        <p style="color: #888; margin: 5px 0 0 0; font-size: 12px;">Enhanced with Equilibrium Intersection Visualization</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
