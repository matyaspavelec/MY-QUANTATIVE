"""
Monte Carlo Simulation of Trading Sequences
Single-page Streamlit web application
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io

st.set_page_config(
    page_title="Monte Carlo Trading Simulator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #1a1a1a;
    color: #e0e0e0;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    color: #ffffff;
    font-weight: 600;
}
.metric-card {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-size: 13px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
}
.metric-value.positive { color: #4ade80; }
.metric-value.negative { color: #f87171; }
.header-container {
    text-align: center;
    padding: 30px 0 10px 0;
}
.header-container h1 {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.header-container p {
    color: #888;
    font-size: 15px;
}
div[data-testid="stFileUploader"] {
    background: #2a2a2a;
    border: 2px dashed #3a3a3a;
    border-radius: 12px;
    padding: 10px;
}
.stSelectbox, .stNumberInput {
    color: #e0e0e0;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stCheckbox"] label {
    color: #cccccc !important;
}
.stButton > button {
    background: #1e3a5f;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 32px;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.3px;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #2a4f7f;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header-container">
    <h1>Monte Carlo Trading Simulator</h1>
    <p>Upload trade P&L data &middot; Shuffle sequences &middot; Visualize equity curves</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


def read_uploaded_file(uploaded_file):
    """Read uploaded file and return a DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            uploaded_file.seek(0)
            # Detect delimiter
            first_line = content.split("\n")[0]
            if "\t" in first_line:
                sep = "\t"
            elif ";" in first_line:
                sep = ";"
            elif "," in first_line:
                sep = ","
            else:
                sep = r"\s+"
            return pd.read_csv(io.StringIO(content), sep=sep, engine="python")
        elif name.endswith((".xls", ".xlsx", ".xlsm")):
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {name}")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def run_monte_carlo(pnl_values, n_simulations):
    """Run Monte Carlo simulations using full random permutations."""
    n_trades = len(pnl_values)
    pnl_array = np.array(pnl_values, dtype=np.float64)

    # Pre-allocate matrix for all equity curves
    all_curves = np.empty((n_simulations, n_trades), dtype=np.float64)

    for i in range(n_simulations):
        shuffled = np.random.permutation(pnl_array)
        all_curves[i] = np.cumsum(shuffled)

    # Original equity curve
    original_curve = np.cumsum(pnl_array)

    # Average curve
    avg_curve = np.mean(all_curves, axis=0)

    return original_curve, all_curves, avg_curve


# --- Controls ---
col_upload, col_settings = st.columns([1, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Trade Data",
        type=["csv", "txt", "xls", "xlsx", "xlsm"],
        help="Supported: .csv, .txt, .xls, .xlsx",
    )

df = None
pnl_column = None

if uploaded_file is not None:
    df = read_uploaded_file(uploaded_file)

with col_settings:
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found in the file.")
            df = None
        else:
            # Auto-detect likely P&L column
            default_idx = 0
            for i, col in enumerate(numeric_cols):
                if any(k in col.lower() for k in ["pnl", "p&l", "profit", "pl", "return"]):
                    default_idx = i
                    break

            pnl_column = st.selectbox(
                "Select P&L Column",
                numeric_cols,
                index=default_idx,
            )

            st.markdown(f"**{len(df)}** trades detected")

# Simulation settings row
if df is not None and pnl_column is not None:
    col_nsim, col_bundle, col_run = st.columns([1, 1, 1], gap="medium")

    with col_nsim:
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=10,
            max_value=50000,
            value=1000,
            step=100,
        )

    with col_bundle:
        bundle_toggle = False
        if n_simulations > 500:
            bundle_toggle = st.checkbox(
                "Bundle every 10 simulations into 1 averaged line",
                value=n_simulations >= 2000,
                help="Reduces visual noise for large simulation counts",
            )

    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("Run Simulation", use_container_width=True)

    # --- Run Simulation ---
    if run_clicked:
        pnl_values = df[pnl_column].dropna().values.astype(np.float64)

        if len(pnl_values) < 2:
            st.error("Need at least 2 trades to run simulation.")
        else:
            with st.spinner("Running simulations..."):
                original_curve, all_curves, avg_curve = run_monte_carlo(
                    pnl_values, int(n_simulations)
                )

            n_trades = len(pnl_values)
            x_axis = np.arange(1, n_trades + 1)

            # --- Build Chart ---
            fig = go.Figure()

            # Simulation lines
            if bundle_toggle and n_simulations > 10:
                # Bundle every 10 into averaged lines
                bundle_size = 10
                n_bundles = n_simulations // bundle_size
                for b in range(n_bundles):
                    bundled = np.mean(
                        all_curves[b * bundle_size : (b + 1) * bundle_size], axis=0
                    )
                    fig.add_trace(go.Scattergl(
                        x=x_axis,
                        y=bundled,
                        mode="lines",
                        line=dict(color="rgba(180, 140, 255, 0.12)", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    ))
            else:
                # Plot individual curves (use sampling for very high counts)
                max_plot = min(n_simulations, 2000)
                if n_simulations > max_plot:
                    indices = np.random.choice(n_simulations, max_plot, replace=False)
                else:
                    indices = np.arange(n_simulations)

                for i in indices:
                    fig.add_trace(go.Scattergl(
                        x=x_axis,
                        y=all_curves[i],
                        mode="lines",
                        line=dict(color="rgba(180, 140, 255, 0.08)", width=0.8),
                        hoverinfo="skip",
                        showlegend=False,
                    ))

            # Original equity curve (light blue, thick)
            fig.add_trace(go.Scattergl(
                x=x_axis,
                y=original_curve,
                mode="lines",
                name="Original Sequence",
                line=dict(color="#60a5fa", width=3),
            ))

            # Average line (white, thickest)
            fig.add_trace(go.Scattergl(
                x=x_axis,
                y=avg_curve,
                mode="lines",
                name="Average (all simulations)",
                line=dict(color="#ffffff", width=3.5),
            ))

            fig.update_layout(
                plot_bgcolor="#2a2a2a",
                paper_bgcolor="#1a1a1a",
                font=dict(family="Inter, sans-serif", color="#cccccc", size=13),
                height=650,
                margin=dict(l=60, r=30, t=50, b=60),
                xaxis=dict(
                    title="Trade Number",
                    gridcolor="#3a3a3a",
                    zerolinecolor="#3a3a3a",
                    title_font=dict(size=14),
                ),
                yaxis=dict(
                    title="Cumulative P&L",
                    gridcolor="#3a3a3a",
                    zerolinecolor="#555555",
                    title_font=dict(size=14),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=13),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- Metrics ---
            final_avg = avg_curve[-1]
            final_original = original_curve[-1]
            avg_class = "positive" if final_avg >= 0 else "negative"
            orig_class = "positive" if final_original >= 0 else "negative"

            # Min/Max final equity across simulations
            final_values = all_curves[:, -1]
            worst = np.min(final_values)
            best = np.max(final_values)
            median_val = np.median(final_values)
            worst_class = "positive" if worst >= 0 else "negative"
            best_class = "positive" if best >= 0 else "negative"
            median_class = "positive" if median_val >= 0 else "negative"

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Simulations</div>
                    <div class="metric-value">{n_simulations:,}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Original Final P&L</div>
                    <div class="metric-value {orig_class}">{final_original:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Final P&L</div>
                    <div class="metric-value {avg_class}">{final_avg:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Worst Case</div>
                    <div class="metric-value {worst_class}">{worst:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Case</div>
                    <div class="metric-value {best_class}">{best:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            # Distribution of final P&L
            st.markdown("<br>", unsafe_allow_html=True)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=final_values,
                nbinsx=80,
                marker_color="rgba(180, 140, 255, 0.6)",
                marker_line=dict(color="rgba(180, 140, 255, 0.9)", width=0.5),
            ))
            fig_hist.add_vline(x=final_original, line_color="#60a5fa", line_width=2,
                               annotation_text="Original", annotation_font_color="#60a5fa")
            fig_hist.add_vline(x=final_avg, line_color="#ffffff", line_width=2,
                               annotation_text="Average", annotation_font_color="#ffffff")
            fig_hist.update_layout(
                title="Distribution of Final P&L Across Simulations",
                plot_bgcolor="#2a2a2a",
                paper_bgcolor="#1a1a1a",
                font=dict(family="Inter, sans-serif", color="#cccccc", size=13),
                height=320,
                margin=dict(l=60, r=30, t=50, b=50),
                xaxis=dict(title="Final P&L", gridcolor="#3a3a3a"),
                yaxis=dict(title="Count", gridcolor="#3a3a3a"),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#555; font-size:12px;">'
    'Monte Carlo Trading Simulator &middot; Full random permutation method'
    '</p>',
    unsafe_allow_html=True,
)
