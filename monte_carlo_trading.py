"""
Monte Carlo Simulation & Kelly Criterion Analysis
Single-page Streamlit web application
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(
    page_title="Anarchy Terminal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background-color: #111317;
    color: #e8eaef;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif;
    color: #f0f2f5;
    font-weight: 600;
}
.header-container {
    text-align: center;
    padding: 40px 0 8px 0;
}
.header-container h1 {
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0;
    background: linear-gradient(135deg, #e0e4ec 0%, #8892a8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-divider {
    border: none;
    border-top: 1px solid #22252b;
    margin: 32px 0;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #e0e4ec;
    letter-spacing: -0.3px;
    margin-top: 12px;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 2px solid #22252b;
}
.metric-card {
    background: #1a1d23;
    border: 1px solid #22252b;
    border-radius: 14px;
    padding: 26px 22px;
    text-align: center;
    transition: border-color 0.2s;
    margin: 6px 4px;
}
.metric-card:hover {
    border-color: #3a3f4b;
}
.metric-label {
    font-size: 11px;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 10px;
    font-weight: 500;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: #f0f2f5;
}
.metric-value.positive { color: #4ade80; }
.metric-value.negative { color: #f87171; }
.metric-value.neutral { color: #60a5fa; }
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 4px;
    border-bottom: 1px solid #22252b;
    font-size: 14px;
    gap: 16px;
}
.stat-label { color: #9ca3af; }
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    color: #d0d4dc;
    font-weight: 500;
}
.info-box {
    background: #1a1d23;
    border: 1px solid #22252b;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 18px 24px;
    font-size: 14px;
    line-height: 1.7;
    color: #c9cdd5;
    margin: 16px 0 24px 0;
}
.kelly-card {
    background: #1a1d23;
    border: 1px solid #22252b;
    border-radius: 14px;
    padding: 26px 22px;
    margin: 6px 4px;
}
.kelly-header {
    font-size: 16px;
    font-weight: 600;
    color: #e0e4ec;
    margin-bottom: 4px;
}
.kelly-sub {
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 14px;
}
.kelly-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}
.kelly-unit {
    font-size: 12px;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div[data-testid="stFileUploader"] {
    background: #1a1d23;
    border: 2px dashed #22252b;
    border-radius: 14px;
    padding: 12px;
}
.stSelectbox, .stNumberInput {
    color: #d0d4dc;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stCheckbox"] label {
    color: #d0d4dc !important;
    font-weight: 500;
}
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2d4a 100%);
    color: #e0e4ec;
    border: 1px solid #2a4a6f;
    border-radius: 10px;
    padding: 12px 32px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.3px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a4f7f 0%, #1e3a5f 100%);
    border-color: #3b6aaf;
    color: #ffffff;
}
.stDownloadButton > button {
    background: #1a1d23;
    color: #c9cdd5;
    border: 1px solid #22252b;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 13px;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    background: #22252b;
    color: #e0e4ec;
    border-color: #3a3f4b;
}
div[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    color: #9ca3af;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e0e4ec;
    border-bottom-color: #3b82f6;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header-container">
    <h1>Anarchy Terminal</h1>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# UTILITIES
# =========================================================================

def read_uploaded_file(uploaded_file):
    """Read uploaded file and return a DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            uploaded_file.seek(0)
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


def run_monte_carlo(pnl_values, n_simulations, method="resample"):
    """Run Monte Carlo simulations.
    method='permutation': shuffle without replacement (same final P&L)
    method='resample': bootstrap with replacement (realistic variance)
    """
    n_trades = len(pnl_values)
    pnl_array = np.array(pnl_values, dtype=np.float64)
    all_curves = np.empty((n_simulations, n_trades), dtype=np.float64)

    for i in range(n_simulations):
        if method == "permutation":
            sampled = np.random.permutation(pnl_array)
        else:  # resample with replacement
            sampled = np.random.choice(pnl_array, size=n_trades, replace=True)
        all_curves[i] = np.cumsum(sampled)

    original_curve = np.cumsum(pnl_array)
    avg_curve = np.mean(all_curves, axis=0)
    return original_curve, all_curves, avg_curve


def calc_max_drawdown(equity_curve):
    """Calculate max drawdown from an equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - peak
    return np.min(drawdown)


def calc_kelly_criterion(pnl_values):
    """Calculate Kelly Criterion from trade P&L values."""
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    win_rate = len(wins) / len(pnl_values)
    loss_rate = 1.0 - win_rate
    avg_win = np.mean(wins)
    avg_loss = np.abs(np.mean(losses))

    if avg_loss == 0:
        return 0.0, win_rate, loss_rate, avg_win, avg_loss

    win_loss_ratio = avg_win / avg_loss
    kelly = win_rate - (loss_rate / win_loss_ratio)

    return kelly, win_rate, loss_rate, avg_win, avg_loss


def apply_kelly_sizing(pnl_values, kelly_fraction, kelly_pct):
    """
    Simulate equity curve with Kelly-based position sizing.
    Uses fixed-fraction of STARTING equity to keep results in realistic range.
    Each trade's actual P&L is scaled by (kelly_pct / base_risk).
    """
    n = len(pnl_values)
    starting_equity = 10000.0
    equity = np.zeros(n + 1)
    equity[0] = starting_equity

    # Scale trades proportionally to kelly fraction
    # Normalize: what fraction of equity does each trade's P&L represent?
    avg_abs_pnl = np.mean(np.abs(pnl_values))
    if avg_abs_pnl == 0:
        return equity[1:]

    # Scale factor: kelly_pct determines how much of equity each "unit" of risk is
    scale = (kelly_pct * starting_equity) / avg_abs_pnl

    for i in range(n):
        equity[i + 1] = equity[i] + pnl_values[i] * scale
        if equity[i + 1] <= 0:
            equity[i + 1:] = 0
            break

    return equity[1:]


def fmt_human(value):
    """Format large numbers into readable estimates like $2.3M, $450K, etc."""
    neg = value < 0
    v = abs(value)
    if v >= 1_000_000_000:
        s = f"${v / 1_000_000_000:,.1f}B"
    elif v >= 1_000_000:
        s = f"${v / 1_000_000:,.1f}M"
    elif v >= 10_000:
        s = f"${v / 1_000:,.0f}K"
    elif v >= 1_000:
        s = f"${v / 1_000:,.1f}K"
    else:
        s = f"${v:,.0f}"
    return f"-{s}" if neg else s


def get_file_extension(filename):
    """Get the file extension from filename."""
    if "." in filename:
        return "." + filename.rsplit(".", 1)[-1].lower()
    return ".csv"


# =========================================================================
# FILE UPLOAD SECTION
# =========================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col_upload, col_spacer, col_settings = st.columns([5, 1, 4], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Trade Data",
        type=["csv", "txt", "xls", "xlsx", "xlsm"],
        help="One column must contain per-trade P&L values",
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

            n_trades_found = df[pnl_column].dropna().shape[0]
            st.markdown(
                f'<p style="font-family: JetBrains Mono, monospace; color: #4ade80; '
                f'font-size: 15px; margin-top: 8px;">'
                f'{n_trades_found} trades detected</p>',
                unsafe_allow_html=True,
            )


# =========================================================================
# TABS: MONTE CARLO | KELLY CRITERION
# =========================================================================

if df is not None and pnl_column is not None:
    pnl_values = df[pnl_column].dropna().values.astype(np.float64)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab_mc, tab_kelly, tab_var, tab_risk = st.tabs(["Monte Carlo Simulator", "Kelly Criterion Analysis", "Risk Analytics (VaR)", "Risk Laboratory"])

    # =====================================================================
    # TAB 1: MONTE CARLO
    # =====================================================================
    with tab_mc:
        st.markdown('<div class="section-title">Monte Carlo Simulation Analysis</div>',
                    unsafe_allow_html=True)

        col_nsim, col_method, col_bundle, col_run = st.columns(
            [2, 2, 3, 2], gap="medium"
        )

        with col_nsim:
            n_simulations = st.number_input(
                "Number of Simulations",
                min_value=10,
                max_value=50000,
                value=1000,
                step=100,
            )

        with col_method:
            mc_method = st.selectbox(
                "Simulation Method",
                ["Resample (with replacement)", "Permutation (shuffle only)"],
                index=0,
                help="Resample: bootstrap sampling creates realistic variance in outcomes. "
                     "Permutation: shuffles trade order, final P&L stays the same.",
            )
            method_key = "resample" if "Resample" in mc_method else "permutation"

        with col_bundle:
            bundle_toggle = False
            if n_simulations > 500:
                st.markdown("<br>", unsafe_allow_html=True)
                bundle_toggle = st.checkbox(
                    "Bundle every 10 simulations into 1 averaged line",
                    value=n_simulations >= 2000,
                    help="Reduces visual clutter by averaging groups of 10 curves",
                )

        with col_run:
            st.markdown("<br>", unsafe_allow_html=True)
            run_clicked = st.button("Run Simulation", use_container_width=True,
                                    key="mc_run")

        # --- Run Simulation ---
        if run_clicked:
            if len(pnl_values) < 2:
                st.error("Need at least 2 trades to run simulation.")
            else:
                with st.spinner("Running simulations..."):
                    original_curve, all_curves, avg_curve = run_monte_carlo(
                        pnl_values, int(n_simulations), method=method_key
                    )
                    # Store everything in session state so results persist across reruns
                    st.session_state["mc_all_curves"] = all_curves
                    st.session_state["mc_original_curve"] = original_curve
                    st.session_state["mc_avg_curve"] = avg_curve
                    st.session_state["mc_n_sims"] = int(n_simulations)
                    st.session_state["mc_method"] = method_key
                    st.session_state["mc_bundle"] = bundle_toggle
                    st.session_state["mc_has_results"] = True

        # --- Display Results (from session state) ---
        if st.session_state.get("mc_has_results", False):
                all_curves = st.session_state["mc_all_curves"]
                original_curve = st.session_state["mc_original_curve"]
                avg_curve = st.session_state["mc_avg_curve"]
                n_sims_display = st.session_state["mc_n_sims"]
                method_key_display = st.session_state["mc_method"]
                bundle_display = st.session_state.get("mc_bundle", False)

                n_trades = len(pnl_values)
                x_axis = np.arange(1, n_trades + 1)

                # --- Build Chart ---
                fig = go.Figure()

                # Colorful line palette for simulations
                sim_colors = [
                    "rgba(255, 107, 107, {a})",  # red
                    "rgba(255, 159, 67, {a})",   # orange
                    "rgba(254, 202, 87, {a})",   # yellow
                    "rgba(46, 213, 115, {a})",   # green
                    "rgba(30, 196, 179, {a})",   # teal
                    "rgba(69, 170, 242, {a})",   # light blue
                    "rgba(140, 122, 230, {a})",  # purple
                    "rgba(232, 67, 147, {a})",   # pink
                    "rgba(162, 210, 81, {a})",   # lime
                    "rgba(0, 210, 211, {a})",    # cyan
                    "rgba(204, 142, 53, {a})",   # gold
                    "rgba(119, 190, 29, {a})",   # bright green
                    "rgba(196, 113, 237, {a})",  # violet
                    "rgba(255, 135, 135, {a})",  # salmon
                    "rgba(72, 219, 251, {a})",   # sky
                ]

                if bundle_display and n_sims_display > 10:
                    bundle_size = 10
                    n_bundles = n_sims_display // bundle_size
                    for b in range(n_bundles):
                        bundled = np.mean(
                            all_curves[b * bundle_size: (b + 1) * bundle_size], axis=0
                        )
                        c = sim_colors[b % len(sim_colors)].format(a=0.35)
                        fig.add_trace(go.Scattergl(
                            x=x_axis, y=bundled, mode="lines",
                            line=dict(color=c, width=1.5),
                            hoverinfo="skip", showlegend=False,
                        ))
                else:
                    max_plot = min(n_sims_display, 2000)
                    indices = (np.random.choice(n_sims_display, max_plot, replace=False)
                               if n_sims_display > max_plot else np.arange(n_sims_display))
                    for idx_pos, i in enumerate(indices):
                        c = sim_colors[idx_pos % len(sim_colors)].format(a=0.25)
                        fig.add_trace(go.Scattergl(
                            x=x_axis, y=all_curves[i], mode="lines",
                            line=dict(color=c, width=1.5),
                            hoverinfo="skip", showlegend=False,
                        ))

                # Original curve (light blue)
                fig.add_trace(go.Scattergl(
                    x=x_axis, y=original_curve, mode="lines",
                    name="Original Sequence",
                    line=dict(color="#60a5fa", width=2),
                ))

                # Average curve (white)
                fig.add_trace(go.Scattergl(
                    x=x_axis, y=avg_curve, mode="lines",
                    name="Average (all simulations)",
                    line=dict(color="#ffffff", width=2),
                ))

                fig.update_layout(
                    plot_bgcolor="#1a1d23",
                    paper_bgcolor="#111317",
                    font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec",
                              size=13),
                    height=680,
                    margin=dict(l=60, r=30, t=50, b=60),
                    xaxis=dict(
                        title="Trade Number", gridcolor="#2a2d35",
                        zerolinecolor="#2a2d35",
                        title_font=dict(size=14, color="#e0e4ec"),
                        tickfont=dict(color="#c9cdd5"),
                    ),
                    yaxis=dict(
                        title="Cumulative P&L", gridcolor="#2a2d35",
                        zerolinecolor="#3a3f4b",
                        title_font=dict(size=14, color="#e0e4ec"),
                        tickfont=dict(color="#c9cdd5"),
                    ),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5,
                        font=dict(size=13, color="#e0e4ec"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- Download Average Curve ---
                st.markdown("")
                avg_df = pd.DataFrame({
                    "Trade_Number": x_axis,
                    "Average_Cumulative_PnL": avg_curve,
                    "Average_Per_Trade_PnL": np.diff(avg_curve, prepend=0),
                })

                file_ext = get_file_extension(uploaded_file.name)

                if file_ext in (".xls", ".xlsx", ".xlsm"):
                    buf = io.BytesIO()
                    avg_df.to_excel(buf, index=False, engine="openpyxl")
                    download_data = buf.getvalue()
                    download_name = "average_equity_curve.xlsx"
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif file_ext == ".txt":
                    download_data = avg_df.to_csv(index=False, sep="\t").encode("utf-8")
                    download_name = "average_equity_curve.txt"
                    mime = "text/plain"
                else:
                    download_data = avg_df.to_csv(index=False).encode("utf-8")
                    download_name = "average_equity_curve.csv"
                    mime = "text/csv"

                st.download_button(
                    label=f"Download Average Equity Curve (.{download_name.split('.')[-1]})",
                    data=download_data,
                    file_name=download_name,
                    mime=mime,
                )

                # --- Metrics ---
                st.markdown("<br>", unsafe_allow_html=True)

                final_original = original_curve[-1]
                final_avg = avg_curve[-1]

                # Drawdown analysis (actually meaningful for permutation MC)
                all_max_dd = np.array([calc_max_drawdown(c) for c in all_curves])
                orig_max_dd = calc_max_drawdown(original_curve)
                worst_dd = np.min(all_max_dd)
                best_dd = np.max(all_max_dd)
                avg_dd = np.mean(all_max_dd)
                median_dd = np.median(all_max_dd)

                # Min equity point across sims
                all_min_equity = np.min(all_curves, axis=1)
                worst_min_eq = np.min(all_min_equity)
                avg_min_eq = np.mean(all_min_equity)

                orig_class = "positive" if final_original >= 0 else "negative"
                avg_class = "positive" if final_avg >= 0 else "negative"

                # Final P&L stats across sims
                final_values = all_curves[:, -1]
                worst_final = np.min(final_values)
                best_final = np.max(final_values)
                worst_final_class = "positive" if worst_final >= 0 else "negative"
                best_final_class = "positive" if best_final >= 0 else "negative"

                # Row 1: P&L results
                c1, c2, c3, c4, c5 = st.columns(5)

                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Simulations</div>
                        <div class="metric-value neutral">{n_sims_display:,}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Original Final P&L</div>
                        <div class="metric-value {orig_class}">{fmt_human(final_original)}</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Final P&L</div>
                        <div class="metric-value {avg_class}">{fmt_human(final_avg)}</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Best Final P&L</div>
                        <div class="metric-value {best_final_class}">{fmt_human(best_final)}</div>
                    </div>""", unsafe_allow_html=True)
                with c5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Worst Final P&L</div>
                        <div class="metric-value {worst_final_class}">{fmt_human(worst_final)}</div>
                    </div>""", unsafe_allow_html=True)

                # Row 2: Drawdown results
                st.markdown("<br>", unsafe_allow_html=True)
                d1, d2, d3, d4 = st.columns(4)

                with d1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Original Max Drawdown</div>
                        <div class="metric-value negative">{fmt_human(orig_max_dd)}</div>
                    </div>""", unsafe_allow_html=True)
                with d2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Max Drawdown</div>
                        <div class="metric-value negative">{fmt_human(avg_dd)}</div>
                    </div>""", unsafe_allow_html=True)
                with d3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Best Drawdown</div>
                        <div class="metric-value negative">{fmt_human(best_dd)}</div>
                    </div>""", unsafe_allow_html=True)
                with d4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Worst Drawdown</div>
                        <div class="metric-value negative">{fmt_human(worst_dd)}</div>
                    </div>""", unsafe_allow_html=True)

                # --- Distribution Section ---
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">Drawdown Distribution</div>',
                    unsafe_allow_html=True,
                )

                if method_key_display == "permutation":
                    dist_text = (
                        '<strong>Why drawdowns?</strong> Permutation shuffling preserves '
                        'all trades (just reorders them), so every simulation ends at the same '
                        'final P&L. What changes is the <em>path</em> — how deep the equity '
                        'dips along the way. A tight distribution means your strategy is robust '
                        'to sequencing; a wide spread means luck in trade ordering mattered.'
                    )
                else:
                    dist_text = (
                        '<strong>Resampling with replacement</strong> draws trades randomly '
                        'from your history, allowing repeats. This creates realistic variance '
                        'in both the path <em>and</em> final outcome. The drawdown distribution '
                        'below shows the range of max drawdowns across all simulations — giving '
                        'you a realistic picture of the risk your strategy carries.'
                    )
                st.markdown(
                    f'<div class="info-box">{dist_text}</div>',
                    unsafe_allow_html=True,
                )

                col_hist, col_stats = st.columns([3, 1], gap="large")

                with col_hist:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=all_max_dd,
                        nbinsx=80,
                        marker_color="rgba(160, 130, 240, 0.5)",
                        marker_line=dict(color="rgba(160, 130, 240, 0.8)", width=0.5),
                    ))
                    fig_hist.add_vline(
                        x=orig_max_dd, line_color="#60a5fa", line_width=2,
                        annotation_text="Your Drawdown",
                        annotation_font=dict(color="#60a5fa", size=12),
                    )
                    fig_hist.add_vline(
                        x=avg_dd, line_color="#ffffff", line_width=2,
                        annotation_text="Average",
                        annotation_font=dict(color="#ffffff", size=12),
                    )
                    fig_hist.update_layout(
                        plot_bgcolor="#1a1d23",
                        paper_bgcolor="#111317",
                        font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec",
                                  size=13),
                        height=340,
                        margin=dict(l=60, r=30, t=30, b=50),
                        xaxis=dict(title="Max Drawdown", gridcolor="#2a2d35",
                                   title_font=dict(color="#e0e4ec"),
                                   tickfont=dict(color="#c9cdd5")),
                        yaxis=dict(title="Frequency", gridcolor="#2a2d35",
                                   title_font=dict(color="#e0e4ec"),
                                   tickfont=dict(color="#c9cdd5")),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col_stats:
                    # Percentile of original drawdown
                    pct_worse = np.mean(all_max_dd <= orig_max_dd) * 100

                    stats_html = f"""
                    <div style="padding: 10px 0;">
                        <div class="stat-row">
                            <span class="stat-label">Worst Drawdown</span>
                            <span class="stat-value">{worst_dd:,.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">5th Percentile</span>
                            <span class="stat-value">{np.percentile(all_max_dd, 5):,.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">25th Percentile</span>
                            <span class="stat-value">{np.percentile(all_max_dd, 25):,.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Median</span>
                            <span class="stat-value">{median_dd:,.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">75th Percentile</span>
                            <span class="stat-value">{np.percentile(all_max_dd, 75):,.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Best (Shallowest)</span>
                            <span class="stat-value">{best_dd:,.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Std Deviation</span>
                            <span class="stat-value">{np.std(all_max_dd):,.2f}</span>
                        </div>
                        <div class="stat-row" style="border-bottom: none; margin-top: 8px;">
                            <span class="stat-label">Your DD Percentile</span>
                            <span class="stat-value" style="color: #60a5fa;">{pct_worse:.1f}%</span>
                        </div>
                        <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">
                            {pct_worse:.0f}% of random orderings had a drawdown
                            equal to or deeper than yours.
                        </div>
                    </div>
                    """
                    st.markdown(stats_html, unsafe_allow_html=True)

    # =====================================================================
    # TAB 2: KELLY CRITERION
    # =====================================================================
    with tab_kelly:
        st.markdown(
            '<div class="section-title">Kelly Criterion Position Sizing</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            'The <strong>Kelly Criterion</strong> is a formula that determines the '
            'optimal fraction of your capital to risk on each trade, based on your '
            'historical win rate and win/loss ratio. It maximizes long-term growth '
            'rate but comes with high volatility — which is why traders often use '
            'fractional Kelly (½ or ¼) to reduce risk while still capturing most of '
            'the growth advantage.'
            '<br><br>'
            '<strong>Formula:</strong> &nbsp; '
            '<span style="font-family: JetBrains Mono, monospace; color: #e0e4ec;">'
            'K% = W − (1 − W) / R</span>'
            '<br>'
            'where <em>W</em> = win rate, <em>R</em> = avg win / avg loss'
            '</div>',
            unsafe_allow_html=True,
        )

        # Calculate Kelly
        kelly, win_rate, loss_rate, avg_win, avg_loss = calc_kelly_criterion(pnl_values)

        # Display Kelly metrics
        st.markdown("<br>", unsafe_allow_html=True)

        kc1, kc2, kc3, kc4, kc5 = st.columns(5)

        kelly_color = "#4ade80" if kelly > 0 else "#f87171"

        with kc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value neutral">{win_rate * 100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with kc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Win</div>
                <div class="metric-value positive">{fmt_human(avg_win)}</div>
            </div>""", unsafe_allow_html=True)
        with kc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Loss</div>
                <div class="metric-value negative">-{fmt_human(avg_loss)}</div>
            </div>""", unsafe_allow_html=True)
        with kc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Win/Loss Ratio</div>
                <div class="metric-value neutral">{avg_win / avg_loss if avg_loss > 0 else 0:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with kc5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Full Kelly %</div>
                <div class="metric-value" style="color: {kelly_color};">{kelly * 100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        if kelly <= 0:
            st.markdown("")
            st.warning(
                "Kelly Criterion is negative or zero — the edge in this dataset "
                "does not support Kelly-based sizing. The strategy may not have a "
                "positive expectancy."
            )
        else:
            st.markdown("<br>", unsafe_allow_html=True)

            # Kelly fractions to simulate
            fixed_frac = 0.02  # 2% fixed fractional
            kelly_variants = {
                "Fixed 2%": fixed_frac,
                "Quarter Kelly": kelly * 0.25,
                "Half Kelly": kelly * 0.5,
                "Full Kelly": kelly,
                "1.5x Kelly": kelly * 1.5,
            }

            colors = {
                "Fixed 2%": "#6b7280",
                "Quarter Kelly": "#60a5fa",
                "Half Kelly": "#4ade80",
                "Full Kelly": "#fbbf24",
                "1.5x Kelly": "#f87171",
            }

            # Run button
            col_kelly_run, _ = st.columns([1, 3])
            with col_kelly_run:
                kelly_run = st.button("Run Kelly Analysis", use_container_width=True,
                                      key="kelly_run")

            if kelly_run:
                # Calculate equity curves and store in session state
                results = {}
                for name, fraction in kelly_variants.items():
                    results[name] = apply_kelly_sizing(pnl_values, kelly, fraction)
                st.session_state["kelly_results"] = results
                st.session_state["kelly_variants"] = kelly_variants
                st.session_state["kelly_colors"] = colors
                st.session_state["kelly_has_results"] = True

            if st.session_state.get("kelly_has_results", False):
                results = st.session_state["kelly_results"]
                kelly_variants_display = st.session_state["kelly_variants"]
                colors_display = st.session_state["kelly_colors"]

                fig_kelly = go.Figure()
                for name, fraction in kelly_variants_display.items():
                    eq = results[name]
                    fig_kelly.add_trace(go.Scattergl(
                        x=np.arange(1, len(eq) + 1),
                        y=eq,
                        mode="lines",
                        name=f"{name} ({fraction * 100:.1f}%)",
                        line=dict(color=colors_display[name], width=2.5),
                    ))

                fig_kelly.update_layout(
                    plot_bgcolor="#1a1d23",
                    paper_bgcolor="#111317",
                    font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec",
                              size=13),
                    height=600,
                    margin=dict(l=60, r=30, t=50, b=60),
                    xaxis=dict(
                        title="Trade Number", gridcolor="#2a2d35",
                        zerolinecolor="#2a2d35",
                        title_font=dict(size=14, color="#e0e4ec"),
                        tickfont=dict(color="#c9cdd5"),
                    ),
                    yaxis=dict(
                        title="Equity ($)", gridcolor="#2a2d35",
                        zerolinecolor="#3a3f4b",
                        title_font=dict(size=14, color="#e0e4ec"),
                        tickfont=dict(color="#c9cdd5"),
                        type="log",
                    ),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5,
                        font=dict(size=13, color="#e0e4ec"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    hovermode="x unified",
                )

                st.plotly_chart(fig_kelly, use_container_width=True)

                # Results cards
                st.markdown("<br>", unsafe_allow_html=True)

                cols = st.columns(len(kelly_variants_display))
                for idx, (name, fraction) in enumerate(kelly_variants_display.items()):
                    eq = results[name]
                    final_eq = eq[-1] if len(eq) > 0 else 0
                    dd = calc_max_drawdown(eq)
                    growth = ((final_eq / 10000) - 1) * 100 if final_eq > 0 else -100

                    growth_color = "#4ade80" if growth >= 0 else "#f87171"

                    # Format growth as human-readable
                    if abs(growth) >= 10000:
                        growth_str = f"{growth / 1000:+,.0f}K%"
                    elif abs(growth) >= 1000:
                        growth_str = f"{growth / 1000:+,.1f}K%"
                    else:
                        growth_str = f"{growth:+,.0f}%"

                    with cols[idx]:
                        st.markdown(f"""
                        <div class="kelly-card">
                            <div class="kelly-header" style="color: {colors_display[name]};">{name}</div>
                            <div class="kelly-sub">Risk {fraction * 100:.1f}% per trade</div>
                            <div class="kelly-val" style="color: {growth_color};">{growth_str}</div>
                            <div class="kelly-unit">Total Return</div>
                            <div style="margin-top: 14px;">
                                <div class="stat-row">
                                    <span class="stat-label">Final Equity</span>
                                    <span class="stat-value">{fmt_human(final_eq)}</span>
                                </div>
                                <div class="stat-row" style="border-bottom: none;">
                                    <span class="stat-label">Max Drawdown</span>
                                    <span class="stat-value" style="color: #f87171;">{fmt_human(dd)}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="info-box">'
                    '<strong>Interpretation:</strong> Full Kelly maximizes theoretical '
                    'long-term growth but produces extreme drawdowns that most traders '
                    'cannot stomach. Half Kelly achieves ~75% of the growth with '
                    'significantly less volatility — widely considered the practical '
                    'sweet spot. Quarter Kelly is conservative but very smooth. '
                    'Anything above Full Kelly (1.5x) is over-leveraged and typically '
                    'leads to ruin over enough trades.'
                    '</div>',
                    unsafe_allow_html=True,
                )


    # =====================================================================
    # TAB 3: RISK ANALYTICS (VaR)
    # =====================================================================
    with tab_var:
        st.markdown(
            '<div class="section-title">Value at Risk (VaR) Analysis</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            '<strong>Value at Risk (VaR)</strong> answers a simple question: '
            '"What is the worst loss I can expect in a given percentage of cases?" '
            'For example, a 95% VaR of -$500 means that 95% of the time, your loss on a single '
            'trade will not exceed $500 — but 5% of the time, it could be worse.'
            '<br><br>'
            '<strong>Two methods:</strong><br>'
            '• <strong>Historical VaR</strong> — sorts your actual trade P&L data and picks '
            'the loss at the relevant percentile. Simple, no assumptions about distribution.<br>'
            '• <strong>Monte Carlo VaR</strong> — uses the simulated equity paths (if you ran '
            'the Monte Carlo tab) to estimate VaR from per-trade changes across thousands of '
            'randomized sequences. Captures a broader range of possible outcomes.'
            '</div>',
            unsafe_allow_html=True,
        )

        # ---- Historical VaR ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="font-size: 18px;">Historical VaR</div>',
            unsafe_allow_html=True,
        )

        sorted_pnl = np.sort(pnl_values)
        n_total = len(sorted_pnl)

        confidence_levels = [0.90, 0.95, 0.99]
        var_results = {}
        for cl in confidence_levels:
            idx = int(np.floor((1 - cl) * n_total))
            idx = max(0, min(idx, n_total - 1))
            var_results[cl] = sorted_pnl[idx]

        # VaR cards
        vc1, vc2, vc3 = st.columns(3)

        var_descriptions = {
            0.90: "In 9 out of 10 trades, your loss will not exceed this amount. "
                  "This is a moderate confidence threshold.",
            0.95: "The industry standard. In 19 out of 20 trades, your loss stays "
                  "within this limit. Used by most risk managers.",
            0.99: "The most conservative measure. Only 1 in 100 trades is expected "
                  "to breach this threshold — your extreme tail risk.",
        }

        for col_var, cl in zip([vc1, vc2, vc3], confidence_levels):
            var_val = var_results[cl]
            val_class = "negative" if var_val < 0 else "positive"
            with col_var:
                st.markdown(f"""
                <div class="kelly-card">
                    <div class="kelly-header" style="color: #60a5fa;">{cl*100:.0f}% Confidence</div>
                    <div class="kelly-sub">Historical Simulation</div>
                    <div class="kelly-val" style="color: {'#f87171' if var_val < 0 else '#4ade80'};">
                        {fmt_human(var_val)}
                    </div>
                    <div class="kelly-unit">Max Expected Loss Per Trade</div>
                    <div style="margin-top: 14px; font-size: 13px; color: #9ca3af; line-height: 1.6;">
                        {var_descriptions[cl]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ---- Additional Historical Stats ----
        st.markdown("<br>", unsafe_allow_html=True)

        col_dist, col_detail = st.columns([3, 1], gap="large")

        with col_dist:
            # P&L distribution histogram
            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(
                x=pnl_values,
                nbinsx=60,
                marker_color="rgba(96, 165, 250, 0.4)",
                marker_line=dict(color="rgba(96, 165, 250, 0.7)", width=0.5),
                name="Trade P&L",
            ))

            # Add VaR lines
            var_line_colors = {0.90: "#fbbf24", 0.95: "#f87171", 0.99: "#ef4444"}
            for cl in confidence_levels:
                fig_var.add_vline(
                    x=var_results[cl],
                    line_color=var_line_colors[cl], line_width=2,
                    line_dash="dash",
                    annotation_text=f"VaR {cl*100:.0f}%",
                    annotation_font=dict(color=var_line_colors[cl], size=11),
                )

            # Mean line
            fig_var.add_vline(
                x=np.mean(pnl_values),
                line_color="#4ade80", line_width=2,
                annotation_text="Mean",
                annotation_font=dict(color="#4ade80", size=11),
            )

            fig_var.update_layout(
                plot_bgcolor="#1a1d23",
                paper_bgcolor="#111317",
                font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec", size=13),
                height=380,
                margin=dict(l=60, r=30, t=30, b=50),
                xaxis=dict(title="Trade P&L", gridcolor="#2a2d35",
                           title_font=dict(color="#e0e4ec"),
                           tickfont=dict(color="#c9cdd5")),
                yaxis=dict(title="Frequency", gridcolor="#2a2d35",
                           title_font=dict(color="#e0e4ec"),
                           tickfont=dict(color="#c9cdd5")),
                showlegend=False,
            )
            st.plotly_chart(fig_var, use_container_width=True)

        with col_detail:
            mean_pnl = np.mean(pnl_values)
            median_pnl = np.median(pnl_values)
            std_pnl = np.std(pnl_values)
            skew_pnl = float(pd.Series(pnl_values).skew())
            worst_trade = np.min(pnl_values)
            best_trade = np.max(pnl_values)
            cvar_95 = np.mean(sorted_pnl[sorted_pnl <= var_results[0.95]])

            st.markdown(f"""
            <div style="padding: 10px 0;">
                <div class="stat-row">
                    <span class="stat-label">Mean P&L</span>
                    <span class="stat-value" style="color: {'#4ade80' if mean_pnl >= 0 else '#f87171'};">
                        {fmt_human(mean_pnl)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Median P&L</span>
                    <span class="stat-value">{fmt_human(median_pnl)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Std Deviation</span>
                    <span class="stat-value">{fmt_human(std_pnl)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Skewness</span>
                    <span class="stat-value">{skew_pnl:+.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Worst Trade</span>
                    <span class="stat-value" style="color: #f87171;">{fmt_human(worst_trade)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Best Trade</span>
                    <span class="stat-value" style="color: #4ade80;">{fmt_human(best_trade)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">CVaR 95%</span>
                    <span class="stat-value" style="color: #f87171;">{fmt_human(cvar_95)}</span>
                </div>
                <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                    CVaR (Expected Shortfall) = average loss<br>
                    when VaR is breached. Captures tail severity.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Monte Carlo VaR ----
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="font-size: 18px;">Monte Carlo VaR</div>',
            unsafe_allow_html=True,
        )

        has_mc_data = "mc_all_curves" in st.session_state

        if has_mc_data:
            mc_n = st.session_state["mc_n_sims"]
            mc_m = st.session_state["mc_method"]
            method_label = "Resample" if mc_m == "resample" else "Permutation"
            st.markdown(
                f'<div class="info-box" style="border-left-color: #4ade80;">'
                f'✓ Using simulation data from Monte Carlo tab — '
                f'<strong>{mc_n:,} simulations</strong> ({method_label}). '
                f'No need to re-run.</div>',
                unsafe_allow_html=True,
            )

            mc_var_curves = st.session_state["mc_all_curves"]

            # Extract per-trade P&L from all simulated curves
            mc_per_trade = np.diff(mc_var_curves, axis=1, prepend=0)
            mc_all_trades = mc_per_trade.flatten()

            mc_sorted = np.sort(mc_all_trades)

            mc_var_results = {}
            mc_cvar_results = {}
            for cl in confidence_levels:
                idx = int(np.floor((1 - cl) * len(mc_sorted)))
                idx = max(0, min(idx, len(mc_sorted) - 1))
                mc_var_results[cl] = mc_sorted[idx]
                mc_cvar_results[cl] = np.mean(mc_sorted[:idx + 1]) if idx > 0 else mc_sorted[0]

            # Comparison table
            st.markdown("<br>", unsafe_allow_html=True)

            mc_vc1, mc_vc2, mc_vc3 = st.columns(3)

            for col_mc, cl in zip([mc_vc1, mc_vc2, mc_vc3], confidence_levels):
                h_var = var_results[cl]
                m_var = mc_var_results[cl]
                diff = m_var - h_var
                with col_mc:
                    st.markdown(f"""
                    <div class="kelly-card">
                        <div class="kelly-header" style="color: #a78bfa;">{cl*100:.0f}% Confidence</div>
                        <div class="kelly-sub">Monte Carlo vs Historical</div>
                        <div style="display: flex; gap: 24px; margin: 14px 0;">
                            <div>
                                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;
                                    letter-spacing: 1px; margin-bottom: 4px;">Historical</div>
                                <div style="font-family: JetBrains Mono, monospace; font-size: 20px;
                                    font-weight: 600; color: #f87171;">{fmt_human(h_var)}</div>
                            </div>
                            <div>
                                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;
                                    letter-spacing: 1px; margin-bottom: 4px;">Monte Carlo</div>
                                <div style="font-family: JetBrains Mono, monospace; font-size: 20px;
                                    font-weight: 600; color: #a78bfa;">{fmt_human(m_var)}</div>
                            </div>
                        </div>
                        <div style="font-size: 12px; color: #9ca3af; border-top: 1px solid #22252b;
                            padding-top: 10px;">
                            Difference: <span style="font-family: JetBrains Mono, monospace;
                            color: {'#4ade80' if diff >= 0 else '#f87171'};">{fmt_human(diff)}</span>
                            <br>MC CVaR: <span style="font-family: JetBrains Mono, monospace;
                            color: #f87171;">{fmt_human(mc_cvar_results[cl])}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # MC VaR distribution chart
            st.markdown("<br>", unsafe_allow_html=True)

            fig_mc_var = go.Figure()

            # Historical distribution
            fig_mc_var.add_trace(go.Histogram(
                x=pnl_values,
                nbinsx=60,
                marker_color="rgba(96, 165, 250, 0.3)",
                marker_line=dict(color="rgba(96, 165, 250, 0.5)", width=0.5),
                name="Historical Trades",
                opacity=0.7,
            ))

            # MC distribution (subsample for performance)
            mc_sample = mc_all_trades[np.random.choice(len(mc_all_trades),
                        size=min(50000, len(mc_all_trades)), replace=False)]
            fig_mc_var.add_trace(go.Histogram(
                x=mc_sample,
                nbinsx=80,
                marker_color="rgba(167, 139, 250, 0.3)",
                marker_line=dict(color="rgba(167, 139, 250, 0.5)", width=0.5),
                name="Monte Carlo Trades",
                opacity=0.7,
            ))

            # VaR lines
            fig_mc_var.add_vline(
                x=var_results[0.95], line_color="#60a5fa", line_width=2,
                line_dash="dash",
                annotation_text="Historical 95%",
                annotation_font=dict(color="#60a5fa", size=11),
            )
            fig_mc_var.add_vline(
                x=mc_var_results[0.95], line_color="#a78bfa", line_width=2,
                line_dash="dash",
                annotation_text="MC 95%",
                annotation_font=dict(color="#a78bfa", size=11),
            )

            fig_mc_var.update_layout(
                plot_bgcolor="#1a1d23",
                paper_bgcolor="#111317",
                font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec", size=13),
                height=400,
                margin=dict(l=60, r=30, t=30, b=50),
                barmode="overlay",
                xaxis=dict(title="Trade P&L", gridcolor="#2a2d35",
                           title_font=dict(color="#e0e4ec"),
                           tickfont=dict(color="#c9cdd5")),
                yaxis=dict(title="Frequency", gridcolor="#2a2d35",
                           title_font=dict(color="#e0e4ec"),
                           tickfont=dict(color="#c9cdd5")),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                    font=dict(size=13, color="#e0e4ec"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )

            st.plotly_chart(fig_mc_var, use_container_width=True)

            st.markdown(
                '<div class="info-box">'
                '<strong>How to read this:</strong> If the Monte Carlo VaR is wider (more negative) '
                'than the Historical VaR, your actual trade history may be understating risk — the '
                'resampled scenarios reveal worse outcomes that haven\'t happened yet but statistically '
                'could. If they\'re close, your historical data already captures the full risk picture.'
                '<br><br>'
                '<strong>CVaR (Conditional VaR)</strong> — also called Expected Shortfall — is the '
                'average loss in the worst cases beyond VaR. It tells you not just <em>where</em> '
                'the tail starts, but <em>how bad</em> it gets on average when things go wrong.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="info-box" style="border-left-color: #fbbf24;">'
                '⚠ No Monte Carlo data available yet. Go to the '
                '<strong>Monte Carlo Simulator</strong> tab first, run a simulation, '
                'then come back here — the results will be automatically loaded for '
                'VaR comparison.</div>',
                unsafe_allow_html=True,
            )


    # =====================================================================
    # TAB 4: RISK LABORATORY
    # =====================================================================
    with tab_risk:
        st.markdown(
            '<div class="section-title">Risk Laboratory — Stress Testing Suite</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            'The <strong>Risk Laboratory</strong> subjects your trading data to a battery '
            'of stress tests designed to answer one question: <em>"How fragile is my edge?"</em> '
            'Each module attacks a different assumption — fat-tail shocks, worst-case sequencing, '
            'execution friction, and edge decay — so you can see how your strategy holds up '
            'before real capital is on the line.'
            '</div>',
            unsafe_allow_html=True,
        )

        # --- Helper: compute backtest metrics from a P&L array ---
        def calc_backtest_metrics(pnl):
            """Return (profit_factor, expectancy, net_profit) for a P&L array."""
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            gross_profit = np.sum(wins) if len(wins) else 0.0
            gross_loss = np.abs(np.sum(losses)) if len(losses) else 0.0
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            expectancy = np.mean(pnl) if len(pnl) else 0.0
            net = np.sum(pnl)
            return pf, expectancy, net

        # --- Helper: risk grade color ---
        def risk_grade(dd_pct):
            """Return (label, color) for a drawdown percentage."""
            dd = abs(dd_pct)
            if dd <= 5:
                return "Moderate", "#4ade80"
            elif dd <= 10:
                return "High", "#fbbf24"
            elif dd <= 20:
                return "Critical", "#f97316"
            else:
                return "Undeployable", "#ef4444"

        mean_pnl_rl = np.mean(pnl_values)
        std_pnl_rl = np.std(pnl_values, ddof=1) if len(pnl_values) > 1 else 0.0
        account_size = np.sum(pnl_values)  # cumulative P&L as proxy for account

        # =================================================================
        # MODULE 1 — SIGMA SHOCK ANALYZER
        # =================================================================
        st.markdown(
            '<div class="section-title" style="font-size: 18px;">1 &mdash; Sigma Shock Analyzer</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            'Simulates the impact of statistically extreme losses (1&sigma;, 2&sigma;, 3&sigma;) '
            'hitting your account as the next 1 or 3 consecutive trades. The resulting drawdown '
            'is graded by severity.'
            '</div>',
            unsafe_allow_html=True,
        )

        col_acc, col_trades = st.columns(2, gap="medium")
        with col_acc:
            sigma_account = st.number_input(
                "Account Size ($)",
                min_value=100.0,
                value=10000.0,
                step=1000.0,
                key="sigma_account",
                help="Your total trading account value used to compute drawdown percentages.",
            )
        with col_trades:
            sigma_n_trades = st.radio(
                "Consecutive shock trades",
                [1, 3],
                horizontal=True,
                key="sigma_n_trades",
            )

        sigma_levels = [1, 2, 3]
        sc1, sc2, sc3 = st.columns(3)

        for col_s, sigma in zip([sc1, sc2, sc3], sigma_levels):
            shock_loss = mean_pnl_rl - sigma * std_pnl_rl  # negative tail
            total_loss = shock_loss * sigma_n_trades
            dd_pct = (total_loss / sigma_account) * 100 if sigma_account else 0
            grade_label, grade_color = risk_grade(dd_pct)

            with col_s:
                st.markdown(f"""
                <div class="kelly-card">
                    <div class="kelly-header" style="color: {grade_color};">{sigma}&sigma; Shock</div>
                    <div class="kelly-sub">{sigma_n_trades} trade(s) at &mu; &minus; {sigma}&sigma;</div>
                    <div class="kelly-val" style="color: {grade_color};">
                        {dd_pct:+.1f}%
                    </div>
                    <div class="kelly-unit">Account Drawdown</div>
                    <div style="margin-top: 14px;">
                        <div class="stat-row">
                            <span class="stat-label">Per-Trade Loss</span>
                            <span class="stat-value" style="color: #f87171;">{fmt_human(shock_loss)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Total Impact</span>
                            <span class="stat-value" style="color: #f87171;">{fmt_human(total_loss)}</span>
                        </div>
                        <div class="stat-row" style="border-bottom: none;">
                            <span class="stat-label">Risk Grade</span>
                            <span class="stat-value" style="color: {grade_color};">{grade_label}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Historical frequency of sigma events
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="font-size: 16px;">Historical Sigma Frequency</div>',
            unsafe_allow_html=True,
        )

        freq_cols = st.columns(3)
        for col_f, sigma in zip(freq_cols, sigma_levels):
            threshold = mean_pnl_rl - sigma * std_pnl_rl
            count = int(np.sum(pnl_values <= threshold))
            pct = count / len(pnl_values) * 100
            # Expected from normal distribution
            expected_pct = {1: 15.87, 2: 2.28, 3: 0.13}[sigma]
            with col_f:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{sigma}&sigma; Events (P&L &le; {fmt_human(threshold)})</div>
                    <div class="metric-value neutral">{count} <span style="font-size:14px; color:#9ca3af;">
                        ({pct:.1f}%)</span></div>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 8px;">
                        Normal distribution expects ~{expected_pct:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # =================================================================
        # MODULE 2 — FATAL SEQUENCE (WORST-CASE)
        # =================================================================
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="font-size: 18px;">2 &mdash; Fatal Sequence (Worst-Case Scenario)</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            'Extracts the <strong>10 largest losses</strong> from your dataset and simulates '
            'them occurring back-to-back — the absolute worst-case scenario for your account.'
            '</div>',
            unsafe_allow_html=True,
        )

        sorted_losses = np.sort(pnl_values)[:10]  # 10 most negative
        fatal_cum = np.cumsum(sorted_losses)
        fatal_max_dd = np.min(fatal_cum)
        fatal_final = sigma_account + np.sum(sorted_losses)
        fatal_dd_pct = (np.sum(sorted_losses) / sigma_account) * 100 if sigma_account else 0
        fatal_grade, fatal_color = risk_grade(fatal_dd_pct)

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sequence Total Loss</div>
                <div class="metric-value negative">{fmt_human(np.sum(sorted_losses))}</div>
            </div>""", unsafe_allow_html=True)
        with fc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{fmt_human(fatal_max_dd)}</div>
            </div>""", unsafe_allow_html=True)
        with fc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Account Balance</div>
                <div class="metric-value {'positive' if fatal_final >= 0 else 'negative'}">{fmt_human(fatal_final)}</div>
            </div>""", unsafe_allow_html=True)
        with fc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Drawdown %</div>
                <div class="metric-value" style="color: {fatal_color};">{fatal_dd_pct:+.1f}%</div>
                <div style="font-size: 11px; color: {fatal_color}; margin-top: 4px;">{fatal_grade}</div>
            </div>""", unsafe_allow_html=True)

        # Fatal sequence equity chart
        st.markdown("<br>", unsafe_allow_html=True)
        fatal_equity = np.concatenate([[sigma_account], sigma_account + fatal_cum])
        fig_fatal = go.Figure()
        fig_fatal.add_trace(go.Scatter(
            x=np.arange(0, len(fatal_equity)),
            y=fatal_equity,
            mode="lines+markers",
            line=dict(color="#f87171", width=3),
            marker=dict(size=8, color="#f87171"),
            name="Fatal Sequence",
            fill="tozeroy",
            fillcolor="rgba(248, 113, 113, 0.08)",
        ))
        fig_fatal.update_layout(
            plot_bgcolor="#1a1d23",
            paper_bgcolor="#111317",
            font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec", size=13),
            height=320,
            margin=dict(l=60, r=30, t=30, b=50),
            xaxis=dict(title="Trade # in Sequence", gridcolor="#2a2d35",
                       title_font=dict(color="#e0e4ec"), tickfont=dict(color="#c9cdd5"),
                       dtick=1),
            yaxis=dict(title="Account Balance ($)", gridcolor="#2a2d35",
                       title_font=dict(color="#e0e4ec"), tickfont=dict(color="#c9cdd5")),
            showlegend=False,
        )
        st.plotly_chart(fig_fatal, use_container_width=True)

        # Show the 10 trades
        st.markdown(
            '<div style="font-size: 12px; color: #6b7280; margin-top: 4px;">'
            'The 10 worst trades in order: '
            + ", ".join(f"<span style='font-family:JetBrains Mono,monospace;color:#f87171;'>"
                        f"{fmt_human(v)}</span>" for v in sorted_losses)
            + '</div>',
            unsafe_allow_html=True,
        )

        # =================================================================
        # MODULE 3 — OPERATIONAL STRESS (FRICTION TEST)
        # =================================================================
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="font-size: 18px;">3 &mdash; Operational Stress (Friction Test)</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            'Applies a flat <strong>4% slippage penalty</strong> to every trade. '
            'Wins are reduced by 4%, losses are increased by 4%. This models real-world '
            'execution friction — slippage, spread, partial fills, and latency costs.'
            '</div>',
            unsafe_allow_html=True,
        )

        friction_pnl = np.where(
            pnl_values >= 0,
            pnl_values * 0.96,   # reduce wins by 4%
            pnl_values * 1.04,   # increase losses by 4%
        )

        orig_pf, orig_exp, orig_net = calc_backtest_metrics(pnl_values)
        fric_pf, fric_exp, fric_net = calc_backtest_metrics(friction_pnl)

        fric_equity_orig = np.cumsum(pnl_values)
        fric_equity_stressed = np.cumsum(friction_pnl)

        fig_fric = go.Figure()
        x_ax = np.arange(1, len(pnl_values) + 1)
        fig_fric.add_trace(go.Scattergl(
            x=x_ax, y=fric_equity_orig, mode="lines",
            name="Original", line=dict(color="#60a5fa", width=2),
        ))
        fig_fric.add_trace(go.Scattergl(
            x=x_ax, y=fric_equity_stressed, mode="lines",
            name="With 4% Friction", line=dict(color="#fbbf24", width=2),
        ))
        fig_fric.update_layout(
            plot_bgcolor="#1a1d23",
            paper_bgcolor="#111317",
            font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec", size=13),
            height=380,
            margin=dict(l=60, r=30, t=50, b=50),
            xaxis=dict(title="Trade Number", gridcolor="#2a2d35",
                       title_font=dict(color="#e0e4ec"), tickfont=dict(color="#c9cdd5")),
            yaxis=dict(title="Cumulative P&L", gridcolor="#2a2d35",
                       zerolinecolor="#3a3f4b",
                       title_font=dict(color="#e0e4ec"), tickfont=dict(color="#c9cdd5")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5,
                        font=dict(size=13, color="#e0e4ec"),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fric, use_container_width=True)

        fr1, fr2, fr3 = st.columns(3)
        net_diff_fric = fric_net - orig_net
        with fr1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Original Net Profit</div>
                <div class="metric-value {'positive' if orig_net >= 0 else 'negative'}">{fmt_human(orig_net)}</div>
            </div>""", unsafe_allow_html=True)
        with fr2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Friction Net Profit</div>
                <div class="metric-value {'positive' if fric_net >= 0 else 'negative'}">{fmt_human(fric_net)}</div>
            </div>""", unsafe_allow_html=True)
        with fr3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Profit Lost to Friction</div>
                <div class="metric-value negative">{fmt_human(net_diff_fric)}</div>
            </div>""", unsafe_allow_html=True)

        # =================================================================
        # MODULE 4 — REGIME CHANGE (EDGE DECAY)
        # =================================================================
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="font-size: 18px;">4 &mdash; Regime Change (Edge Decay)</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            'Applies a <strong>16% haircut to all winning trades</strong> while leaving losses '
            'unchanged. This models what happens when market conditions shift — your edge '
            'decays, winners shrink, but losers stay the same size.'
            '</div>',
            unsafe_allow_html=True,
        )

        decay_pnl = np.where(
            pnl_values > 0,
            pnl_values * 0.84,   # reduce wins by 16%
            pnl_values,           # losses unchanged
        )

        decay_pf, decay_exp, decay_net = calc_backtest_metrics(decay_pnl)

        decay_equity_orig = np.cumsum(pnl_values)
        decay_equity_stressed = np.cumsum(decay_pnl)

        fig_decay = go.Figure()
        fig_decay.add_trace(go.Scattergl(
            x=x_ax, y=decay_equity_orig, mode="lines",
            name="Original", line=dict(color="#60a5fa", width=2),
        ))
        fig_decay.add_trace(go.Scattergl(
            x=x_ax, y=decay_equity_stressed, mode="lines",
            name="With 16% Edge Decay", line=dict(color="#f97316", width=2),
        ))
        fig_decay.update_layout(
            plot_bgcolor="#1a1d23",
            paper_bgcolor="#111317",
            font=dict(family="Space Grotesk, sans-serif", color="#e0e4ec", size=13),
            height=380,
            margin=dict(l=60, r=30, t=50, b=50),
            xaxis=dict(title="Trade Number", gridcolor="#2a2d35",
                       title_font=dict(color="#e0e4ec"), tickfont=dict(color="#c9cdd5")),
            yaxis=dict(title="Cumulative P&L", gridcolor="#2a2d35",
                       zerolinecolor="#3a3f4b",
                       title_font=dict(color="#e0e4ec"), tickfont=dict(color="#c9cdd5")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5,
                        font=dict(size=13, color="#e0e4ec"),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_decay, use_container_width=True)

        dr1, dr2, dr3 = st.columns(3)
        net_diff_decay = decay_net - orig_net
        with dr1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Original Net Profit</div>
                <div class="metric-value {'positive' if orig_net >= 0 else 'negative'}">{fmt_human(orig_net)}</div>
            </div>""", unsafe_allow_html=True)
        with dr2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Decayed Net Profit</div>
                <div class="metric-value {'positive' if decay_net >= 0 else 'negative'}">{fmt_human(decay_net)}</div>
            </div>""", unsafe_allow_html=True)
        with dr3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Profit Lost to Decay</div>
                <div class="metric-value negative">{fmt_human(net_diff_decay)}</div>
            </div>""", unsafe_allow_html=True)

        # =================================================================
        # MODULE 5 — STRESS SUMMARY TABLE
        # =================================================================
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title" style="font-size: 18px;">Stress Summary &mdash; Scenario Comparison</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="info-box">'
            'Side-by-side comparison of your <strong>Current Backtest</strong> against '
            'the Friction Stress and Edge Decay scenarios. The scenario causing the '
            'largest deviation is highlighted.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Determine which scenario causes the largest deviation from original net profit
        fric_deviation = abs(orig_net - fric_net)
        decay_deviation = abs(orig_net - decay_net)
        worst_scenario = "Friction Stress" if fric_deviation >= decay_deviation else "Edge Decay"

        def fmt_pf(pf):
            return f"{pf:.2f}" if pf != float("inf") else "INF"

        # Build table as styled HTML
        highlight_fric = "border: 2px solid #f87171;" if worst_scenario == "Friction Stress" else ""
        highlight_decay = "border: 2px solid #f87171;" if worst_scenario == "Edge Decay" else ""

        summary_html = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div class="kelly-card">
                <div class="kelly-header" style="color: #60a5fa;">Current Backtest</div>
                <div class="kelly-sub">Baseline (no stress applied)</div>
                <div style="margin-top: 14px;">
                    <div class="stat-row">
                        <span class="stat-label">Profit Factor</span>
                        <span class="stat-value">{fmt_pf(orig_pf)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Expectancy</span>
                        <span class="stat-value" style="color: {'#4ade80' if orig_exp >= 0 else '#f87171'};">
                            {fmt_human(orig_exp)}</span>
                    </div>
                    <div class="stat-row" style="border-bottom: none;">
                        <span class="stat-label">Net Profit</span>
                        <span class="stat-value" style="color: {'#4ade80' if orig_net >= 0 else '#f87171'};">
                            {fmt_human(orig_net)}</span>
                    </div>
                </div>
            </div>
            <div class="kelly-card" style="{highlight_fric}">
                <div class="kelly-header" style="color: #fbbf24;">Friction Stress (4%)</div>
                <div class="kelly-sub">Slippage penalty on every trade</div>
                <div style="margin-top: 14px;">
                    <div class="stat-row">
                        <span class="stat-label">Profit Factor</span>
                        <span class="stat-value">{fmt_pf(fric_pf)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Expectancy</span>
                        <span class="stat-value" style="color: {'#4ade80' if fric_exp >= 0 else '#f87171'};">
                            {fmt_human(fric_exp)}</span>
                    </div>
                    <div class="stat-row" style="border-bottom: none;">
                        <span class="stat-label">Net Profit</span>
                        <span class="stat-value" style="color: {'#4ade80' if fric_net >= 0 else '#f87171'};">
                            {fmt_human(fric_net)}</span>
                    </div>
                </div>
            </div>
            <div class="kelly-card" style="{highlight_decay}">
                <div class="kelly-header" style="color: #f97316;">Edge Decay (16%)</div>
                <div class="kelly-sub">Haircut on all winning trades</div>
                <div style="margin-top: 14px;">
                    <div class="stat-row">
                        <span class="stat-label">Profit Factor</span>
                        <span class="stat-value">{fmt_pf(decay_pf)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Expectancy</span>
                        <span class="stat-value" style="color: {'#4ade80' if decay_exp >= 0 else '#f87171'};">
                            {fmt_human(decay_exp)}</span>
                    </div>
                    <div class="stat-row" style="border-bottom: none;">
                        <span class="stat-label">Net Profit</span>
                        <span class="stat-value" style="color: {'#4ade80' if decay_net >= 0 else '#f87171'};">
                            {fmt_human(decay_net)}</span>
                    </div>
                </div>
            </div>
        </div>
        <div style="margin-top: 16px; padding: 14px 20px; background: #1a1d23;
            border: 1px solid #22252b; border-left: 3px solid #f87171;
            border-radius: 8px; font-size: 14px; color: #c9cdd5;">
            <strong style="color: #f87171;">Largest deviation:</strong>
            <strong>{worst_scenario}</strong> causes the biggest impact —
            net profit drops by <span style="font-family: JetBrains Mono, monospace;
            color: #f87171;">{fmt_human(max(fric_deviation, decay_deviation))}</span>
            from the original backtest
            ({((max(fric_deviation, decay_deviation) / abs(orig_net)) * 100) if orig_net != 0 else 0:,.1f}% deviation).
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)


# =========================================================================
# FOOTER
# =========================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#3a3f4b; font-size:12px; '
    'font-family: Space Grotesk, sans-serif; letter-spacing: 0.5px;">'
    'Anarchy Terminal &middot; Monte Carlo &middot; '
    'Kelly Criterion &middot; Value at Risk &middot; Risk Laboratory</p>',
    unsafe_allow_html=True,
)
