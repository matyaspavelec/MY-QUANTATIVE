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
    page_title="Quantitative Trading Lab",
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
    <h1>Quantitative Trading Lab</h1>
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

    tab_mc, tab_kelly, tab_var = st.tabs(["Monte Carlo Simulator", "Kelly Criterion Analysis", "Risk Analytics (VaR)"])

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

                if bundle_toggle and n_simulations > 10:
                    bundle_size = 10
                    n_bundles = n_simulations // bundle_size
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
                    max_plot = min(n_simulations, 2000)
                    indices = (np.random.choice(n_simulations, max_plot, replace=False)
                               if n_simulations > max_plot else np.arange(n_simulations))
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
                        <div class="metric-value neutral">{n_simulations:,}</div>
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

                if method_key == "permutation":
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
                # Calculate equity curves
                fig_kelly = go.Figure()

                results = {}
                for name, fraction in kelly_variants.items():
                    eq = apply_kelly_sizing(pnl_values, kelly, fraction)
                    results[name] = eq
                    fig_kelly.add_trace(go.Scattergl(
                        x=np.arange(1, len(eq) + 1),
                        y=eq,
                        mode="lines",
                        name=f"{name} ({fraction * 100:.1f}%)",
                        line=dict(color=colors[name], width=2.5),
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

                cols = st.columns(len(kelly_variants))
                for idx, (name, fraction) in enumerate(kelly_variants.items()):
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
                            <div class="kelly-header" style="color: {colors[name]};">{name}</div>
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

        col_mc_var_nsim, col_mc_var_run, _ = st.columns([2, 2, 4], gap="medium")

        with col_mc_var_nsim:
            var_n_sims = st.number_input(
                "Number of Simulations",
                min_value=100,
                max_value=50000,
                value=2000,
                step=500,
                key="var_nsims",
            )

        with col_mc_var_run:
            st.markdown("<br>", unsafe_allow_html=True)
            var_run_clicked = st.button("Run Monte Carlo VaR", use_container_width=True,
                                         key="var_mc_run")

        if var_run_clicked:
            with st.spinner("Running Monte Carlo VaR simulations..."):
                _, mc_var_curves, _ = run_monte_carlo(
                    pnl_values, int(var_n_sims), method="resample"
                )

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


# =========================================================================
# FOOTER
# =========================================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#3a3f4b; font-size:12px; '
    'font-family: Space Grotesk, sans-serif; letter-spacing: 0.5px;">'
    'Quantitative Trading Lab &middot; Monte Carlo &middot; '
    'Kelly Criterion &middot; Value at Risk</p>',
    unsafe_allow_html=True,
)
