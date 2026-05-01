"""
app.py
RetailSense — AI-Powered Retail Demand Intelligence
Main Streamlit application.
"""

import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailSense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Guard: database must exist before anything else
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "retail.db")


def _db_ready() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM weekly_sales").fetchone()[0]
        conn.close()
        return n > 0
    except Exception:
        return False


if not _db_ready():
    # On Streamlit Cloud: auto-build from demo_data/ if available
    try:
        import data_loader
        with st.spinner("Building database from demo data..."):
            data_loader.load_rossmann_data()
    except Exception as exc:
        st.error(
            "**Database not found.**\n\n"
            "Place `train.csv` and `store.csv` in `/data` then run:\n\n"
            "```\npython data_loader.py\n```\n\n"
            f"Error: {exc}"
        )
        st.stop()

# ---------------------------------------------------------------------------
# Lazy imports (prophet is slow)
# ---------------------------------------------------------------------------
import analyst
import forecaster

# ---------------------------------------------------------------------------
# Custom CSS — portfolio-grade styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global ────────────────────────────────────────────────── */
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
    }
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ── Metric cards ──────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    /* ── Analyst card ──────────────────────────────────────────── */
    .analyst-card {
        background: #161b22;
        border-left: 4px solid #1f6feb;
        border-radius: 0 10px 10px 0;
        padding: 18px 22px;
        margin: 12px 0;
        color: #c9d1d9;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .analyst-card h4 {
        color: #58a6ff;
        margin: 0 0 10px 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Section headers ───────────────────────────────────────── */
    .section-header {
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin: 24px 0 14px 0;
    }

    /* ── Anomaly pill ──────────────────────────────────────────── */
    .anomaly-pill {
        display: inline-block;
        background: #3d1a1a;
        color: #f85149;
        border: 1px solid #6e1a1a;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* ── Free-mode banner ──────────────────────────────────────── */
    .free-mode-banner {
        background: #2d2208;
        border: 1px solid #6e4a08;
        border-radius: 8px;
        padding: 10px 14px;
        color: #d29922;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }

    /* ── Sidebar store info ────────────────────────────────────── */
    .store-info-block {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 RetailSense")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.85rem;'>"
        "AI-Powered Retail Demand Intelligence"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not analyst.AI_AVAILABLE:
        st.markdown(
            "<div class='free-mode-banner'>"
            "⚡ <strong>Free Mode</strong><br>"
            "Add <code>ANTHROPIC_API_KEY</code> to <code>.env</code> "
            "to unlock AI analyst narratives."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ AI Analyst Active", icon="🤖")

    st.divider()

    # Store selector
    store_ids = forecaster.get_all_store_ids()
    selected_store = st.selectbox(
        "Select Store",
        options=store_ids,
        format_func=lambda x: f"Store {x}",
    )

    forecast_weeks = st.slider(
        "Weeks to Forecast",
        min_value=4,
        max_value=26,
        value=12,
        step=1,
    )

    run_btn = st.button("▶  Run Forecast", type="primary", use_container_width=True)

    st.divider()

    # About this data
    try:
        meta = forecaster.get_db_meta()
        st.markdown("**About This Data**")
        st.markdown(
            f"<div class='store-info-block'>"
            f"📦 <strong>Source:</strong> Rossmann Store Sales (Kaggle)<br>"
            f"🏪 <strong>Stores loaded:</strong> {meta['total_stores']}<br>"
            f"📅 <strong>Date range:</strong> {meta['min_date']} → {meta['max_date']}<br>"
            f"🔮 <strong>Model:</strong> Facebook Prophet"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Main area — page title
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#e6edf3;font-size:2rem;font-weight:700;margin-bottom:4px;'>"
    "📈 RetailSense</h1>"
    "<p style='color:#8b949e;margin-top:0;'>AI-Powered Retail Demand Forecasting</p>",
    unsafe_allow_html=True,
)

tab_forecast, tab_compare = st.tabs(["📊 Store Forecast", "🏪 Store Comparison"])

# ===========================================================================
# TAB 1 — Store Forecast
# ===========================================================================
with tab_forecast:
    if not run_btn:
        st.markdown(
            "<div style='text-align:center;color:#8b949e;padding:60px 0;'>"
            "<div style='font-size:3rem;'>🔮</div>"
            "<div style='font-size:1.1rem;margin-top:12px;'>"
            "Select a store and click <strong>Run Forecast</strong> to begin.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner(f"Running Prophet model for Store {selected_store}… this takes about 10 seconds"):
            try:
                result = forecaster.run_forecast(selected_store, forecast_weeks)
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                st.stop()

        forecast_df = result["forecast_df"]
        historical_df = result["historical_df"]
        anomalies = result["anomalies"]
        meta_store = result["store_metadata"]
        stats = result["summary_stats"]

        # ── Metric row ──────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Forecast Sales",
            f"€{stats['total_forecast_sales']:,.0f}",
        )
        c2.metric("Peak Week", stats["peak_week"])
        c3.metric("Trough Week", stats["trough_week"])
        c4.metric("Anomalies Detected", len(anomalies))

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        # ── Store badge row ──────────────────────────────────────────────────
        st.markdown(
            f"<span style='color:#8b949e;font-size:0.85rem;'>"
            f"Store Type: <strong style='color:#58a6ff;'>{meta_store['store_type']}</strong> &nbsp;|&nbsp; "
            f"Assortment: <strong style='color:#58a6ff;'>{meta_store['assortment']}</strong> &nbsp;|&nbsp; "
            f"Competition Distance: <strong style='color:#58a6ff;'>{meta_store['competition_distance']:,.0f} m</strong>"
            f"</span>",
            unsafe_allow_html=True,
        )

        # ── Plotly chart ────────────────────────────────────────────────────
        hist_dates = pd.to_datetime(historical_df["week_start"])
        hist_sales = historical_df["total_sales"]

        future_mask = ~forecast_df["ds"].isin(hist_dates)
        fc_future = forecast_df[future_mask]
        fc_insample = forecast_df[~future_mask]

        # Anomaly points
        anomaly_dates = [pd.Timestamp(a["date"]) for a in anomalies]
        anomaly_sales = [a["actual"] for a in anomalies]

        split_date = hist_dates.max()

        fig = go.Figure()

        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fc_future["ds"], fc_future["ds"].iloc[::-1]]),
                y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(255,165,0,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval",
                hoverinfo="skip",
            )
        )

        # Historical actuals
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=hist_sales,
                mode="lines",
                name="Actual Sales",
                line=dict(color="#2188ff", width=2),
            )
        )

        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=fc_future["ds"],
                y=fc_future["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color="#f97316", width=2.5, dash="dash"),
            )
        )

        # Anomaly dots
        if anomaly_dates:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_sales,
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color="#f85149", size=9, symbol="circle-open", line=dict(width=2)),
                )
            )

        # Vertical split line (add_vline with annotations is broken in Plotly 6.7 on date axes)
        split_str = split_date.strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=split_str, x1=split_str,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot", color="#8b949e", width=1),
        )
        fig.add_annotation(
            x=split_str, y=1,
            xref="x", yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="#8b949e", size=11),
            yanchor="bottom",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            title=dict(
                text=f"Store {selected_store} — {forecast_weeks}-Week Demand Forecast",
                font=dict(color="#e6edf3", size=16),
            ),
            xaxis=dict(
                title="Week",
                gridcolor="#21262d",
                linecolor="#30363d",
                tickfont=dict(color="#8b949e"),
                title_font=dict(color="#8b949e"),
            ),
            yaxis=dict(
                title="Sales (EUR)",
                gridcolor="#21262d",
                linecolor="#30363d",
                tickfont=dict(color="#8b949e"),
                title_font=dict(color="#8b949e"),
                tickprefix="€",
            ),
            legend=dict(
                font=dict(color="#8b949e"),
                bgcolor="#161b22",
                bordercolor="#30363d",
                borderwidth=1,
            ),
            hovermode="x unified",
            margin=dict(t=50, b=40, l=60, r=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Analyst narrative ────────────────────────────────────────────────
        st.markdown(
            "<div class='section-header'>🧠 Analyst Summary</div>",
            unsafe_allow_html=True,
        )

        narrative = analyst.generate_forecast_narrative(
            store_id=selected_store,
            store_type=meta_store["store_type"],
            forecast_weeks=forecast_weeks,
            summary_stats=stats,
            anomaly_count=len(anomalies),
        )

        st.markdown(
            f"<div class='analyst-card'><h4>AI Analyst · Store {selected_store}</h4>{narrative}</div>",
            unsafe_allow_html=True,
        )

        # ── Anomaly detail ───────────────────────────────────────────────────
        if anomalies:
            st.markdown(
                "<div class='section-header'>🔴 Anomalies Detected</div>",
                unsafe_allow_html=True,
            )
            for a in anomalies:
                direction = "above" if a["pct_diff"] > 0 else "below"
                label = (
                    f"Week of {a['date']}  —  "
                    f"{abs(a['pct_diff']):.1f}% {direction} expected"
                )
                with st.expander(label):
                    cols = st.columns(3)
                    cols[0].metric("Actual Sales", f"€{a['actual']:,.0f}")
                    cols[1].metric("Expected Sales", f"€{a['expected']:,.0f}")
                    cols[2].metric("Deviation", f"{a['pct_diff']:+.1f}%")

                    flags = []
                    if a["had_promo"]:
                        flags.append("🏷️ Promotion Active")
                    if a["had_holiday"]:
                        flags.append("🎌 State Holiday")
                    if flags:
                        st.markdown(" &nbsp; ".join(flags), unsafe_allow_html=True)

                    explanation = analyst.explain_anomaly(
                        store_id=selected_store,
                        anomaly_date=a["date"],
                        actual=a["actual"],
                        expected=a["expected"],
                        pct_diff=a["pct_diff"],
                        had_promo=a["had_promo"],
                        had_holiday=a["had_holiday"],
                    )
                    st.markdown(
                        f"<div class='analyst-card' style='margin-top:8px;'>"
                        f"<h4>AI Observation</h4>{explanation}</div>",
                        unsafe_allow_html=True,
                    )

# ===========================================================================
# TAB 2 — Store Comparison
# ===========================================================================
with tab_compare:
    st.markdown(
        "<div class='section-header'>🏪 All Stores — Performance Overview</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading store summaries…"):
        all_store_ids = forecaster.get_all_store_ids()
        summaries = []
        for sid in all_store_ids:
            try:
                summaries.append(forecaster.get_store_summary(sid))
            except Exception:
                pass

    if summaries:
        # ── AI comparison paragraph ──────────────────────────────────────────
        comparison_text = analyst.generate_store_comparison(summaries)
        st.markdown(
            f"<div class='analyst-card'><h4>AI Analyst · Store Comparison</h4>{comparison_text}</div>",
            unsafe_allow_html=True,
        )

        # ── Summary table ────────────────────────────────────────────────────
        df_summary = pd.DataFrame(summaries)
        df_display = df_summary.rename(
            columns={
                "store_id": "Store ID",
                "store_type": "Store Type",
                "assortment": "Assortment",
                "avg_weekly_sales": "Avg Weekly Sales (€)",
                "competition_distance": "Competition Distance (m)",
                "last_4wk_avg": "Last 4-Wk Avg Sales (€)",
            }
        )

        # Format currency columns
        for col in ["Avg Weekly Sales (€)", "Last 4-Wk Avg Sales (€)"]:
            df_display[col] = df_display[col].apply(lambda x: f"€{x:,.0f}")
        df_display["Competition Distance (m)"] = df_display["Competition Distance (m)"].apply(
            lambda x: f"{x:,.0f} m"
        )

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )

        # ── Bar chart: avg weekly sales by store type ────────────────────────
        st.markdown(
            "<div class='section-header'>📊 Average Weekly Sales by Store Type</div>",
            unsafe_allow_html=True,
        )

        by_type = (
            df_summary.groupby("store_type")["avg_weekly_sales"]
            .mean()
            .reset_index()
            .sort_values("avg_weekly_sales", ascending=False)
        )

        bar_fig = go.Figure(
            go.Bar(
                x=by_type["store_type"],
                y=by_type["avg_weekly_sales"],
                marker=dict(
                    color=by_type["avg_weekly_sales"],
                    colorscale=[[0, "#1f6feb"], [0.5, "#388bfd"], [1, "#79c0ff"]],
                    showscale=False,
                ),
                text=by_type["avg_weekly_sales"].apply(lambda v: f"€{v:,.0f}"),
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            )
        )
        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            xaxis=dict(
                title="Store Type",
                gridcolor="#21262d",
                tickfont=dict(color="#8b949e"),
                title_font=dict(color="#8b949e"),
            ),
            yaxis=dict(
                title="Avg Weekly Sales (EUR)",
                gridcolor="#21262d",
                tickfont=dict(color="#8b949e"),
                title_font=dict(color="#8b949e"),
                tickprefix="€",
            ),
            margin=dict(t=30, b=40, l=60, r=20),
            height=350,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.warning("No store summaries could be loaded. Ensure the database is populated.")
