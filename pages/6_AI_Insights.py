import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_data, init_session_state, next_page_button
from utils.mini_llm import MiniLLM
from utils.decision_panel import (
    flow_indicator,
    business_label,
    decision_card,
    biz_label,
    ai_insight_strip,
)
from utils.ui import load_css, topnav, page_title, section_header, kpi_grid, insight_card, risk_bar, apply_theme, divider

st.set_page_config(page_title="AI Insights - BOIDE", layout="wide", page_icon=":bar_chart:")
load_css()
init_session_state()

topnav("AI Insights")
flow_indicator("decision")
page_title(
    "AI Insights Engine",
    "MINI-LLM  RULE-BASED REASONING  EXPLAINABLE INTELLIGENCE  NO EXTERNAL APIS",
)

df = load_data()
llm = MiniLLM()
forecast = st.session_state.get("forecast")
segments = st.session_state.get("segments")
anomalies = st.session_state.get("anomalies", 0)

missing = []
if forecast is None:
    missing.append("Forecasting")
if segments is None:
    missing.append("Segmentation")
if anomalies == 0:
    missing.append("Anomaly Detection")

if missing:
    st.warning(
        f"Modules not yet run: **{', '.join(missing)}**  \n"
        "Insights will still generate, but confidence will be lower."
    )

divider()

section_header("", "Signals the engine will analyse")
signals_data = [
    ("Revenue", "Total + monthly MoM trend", "payment_value"),
    ("AOV", "Avg order value", "payment_value.mean()"),
    ("Volatility (CV)", "Std / mean", "payment_value.std()/mean"),
    ("Review quality", "Avg score + low-star %", "review_score"),
    ("Cancel rate", "% of cancelled orders", "order_status"),
    ("Delivery delay", "% late + avg lag days", "delivery_delay_days"),
    ("Category spread", "Top-3 revenue concentration", "category_en"),
    ("Forecast signal", "% vs 7-day rolling baseline", "session: forecast"),
    ("VIP customer %", "High-value cluster share", "session: segments"),
    ("Anomaly count", "Flagged days by IsolationForest", "session: anomalies"),
]
st.dataframe(pd.DataFrame(signals_data, columns=["Signal", "Description", "Source"]), use_container_width=True, hide_index=True)

divider()

if st.button("Generate AI Insights & Recommendations", use_container_width=True, type="primary"):
    with st.spinner("Running Mini-LLM reasoning engine..."):
        result = llm.run(
            df,
            forecast,
            segments,
            anomalies,
            forecast_details=st.session_state.get("forecast_details"),
            segment_details=st.session_state.get("segment_details"),
            anomaly_details=st.session_state.get("anomaly_details"),
        )

    ctx = result["context"]
    recs = result["recommendations"]
    conf = result["confidence"]
    risk = result["risk_score"]

    section_header("", "Executive Summary")
    st.info(result["summary"])

    section_header("", "Data Signals Used")
    kpi_grid([
        {"label": "Revenue", "value": f"R${ctx['revenue']/1e6:.2f}M"},
        {"label": "Avg Order", "value": f"R${ctx['avg_order']:.0f}"},
        {"label": "Volatility CV", "value": f"{ctx['cv']:.2f}"},
        {"label": "Avg Review", "value": f"{ctx['avg_review']:.2f}/5"},
        {"label": "Cancel Rate", "value": f"{ctx['cancel_rate']:.1f}%"},
        {"label": "Late Delivery", "value": f"{ctx['late_pct']:.1f}%"},
    ])

    divider()

    section_header("", "Risk & Confidence Scores")
    col_r, col_c = st.columns(2)

    with col_r:
        risk_bar("Composite Risk Score", risk, 100)
        if risk > 65:
            st.error("High risk - immediate action recommended")
        elif risk > 35:
            st.warning("Moderate risk - monitor KPIs closely")
        else:
            st.success("Risk within acceptable bounds")

    with col_c:
        conf_color = "#10b981" if conf > 70 else "#f59e0b" if conf > 40 else "#ef4444"
        st.markdown(
            f"""
            <div class="risk-bar-wrap">
                <div class="risk-bar-label">
                    <span>Confidence Score</span>
                    <span style="color:{conf_color}; font-weight:500;">{conf}%</span>
                </div>
                <div class="risk-bar-track">
                    <div class="risk-bar-fill"
                         style="width:{conf}%; background:linear-gradient(90deg,{conf_color},{conf_color}88);">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if conf > 70:
            st.success(f"High confidence ({conf}%) - all modules connected")
        elif conf > 40:
            st.warning(f"Moderate confidence ({conf}%) - run missing modules")
        else:
            st.error(f"Low confidence ({conf}%) - manual review recommended")

    section_header("", "Risk Dimension Breakdown")
    dims = ["Revenue", "Operations", "Product Quality", "Delivery", "Customer"]
    cv_n = min(100, ctx["cv"] * 25)
    op_n = min(100, ctx["cancel_rate"] * 12 + ctx["anomaly_count"] * 3)
    qual_n = min(100, max(0, (4.0 - ctx["avg_review"]) * 33) + ctx["low_review_pct"] * 1.5)
    del_n = min(100, ctx["late_pct"] * 1.5 + max(0, ctx["avg_delay"]) * 4)
    cust_n = min(100, (ctx["lv_pct"] or 30))

    fig_radar = go.Figure(go.Scatterpolar(
        r=[cv_n, op_n, qual_n, del_n, cust_n, cv_n],
        theta=dims + [dims[0]],
        fill="toself",
        fillcolor="rgba(59,130,246,0.1)",
        line=dict(color="#3b82f6", width=1.5),
        marker=dict(color="#06b6d4", size=6),
    ))
    fig_radar = apply_theme(fig_radar)
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(148,163,184,0.20)",
                tickfont=dict(family="DM Mono", size=9, color="#64748b"),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.20)",
                tickfont=dict(family="DM Sans", size=11, color="#334155"),
            ),
        ),
        showlegend=False,
        height=320,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    divider()

    section_header("", f"Insights & Recommendations - {len(recs)} triggered")
    if not recs:
        st.success("All metrics within normal ranges - no significant issues detected.")
    else:
        for group_label, tag_key, icon in [
            ("Risk Alerts", "risk", ""),
            ("Growth Signals", "growth", ""),
            ("Strategic Opportunities", "opportunity", ""),
        ]:
            group = [r for r in recs if r["tag"] == tag_key]
            if not group:
                continue
            st.markdown(f"**{icon} {group_label}**")
            for rec in group:
                insight_card(rec)

    divider()

    section_header("", "Simulate a Strategy in Digital Twin")
    top_tag = recs[0]["tag"] if recs else "neutral"
    default = {
        "risk": "Reduce Risk",
        "growth": "Increase Marketing",
        "opportunity": "Optimise Pricing",
        "neutral": "No Action",
    }.get(top_tag, "No Action")
    options = ["Increase Marketing", "Optimise Pricing", "Reduce Risk", "No Action"]
    strategy = st.selectbox("Select a strategy", options, index=options.index(default))
    if st.button("Send to Digital Twin"):
        st.session_state["decision"] = strategy
        st.success(f"Strategy **{strategy}** saved - navigate to **Digital Twin**")

else:
    st.markdown(
        """
        <div style="background:#ffffff; border:1px solid rgba(148,163,184,0.22);
                    border-radius:12px; padding:32px; text-align:center; margin-top:8px;">
            <div style="font-family:'Syne',sans-serif; font-size:20px;
                        color:#0f172a; font-weight:700; margin-bottom:10px;">
                Mini-LLM ready to run
            </div>
            <div style="font-family:'DM Mono',monospace; font-size:12px;
                        color:#64748b; line-height:2;">
                19 weighted rules &nbsp;&nbsp; 5 business domains &nbsp;&nbsp; richer model-quality signals<br>
                No external API &nbsp;&nbsp; Fully explainable &nbsp;&nbsp; Presentation-ready
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
ai_insight_strip(
    "The Mini-LLM combines business signals from forecasting, segmentation, and anomaly detection to generate explainable recommendations.",
    label="AI: How these recommendations were generated",
)

st.subheader("What should you do?")
decision_card(
    title="Run the top recommendation through the Digital Twin simulator",
    action="Select the highest-priority recommendation above and click Simulate. The Digital Twin will show the projected revenue impact before you commit.",
    priority="high",
    icon="",
    evidence="Mini-LLM decision layer",
)

if "ai_insights_result" not in st.session_state:
    with st.spinner("Generating AI summary..."):
        _ai_res = llm.run(
            df,
            forecast,
            segments,
            anomalies,
            forecast_details=st.session_state.get("forecast_details"),
            segment_details=st.session_state.get("segment_details"),
            anomaly_details=st.session_state.get("anomaly_details"),
        )
        st.session_state["ai_insights_result"] = _ai_res
_ai_res = st.session_state["ai_insights_result"]

st.divider()
ai_insight_strip(_ai_res["summary"], label="AI: Top Business Recommendation")

if _ai_res["recommendations"]:
    _top = _ai_res["recommendations"][0]
    st.subheader("Top recommended action")
    decision_card(
        title=_top["insight"],
        action=_top["action"],
        priority="high" if _top["tag"] == "risk" else "medium" if _top["tag"] == "growth" else "low",
        evidence=_top["evidence"],
        icon="",
    )

st.divider()
st.markdown("**Simulate this recommendation in Digital Twin:**")
_col1, _col2 = st.columns([3, 1])
with _col1:
    _action = _ai_res["recommendations"][0]["action"][:80] if _ai_res["recommendations"] else "Review strategy"
    st.markdown(
        f'<div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);'
        f'border-radius:8px;padding:10px 14px;font-size:13px;color:#334155;">'
        f'<strong>Strategy:</strong> {_action}</div>',
        unsafe_allow_html=True,
    )
with _col2:
    if st.button("Simulate in Digital Twin", key="goto_twin_ai", use_container_width=True, type="primary"):
        st.session_state["decision"] = _ai_res["recommendations"][0]["action"] if _ai_res["recommendations"] else "Review strategy"
        st.session_state["decision_source"] = "ai_insights"
        st.switch_page("pages/7_Digital_Twin.py")

next_page_button("Digital Twin", "pages/7_Digital_Twin.py")
