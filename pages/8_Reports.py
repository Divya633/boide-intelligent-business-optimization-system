import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.anomaly_model import detect_anomalies
from utils.data_loader import get_monthly_order_revenue, init_session_state, load_data, next_page_button
from utils.decision_panel import ai_insight_strip, decision_card, flow_indicator
from utils.mini_llm import MiniLLM
from utils.segmentation_model import run_segmentation
from utils.ui import divider, kpi_grid, load_css, page_title, section_header, topnav


st.set_page_config(page_title="Reports - BOIDE", layout="wide", page_icon=":bar_chart:")
load_css()
init_session_state()
topnav("Reports")
flow_indicator("outcome")
page_title("Reports", "VIEW KEY SUMMARIES FIRST, THEN DOWNLOAD CSV AND JSON EXPORTS")

df = load_data()
forecast = st.session_state.get("forecast")
segments = st.session_state.get("segments")
anomalies = st.session_state.get("anomalies", 0)
decision = st.session_state.get("decision") or "Not set"

forecast_details = st.session_state.get("forecast_details") or {}
segment_details = st.session_state.get("segment_details") or {}
anomaly_details = st.session_state.get("anomaly_details") or {}

if not segment_details:
    seg = run_segmentation(df)
    segment_details = {
        "silhouette_score": seg.silhouette_score,
        "selected_k": seg.selected_k,
        "candidate_scores": seg.candidate_scores,
        "segment_counts": seg.segment_counts,
    }
else:
    seg = run_segmentation(df)

anom = detect_anomalies(df)
if not anomaly_details:
    anomaly_details = {
        "anomaly_rate": anom.anomaly_rate,
        "high_severity_count": anom.high_severity_count,
        "severity_counts": anom.severity_counts,
        "contamination": anom.contamination,
    }

llm = MiniLLM()
result = llm.run(
    df,
    forecast,
    segments,
    anomalies,
    forecast_details=forecast_details,
    segment_details=segment_details,
    anomaly_details=anomaly_details,
)
ctx = result["context"]
recs = result["recommendations"]

kpi_grid([
    {"label": "Downloads", "value": "9"},
    {"label": "Forecast Model", "value": str(forecast_details.get("model_label", "Not run"))[:16]},
    {"label": "AI Confidence", "value": f"{result['confidence']}%"},
    {"label": "Risk Score", "value": f"{result['risk_score']}/100"},
])

divider()

section_header("", "At A Glance")
col1, col2 = st.columns(2)
with col1:
    st.dataframe(
        pd.DataFrame(
            [
                ["Total Revenue (R$)", round(ctx["revenue"], 2)],
                ["Avg Order Value (R$)", round(ctx["avg_order"], 2)],
                ["Total Orders", ctx["total_orders"]],
                ["Top Category", ctx["top_category"]],
                ["Late Delivery Rate (%)", round(ctx["late_pct"], 2)],
            ],
            columns=["Metric", "Value"],
        ),
        use_container_width=True,
        hide_index=True,
    )
with col2:
    st.dataframe(
        pd.DataFrame(
            [
                ["Selected Forecast Model", forecast_details.get("model_label", "Not run")],
                ["Forecast MAPE (%)", round(float(forecast_details.get("mape", 0.0)), 2) if forecast_details else "Not run"],
                ["Segmentation k", segment_details.get("selected_k", "Not run")],
                ["Segmentation Silhouette", round(float(segment_details.get("silhouette_score", 0.0)), 3) if segment_details else "Not run"],
                ["Anomaly Rate (%)", round(float(anomaly_details.get("anomaly_rate", 0.0)), 2)],
            ],
            columns=["Metric", "Value"],
        ),
        use_container_width=True,
        hide_index=True,
    )

divider()

section_header("", "Visible Report Previews")

st.subheader("Forecast Candidate Comparison")
forecast_candidates = pd.DataFrame(forecast_details.get("candidate_scores", []))
if forecast_candidates.empty:
    st.info("Run Forecasting to populate model comparison data.")
else:
    st.dataframe(forecast_candidates, use_container_width=True, hide_index=True)

st.subheader("Segmentation Summary")
segmentation_report = seg.segment_summary.rename(
    columns={
        "segment": "Segment",
        "customers": "Customers",
        "avg_spend": "Avg Spend (R$)",
        "total_revenue": "Total Revenue (R$)",
        "avg_orders": "Avg Orders",
        "avg_recency": "Days Since Last Buy",
    }
).round(2)
st.dataframe(segmentation_report, use_container_width=True, hide_index=True)

st.subheader("Latest Anomaly Findings")
anomaly_preview = anom.anomalies[["date", "revenue", "severity", "possible_cause"]].copy().tail(10)
anomaly_preview.columns = ["Date", "Revenue (R$)", "Severity", "Possible Cause"]
if anomaly_preview.empty:
    st.info("No anomalies available.")
else:
    st.dataframe(anomaly_preview, use_container_width=True, hide_index=True)

divider()

section_header("", "Download Reports")

kpi_report = pd.DataFrame(
    [
        ["Total Revenue (R$)", round(ctx["revenue"], 2)],
        ["Avg Order Value (R$)", round(ctx["avg_order"], 2)],
        ["Total Orders", ctx["total_orders"]],
        ["Avg Review Score", round(ctx["avg_review"], 2)],
        ["Cancellation Rate (%)", round(ctx["cancel_rate"], 2)],
        ["Revenue Volatility (CV)", round(ctx["cv"], 3)],
        ["Late Delivery Rate (%)", round(ctx["late_pct"], 2)],
        ["Top Category", ctx["top_category"]],
    ],
    columns=["Metric", "Value"],
)

category_report = (
    df.groupby("category_en")
    .agg(
        revenue=("item_revenue", "sum"),
        orders=("order_id", "nunique"),
        avg_review=("review_score", "mean"),
        avg_price=("price", "mean"),
    )
    .reset_index()
    .rename(
        columns={
            "category_en": "Category",
            "revenue": "Revenue (R$)",
            "orders": "Orders",
            "avg_review": "Avg Review",
            "avg_price": "Avg Price (R$)",
        }
    )
    .round(2)
    .sort_values("Revenue (R$)", ascending=False)
)

insight_report = pd.DataFrame(
    [{"Priority": r["badge"], "Tag": r["tag"].upper(), "Insight": r["insight"], "Action": r["action"], "Evidence": r["evidence"]} for r in recs]
) if recs else pd.DataFrame(columns=["Priority", "Tag", "Insight", "Action", "Evidence"])

monthly_report = get_monthly_order_revenue(df)
monthly_report.columns = ["Month", "Revenue (R$)"]
monthly_report["Month"] = monthly_report["Month"].astype(str)
monthly_report["Revenue (R$)"] = monthly_report["Revenue (R$)"].round(2)

forecast_report = pd.DataFrame(
    [
        ["Selected Forecast Model", forecast_details.get("model_label", "Not run")],
        ["Forecast MAPE (%)", round(float(forecast_details.get("mape", 0.0)), 2) if forecast_details else "Not run"],
        ["Forecast RMSE", round(float(forecast_details.get("rmse", 0.0)), 2) if forecast_details else "Not run"],
        ["Directional Accuracy (%)", round(float(forecast_details.get("directional_accuracy", 0.0)), 2) if forecast_details else "Not run"],
        ["Forecast Confidence (%)", forecast_details.get("confidence", "Not run")],
    ],
    columns=["Metric", "Value"],
)

anomaly_report = anom.anomalies[
    ["date", "revenue", "orders", "avg_order_value", "cancel_rate", "severity", "possible_cause", "impact"]
].copy()
anomaly_report.columns = ["Date", "Revenue (R$)", "Orders", "Avg Order Value (R$)", "Cancellation Rate (%)", "Severity", "Possible Cause", "Impact"]

simulation_report = pd.DataFrame(
    [
        ["Decision Strategy", decision],
        ["7-Day Forecast (R$)", round(forecast, 2) if forecast else "Not run"],
        ["Anomalies Detected", anomalies],
        ["AI Confidence (%)", result["confidence"]],
        ["Risk Score", result["risk_score"]],
        ["Selected Segment k", segment_details.get("selected_k", "Not run")],
    ],
    columns=["Parameter", "Value"],
)

reports = [
    ("KPI Summary", kpi_report, "kpi_summary"),
    ("Category Performance", category_report, "category_performance"),
    ("AI Insights", insight_report, "ai_insights"),
    ("Forecast Model Summary", forecast_report, "forecast_model_summary"),
    ("Forecast Candidate Comparison", forecast_candidates, "forecast_candidates"),
    ("Segmentation Summary", segmentation_report, "segmentation_summary"),
    ("Anomaly Report", anomaly_report, "anomaly_report"),
    ("Monthly Revenue", monthly_report, "monthly_revenue"),
    ("Simulation Summary", simulation_report, "simulation_summary"),
]

for title, data, fname in reports:
    with st.expander(f"{title} - {len(data)} rows", expanded=False):
        st.dataframe(data, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                f"Download {fname}.csv",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name=f"{fname}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"csv_{fname}",
            )
        with c2:
            st.download_button(
                f"Download {fname}.json",
                data=data.to_json(orient="records", indent=2).encode("utf-8"),
                file_name=f"{fname}.json",
                mime="application/json",
                use_container_width=True,
                key=f"json_{fname}",
            )

divider()

ai_insight_strip(
    f"{result['summary']} Confidence: {result['confidence']}%. Risk: {result['risk_score']}/100.",
    label="AI: Final Business Intelligence Summary",
)

if recs:
    top = recs[0]
    decision_card(
        title=top["insight"],
        action=top["action"],
        priority="high" if top["tag"] == "risk" else "medium" if top["tag"] == "growth" else "low",
        evidence=top["evidence"],
        icon="",
    )

decision_card(
    title="Continue to Methodology",
    action="Review the dataset, preprocessing, models, metrics, and limitations behind the BOIDE workflow.",
    priority="low",
    icon="",
    evidence="Reports page -> Methodology",
)

next_page_button("Methodology", "pages/9_Methodology.py")
