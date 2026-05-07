import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_data, init_session_state, next_page_button
from utils.mini_llm import MiniLLM
from utils.anomaly_model import detect_anomalies
from utils.decision_panel import (
    decision_panel,
    ai_insight_strip,
    flow_indicator,
    simulate_decision_button,
    anomaly_decisions,
    business_label,
    model_explain,
    decision_card,
)
from utils.ui import load_css, topnav, apply_theme, divider

st.set_page_config(page_title="Anomaly Detection - BOIDE", layout="wide")
init_session_state()
load_css()
topnav("Anomaly")
flow_indicator("insight")

st.title("When did your revenue behave unexpectedly?")
st.caption("Multivariate anomaly detection using Isolation Forest across revenue, orders, AOV, cancellation rate, and delivery signals")

df = load_data()


@st.cache_data(show_spinner=False)
def get_anomaly_payload(_df_len: int):
    return detect_anomalies(load_data())


result = get_anomaly_payload(len(df))
daily = result.daily_features
anomalies_df = result.anomalies.copy()

st.session_state["anomalies"] = result.anomaly_count
st.session_state["anomaly_details"] = {
    "anomaly_rate": float(result.anomaly_rate),
    "high_severity_count": int(result.high_severity_count),
    "severity_counts": result.severity_counts,
    "contamination": float(result.contamination),
}

normal_days_rate = f"{(1 - len(anomalies_df) / max(len(daily), 1)) * 100:.1f}%"
anomaly_indices = anomalies_df.index.tolist()
avg_detect = sum(anomaly_indices[i + 1] - anomaly_indices[i] for i in range(len(anomaly_indices) - 1)) / max(len(anomaly_indices) - 1, 1) if len(anomaly_indices) > 1 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Anomalies Detected", result.anomaly_count)
col2.metric("High Severity", result.high_severity_count)
col3.metric("Normal Days Rate", normal_days_rate)
col4.metric("Avg Gap Between Anomalies", f"{avg_detect:.0f} days")
col5.metric("Anomaly Rate", f"{result.anomaly_rate:.2f}%")

divider()

st.subheader("Sales Pattern & Anomaly Timeline")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=daily["date"],
    y=daily["revenue"],
    mode="lines",
    name="Daily Revenue",
    line=dict(color="#2563eb", width=1.5),
))
fig1.add_hrect(
    y0=float(daily["revenue"].mean() - daily["revenue"].std()),
    y1=float(daily["revenue"].mean() + daily["revenue"].std()),
    fillcolor="rgba(37,99,235,0.08)",
    line_width=0,
    annotation_text="Revenue normal zone",
    annotation_position="top left",
)
for sev, color in [("High", "#ef4444"), ("Medium", "#f59e0b"), ("Low", "#10b981")]:
    subset = anomalies_df[anomalies_df["severity"] == sev]
    if not subset.empty:
        fig1.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["revenue"],
            mode="markers",
            name=f"{sev} Anomaly",
            marker=dict(color=color, size=11, symbol="x"),
        ))
fig1 = apply_theme(fig1)
fig1.update_layout(title="Revenue with Anomaly Markers", xaxis_title="Date", yaxis_title="Revenue (R$)")
st.plotly_chart(fig1, use_container_width=True)

divider()

st.subheader("Detected Anomalies")
table = anomalies_df[["date", "revenue", "orders", "avg_order_value", "cancel_rate", "severity", "possible_cause", "impact"]].copy()
table.columns = ["Date", "Revenue (R$)", "Orders", "Avg Order Value (R$)", "Cancellation Rate (%)", "Severity", "Possible Cause", "Impact"]
st.dataframe(table, use_container_width=True, hide_index=True)

divider()

st.subheader("Anomaly Severity by Month")
anom_monthly = anomalies_df.copy()
anom_monthly["month"] = anom_monthly["date"].dt.to_period("M").astype(str)
monthly = anom_monthly.groupby(["month", "severity"]).size().reset_index(name="Count")
fig2 = px.bar(monthly, x="month", y="Count", color="severity", barmode="stack", title="Monthly anomaly severity distribution")
fig2 = apply_theme(fig2)
fig2.update_layout(xaxis_title="Month", yaxis_title="Anomaly Count")
st.plotly_chart(fig2, use_container_width=True)

divider()

if result.high_severity_count > 3:
    st.error(f"{result.high_severity_count} high-severity anomalies detected - investigate payment, fraud, and logistics incidents immediately")
elif result.anomaly_count > 5:
    st.warning(f"{result.anomaly_count} anomalies detected - monitor the anomaly pattern closely and validate business events on flagged dates")
else:
    st.success("Anomaly rate is within a manageable band")

decisions = anomaly_decisions(result.high_severity_count, result.anomaly_count, result.anomaly_rate)
decision_panel(decisions, title="Anomaly Risk Decision Panel")

with st.expander("What do these numbers mean?", expanded=False):
    st.markdown(f"""
    | Business Term | Technical Term | Your Value |
    |---|---|---|
    | **{business_label("anomaly_count")}** | Anomaly Count | {result.anomaly_count} days |
    | **{business_label("contamination")}** | Contamination | {result.contamination:.2%} |
    | **Revenue Disruption Rate** | Anomaly Rate | {result.anomaly_rate:.1f}% |
    """)
    st.markdown(f"> **Isolation Forest:** {model_explain('IsolationForest')}")

ai_insight_strip(
    f"Isolation Forest scanned {len(daily):,} daily records using multiple business signals. "
    f"It found {result.anomaly_count} anomalies, including {result.high_severity_count} high-severity days that may reflect operational incidents rather than normal demand noise.",
    label="AI: Revenue Disruption Analysis",
)

with st.spinner("AI interpreting anomalies..."):
    llm_result = MiniLLM().run(
        df,
        anomalies=st.session_state.get("anomalies", 0),
        anomaly_details=st.session_state.get("anomaly_details"),
    )

ai_insight_strip(llm_result["summary"], label="AI: Revenue Disruption Summary")
anom_rec = next(
    (
        rec
        for rec in llm_result["recommendations"]
        if any(k in rec["insight"].lower() for k in ["anomal", "disrupt", "cancel", "risk", "unstable"])
    ),
    llm_result["recommendations"][0] if llm_result["recommendations"] else None,
)
if anom_rec:
    decision_card(
        title=anom_rec["insight"],
        action=anom_rec["action"],
        priority="high" if anom_rec["tag"] == "risk" else "medium",
        evidence=anom_rec["evidence"],
        icon="",
    )

simulate_decision_button("Reduce Risk", source_page="anomaly")
next_page_button("AI Insights", "pages/6_AI_Insights.py")
