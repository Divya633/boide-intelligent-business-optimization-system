import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mini_llm import MiniLLM
from utils.data_loader import load_data, init_session_state, next_page_button
from utils.segmentation_model import run_segmentation
from utils.ui import load_css, topnav, apply_theme, kpi_grid, divider
from utils.decision_panel import (
    ai_insight_strip,
    flow_indicator,
    simulate_decision_button,
    model_explain,
    decision_card,
    biz_label,
)

st.set_page_config(page_title="Segmentation - BOIDE", layout="wide", page_icon=":bar_chart:")
init_session_state()
load_css()
topnav("Segmentation")

flow_indicator("insight")
st.title("Who are your customers and what do they need?")
st.caption("RFM-based clustering with automatic cluster selection to identify VIP customers, loyal buyers, and lapsed segments")

df = load_data()


@st.cache_data(show_spinner=False)
def get_segmentation_payload(_df_len: int):
    return run_segmentation(load_data())


with st.spinner("Clustering customers with RFM model selection..."):
    segmentation = get_segmentation_payload(len(df))

customer = segmentation.customers
order_level = segmentation.order_level
sil = segmentation.silhouette_score

st.success(f"{len(customer):,} unique customers segmented into {segmentation.selected_k} groups")

high_count = int(segmentation.segment_counts.get("High Value", 0))
mid_count = int(segmentation.segment_counts.get("Mid Value", 0) + segmentation.segment_counts.get("Loyal", 0))
low_count = int(len(customer) - high_count - mid_count)
hv_pct = high_count / max(len(customer), 1) * 100

st.session_state["segments"] = {
    "high_value": high_count,
    "total": len(customer),
    "mid_value": mid_count,
    "low_value": low_count,
}
st.session_state["segment_details"] = {
    "silhouette_score": float(sil),
    "selected_k": int(segmentation.selected_k),
    "candidate_scores": segmentation.candidate_scores,
    "segment_counts": segmentation.segment_counts,
}

color_map = {
    "High Value": "#2563eb",
    "Mid Value": "#7c3aed",
    "Loyal": "#0f766e",
    "Discount-Sensitive": "#f59e0b",
    "Occasional": "#10b981",
}

st.subheader("Customer Group Overview")
kpi_grid([
    {"label": "Total Customers", "value": f"{len(customer):,}"},
    {"label": "VIP Customers", "value": f"{high_count:,}", "delta": f"{hv_pct:.1f}% of total", "delta_type": "positive"},
    {"label": "Selected Clusters", "value": f"{segmentation.selected_k}"},
    {"label": biz_label("Silhouette Score"), "value": f"{sil:.3f}", "delta": "Auto-selected best separation", "delta_type": "positive"},
])

divider()

st.subheader("Model Selection")
st.dataframe(segmentation.candidate_scores, use_container_width=True, hide_index=True)
st.caption("The dashboard tests multiple cluster counts and keeps the best silhouette score instead of forcing a fixed k.")

divider()

st.subheader("Revenue from Each Customer Group")
seg_rev = customer.groupby("segment")["monetary"].sum().reset_index().rename(columns={"monetary": "Total Revenue"})
fig_pie = go.Figure(go.Pie(
    values=seg_rev["Total Revenue"],
    labels=seg_rev["segment"],
    hole=0.5,
    marker=dict(colors=[color_map.get(s, "#94a3b8") for s in seg_rev["segment"]]),
    textinfo="label+percent",
))
fig_pie = apply_theme(fig_pie)
fig_pie.update_layout(title="Which customer group drives the most revenue?", showlegend=True)
st.plotly_chart(fig_pie, use_container_width=True)

divider()

st.subheader("Customer Cluster Map")
sample = customer.sample(min(5000, len(customer)), random_state=42)
fig_scatter = px.scatter(
    sample,
    x="frequency",
    y="monetary",
    color="segment",
    title="Purchase Frequency vs Total Spend",
    labels={"frequency": "Orders", "monetary": "Total Spend (R$)", "segment": "Customer Group"},
    color_discrete_map=color_map,
    opacity=0.65,
    hover_data={"recency": True},
)
fig_scatter = apply_theme(fig_scatter)
fig_scatter.update_traces(marker=dict(size=5))
st.plotly_chart(fig_scatter, use_container_width=True)

divider()

st.subheader("Group Performance Metrics")
seg_stats = segmentation.segment_summary.rename(
    columns={
        "segment": "Customer Group",
        "customers": "Customers",
        "avg_spend": "Avg Spend (R$)",
        "total_revenue": "Total Revenue (R$)",
        "avg_orders": "Avg Orders",
        "avg_recency": "Days Since Last Buy",
    }
)
for col in ["Avg Spend (R$)", "Total Revenue (R$)", "Avg Orders", "Days Since Last Buy"]:
    seg_stats[col] = seg_stats[col].round(2)
st.dataframe(seg_stats, use_container_width=True, hide_index=True)

divider()

cust_orders_cnt = order_level.groupby("customer_unique_id")["order_id"].nunique()
repeat_count = int((cust_orders_cnt > 1).sum())
repeat_rt = repeat_count / max(len(cust_orders_cnt), 1) * 100
benchmark_gap = 20 - repeat_rt
gap_label = f"Gap: {benchmark_gap:.1f}%" if benchmark_gap >= 0 else f"Above benchmark by {abs(benchmark_gap):.1f}%"

st.subheader("Customer Loyalty Rate")
col_rp1, col_rp2, col_rp3 = st.columns(3)
col_rp1.metric(biz_label("repeat_rate"), f"{repeat_rt:.1f}%", delta=f"{repeat_count:,} repeat buyers", delta_color="off")
col_rp2.metric("One-Time Buyers", f"{len(cust_orders_cnt) - repeat_count:,}", delta="bought only once", delta_color="off")
col_rp3.metric("Industry Benchmark", "20-30%", delta=gap_label, delta_color="inverse" if benchmark_gap >= 0 else "normal")

if hv_pct < 5:
    decision_card(
        title=f"VIP base is too small at {hv_pct:.1f}%",
        action="Create a high-value retention program with exclusive support, early access, and shipping perks.",
        priority="high",
        icon="",
        evidence=f"VIP customers = {hv_pct:.1f}% of base",
    )
else:
    decision_card(
        title=f"VIP base is healthy at {hv_pct:.1f}%",
        action="Protect high-value segments with loyalty offers and personalized premium upsell journeys.",
        priority="low",
        icon="",
        evidence=f"VIP customers = {hv_pct:.1f}% of base",
    )

with st.expander("What do these technical terms mean?", expanded=False):
    st.markdown(f"""
    | Business Term | Technical Term | Your Value |
    |---|---|---|
    | **{biz_label("Silhouette Score")}** | Silhouette Score | {sil:.3f} |
    | **VIP Customer Share** | hv_pct | {hv_pct:.1f}% |
    | **{biz_label("repeat_rate")}** | Repeat Purchase Rate | {repeat_rt:.1f}% |
    """)
    st.markdown(f"> **{model_explain('KMeans')}**")
    st.markdown(f"> **{model_explain('Silhouette Score')}**")

ai_insight_strip(
    f"{len(customer):,} customers were segmented with the best-performing cluster count of {segmentation.selected_k}. "
    f"Silhouette score is {sil:.3f}, repeat purchase rate is {repeat_rt:.2f}%, and VIP customers make up {hv_pct:.2f}% of the base.",
    label="AI: Customer Intelligence Summary",
)

with st.spinner("AI analysing customer segments..."):
    result = MiniLLM().run(
        df,
        segments=st.session_state.get("segments"),
        segment_details=st.session_state.get("segment_details"),
    )

ai_insight_strip(result["summary"], label="AI: Customer Segmentation Summary")
segment_rec = next(
    (
        rec
        for rec in result["recommendations"]
        if any(k in rec["insight"].lower() for k in ["customer", "repeat", "loyal", "vip", "segment"])
    ),
    result["recommendations"][0] if result["recommendations"] else None,
)
if segment_rec:
    decision_card(
        title=segment_rec["insight"],
        action=segment_rec["action"],
        priority="high" if segment_rec["tag"] == "risk" else "medium",
        evidence=segment_rec["evidence"],
        icon="",
    )

simulate_decision_button("Increase Marketing", source_page="segmentation")
next_page_button("Anomaly Detection", "pages/5_Anomaly_Detection.py")
