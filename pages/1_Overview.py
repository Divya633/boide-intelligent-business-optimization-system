import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import load_data, init_session_state, next_page_button
from utils.ui import load_css, topnav, page_title, kpi_grid, section_header, apply_theme, divider
from utils.decision_panel import (ai_insight_strip, flow_indicator,
                                   decision_card, simulate_decision_button, biz_label)
from utils.mini_llm import MiniLLM

st.set_page_config(page_title="Overview – BOIDE", layout="wide", page_icon="🧠")
load_css()
init_session_state()
topnav("Overview")

# ── ZONE 1: Flow + Context ─────────────────────────────────────
flow_indicator("data")
st.title("What is your business doing?")
st.caption("Revenue health, order trends, and category intelligence — based on 99,441 real Olist orders")

df = load_data()

# ── Compute KPIs ───────────────────────────────────────────────
revenue     = df["payment_value"].sum()
orders      = df["order_id"].nunique()
avg_order   = df["payment_value"].mean()
avg_review  = df["review_score"].mean() if "review_score" in df.columns else 0
cancel_rate = (df["order_status"] == "canceled").sum() / len(df) * 100
unique_cats = df["category_en"].nunique() if "category_en" in df.columns else 0

monthly        = df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["payment_value"].sum()
monthly_median = monthly.median()
monthly_complete = monthly[monthly >= monthly_median * 0.5]
mom = float((monthly_complete.iloc[-1] - monthly_complete.iloc[-2]) / monthly_complete.iloc[-2] * 100) \
      if len(monthly_complete) >= 2 else 0.0

cust_orders   = df.groupby("customer_unique_id")["order_id"].nunique()
repeat_rate   = float((cust_orders > 1).sum() / max(len(cust_orders), 1) * 100)
freight_ratio = 0.0
if "freight_value" in df.columns and "price" in df.columns:
    valid = df[(df["price"] > 0) & df["freight_value"].notna()]
    freight_ratio = float((valid["freight_value"] / valid["price"]).mean() * 100)

# ── ZONE 2: Evidence — KPIs ───────────────────────────────────
st.subheader("📊 Business Health Snapshot")
kpi_grid([
    {"label": "Total Revenue",              "value": f"R${revenue/1e6:.2f}M",
     "delta": f"{mom:+.1f}% MoM",          "delta_type": "positive" if mom >= 0 else "negative"},
    {"label": "Total Orders",               "value": f"{orders:,}"},
    {"label": biz_label("avg_order"),       "value": f"R${avg_order:.0f}"},
    {"label": "Customer Satisfaction",      "value": f"{avg_review:.2f}/5"},
    {"label": biz_label("cancel_rate"),     "value": f"{cancel_rate:.2f}%",
     "delta": "target < 2%",               "delta_type": "negative" if cancel_rate > 2 else "positive"},
    {"label": biz_label("repeat_rate"),     "value": f"{repeat_rate:.1f}%",
     "delta": "industry avg 20–30%",       "delta_type": "negative" if repeat_rate < 10 else "neutral"},
])

if freight_ratio > 25:
    st.warning(f"⚠️ **{biz_label('freight_ratio')} = {freight_ratio:.1f}%** of product price "
               f"— free shipping at R$150+ could improve conversion by 10–15%")

divider()

# ── Revenue trend chart ───────────────────────────────────────
st.subheader("📈 Revenue Over Time")
monthly_df = (
    df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["payment_value"]
    .sum().reset_index()
)
monthly_df["order_purchase_timestamp"] = monthly_df["order_purchase_timestamp"].astype(str)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=monthly_df["order_purchase_timestamp"], y=monthly_df["payment_value"],
    mode="lines+markers", line=dict(color="#3b82f6", width=2),
    marker=dict(color="#06b6d4", size=5),
    fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"
))
fig1 = apply_theme(fig1)
fig1.update_layout(title="Monthly Revenue (R$)", showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# AI strip after revenue chart
ai_insight_strip(
    f"Revenue totals R${revenue/1e6:.2f}M across {orders:,} orders. "
    f"Month-over-month trend is {mom:+.1f}%. "
    f"{'Growth is positive — consider scaling top categories.' if mom > 0 else 'Revenue softening — a promotional push is recommended.'} "
    f"Only {repeat_rate:.1f}% of customers reorder vs the 20–30% industry average — "
    f"retention is your highest-leverage growth opportunity.",
    label="AI: Revenue Health Interpretation"
)

divider()

# ── Category + Payment charts ─────────────────────────────────
st.subheader("🗂 Category & Payment Intelligence")
col_a, col_b = st.columns([2, 1])
with col_a:
    cat = df.groupby("category_en")["payment_value"].sum().nlargest(15).reset_index()
    cat.columns = ["Category", "Revenue"]
    fig3 = go.Figure(go.Bar(
        x=cat["Revenue"], y=cat["Category"], orientation="h",
        marker=dict(color=cat["Revenue"],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#06b6d4"]],
                    line_width=0)
    ))
    fig3 = apply_theme(fig3)
    fig3.update_layout(title="Top 15 Categories by Revenue",
                       yaxis=dict(categoryorder="total ascending"), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
with col_b:
    pay = df.groupby("payment_type")["payment_value"].sum().reset_index()
    fig4 = go.Figure(go.Pie(
        values=pay["payment_value"], labels=pay["payment_type"], hole=0.55,
        marker=dict(colors=["#3b82f6","#06b6d4","#10b981","#8b5cf6"]),
        textfont=dict(family="DM Mono", size=10),
    ))
    fig4 = apply_theme(fig4)
    fig4.update_layout(title="How Customers Pay", showlegend=True)
    st.plotly_chart(fig4, use_container_width=True)

top3_share = cat["Revenue"].nlargest(3).sum() / cat["Revenue"].sum() * 100
ai_insight_strip(
    f"Credit card dominates at ~74% of payments. Boleto (19%) signals a significant "
    f"unbanked customer base — installment-based promotions could unlock this group. "
    f"Top 3 categories hold {top3_share:.1f}% of revenue — "
    f"{'concentration risk: over-reliance on a few categories.' if top3_share > 40 else 'healthy diversification across categories.'}",
    label="AI: Payment & Category Insight"
)

divider()

# ── ZONE 3: Decision Panel ────────────────────────────────────
st.subheader("🎯 What should you do?")

with st.spinner("Running Mini-LLM analysis…"):
    _result = MiniLLM().run(df)

_recs = _result["recommendations"]
_conf = _result["confidence"]
_risk = _result["risk_score"]

col_m1, col_m2 = st.columns(2)
col_m1.metric("AI Confidence",  f"{_conf}%",  delta="Good" if _conf > 60 else "Run more modules",
              delta_color="normal" if _conf > 60 else "off")
col_m2.metric("Business Risk",  f"{_risk}/100",
              delta="High" if _risk > 60 else "Low" if _risk < 30 else "Moderate",
              delta_color="inverse")

# Decision cards from Mini-LLM top findings
if _recs:
    for _r in _recs[:3]:
        priority = "high" if _r["tag"] == "risk" else "medium" if _r["tag"] == "growth" else "low"
        icon     = "🔴" if _r["tag"] == "risk" else "🟢" if _r["tag"] == "growth" else "🔵"
        decision_card(title=_r["insight"], action=_r["action"],
                      priority=priority, evidence=_r["evidence"], icon=icon)

# Always show repeat rate card — most critical finding
decision_card(
    title    = f"Customer loyalty rate is only {repeat_rate:.1f}% — industry avg is 20–30%",
    action   = "Launch post-purchase email within 7 days of delivery. "
               "Offer 10% discount on next order. Target Occasional segment first.",
    priority = "high", icon = "🔄",
    evidence = f"repeat_rate = {repeat_rate:.1f}% across {len(cust_orders):,} customers"
)

st.divider()
simulate_decision_button("Increase Marketing", source_page="overview")

# ══ AI STRIP + DECISION + SIMULATE ═══════════════════════════
st.divider()
with st.spinner("🤖 AI analysing your business…"):
    _r = MiniLLM().run(df)
ai_insight_strip(_r["summary"], label="AI: Business Health Summary")
_top = _r["recommendations"][0] if _r["recommendations"] else None
if _top:
    decision_card(
        title    = _top["insight"],
        action   = _top["action"],
        priority = "high" if _top["tag"]=="risk" else "medium" if _top["tag"]=="growth" else "low",
        evidence = _top["evidence"], icon="🤖"
    )
simulate_decision_button(
    _top["action"][:60] if _top else "Review business strategy",
    source_page="overview"
)
next_page_button("Product Analysis", "pages/2_Product_Analysis.py")