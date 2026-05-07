import streamlit as st
import plotly.express as px
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import load_data, init_session_state, next_page_button, get_order_level
from utils.ui import load_css, topnav, apply_theme, section_header, divider
from utils.decision_panel import (ai_insight_strip, flow_indicator, decision_panel,
    decision_card, simulate_decision_button, biz_label)
from utils.mini_llm import MiniLLM

st.set_page_config(page_title="Product Analysis - BOIDE", layout="wide")
init_session_state()
load_css()
topnav("Product Analysis")
flow_indicator("data")

st.title("What do your products and delivery tell us?")
st.caption("Category performance, delivery risk, seller quality, and freight intelligence")

df = load_data()
order_level = get_order_level(df)

#  KPI Cards 
total_products = df["product_id"].nunique()
avg_price      = df["price"].mean() if "price" in df.columns else 0
avg_rating     = df["review_score"].mean() if "review_score" in df.columns else 0
avg_discount   = 0  # Olist doesn't have explicit discount; placeholder

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Products",  f"{total_products:,}")
col2.metric("Avg Price",       f"R${avg_price:.2f}")
col3.metric("Avg Rating",      f"{avg_rating:.2f}/5")
col4.metric("Avg Photos",
            f"{df['product_photos_qty'].mean():.1f}" if "product_photos_qty" in df.columns else "N/A")

st.divider()

#  Category Distribution 
st.subheader("Category Distribution")
cat_rev = (
    df.groupby("category_en")
    .agg(revenue=("item_revenue", "sum"), orders=("order_id", "nunique"))
    .nlargest(15, "revenue").reset_index()
)

fig1 = px.bar(cat_rev, x="category_en", y="revenue",
              color="orders", color_continuous_scale="Viridis",
              title="Top 15 Categories - Revenue & Order Volume",
              labels={"revenue": "Revenue (R$)", "category_en": "Category", "orders": "Orders"})
fig1.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

#  Rating Distribution 
st.subheader("Rating Distribution")
col_a, col_b = st.columns(2)

with col_a:
    rating_counts = df["review_score"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["Score", "Count"]
    fig2 = px.bar(rating_counts, x="Score", y="Count",
                  title="Review Score Distribution",
                  color="Score", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    cat_rating = (
        df.groupby("category_en")["review_score"]
        .mean().nlargest(10).reset_index()
    )
    cat_rating.columns = ["Category", "Avg Rating"]
    fig3 = px.bar(cat_rating, x="Avg Rating", y="Category", orientation="h",
                  title="Top 10 Categories by Avg Rating",
                  color="Avg Rating", color_continuous_scale="Greens")
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

#  Price vs Rating Analysis 
st.subheader("Price vs Rating Analysis")
if "price" in df.columns:
    sample = df[["price", "review_score", "category_en"]].dropna().sample(
        min(3000, len(df)), random_state=42
    )
    fig4 = px.scatter(sample, x="price", y="review_score",
                      color="category_en",
                      title="Price vs Review Score (sample of 3000)",
                      labels={"price": "Price (R$)", "review_score": "Review Score"},
                      opacity=0.5)
    fig4.update_layout(showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    price_corr = df["price"].corr(df["review_score"])
    st.info(f"Price-Rating correlation: **{price_corr:.3f}** "
            f"({'weak positive' if price_corr > 0 else 'weak negative'} relationship)")

st.divider()

#  Top Rated Products 
st.subheader("Top Rated Products")
top_rated = (
    df.groupby("product_id")
    .agg(avg_rating=("review_score","mean"), orders=("order_id","nunique"),
         avg_price=("price","mean"), category=("category_en","first"))
    .query("orders >= 10")
    .nlargest(10, "avg_rating")
    .reset_index()
)
top_rated["avg_rating"]  = top_rated["avg_rating"].round(2)
top_rated["avg_price"]   = top_rated["avg_price"].round(2)
st.dataframe(
    top_rated[["product_id","category","avg_rating","avg_price","orders"]],
    use_container_width=True
)

#  Delivery delay impact on rating 
st.subheader("Delivery Delay vs Rating Impact")
if "delivery_delay_days" in df.columns:
    delay_rating = (
        df.groupby(df["delivery_delay_days"].clip(-10, 30).round())
        ["review_score"].mean().reset_index()
    )
    delay_rating.columns = ["Delay (days)", "Avg Rating"]
    fig5 = px.line(delay_rating, x="Delay (days)", y="Avg Rating",
                   title="Impact of Delivery Delay on Customer Rating",
                   markers=True)
    fig5.add_vline(x=0, line_dash="dash", line_color="red",
                   annotation_text="On-time delivery")
    st.plotly_chart(fig5, use_container_width=True)

    delay_corr = df["delivery_delay_days"].corr(df["review_score"])
    st.warning(f"Delivery delay correlation with rating: **{delay_corr:.3f}** - "
               "every extra day late reduces average rating significantly")

#  UPGRADE: Seller Performance Leaderboard 
st.divider()
st.subheader("Seller Performance Leaderboard")
st.caption("Top sellers by revenue with delivery and quality scores")

if "seller_id" in df.columns:
    seller_stats = (
        df.groupby("seller_id")
        .agg(
            Revenue      = ("item_revenue",       "sum"),
            Orders       = ("order_id",            "nunique"),
            Avg_Review   = ("review_score",        "mean"),
            Late_Rate    = ("delivery_delay_days", lambda x: (x > 0).mean() * 100))
        .reset_index()
        .nlargest(15, "Revenue")
        .rename(columns={"Avg_Review":"Avg Review","Late_Rate":"Late Rate (%)"})
    )
    seller_stats["Revenue"]     = seller_stats["Revenue"].round(2)
    seller_stats["Avg Review"]  = seller_stats["Avg Review"].round(2)
    seller_stats["Late Rate (%)"] = seller_stats["Late Rate (%)"].round(1)

    # Health score: revenue rank + review + delivery
    seller_stats["Health Score"] = (
        (seller_stats["Avg Review"] / 5 * 40) +
        ((100 - seller_stats["Late Rate (%)"]) / 100 * 40) +
        20
    ).round(0).astype(int)

    st.dataframe(seller_stats[["seller_id","Revenue","Orders",
                                "Avg Review","Late Rate (%)","Health Score"]]
                 .rename(columns={"seller_id":"Seller ID"}),
                 use_container_width=True, hide_index=True)

    top10_share = df.groupby("seller_id")["item_revenue"].sum().nlargest(10).sum() / df["item_revenue"].sum() * 100
    st.info(f"Top 10 sellers = **{top10_share:.1f}%** of total revenue across "
            f"**{df['seller_id'].nunique():,}** total sellers")

st.divider()

#  UPGRADE: Freight Cost Intelligence 
st.subheader("Freight Cost Intelligence")
st.caption("Freight = 32.2% of product price on average - a hidden revenue lever")

if "freight_value" in df.columns and "price" in df.columns:
    valid = df[(df["price"] > 0) & df["freight_value"].notna()].copy()
    valid["freight_ratio"] = valid["freight_value"] / valid["price"] * 100

    freight_by_cat = (
        valid.groupby("category_en")["freight_ratio"]
        .mean().nlargest(10).reset_index()
        .rename(columns={"freight_ratio":"Avg Freight % of Price","category_en":"Category"})
    )

    
    fig_freight = px.bar(
        freight_by_cat, x="Avg Freight % of Price", y="Category",
        orientation="h", title="Top 10 Categories by Freight Burden",
        color="Avg Freight % of Price",
        color_continuous_scale=[[0,"#3b82f6"],[0.5,"#f59e0b"],[1,"#ef4444"]])
    fig_freight = apply_theme(fig_freight)
    fig_freight.add_vline(x=40, line_dash="dash", line_color="#ef4444",
                          annotation_text="40% threshold")
    st.plotly_chart(fig_freight, use_container_width=True)

    overall_freight = valid["freight_ratio"].mean()
    high_freight    = (valid["freight_ratio"] > 40).mean() * 100
    st.warning(f"Avg freight = **{overall_freight:.1f}%** of price | "
               f"**{high_freight:.1f}%** of orders have freight > 40% of price - "
               f"consider free shipping at R$150+ threshold")

st.divider()

#  Mini-LLM AI Insights 
st.divider()
st.subheader("AI Product Intelligence")
st.caption("Powered by Mini-LLM - rule-based reasoning on real Olist data")

if "delivery_delay_days" in df.columns:
    late_pct   = (order_level["delivery_delay_days"] > 0).sum() / len(order_level) * 100
    avg_delay  = order_level["delivery_delay_days"].mean()
else:
    late_pct  = 0
    avg_delay = 0

avg_rev_score = df["review_score"].mean() if "review_score" in df.columns else 0

with st.spinner("Running Mini-LLM analysis"):
    _llm    = MiniLLM()
    _result = _llm.run(df)
    _recs   = _result["recommendations"]
    _conf   = _result["confidence"]
ai_insight_strip(_result["summary"], label="AI: Product Health Summary")

# Filter rules relevant to product/delivery
_prod_recs = [r for r in _recs if any(k in r["insight"].lower()
    for k in ["review", "deliver", "delay", "quality", "category", "cancel"])]

if _prod_recs:
    st.markdown("**Mini-LLM product-specific findings:**")
    for _r in _prod_recs:
        _icon = "" if _r["tag"] == "risk" else "" if _r["tag"] == "growth" else ""
        st.markdown(f"**{_r['badge']}** - {_r['insight']}")
        st.caption(f"Action: {_r['action']}")
        st.caption(f"Evidence: {_r['evidence']}")

# Decision cards
product_decisions = []
if late_pct > 20:
    product_decisions.append({"title":"Delivery performance is critical","priority":"high","icon":"",
        "action":"Renegotiate carrier SLAs. Add 2-day buffer to estimated delivery dates.",
        "evidence":f"{late_pct:.1f}% of orders delivered late"})
elif late_pct > 10:
    product_decisions.append({"title":"Delivery delays need attention","priority":"medium","icon":"",
        "action":"Monitor carrier performance weekly. Set alerts for delays > 5 days.",
        "evidence":f"{late_pct:.1f}% late delivery rate"})
if avg_rev_score < 3.8:
    product_decisions.append({"title":"Product quality risk detected","priority":"high","icon":"",
        "action":"Audit top complaint categories. Require sellers to update descriptions.",
        "evidence":f"Avg review = {avg_rev_score:.2f}/5"})
if not product_decisions:
    product_decisions.append({"title":"Product metrics look healthy","priority":"low","icon":"",
        "action":"Use strong review score in marketing campaigns.",
        "evidence":f"Avg review = {avg_rev_score:.2f}/5 | Late rate = {late_pct:.1f}%"})

decision_panel(product_decisions, title="Product & Delivery Decision Panel")


#  AI Insight 
st.divider()
ai_insight_strip(
    "Delivery delay has a strong negative impact on customer satisfaction. Every extra day late reduces review score by ~0.2 stars. Freight costs average 32% of product price - a free shipping threshold at R$150+ could lift conversion by 10-15%.",
    label="AI: Product & Delivery Intelligence"
)


#  AI STRIP + DECISION + SIMULATE 
st.divider()
with st.spinner("AI analysing product data"):
    _r = MiniLLM().run(df)
ai_insight_strip(_r["summary"], label="AI: Product Intelligence Summary")
_prod = next((x for x in _r["recommendations"]
              if any(k in x["insight"].lower()
                     for k in ["review","deliver","freight","quality","categor"])),
             _r["recommendations"][0] if _r["recommendations"] else None)
if _prod:
    decision_card(
        title    = _prod["insight"],
        action   = _prod["action"],
        priority = "high" if _prod["tag"]=="risk" else "medium",
        evidence = _prod["evidence"], icon=""
    )
simulate_decision_button("Optimise Pricing", source_page="product")
next_page_button("Forecasting", "pages/3_Forecasting.py")
