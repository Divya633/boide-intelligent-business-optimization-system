import streamlit as st
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import get_order_level, init_session_state, load_data, next_page_button
from utils.decision_panel import ai_insight_strip, decision_card, flow_indicator
from utils.ui import apply_theme, divider, load_css, topnav
from utils.simulation import compute_elasticity, simulate_business
from utils.mini_llm import MiniLLM


st.set_page_config(page_title="Digital Twin - BOIDE", layout="wide")
init_session_state()
load_css()
topnav("Digital Twin")

flow_indicator("simulation")
st.title("Simulate a Business Decision")
st.caption("Adjust price, marketing and inventory to see projected revenue impact instantly")

df = load_data()
order_level = get_order_level(df)
elasticity = compute_elasticity(df)
base_rev = order_level["payment_value"].sum()

preset = st.session_state.get("decision") or "No Action"
source = st.session_state.get("decision_source") or ""

PRESETS = {
    "No Action": (0, 0, 0),
    "Increase Marketing": (0, 30, 10),
    "Optimise Pricing": (-10, 10, 5),
    "Reduce Risk": (-5, 5, 20),
}


def map_to_preset(decision: str) -> str:
    if not decision or not isinstance(decision, str):
        return "No Action"
    lowered = decision.lower()
    if any(token in lowered for token in ["market", "promot", "campaign", "loyalty", "email"]):
        return "Increase Marketing"
    if any(token in lowered for token in ["price", "pricing", "discount", "bundl", "freight"]):
        return "Optimise Pricing"
    if any(token in lowered for token in ["risk", "anomal", "cancel", "reduce", "safe", "investi"]):
        return "Reduce Risk"
    return "No Action"


mapped = map_to_preset(preset)
defaults = PRESETS.get(mapped, (0, 0, 0))

if preset != "No Action":
    from_page = source.replace("_", " ").title() if source else "AI Insights"
    st.markdown(
        f"""
        <div style="background:rgba(59,130,246,0.10);border:1px solid rgba(59,130,246,0.35);
            border-left:5px solid #3b82f6;border-radius:8px;padding:16px 20px;margin-bottom:16px;">
            <div style="font-size:11px;color:#3b82f6;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">
                Decision received from: {from_page}
            </div>
            <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:6px;">
                {preset}
            </div>
            <div style="font-size:12px;color:#334155;font-weight:700;">
                Mapped to strategy: <strong style="color:#60a5fa;">{mapped}</strong>.
                Click <strong>Run Simulation</strong> to see results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Navigate to any page and click Simulate to pre-fill a strategy here, or adjust the sliders manually below.")

st.markdown(
    f"<div style='font-weight:700; color:#334155;'>Price elasticity from Olist data: {elasticity:.3f}</div>",
    unsafe_allow_html=True,
)

divider()

st.subheader("Strategy Levers")
st.caption("Move sliders to test your strategy. Results update when you run the simulation.")

col1, col2, col3 = st.columns(3)
with col1:
    price_change = st.slider(
        "Price Change (%)",
        -30,
        30,
        int(defaults[0]),
        step=1,
        help="Negative = price cut. Positive = price rise.",
    )
with col2:
    marketing = st.slider(
        "Marketing Spend (%)",
        0,
        50,
        int(defaults[1]),
        step=1,
        help="Each 1% marketing increase adds 0.3% revenue in this heuristic model.",
    )
with col3:
    inventory = st.slider(
        "Inventory Increase (%)",
        0,
        50,
        int(defaults[2]),
        step=1,
        help="Inventory uplift is capped at 15% in the simulation model.",
    )

run_auto = defaults != (0, 0, 0)

if st.button("Run Simulation", type="primary", use_container_width=True, key="run_sim_main") or run_auto:
    result = simulate_business(df, price_change, marketing, inventory)
    delta_pct = result["delta_pct"]
    risk = result["risk_score"]

    divider()

    st.subheader("Simulation Results")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Projected Revenue", f"R${result['projected_revenue']/1e6:.2f}M", delta=f"{delta_pct:+.2f}%")
    col_b.metric("Revenue Change", f"R${result['delta']:+,.0f}")
    col_c.metric("Risk Score", f"{risk:.0f}/100", delta="High" if risk > 65 else "Low" if risk < 30 else "Medium", delta_color="inverse")
    col_d.metric("Price Elasticity", f"{elasticity:.3f}", delta="High sensitivity" if elasticity > 0.6 else "Moderate")

    st.subheader("How Each Lever Contributes")
    fig = go.Figure(
        go.Waterfall(
            name="Revenue drivers",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Base Revenue", "Price Effect", "Marketing Effect", "Inventory Effect", "Projected Total"],
            y=[
                result["base_revenue"],
                result["base_revenue"] * result["price_effect_pct"] / 100,
                result["base_revenue"] * result["marketing_effect_pct"] / 100,
                result["base_revenue"] * result["inventory_effect_pct"] / 100,
                0,
            ],
            connector={"line": {"color": "rgba(255,255,255,0.2)"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#4f8ef7"}},
            text=[
                f"R${result['base_revenue']/1e6:.1f}M",
                f"{result['price_effect_pct']:+.1f}%",
                f"{result['marketing_effect_pct']:+.1f}%",
                f"{result['inventory_effect_pct']:+.1f}%",
                f"R${result['projected_revenue']/1e6:.1f}M",
            ],
            textposition="outside",
        )
    )
    fig = apply_theme(fig)
    fig.update_layout(title="Revenue waterfall", yaxis_title="Revenue (R$)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Assessment")
    st.progress(
        int(risk) / 100,
        text=f"Risk score: {risk:.0f}/100 - {'High' if risk > 65 else 'Moderate' if risk > 40 else 'Low'}",
    )

    ai_insight_strip(
        f"Strategy: {mapped}. Projected revenue change = {delta_pct:+.1f}%. "
        f"Risk score = {risk:.0f}/100. Price elasticity = {elasticity:.3f}.",
        label="AI: Simulation Outcome Interpretation",
    )

    if delta_pct > 5:
        decision_card(
            title=f"Strategy projects +{delta_pct:.1f}% revenue growth",
            action="A/B test on a smaller audience before a full rollout.",
            priority="low",
            icon="",
            evidence=f"delta = {delta_pct:+.1f}% | risk = {risk:.0f}/100 | elasticity = {elasticity:.3f}",
        )
    elif delta_pct < -3:
        decision_card(
            title=f"Strategy reduces revenue by {abs(delta_pct):.1f}%",
            action="Reduce the price change and rely more on marketing support.",
            priority="high",
            icon="",
            evidence=f"delta = {delta_pct:+.1f}% | price_change = {price_change}%",
        )
    else:
        decision_card(
            title="Strategy is balanced",
            action="Roll out in phases and monitor conversion after each change.",
            priority="medium",
            icon="",
            evidence=f"delta = {delta_pct:+.1f}% | risk = {risk:.0f}/100",
        )

    if risk > 65:
        decision_card(
            title=f"Risk score {risk:.0f}/100 is high",
            action="Implement one lever at a time: marketing first, then inventory, then price.",
            priority="high",
            icon="",
            evidence=f"risk = {risk:.0f}/100 (high threshold = 65)",
        )

    divider()
    st.subheader("Mini-LLM Recommendations")
    st.markdown(
        "<div style='font-weight:700; color:#334155; margin-bottom:12px;'>"
        "Rules fired on real Olist signals + your simulation inputs"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Running Mini-LLM..."):
        llm_res = MiniLLM().run(
            df,
            forecast=st.session_state.get("forecast"),
            segments=st.session_state.get("segments"),
            anomalies=st.session_state.get("anomalies", 0),
            forecast_details=st.session_state.get("forecast_details"),
            segment_details=st.session_state.get("segment_details"),
            anomaly_details=st.session_state.get("anomaly_details"),
        )

    for rec in llm_res["recommendations"][:4]:
        icon = "" if rec["tag"] == "risk" else "" if rec["tag"] == "growth" else ""
        st.markdown(f"{icon} **{rec['badge']}** - {rec['insight']}")
        st.markdown(
            f"<div style='font-weight:700; color:#334155; margin:6px 0 18px 0; line-height:1.7;'>"
            f"-> {rec['action']} | Evidence: {rec['evidence']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='font-weight:700; color:#334155; margin-top:10px;'>"
        f"Confidence: {llm_res['confidence']}% | "
        f"Risk: {llm_res['risk_score']}/100 | "
        f"Rules fired: {len(llm_res['recommendations'])}"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("Adjust the sliders above and click Run Simulation to see projected results.")
    if run_auto:
        st.success(f"Strategy **{mapped}** pre-loaded from {source or 'AI Insights'} - click Run Simulation to see results.")

divider()
next_page_button("Reports", "pages/8_Reports.py")
