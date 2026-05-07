"""
utils/decision_panel.py
-----------------------
Reusable Decision Panel components for all BOIDE pages.
"""

import streamlit as st

_CSS_INJECTED = False

def _inject_css():
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    st.markdown("""
    <style>
    .decision-card { background:#ffffff; border-radius:10px; padding:18px 20px;
        margin-bottom:10px; border-left:4px solid #4f8ef7;
        border-top:1px solid rgba(148,163,184,0.22);
        border-right:1px solid rgba(148,163,184,0.22);
        border-bottom:1px solid rgba(148,163,184,0.22);
        box-shadow:0 10px 24px rgba(15,23,42,0.05); }
    .decision-card.high   { border-left-color:#ef4444; }
    .decision-card.medium { border-left-color:#f59e0b; }
    .decision-card.low    { border-left-color:#10b981; }
    .priority-badge { display:inline-block; font-family:'DM Mono',monospace;
        font-size:10px; font-weight:600; letter-spacing:0.1em;
        padding:2px 10px; border-radius:4px; margin-bottom:8px; }
    .badge-high   { background:rgba(239,68,68,0.10);  color:#b91c1c; }
    .badge-medium { background:rgba(245,158,11,0.12); color:#92400e; }
    .badge-low    { background:rgba(16,185,129,0.12); color:#047857; }
    .decision-title  { font-family:'DM Sans',sans-serif; font-size:14px;
        font-weight:600; color:#0f172a; margin-bottom:5px; }
    .decision-action { font-family:'DM Sans',sans-serif; font-size:13px;
        color:#334155; line-height:1.6; }
    .decision-evidence { font-family:'DM Mono',monospace; font-size:10.5px;
        color:#64748b; margin-top:6px; padding-top:6px;
        border-top:1px solid rgba(148,163,184,0.18); }
    .ai-strip { background:linear-gradient(135deg,rgba(59,130,246,0.08),rgba(139,92,246,0.08));
        border:1px solid rgba(59,130,246,0.2); border-radius:10px;
        padding:14px 18px; margin:16px 0; }
    .ai-strip-label { font-family:'DM Mono',monospace; font-size:10px; color:#3b82f6;
        letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px; }
    .ai-strip-text  { font-family:'DM Sans',sans-serif; font-size:13.5px;
        color:#334155; line-height:1.6; }
    </style>
    """, unsafe_allow_html=True)
    _CSS_INJECTED = True

#  1. Flow indicator using native st.columns 

def flow_indicator(current_step: str):
    """Horizontal pipeline bar using st.columns; always renders horizontally."""
    steps = [
        ("data",       "01", "Data",       "Olist CSVs"),
        ("insight",    "02", "Insight",    "Modules run"),
        ("decision",   "03", "Decision",   "AI Insights"),
        ("simulation", "04", "Simulation", "Digital Twin"),
        ("outcome",    "05", "Outcome",    "Reports"),
    ]
    order       = [s[0] for s in steps]
    current_idx = order.index(current_step) if current_step in order else 0

    # Inject step styles once
    st.markdown("""
    <style>
    .fstep-done    {background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.4);
        border-radius:8px;padding:8px 4px;text-align:center;}
    .fstep-active  {background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.6);
        border-radius:8px;padding:8px 4px;text-align:center;}
    .fstep-pending {background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
        border-radius:8px;padding:8px 4px;text-align:center;}
    .fstep-icon {font-size:17px;line-height:1.3;}
    .fstep-lbl-done    {font-size:11px;font-weight:600;color:#10b981;margin-top:2px;}
    .fstep-lbl-active  {font-size:11px;font-weight:600;color:#60a5fa;margin-top:2px;}
    .fstep-lbl-pending {font-size:11px;font-weight:600;color:#4a5568;margin-top:2px;}
    .fstep-sub {font-size:9px;opacity:0.65;}
    </style>
    """, unsafe_allow_html=True)

    # 9 columns: step, arrow, step, arrow, step, arrow, step, arrow, step
    cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])

    step_positions  = [0, 2, 4, 6, 8]
    arrow_positions = [1, 3, 5, 7]

    for col_idx, (i, (key, icon, label, sub)) in zip(step_positions, enumerate(steps)):
        cls = "done" if i < current_idx else ("active" if i == current_idx else "pending")
        with cols[col_idx]:
            st.markdown(
                f'<div class="fstep-{cls}">'
                f'<div class="fstep-icon">{icon}</div>'
                f'<div class="fstep-lbl-{cls}">{label}</div>'
                f'<div class="fstep-lbl-{cls} fstep-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    for ap in arrow_positions:
        with cols[ap]:
            st.markdown(
                '<p style="text-align:center;padding-top:12px;'
                'font-size:16px;color:#4a5568;margin:0;">&gt;</p>',
                unsafe_allow_html=True
            )

#  2. Decision card 

def decision_card(title, action, priority="medium", evidence="", icon=""):
    _inject_css()
    badge_text = {"high":"HIGH PRIORITY","medium":"MEDIUM PRIORITY","low":"LOW PRIORITY"}.get(priority,"INFO")
    icon_html  = f"{icon} " if icon else ""
    ev_html    = f"<div class='decision-evidence'>Evidence: {evidence}</div>" if evidence else ""
    st.markdown(
        f'<div class="decision-card {priority}">'
        f'<span class="priority-badge badge-{priority}">{badge_text}</span>'
        f'<div class="decision-title">{icon_html}{title}</div>'
        f'<div class="decision-action">Action: {action}</div>'
        f'{ev_html}</div>',
        unsafe_allow_html=True
    )

#  3. Decision panel 

def decision_panel(decisions, title="Decision Panel"):
    _inject_css()
    st.markdown(f"### {title}")
    if not decisions:
        st.success("No critical decisions needed - all metrics are within normal range.")
        return
    for d in decisions:
        decision_card(
            title    = d.get("title",""),
            action   = d.get("action",""),
            priority = d.get("priority","medium"),
            evidence = d.get("evidence",""),
            icon     = d.get("icon",""),
        )

#  5. Simulate decision button 

def simulate_decision_button(decision, source_page="page"):
    _inject_css()
    # Counter makes key unique even when called multiple times on same page
    _ck = f"_sim_ctr_{source_page}"
    if _ck not in st.session_state:
        st.session_state[_ck] = 0
    st.session_state[_ck] += 1
    _ukey = f"sim_{source_page}_{st.session_state[_ck]}"
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f'<div style="background:rgba(59,130,246,0.08);border:1px solid '
            f'rgba(59,130,246,0.2);border-radius:8px;padding:10px 14px;'
            f'font-family:DM Sans,sans-serif;font-size:13px;color:#334155;">'
            f'<strong>Recommended strategy:</strong> {decision}</div>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button("Simulate", key=_ukey, use_container_width=True,
                     type="primary"):
            # Store decision so Digital Twin can read it
            st.session_state["decision"]       = decision
            st.session_state["decision_source"] = source_page
            st.session_state["nav_twin"]        = True
            # Navigate directly to Digital Twin
            st.switch_page("pages/7_Digital_Twin.py")

#  6. Business language 

BUSINESS_LABELS = {
    "MAPE":             "Demand Trend Confidence",
    "RMSE":             "Forecast Error (in R$)",
    "CV":               "Revenue Stability Index",
    "silhouette_score": "Customer Group Clarity",
    "contamination":    "Anomaly Sensitivity",
    "directional_acc":  "Trend Direction Accuracy",
    "anomaly_count":    "Revenue Disruption Days",
    "cancel_rate":      "Order Failure Rate",
    "late_pct":         "Late Delivery Rate",
    "hv_pct":           "Premium Customer Share",
    "low_review_pct":   "Unhappy Customer Rate",
    "avg_delay":        "Avg Delivery Lateness",
    "top3_share":       "Revenue Concentration Risk",
    "forecast_vs_baseline": "Expected Demand Change",
}

MODEL_EXPLANATIONS = {
    "ARIMA":            "A formula that learns your revenue rhythm and predicts future sales from past patterns.",
    "KMeans":           "Groups customers by how much they spend, how often they buy, and when they last bought.",
    "IsolationForest":  "Flags days where revenue was unusually high or low, like an alert system for your data.",
    "Mini-LLM":         "14 business rules in Python that read 12 data signals and recommend actions without a black box.",
    "Silhouette Score": "Above 0.4 means your customer groups are meaningfully different from each other.",
    "MAPE":             "Average % gap between predicted and actual revenue. For daily retail, 30-60% is normal.",
    "RMSE":             "Average error in R$ between forecast and actual revenue.",
    "CV":               "How unpredictable daily revenue is. High CV = big swings = business risk.",
}

def business_label(term):
    return BUSINESS_LABELS.get(term, term)

def model_explain(name):
    return MODEL_EXPLANATIONS.get(name, "")

#  7. Page-specific decision generators 

def forecast_decisions(growth, confidence, mape):
    decisions = []
    if growth > 10:
        decisions.append({"title":"Strong demand growth expected","priority":"high","icon":"",
            "action":"Pre-position inventory for top 3 categories. Increase ad spend 20% in first 3 forecast days.",
            "evidence":f"7-day forecast growth = +{growth:.1f}%"})
    elif growth > 3:
        decisions.append({"title":"Moderate growth signal","priority":"medium","icon":"",
            "action":"Maintain stock. Monitor daily and scale if trend continues 3+ days.",
            "evidence":f"7-day forecast = +{growth:.1f}%"})
    elif growth < -10:
        decisions.append({"title":"Revenue decline predicted - act now","priority":"high","icon":"",
            "action":"Launch promotional campaign on slow SKUs. Send re-engagement emails to dormant customers.",
            "evidence":f"7-day forecast decline = {growth:.1f}%"})
    elif growth < -3:
        decisions.append({"title":"Mild demand softness expected","priority":"medium","icon":"",
            "action":"Prepare discount offers for Discount-Sensitive segment. Reduce ad spend until stable.",
            "evidence":f"7-day forecast = {growth:.1f}%"})
    else:
        decisions.append({"title":"Revenue stable - maintain strategy","priority":"low","icon":"",
            "action":"Focus on customer retention. Optimise checkout conversion.",
            "evidence":f"7-day forecast = {growth:.1f}% (stable)"})
    if confidence < 50:
        decisions.append({"title":"Forecast uncertainty is high","priority":"medium","icon":"",
            "action":"Use 8-week forecast for planning. Build 20% safety stock buffer.",
            "evidence":f"Demand Trend Confidence = {confidence}%"})
    return decisions

def segmentation_decisions(high_count, total, sil):
    hv_pct    = high_count / max(total, 1) * 100
    decisions = []
    if hv_pct < 5:
        decisions.append({"title":"Critical: too few premium customers","priority":"high","icon":"",
            "action":"Launch VIP loyalty programme immediately. Offer early access and free shipping to top spenders.",
            "evidence":f"Premium Customer Share = {hv_pct:.1f}% (target > 10%)"})
    elif hv_pct < 10:
        decisions.append({"title":"Grow your premium customer base","priority":"medium","icon":"",
            "action":"Introduce spend-more-save-more programme. Upsell mid-value customers with bundles.",
            "evidence":f"Premium Customer Share = {hv_pct:.1f}%"})
    else:
        decisions.append({"title":"Strong VIP base - scale the programme","priority":"low","icon":"",
            "action":"Invest in loyalty rewards and exclusive early access. This group drives outsized revenue.",
            "evidence":f"Premium Customer Share = {hv_pct:.1f}% (healthy)"})
    decisions.append({"title":"Re-engage occasional buyers","priority":"medium","icon":"",
        "action":"Send win-back emails to customers inactive 180+ days. Offer 10% discount valid 7 days.",
        "evidence":"Occasional segment = high recency, low frequency"})
    return decisions

def anomaly_decisions(high_count, total_anomalies, anomaly_rate):
    decisions = []
    if high_count > 3:
        decisions.append({"title":"Multiple high-severity revenue spikes detected","priority":"high","icon":"",
            "action":"Investigate flagged dates: check payment gateway failures, fraud patterns, server logs.",
            "evidence":f"{high_count} high-severity anomalies (Z-score > 3)"})
    elif high_count > 0:
        decisions.append({"title":"High-severity anomaly requires investigation","priority":"medium","icon":"",
            "action":"Review the specific flagged date. Document findings for future reference.",
            "evidence":f"{high_count} high-severity anomaly"})
    if anomaly_rate > 3:
        decisions.append({"title":"Revenue disruptions are too frequent","priority":"high","icon":"",
            "action":"Set up real-time revenue alert: flag if daily revenue drops 40%+ below 7-day average.",
            "evidence":f"Revenue Disruption Rate = {anomaly_rate:.1f}% (target < 2%)"})
    else:
        decisions.append({"title":"Revenue pattern is broadly stable","priority":"low","icon":"",
            "action":"Continue weekly monitoring. Review monthly anomaly clusters for seasonal patterns.",
            "evidence":f"{total_anomalies} anomalous days total"})
    return decisions

#  Alias so pages can use shorter name 
def biz_label(term: str) -> str:
    """Short alias for business_label()."""
    extended = {
        "avg_order":       "Avg Order Value",
        "cancel_rate":     "Order Failure Rate",
        "repeat_rate":     "Customer Loyalty Rate",
        "freight_ratio":   "Shipping Cost Burden",
        "Silhouette Score":"Customer Group Clarity",
        "Z-score":         "Disruption Severity",
        "MAPE":            "Demand Trend Confidence",
        "RMSE":            "Forecast Error (R$)",
        "CV":              "Revenue Stability Index",
    }
    merged = {**BUSINESS_LABELS, **extended}
    return merged.get(term, term)

#  AI insight strip 
def ai_insight_strip(insight: str, label: str = "AI Interpretation"):
    """Render a styled AI interpretation band."""
    _inject_css()
    st.markdown(
        f'''<div class="ai-strip">
        <div class="ai-strip-label">{label}</div>
        <div class="ai-strip-text">{insight}</div>
        </div>''',
        unsafe_allow_html=True
    )
