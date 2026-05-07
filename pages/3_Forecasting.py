import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import get_daily_order_revenue, init_session_state, load_data, next_page_button
from utils.decision_panel import (
    ai_insight_strip,
    business_label,
    decision_card,
    decision_panel,
    flow_indicator,
    forecast_decisions,
    model_explain,
    simulate_decision_button,
)
from utils.forecasting_model import evaluate_forecast, forecast_future, train_best_forecast_model
from utils.mini_llm import MiniLLM
from utils.ui import load_css, topnav


def _safe_metric_value(value: float) -> float:
    return float(value) if pd.notna(value) and np.isfinite(value) else 0.0


def _forecast_band(forecast_series: pd.Series, error_value: float, multiplier: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(forecast_series.values, dtype=float)
    error = max(_safe_metric_value(error_value), float(np.nanstd(values)) if len(values) else 0.0)
    lower = np.maximum(0.0, values - multiplier * error)
    upper = values + multiplier * error
    return lower, upper


def _seasonality_signal(series: pd.Series) -> tuple[str, float]:
    if len(series) < 28:
        return "Limited history", 0.0
    prepared = series.copy().astype(float)
    weekly_profile = prepared.groupby(prepared.index.dayofweek).mean()
    baseline = max(float(prepared.mean()), 1.0)
    strength = float((weekly_profile.max() - weekly_profile.min()) / baseline * 100)
    if strength >= 30:
        return "Strong weekly seasonality", strength
    if strength >= 15:
        return "Moderate weekly seasonality", strength
    return "Low weekly seasonality", strength


def _planning_mode(growth_pct: float, confidence_score: int, mape_value: float) -> tuple[str, str, str]:
    if confidence_score < 50 or mape_value > 55:
        return (
            "Guarded plan",
            "Use the conservative case for inventory and cash planning until another module confirms the signal.",
            "medium",
        )
    if growth_pct >= 5:
        return (
            "Growth plan",
            "Prepare inventory and marketing capacity for the optimistic case, but launch in controlled steps.",
            "low",
        )
    if growth_pct <= -5:
        return (
            "Defense plan",
            "Protect margin, reduce slow-moving stock exposure, and use targeted promotions before broad discounts.",
            "high",
        )
    return (
        "Steady plan",
        "Hold current operating levels and watch for three consecutive days above or below the base forecast.",
        "low",
    )


st.set_page_config(page_title="Forecasting - BOIDE", layout="wide")
init_session_state()
load_css()
topnav("Forecasting")
flow_indicator("insight")

st.title("Sales Forecasting")
st.caption("Automatic time-series model selection with direct same-scale evaluation and stable error metrics")

df = load_data()
daily = get_daily_order_revenue(df)
sales = pd.Series(daily["revenue"].values, index=pd.to_datetime(daily["date"]))

with st.spinner("Training forecasting models..."):
    train = sales[:-7]
    test = sales[-7:]
    candidate_orders = [(2, 1, 2), (5, 1, 0), (1, 1, 1)]
    selected_order = candidate_orders[0]
    diagnostics = evaluate_forecast(train, test, order=selected_order, candidate_orders=candidate_orders)
    model = train_best_forecast_model(sales, diagnostics)
    forecast = forecast_future(model, steps=7).clip(lower=0)
    rmse = diagnostics.rmse
    mape = diagnostics.mape

forecast_value = float(forecast.sum())
growth = float(((forecast.mean() - train.tail(7).mean()) / train.tail(7).mean()) * 100) if train.tail(7).mean() else 0.0
st.session_state["forecast"] = forecast_value
st.session_state["forecast_growth"] = growth
st.session_state["forecast_details"] = {
    "model_label": diagnostics.model_label if diagnostics.model_label != "ARIMA" else f"ARIMA {diagnostics.order}",
    "mape": float(mape),
    "rmse": float(rmse),
    "confidence": int(diagnostics.confidence_score or 20),
    "directional_accuracy": float(diagnostics.directional_accuracy or 50.0),
    "candidate_scores": diagnostics.candidate_scores,
}

rmse = _safe_metric_value(diagnostics.rmse)
mape = _safe_metric_value(diagnostics.mape)
baseline_rmse = _safe_metric_value(diagnostics.baseline_rmse or 0.0)
baseline_mape = _safe_metric_value(diagnostics.baseline_mape or 0.0)
improvement_rmse_pct = _safe_metric_value(diagnostics.improvement_rmse_pct or 0.0)
improvement_mape_pct = _safe_metric_value(diagnostics.improvement_mape_pct or 0.0)
directional_acc = float(diagnostics.directional_accuracy or 50.0)
confidence = int(diagnostics.confidence_score or 20)
future_dates_7 = pd.date_range(start=sales.index[-1], periods=8, freq="D")[1:]
lower_7, upper_7 = _forecast_band(forecast, rmse, multiplier=1.15)
conservative_7 = float(lower_7.sum())
optimistic_7 = float(upper_7.sum())
seasonality_label, seasonality_strength = _seasonality_signal(sales)
seasonality_short = seasonality_label.replace(" weekly seasonality", " weekly")
planning_label, planning_action, planning_priority = _planning_mode(growth, confidence, mape)
st.session_state["forecast_details"].update(
    {
        "base_case_7d": forecast_value,
        "conservative_case_7d": conservative_7,
        "optimistic_case_7d": optimistic_7,
        "seasonality": seasonality_label,
        "planning_mode": planning_label,
    }
)

rmse_delta = f"{abs(improvement_rmse_pct):.1f}% lower than baseline" if improvement_rmse_pct >= 0 else f"{abs(improvement_rmse_pct):.1f}% higher than baseline"
mape_delta = f"{abs(improvement_mape_pct):.1f}% lower than baseline" if improvement_mape_pct >= 0 else f"{abs(improvement_mape_pct):.1f}% higher than baseline"

st.subheader("Forecast KPIs")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Next-week Forecast", f"R${forecast_value:,.0f}")
col2.metric("Revenue Growth", f"{growth:.2f}%", delta=f"{growth:.2f}%")
col3.metric("Model RMSE", f"{rmse:.2f}", delta=rmse_delta, delta_color="normal")
col4.metric("Model MAPE", f"{mape:.2f}%", delta=mape_delta, delta_color="normal")
col5.metric("Selected Model", diagnostics.model_label if diagnostics.model_label != "ARIMA" else f"ARIMA {diagnostics.order}")

st.subheader("Forecast Planning Cockpit")
plan_col1, plan_col2, plan_col3, plan_col4 = st.columns(4)
plan_col1.metric("Base Case", f"R${forecast_value:,.0f}", delta="next 7 days")
plan_col2.metric("Conservative Case", f"R${conservative_7:,.0f}", delta="lower range")
plan_col3.metric("Optimistic Case", f"R${optimistic_7:,.0f}", delta="upper range")
plan_col4.metric("Seasonality", seasonality_short, delta=f"{seasonality_strength:.1f}% weekly spread")
st.caption(f"Seasonality signal: {seasonality_label}. Weekly spread compares the strongest and weakest weekdays against average daily revenue.")

decision_card(
    title=planning_label,
    action=planning_action,
    priority=planning_priority,
    evidence=(
        f"7-day growth = {growth:+.1f}% | confidence = {confidence}% | "
        f"MAPE = {mape:.1f}% | selected model = {diagnostics.model_label}"
    ),
    icon="",
)

dir_score = directional_acc
mape_score = max(0.0, 100.0 - max(0.0, mape - 20.0) * 1.2)
vol_score = min(100.0, len(test) * 8.0)

st.subheader("Confidence Score")
st.progress(confidence / 100, text=f"Confidence: {confidence}%")

with st.expander("How is confidence calculated?", expanded=False):
    st.markdown(f"""
| Component | Weight | Value | Score |
|---|---|---|---|
| Directional accuracy | 50% | {directional_acc:.1f}% | {dir_score * 0.5:.1f} |
| MAPE bracket score | 30% | MAPE = {mape:.1f}% | {mape_score * 0.3:.1f} |
| Data volume score | 20% | {len(sales)} days | {vol_score * 0.2:.1f} |
| **Total** | | | **{confidence}%** |
    """)

st.divider()

st.subheader("Scale Check")
scale_ratio = _safe_metric_value(diagnostics.scale_ratio or 0.0)
scale_note = "Good" if 0.7 <= scale_ratio <= 1.3 else "Check scale"
scale_col1, scale_col2 = st.columns(2)
scale_col1.metric("Prediction / Actual Scale", f"{scale_ratio:.2f}x")
scale_col2.metric("Scale Status", scale_note)

with st.expander("Validation sample: first 5 actual vs predicted", expanded=False):
    st.dataframe(pd.DataFrame(diagnostics.debug_rows), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Candidate Comparison")
st.dataframe(pd.DataFrame(diagnostics.candidate_scores), use_container_width=True, hide_index=True)

st.divider()

st.subheader("7-Day Validation")
val_fig = go.Figure()
val_fig.add_trace(go.Scatter(
    x=diagnostics.validation_actual.index,
    y=diagnostics.validation_actual.values,
    mode="lines+markers",
    name="Actual",
    line=dict(color="#2563eb", width=2),
))
val_fig.add_trace(go.Scatter(
    x=diagnostics.validation_forecast.index,
    y=diagnostics.validation_forecast.values,
    mode="lines+markers",
    name=f"{diagnostics.model_label} Forecast" if diagnostics.model_label != "ARIMA" else f"ARIMA {diagnostics.order}",
    line=dict(color="#10b981", width=2),
))
val_fig.add_trace(go.Scatter(
    x=diagnostics.baseline_forecast.index,
    y=diagnostics.baseline_forecast.values,
    mode="lines",
    name="Baseline (last value)",
    line=dict(color="#f97316", dash="dot", width=2),
))
val_fig.update_layout(
    title="Actual vs Forecast on Test Window",
    xaxis_title="Date",
    yaxis_title="Revenue (R$)",
)
st.plotly_chart(val_fig, use_container_width=True)

comparison = pd.DataFrame([
    {
        "Model": "Baseline",
        "RMSE": round(baseline_rmse, 2),
        "MAPE (%)": round(baseline_mape, 2),
    },
    {
        "Model": diagnostics.model_label if diagnostics.model_label != "ARIMA" else f"ARIMA {diagnostics.order}",
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
    },
])
st.dataframe(comparison, use_container_width=True, hide_index=True)
st.caption(
    "Preprocessing applied: datetime sorting, missing-value filling, and model selection across ARIMA, Holt-Winters, moving average, and seasonal naive baselines."
)

st.divider()

st.subheader("Next-Week Forecast (7 days)")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=sales.index[-30:], y=sales.values[-30:], mode="lines", name="Actual (last 30d)", line=dict(color="#4f8ef7")))
fig1.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="Test (actual)", line=dict(color="#f59e0b", dash="dot")))
fig1.add_trace(go.Scatter(
    x=list(future_dates_7) + list(future_dates_7[::-1]),
    y=list(upper_7) + list(lower_7[::-1]),
    fill="toself",
    fillcolor="rgba(16,185,129,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Planning Range",
))
fig1.add_trace(go.Scatter(x=future_dates_7, y=upper_7, mode="lines", name="Optimistic Case", line=dict(color="#22c55e", dash="dot")))
fig1.add_trace(go.Scatter(x=future_dates_7, y=forecast.values, mode="lines+markers", name="Base Forecast", line=dict(color="#10b981", width=3)))
fig1.add_trace(go.Scatter(x=future_dates_7, y=lower_7, mode="lines", name="Conservative Case", line=dict(color="#f59e0b", dash="dot")))
fig1.add_trace(go.Scatter(
    x=test.index,
    y=[train.iloc[-1]] * len(test.index),
    mode="lines",
    name="Baseline",
    line=dict(color="#ef4444", dash="dash"),
))
fig1.update_layout(title="Actual vs Forecast - Next 7 Days", xaxis_title="Date", yaxis_title="Revenue (R$)")
st.plotly_chart(fig1, use_container_width=True)

st.divider()

st.subheader("30-Day Scale Forecast")
with st.spinner("Computing 30-day forecast..."):
    diagnostics_30 = evaluate_forecast(
        sales[:-30],
        sales[-30:],
        order=selected_order,
        candidate_orders=candidate_orders,
    )
    model_30 = train_best_forecast_model(sales, diagnostics_30)
    forecast_30 = forecast_future(model_30, steps=30).clip(lower=0)
    future_dates_30 = pd.date_range(start=sales.index[-1], periods=31, freq="D")[1:]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sales.index, y=sales.values, mode="lines", name="Actual", line=dict(color="#4f8ef7")))
lower, upper = _forecast_band(forecast_30, diagnostics_30.rmse, multiplier=1.35)
fig2.add_trace(go.Scatter(x=future_dates_30, y=forecast_30.values, mode="lines", name="30-Day Base Forecast", line=dict(color="#f43f5e", width=2)))
fig2.add_trace(go.Scatter(x=list(future_dates_30) + list(future_dates_30[::-1]), y=list(upper) + list(lower[::-1]), fill="toself", fillcolor="rgba(244,63,94,0.1)", line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))
fig2.update_layout(title="30-Day Revenue Forecast", xaxis_title="Date", yaxis_title="Revenue (R$)")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Revenue Forecast (8 Weeks)")
weekly = daily.set_index("date").resample("W")["revenue"].sum()
with st.spinner("Computing 8-week forecast..."):
    diagnostics_8w = evaluate_forecast(
        weekly[:-8],
        weekly[-8:],
        order=selected_order,
        candidate_orders=candidate_orders,
    )
    model_8w = train_best_forecast_model(weekly, diagnostics_8w)
    forecast_8w = forecast_future(model_8w, steps=8).clip(lower=0)
    future_weeks = pd.date_range(start=weekly.index[-1], periods=9, freq="W")[1:]

fig3 = go.Figure()
fig3.add_trace(go.Bar(x=weekly.index[-20:], y=weekly.values[-20:], name="Actual Weekly", marker_color="#4f8ef7"))
lower_8w, upper_8w = _forecast_band(forecast_8w, diagnostics_8w.rmse, multiplier=1.1)
fig3.add_trace(go.Scatter(
    x=list(future_weeks) + list(future_weeks[::-1]),
    y=list(upper_8w) + list(lower_8w[::-1]),
    fill="toself",
    fillcolor="rgba(245,158,11,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Planning Range",
))
fig3.add_trace(go.Scatter(x=future_weeks, y=forecast_8w.values, mode="lines+markers", name="8-Week Forecast", line=dict(color="#f59e0b", width=2)))
fig3.update_layout(title="8-Week Revenue Forecast", xaxis_title="Week", yaxis_title="Revenue (R$)")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

st.subheader("Key Insights")
if growth > 5:
    st.success(f"Strong growth expected ({growth:.1f}%)")
elif growth < 0:
    st.error(f"Sales decline predicted ({growth:.1f}%)")
else:
    st.warning(f"Stable trend ({growth:.1f}%)")

st.info(
    f"Test-window accuracy: RMSE = R${rmse:,.2f} vs baseline R${baseline_rmse:,.2f} | "
    f"MAPE = {mape:.1f}% vs baseline {baseline_mape:.1f}% | Confidence = {confidence}%"
)

st.subheader("AI Recommendations")
rec_col1, rec_col2 = st.columns(2)
with rec_col1:
    st.markdown("**Based on forecast data:**")
    if growth > 5:
        st.markdown("- Pre-position stock for top categories\n- Increase ad spend in the first forecast window\n- Alert logistics for higher shipment volume")
    elif growth < 0:
        st.markdown("- Run promotional campaigns on slow categories\n- Review pricing strategy\n- Re-engage dormant customers")
    else:
        st.markdown("- Maintain inventory levels\n- Monitor weekly and adjust if trend breaks\n- Focus on retention")
with rec_col2:
    st.markdown("**Model health:**")
    st.markdown(f"- Test-window MAPE {mape:.1f}%")
    st.markdown(f"- Test-window RMSE R${rmse:,.0f} relative to avg R${sales.mean():,.0f}/day")
    st.markdown(f"- Directional accuracy: {directional_acc:.0f}%")

st.divider()
decisions = forecast_decisions(growth, confidence, mape)

with st.spinner("Running Mini-LLM analysis..."):
    llm_result = MiniLLM().run(
        df,
        forecast=st.session_state.get("forecast"),
        segments=st.session_state.get("segments"),
        anomalies=st.session_state.get("anomalies", 0),
        forecast_details=st.session_state.get("forecast_details"),
        segment_details=st.session_state.get("segment_details"),
        anomaly_details=st.session_state.get("anomaly_details"),
    )

filtered = [r for r in llm_result["recommendations"] if any(k in r["insight"].lower() for k in ["forecast", "revenue", "demand"])]
if filtered:
    with st.expander("Mini-LLM cross-module findings", expanded=False):
        for rec in filtered:
            icon = "" if rec["tag"] == "risk" else "" if rec["tag"] == "growth" else ""
            st.markdown(f"{icon} **{rec['badge']}** - {rec['insight']}")
            st.caption(f"-> {rec['action']} | {rec['evidence']}")

decision_panel(decisions, title="Forecasting Decision Panel")

st.divider()
rec_strategy = "Increase Marketing" if growth > 5 else "Reduce Risk" if growth < -5 else "No Action"
simulate_decision_button(rec_strategy, source_page="forecasting")

with st.expander("What do these numbers mean?", expanded=False):
    st.markdown(f"""
| Business Term | Technical Term | Your Value |
|---|---|---|
| **{business_label("MAPE")}** | MAPE | {mape:.1f}% |
| **{business_label("RMSE")}** | RMSE | R${rmse:,.0f} |
| **{business_label("directional_acc")}** | Directional Accuracy | {directional_acc:.0f}% |
    """)
    st.markdown(f"> **{model_explain('ARIMA')}**")

st.divider()
ai_insight_strip(
    llm_result["summary"],
    label="AI: Demand Forecast Summary",
)
forecast_rec = next(
    (rec for rec in llm_result["recommendations"] if any(k in rec["insight"].lower() for k in ["forecast", "revenue", "demand", "trend"])),
    llm_result["recommendations"][0] if llm_result["recommendations"] else None,
)
if forecast_rec:
    decision_card(
        title=forecast_rec["insight"],
        action=forecast_rec["action"],
        priority="high" if forecast_rec["tag"] == "risk" else "medium",
        evidence=forecast_rec["evidence"],
        icon="",
    )
next_page_button("Segmentation", "pages/4_Segmentation.py")
