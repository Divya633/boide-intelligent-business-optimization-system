import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import init_session_state, load_data, next_page_button
from utils.decision_panel import ai_insight_strip, flow_indicator
from utils.ui import divider, kpi_grid, load_css, page_title, section_header, topnav


st.set_page_config(page_title="Methodology - BOIDE", layout="wide", page_icon=":bar_chart:")
load_css()
init_session_state()
topnav("Methodology")
flow_indicator("outcome")
page_title(
    "Project Methodology",
    "DATASET, PREPROCESSING, MODELS, EVALUATION METRICS, DECISION FLOW, AND LIMITATIONS",
)

df = load_data()
forecast_details = st.session_state.get("forecast_details") or {}
segment_details = st.session_state.get("segment_details") or {}
anomaly_details = st.session_state.get("anomaly_details") or {}

orders = int(df["order_id"].nunique()) if "order_id" in df.columns else len(df)
customers = int(df["customer_unique_id"].nunique()) if "customer_unique_id" in df.columns else 0
categories = int(df["category_en"].nunique()) if "category_en" in df.columns else 0
sellers = int(df["seller_id"].nunique()) if "seller_id" in df.columns else 0

kpi_grid(
    [
        {"label": "Merged Rows", "value": f"{len(df):,}"},
        {"label": "Unique Orders", "value": f"{orders:,}"},
        {"label": "Customers", "value": f"{customers:,}"},
        {"label": "Categories", "value": f"{categories:,}"},
        {"label": "Sellers", "value": f"{sellers:,}"},
    ]
)

divider()

section_header("", "Dataset and Source")
st.markdown(
    """
BOIDE uses the public Olist Brazilian e-commerce dataset stored locally in the `data/` folder.
The project joins order, payment, customer, item, product, seller, review, geolocation, and
category translation data to create one business intelligence workflow.
"""
)

section_header("", "Preprocessing")
preprocessing = pd.DataFrame(
    [
        ["Payment aggregation", "Multiple payments per order are aggregated before joining to avoid inflated order-level revenue."],
        ["Order-level deduplication", "KPIs and forecasting use one row per order to avoid item-level double counting."],
        ["Date parsing", "Purchase, delivery, and estimated delivery dates are converted to datetime fields."],
        ["Delivery delay feature", "Delivery delay is calculated as delivered date minus estimated delivery date."],
        ["Category translation", "English category names are used where available for clearer business reporting."],
        ["Missing values", "Forecasting fills time-series gaps using interpolation and forward/backward filling."],
    ],
    columns=["Step", "Purpose"],
)
st.dataframe(preprocessing, use_container_width=True, hide_index=True)

divider()

section_header("", "Models and Techniques")
models = pd.DataFrame(
    [
        ["Forecasting", "ARIMA, Holt-Winters, Moving Average, Seasonal Naive, Baseline", "RMSE, MAPE, directional accuracy, confidence score"],
        ["Segmentation", "RFM feature engineering with KMeans", "Silhouette score and selected cluster count"],
        ["Anomaly Detection", "Isolation Forest over daily revenue and operations signals", "Anomaly rate, high-severity count, severity mix"],
        ["AI Insights", "Rule-based Mini-LLM", "Weighted business rules, confidence, risk score, recommendation ranking"],
        ["Digital Twin", "Heuristic what-if simulation", "Projected revenue, revenue delta, risk score"],
        ["Reports", "CSV and JSON export generation", "Visible previews and downloadable outputs"],
    ],
    columns=["Module", "Method", "Evaluation / Output"],
)
st.dataframe(models, use_container_width=True, hide_index=True)

divider()

section_header("", "Current Run Signals")
run_signals = pd.DataFrame(
    [
        ["Forecast model", forecast_details.get("model_label", "Not run")],
        ["Forecast MAPE", forecast_details.get("mape", "Not run")],
        ["Forecast confidence", forecast_details.get("confidence", "Not run")],
        ["Segmentation k", segment_details.get("selected_k", "Not run")],
        ["Segmentation silhouette", segment_details.get("silhouette_score", "Not run")],
        ["Anomaly rate", anomaly_details.get("anomaly_rate", "Not run")],
        ["High-severity anomalies", anomaly_details.get("high_severity_count", "Not run")],
    ],
    columns=["Signal", "Value"],
)
st.dataframe(run_signals, use_container_width=True, hide_index=True)

divider()

section_header("", "Decision Flow")
st.markdown(
    """
1. Load and merge Olist business data.
2. Compute revenue, customer, delivery, review, product, seller, and payment signals.
3. Run forecasting, segmentation, and anomaly modules to create specialized signals.
4. Feed module outputs into the Mini-LLM rule engine for explainable recommendations.
5. Send the selected recommendation to the Digital Twin to test revenue and risk impact.
6. Export reports for review, presentation, and documentation.
"""
)

section_header("", "Limitations")
limitations = pd.DataFrame(
    [
        ["Historical dataset", "The Olist data is historical, so forecasts demonstrate method rather than live business prediction."],
        ["Rule-based AI", "The Mini-LLM is explainable and offline, but it is not a generative language model."],
        ["Upload preview", "Uploaded CSVs are previewed only and do not yet replace the built-in analytics dataset."],
        ["Simulation model", "Digital Twin projections are heuristic what-if estimates, not causal proof."],
        ["External factors", "Marketing campaigns, holidays, macroeconomic conditions, and competitor actions are not fully modeled."],
    ],
    columns=["Limitation", "How to interpret it"],
)
st.dataframe(limitations, use_container_width=True, hide_index=True)

ai_insight_strip(
    "BOIDE is designed as an explainable decision intelligence prototype: each module produces measurable business signals, and every recommendation can be traced back to model outputs or dataset features.",
    label="AI: Methodology Summary",
)

next_page_button("Data Preview", "pages/10_Data_Preview.py")
