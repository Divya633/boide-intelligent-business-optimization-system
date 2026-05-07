import os
import sys

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.data_loader import init_session_state
from utils.ui import load_css


st.set_page_config(
    page_title="BOIDE Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
init_session_state()

st.sidebar.markdown(
    """
    <div style="padding:16px 8px 12px 8px;
                border-bottom:1px solid rgba(59,130,246,0.15);
                margin-bottom:12px;">
        <div style="font-family:'Syne',sans-serif; font-size:20px;
                    font-weight:800; color:#0f172a; letter-spacing:-0.3px;">
            BOIDE
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:10px;
                    color:#64748b; letter-spacing:0.08em; margin-top:2px;">
            INTELLIGENCE ENGINE v1.0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Run modules in order for best AI Insights confidence.")

pages_map = {
    "nav_overview": "pages/1_Overview.py",
    "nav_product": "pages/2_Product_analysis.py",
    "nav_forecast": "pages/3_Forecasting.py",
    "nav_segment": "pages/4_Segmentation.py",
    "nav_anomaly": "pages/5_Anomaly_Detection.py",
    "nav_insights": "pages/6_AI_Insights.py",
    "nav_twin": "pages/7_Digital_Twin.py",
    "nav_reports": "pages/8_Reports.py",
    "nav_upload": "pages/9_Data_Upload.py",
}
for key, path in pages_map.items():
    if st.session_state.get(key):
        st.session_state[key] = False
        st.switch_page(path)

st.markdown(
    """
    <div style="text-align:center; padding:48px 40px 28px;">
        <div style="font-family:'DM Mono',monospace; font-size:11px;
                    letter-spacing:0.22em; color:#3b82f6;
                    text-transform:uppercase; margin-bottom:14px;">
            Business · Operations · Intelligence · Decision · Engine
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:48px;
                    font-weight:800; color:#0f172a;
                    letter-spacing:-1.5px; line-height:1.05; margin-bottom:16px;">
            BOIDE
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:15px;
                    color:#64748b; max-width:560px;
                    margin:0 auto; line-height:1.75;">
            An explainable AI-powered decision intelligence system built on the Olist e-commerce dataset.<br>
            Forecasting, segmentation, anomaly detection, AI insights, digital twin simulation, reports, and data upload in one place.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

modules = [
    ("🏠", "Overview", "KPIs, revenue trends, category distribution", "nav_overview"),
    ("📦", "Product Analysis", "Ratings, pricing, delivery delay impact", "nav_product"),
    ("📈", "Forecasting", "Model-selected 7-day, 30-day, and 8-week forecasts", "nav_forecast"),
    ("👥", "Segmentation", "RFM clustering with best-k selection", "nav_segment"),
    ("🚨", "Anomaly Detection", "Multivariate anomaly detection and severity", "nav_anomaly"),
    ("🤖", "AI Insights", "Decision engine, confidence, and recommendations", "nav_insights"),
    ("🔬", "Digital Twin", "What-if simulation and strategy testing", "nav_twin"),
    ("📄", "Reports", "Visible summaries plus downloadable exports", "nav_reports"),
    ("📤", "Data Upload", "Upload CSV files and preview business data", "nav_upload"),
]

st.markdown(
    """
    <style>
    div[data-testid="column"] .stButton > button {
        background: linear-gradient(135deg, #2563eb, #0f766e) !important;
        border: 1px solid rgba(37,99,235,0.25) !important;
        color: #ffffff !important;
        border-radius: 0 0 10px 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 10px 12px !important;
        margin-top: 0 !important;
        width: 100% !important;
        min-height: 52px !important;
        transition: all 0.18s !important;
        box-shadow: 0 10px 18px rgba(37,99,235,0.15) !important;
        transform: none !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #0369a1) !important;
        border-color: rgba(37,99,235,0.45) !important;
        color: #ffffff !important;
        transform: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(3)
for i, (icon, name, desc, nav_key) in enumerate(modules):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div style="background:#ffffff;
                        border:1px solid rgba(148,163,184,0.22);
                        border-bottom:none;
                        border-radius:12px 12px 0 0;
                        padding:18px 16px 14px 16px;
                        min-height:162px;
                        display:flex;
                        flex-direction:column;
                        justify-content:flex-start;
                        box-shadow:0 10px 24px rgba(15,23,42,0.06);">
                <div style="font-size:20px; margin-bottom:8px;">{icon}</div>
                <div style="font-family:'DM Sans',sans-serif; font-size:13px;
                            font-weight:700; color:#0f172a; margin-bottom:6px;">{name}</div>
                <div style="font-family:'DM Mono',monospace; font-size:10px;
                            color:#64748b; line-height:1.7;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(f"Open {name}", key=nav_key, use_container_width=True):
            st.session_state[nav_key] = True
            st.rerun()

st.markdown(
    """
    <div style="text-align:center; padding:20px 0 6px;
                font-family:'DM Mono',monospace; font-size:10.5px; color:#475569;">
        Recommended order → Overview · Products · Forecasting · Segmentation · Anomaly Detection · AI Insights · Digital Twin · Reports · Data Upload
    </div>
    """,
    unsafe_allow_html=True,
)
