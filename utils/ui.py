"""
utils/ui.py
-----------
Shared UI helpers for BOIDE.
"""

import os

import streamlit as st


def load_css():
    """Inject the global BOIDE stylesheet into the page."""
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&family=Syne:wght@700;800&display=swap');
            html,body,[data-testid="stAppViewContainer"]{background:#f8fafc!important;color:#0f172a!important;font-family:'DM Sans',sans-serif;}
            h1{font-family:'Syne',sans-serif!important;font-weight:800!important;}
            [data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid rgba(148,163,184,0.25)!important;}
            .stButton>button{background:linear-gradient(135deg,#1d4ed8,#0369a1)!important;color:#e0f2fe!important;border-radius:8px!important;border:1px solid rgba(59,130,246,0.4)!important;}
            </style>
            """,
            unsafe_allow_html=True,
        )


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="DM Sans, sans-serif", color="#334155", size=12),
    xaxis=dict(
        gridcolor="rgba(148,163,184,0.20)",
        linecolor="rgba(148,163,184,0.35)",
        tickfont=dict(family="DM Mono, monospace", size=10, color="#64748b"),
    ),
    yaxis=dict(
        gridcolor="rgba(148,163,184,0.20)",
        linecolor="rgba(148,163,184,0.35)",
        tickfont=dict(family="DM Mono, monospace", size=10, color="#64748b"),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(148,163,184,0.25)",
        borderwidth=1,
        font=dict(family="DM Sans", size=11, color="#334155"),
    ),
    margin=dict(t=40, b=40, l=40, r=20),
    title_font=dict(family="DM Sans", size=14, color="#0f172a"),
)


def apply_theme(fig):
    """Apply the BOIDE light theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


NAV_PAGES = [
    ("01", "Overview", "pages/1_Overview.py"),
    ("02", "Products", "pages/2_Product_analysis.py"),
    ("03", "Forecasting", "pages/3_Forecasting.py"),
    ("04", "Segmentation", "pages/4_Segmentation.py"),
    ("05", "Anomaly", "pages/5_Anomaly_Detection.py"),
    ("06", "AI Insights", "pages/6_AI_Insights.py"),
    ("07", "Digital Twin", "pages/7_Digital_Twin.py"),
    ("08", "Reports", "pages/8_Reports.py"),
    ("09", "Methodology", "pages/9_Methodology.py"),
    ("10", "Data Preview", "pages/10_Data_Preview.py"),
]


def topnav(active: str = ""):
    """Render the horizontal top navigation bar."""
    st.markdown(
        """
        <style>
        [data-testid="stPageLink"] a,
        [data-testid="stPageLink"] p,
        [data-testid="stPageLink"] span,
        [data-testid="stPageLink"] div {
            color: #334155 !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em !important;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }
        [data-testid="stPageLink"]:hover a,
        [data-testid="stPageLink"]:hover p,
        [data-testid="stPageLink"]:hover span {
            color: #2563eb !important;
        }
        [data-testid="stPageLink"] {
            border-radius: 6px !important;
            transition: background 0.15s !important;
            overflow: visible !important;
            min-width: fit-content !important;
        }
        [data-testid="stPageLink"]:hover {
            background: rgba(59,130,246,0.08) !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            overflow: visible !important;
            min-width: 0 !important;
            flex: 1 1 auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="boide-topnav" style="padding:6px 10px; margin-bottom:16px;">', unsafe_allow_html=True)
    cols = st.columns(len(NAV_PAGES))
    for col, (icon, lbl, path) in zip(cols, NAV_PAGES):
        with col:
            st.page_link(path, label=f"{icon} {lbl}", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_title(title: str, subtitle: str = ""):
    """Render a styled page title and subtitle."""
    st.markdown(
        f'<div class="boide-page-title">{title}</div>'
        f'<div class="boide-page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str):
    """Render a styled section header with icon."""
    icon_html = f'<span class="section-icon">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="boide-section">{icon_html}{title}</div>',
        unsafe_allow_html=True,
    )


def kpi_grid(cards: list):
    """Render a KPI grid using native Streamlit metrics."""
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 22px !important;
            font-family: 'DM Mono', monospace !important;
            font-weight: 500 !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 11px !important;
            font-family: 'DM Mono', monospace !important;
            letter-spacing: 0.07em !important;
            text-transform: uppercase !important;
            opacity: 0.6 !important;
        }
        [data-testid="stMetricDelta"] svg { display: none; }
        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid rgba(148,163,184,0.22) !important;
            border-radius: 10px !important;
            padding: 14px 16px !important;
            border-top: 2px solid rgba(59,130,246,0.4) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    delta_color_map = {"positive": "normal", "negative": "inverse", "neutral": "off"}
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        delta = card.get("delta")
        delta_type = card.get("delta_type", "neutral")
        col.metric(
            label=card["label"],
            value=card["value"],
            delta=delta if delta else None,
            delta_color=delta_color_map.get(delta_type, "off"),
        )


_BADGE_CLASS = {
    "CRITICAL": "badge-critical",
    "HIGH": "badge-high",
    "MEDIUM": "badge-medium",
    "GROWTH": "badge-growth",
    "OPPORTUNITY": "badge-opportunity",
    "INFO": "badge-medium",
}


def insight_card(rec: dict):
    """Render a recommendation card."""
    tag = rec.get("tag", "neutral")
    badge = rec.get("badge", "INFO").upper()
    badge_cls = _BADGE_CLASS.get(badge, "badge-medium")
    card_cls = tag if tag in ("risk", "growth", "opportunity") else "neutral"

    st.markdown(
        f"""
        <div class="insight-card {card_cls}">
            <span class="insight-badge {badge_cls}">{badge}</span>
            <div class="insight-text">{rec['insight']}</div>
            <div class="insight-action">Action: {rec['action']}</div>
            <div class="insight-evidence">evidence: {rec['evidence']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_bar(label: str, value: int, max_val: int = 100):
    """Render a styled risk/confidence bar."""
    pct = min(100, int(value / max_val * 100))
    if pct > 65:
        color = "#ef4444"
    elif pct > 35:
        color = "#f59e0b"
    else:
        color = "#10b981"

    st.markdown(
        f"""
        <div class="risk-bar-wrap">
            <div class="risk-bar-label">
                <span>{label}</span>
                <span style="color:{color}; font-weight:500;">{value}/{max_val}</span>
            </div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill"
                     style="width:{pct}%; background:linear-gradient(90deg,{color},{color}aa);">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<div class="boide-divider"></div>', unsafe_allow_html=True)
