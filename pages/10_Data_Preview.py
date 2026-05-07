import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import init_session_state
from utils.decision_panel import ai_insight_strip, flow_indicator
from utils.ui import divider, load_css, page_title, section_header, topnav


st.set_page_config(page_title="Data Preview - BOIDE", layout="wide", page_icon=":bar_chart:")
load_css()
init_session_state()
topnav("Data Preview")
flow_indicator("data")
page_title("Data Preview", "UPLOAD CSV FILES, PREVIEW THEM, AND CHECK STRUCTURE BEFORE FUTURE INTEGRATION")

st.info("This page previews your own business CSV files. It does not replace the built-in Olist dataset in the analytics pipeline yet.")

uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.markdown("Upload CSV files to preview their columns, row counts, and sample records.")
else:
    section_header("", "Uploaded Files")
    summary_rows = []
    previews = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            summary_rows.append(
                {
                    "File": file.name,
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "Size (KB)": round(file.size / 1024, 2),
                }
            )
            previews.append((file.name, df))
        except Exception as exc:
            summary_rows.append(
                {
                    "File": file.name,
                    "Rows": "Error",
                    "Columns": "Error",
                    "Size (KB)": round(file.size / 1024, 2),
                }
            )
            st.error(f"Could not read `{file.name}`: {exc}")

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    divider()

    section_header("", "File Previews")
    for name, df in previews:
        with st.expander(f"{name} - {len(df)} rows, {len(df.columns)} columns", expanded=False):
            st.write("Columns:", list(df.columns))
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)

ai_insight_strip(
    "Use this page to inspect incoming CSV files before integrating them into the BOIDE analytics workflow.",
    label="AI: Data Preview Guidance",
)
