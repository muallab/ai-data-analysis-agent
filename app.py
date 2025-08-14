import streamlit as st
from pathlib import Path
import pandas as pd

from agent.dataio import save_upload, load_dataframe, summarize_schema

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("AI Data Analysis Agent (from scratch)")

# Sidebar: file upload
st.sidebar.header("1) Upload a dataset")
uploaded = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

@st.cache_data(show_spinner=False)
def _cache_load_dataframe(path_str: str):
    """Cache the loaded DataFrame by file path string."""
    return load_dataframe(Path(path_str))

def _schema_metrics(schema: dict):
    cols = st.columns(3)
    cols[0].metric("Rows", schema["shape"]["rows"])
    cols[1].metric("Columns", schema["shape"]["cols"])
    total_missing = sum(schema["missing_pct"].values())
    cols[2].metric("Missing (%) total", f"{total_missing:.2f}")

# Intro state
if not uploaded:
    st.markdown("""
This step adds **data ingest**:
- Upload a **CSV/XLSX** on the left
- We’ll cache the dataset and show a **schema summary** (types, missing %, sample rows)
""")
    with st.expander("Project Health Check", expanded=True):
        sample = Path("data/samples/sales.csv")
        st.write("✅ Streamlit is running.")
        st.write(f"📄 Sample data exists: {sample.exists()}  →  {sample}")
    st.stop()

# Save uploaded file & load cached DataFrame
dest_path = save_upload(uploaded.name, uploaded.getvalue())
df = _cache_load_dataframe(str(dest_path))

# Schema summary
schema = summarize_schema(df)
st.subheader("Schema Summary")
_schema_metrics(schema)

with st.expander("Column types", expanded=True):
    st.json(schema["dtypes"])

with st.expander("Missing values (%)", expanded=False):
    st.json(schema["missing_pct"])

st.subheader("Sample Rows (first 5)")
st.dataframe(df.head(5), use_container_width=True)

st.info("✅ Step 2 complete when you can upload a file and see this summary. Next: the LLM planner.")
