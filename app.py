import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("AI Data Analysis Agent (from scratch)")

st.markdown("""
This is the **scaffold**. Next we will:
1) Add data upload & schema summary
2) Wire an LLM planner
3) Execute SQL/Pandas and show charts
""")

# Quick smoke test: show sample file existence
sample = Path("data/samples/sales.csv")
with st.expander("Project Health Check", expanded=True):
    st.write("✅ Streamlit is running.")
    st.write(f"📄 Sample data exists: {sample.exists()}  →  {sample}")
    st.info("Proceed to Step 2 after you can run this page locally.")
