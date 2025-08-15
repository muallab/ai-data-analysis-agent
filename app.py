import streamlit as st
from pathlib import Path
import pandas as pd
import json

from agent.dataio import save_upload, load_dataframe, summarize_schema
from agent.llm import get_client, ChatMessage
from agent.prompts import SYSTEM_PROMPT, PLANNER_PROMPT
from agent.executor import execute_sql, execute_pandas, render_chart

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

# If no file uploaded yet
if not uploaded:
    st.markdown("""
**Current features:**
1. Step 2: Upload CSV/XLSX and see schema summary  
2. Step 3: LLM reads schema and suggests analysis ideas  
3. Step 4: Planner creates a structured JSON plan  
4. Step 5: Execute plan and display results (SQL or Pandas)  
5. Step 6: Render charts when a chart spec is provided
""")
    with st.expander("Project Health Check", expanded=True):
        sample = Path("data/samples/sales.csv")
        st.write("✅ Streamlit is running.")
        st.write(f"📄 Sample data exists: {sample.exists()}  →  {sample}")
    st.stop()

# Save and load
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

# --- Step 3: LLM Sanity Check ---
st.subheader("Step 3: LLM Sanity Check (no code execution yet)")
if st.button("Generate analysis ideas"):
    schema_txt = (
        f"Columns: {schema['columns']}\n"
        f"Dtypes: {schema['dtypes']}\n"
        f"Missing%: {schema['missing_pct']}\n"
        f"Shape: {schema['shape']}\n"
        f"Sample rows: {schema['sample_rows']}"
    )
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"Here is the dataset schema:\n{schema_txt}\n\nPropose ideas."),
    ]
    try:
        client = get_client()
        with st.spinner("Asking the model..."):
            answer = client.chat(messages, temperature=0.2)
        st.success("Ideas generated")
        st.write(answer)
    except Exception as e:
        st.error(f"LLM call failed: {e}")

# --- Step 4: Planning Agent ---
st.subheader("Step 4: Planning Agent")
question = st.text_input(
    "Enter your analysis question",
    "Which category has the highest total revenue?"
)

if st.button("Generate plan"):
    schema_txt = (
        f"Columns: {schema['columns']}\n"
        f"Dtypes: {schema['dtypes']}\n"
        f"Missing%: {schema['missing_pct']}\n"
        f"Shape: {schema['shape']}\n"
    )
    messages = [
        ChatMessage(role="system", content=PLANNER_PROMPT),
        ChatMessage(role="user", content=f"Schema:\n{schema_txt}\nQuestion:\n{question}"),
    ]
    try:
        client = get_client()
        with st.spinner("Planning..."):
            answer = client.chat(messages, temperature=0.2)
        st.success("Plan generated")
        st.code(answer, language="json")
        st.session_state.plan_output = answer  # persist plan for Step 5
    except Exception as e:
        st.error(f"Planner failed: {e}")

# --- Step 5 & 6: Execute Plan + Render Chart ---
if "plan_output" in st.session_state:
    try:
        plan_data = json.loads(st.session_state.plan_output)
    except json.JSONDecodeError as e:
        st.error(f"Planner output is not valid JSON: {e}")
        plan_data = None

    if plan_data and st.button("Run plan"):
        try:
            # Execute code per approach
            if plan_data["approach"] == "sql":
                result_df = execute_sql(df, plan_data["code"])
            elif plan_data["approach"] == "pandas":
                result_df = execute_pandas(df, plan_data["code"])
            else:
                st.error(f"Unknown approach: {plan_data['approach']}")
                st.stop()

            st.subheader("Execution Results")
            st.dataframe(result_df, use_container_width=True)

            # Step 6: Render chart if provided
            chart_spec = plan_data.get("chart")
            if chart_spec:
                try:
                    st.subheader("Chart")
                    fig = render_chart(result_df, chart_spec)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as ce:
                    st.warning(f"Chart rendering skipped: {ce}")

        except Exception as e:
            st.error(f"Execution failed: {e}")