import json
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.dataio import save_upload, load_dataframe, summarize_schema
from agent.llm import get_client, ChatMessage
from agent.prompts import SYSTEM_PROMPT, PLANNER_PROMPT, EXPLAIN_PROMPT
from agent.executor import execute_sql, execute_pandas, render_chart


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="AI Data Analysis Agent (from scratch)", layout="wide")
st.title("AI Data Analysis Agent (from scratch)")


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def _cache_load_dataframe(path_str: str):
    """Cache the loaded DataFrame by file path string."""
    return load_dataframe(Path(path_str))

def _schema_metrics(schema: dict):
    cols = st.columns(3)
    cols[0].metric("Rows", schema["shape"]["rows"])
    cols[1].metric("Columns", schema["shape"]["cols"])
    total_missing = sum(schema.get("missing_pct", {}).values()) if schema.get("missing_pct") else 0.0
    cols[2].metric("Missing (%) total", f"{total_missing:.2f}")

def _df_preview_records(df: pd.DataFrame, max_rows: int = 50) -> list[dict]:
    """Small JSON preview (records) to keep token usage sensible."""
    return df.head(max_rows).to_dict(orient="records")

def _safe_loads(s: str):
    """Safely parse JSON; return None on failure."""
    try:
        return json.loads(s)
    except Exception:
        return None


# ----------------------------
# Step 2: Upload + schema summary
# ----------------------------
st.sidebar.header("1) Upload a dataset")
uploaded = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.markdown("""
**Features:**
1. Upload CSV/XLSX and view schema summary  
2. LLM reads schema and suggests analysis ideas  
3. Planner produces a structured JSON plan (SQL or Pandas)  
4. Execute the plan and render results (with charts)  
5. Explain results in plain English
""")
    with st.expander("Project Health Check", expanded=True):
        sample = Path("data/samples/sales.csv")
        st.write("✅ Streamlit is running.")
        st.write(f"📄 Sample data exists: {sample.exists()}  →  {sample}")
    st.stop()

# Persist upload and load DataFrame
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


# ----------------------------
# Step 3: LLM Sanity Check (ideas; no code execution)
# ----------------------------
st.subheader("Step 3: LLM Sanity Check (no code execution)")
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


# ----------------------------
# Step 4: Planning Agent (structured JSON)
# ----------------------------
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
            plan_json = client.chat(messages, temperature=0.2)
        st.success("Plan generated")
        st.code(plan_json, language="json")
        # Persist plan for Step 5
        st.session_state.plan_output = plan_json
        st.session_state.plan_question = question
        st.session_state.plan_schema = schema
    except Exception as e:
        st.error(f"Planner failed: {e}")


# ----------------------------
# Step 5 & 6: Execute plan + render chart (hardened)
# ----------------------------
if "plan_output" in st.session_state:
    plan_data = _safe_loads(st.session_state.plan_output)
    if not plan_data or not isinstance(plan_data, dict) or "approach" not in plan_data or "code" not in plan_data:
        st.error("Planner returned invalid JSON. Please click **Generate plan** again.")
    else:
        if st.button("Run plan"):
            try:
                # Execute per approach
                approach = plan_data.get("approach")
                code = plan_data.get("code", "")

                if approach == "sql":
                    result_df = execute_sql(df, code)
                elif approach == "pandas":
                    result_df = execute_pandas(df, code)  # must set: result = <DataFrame>
                else:
                    st.error(f"Unknown approach: {approach}")
                    st.stop()

                # Optional: light number formatting for common column names
                if "total_revenue" in result_df.columns:
                    try:
                        result_df["total_revenue"] = result_df["total_revenue"].astype(float).round(2)
                    except Exception:
                        pass

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

                # Save context for explanations (Step 7)
                st.session_state.last_result = result_df
                st.session_state.last_plan = plan_data
                st.session_state.last_question = st.session_state.get("plan_question", question)
                st.session_state.last_schema = st.session_state.get("plan_schema", schema)

            except Exception as e:
                st.error(f"Execution failed: {e}")


# ----------------------------
# Step 7: Explain results (natural language)
# ----------------------------
st.subheader("Step 7: Explain Results")

if (
    "last_result" in st.session_state
    and isinstance(st.session_state.last_result, pd.DataFrame)
    and not st.session_state.last_result.empty
):
    if st.button("Explain these results"):
        try:
            result_preview = _df_preview_records(st.session_state.last_result, max_rows=50)
            payload = {
                "question": st.session_state.get("last_question", ""),
                "approach": st.session_state.last_plan.get("approach", ""),
                "code": st.session_state.last_plan.get("code", "")[:4000],  # trim long code
                "chart": st.session_state.last_plan.get("chart", None),
                "result_preview": result_preview,
                "result_shape": list(st.session_state.last_result.shape),
                "schema": {
                    "columns": st.session_state.last_schema.get("columns", []),
                    "dtypes": st.session_state.last_schema.get("dtypes", {}),
                    "missing_pct": st.session_state.last_schema.get("missing_pct", {}),
                    "shape": st.session_state.last_schema.get("shape", {}),
                },
            }

            messages = [
                ChatMessage(role="system", content=EXPLAIN_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ]
            client = get_client()
            with st.spinner("Generating explanation..."):
                explanation = client.chat(messages, temperature=0.2)

            st.success("Explanation")
            st.markdown(explanation)  # markdown renders bullets better

        except Exception as e:
            st.error(f"Explanation failed: {e}")
else:
    st.caption("Run a plan first to enable explanations.")