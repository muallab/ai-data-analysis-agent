import json
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from agent.dataio import save_upload, load_dataframe, summarize_schema
from agent.llm import get_client, ChatMessage
from agent.prompts import SYSTEM_PROMPT, PLANNER_PROMPT, EXPLAIN_PROMPT
from agent.executor import execute_sql, execute_pandas, render_chart


# ----------------------------
# UI copy (centralized)
# ----------------------------
UI = {
    "title": "Data Copilot — Ask, Analyze, Visualize.",
    "sidebar_header": "Upload data",
    "uploader_label": "Drop a CSV/XLSX here (≤200 MB)",
    "schema_header": "Dataset overview",
    "ideas_header": "Suggestions (no code execution)",
    "planner_header": "Plan (structured JSON: SQL or Pandas)",
    "results_header": "Results",
    "viz_header": "Visualization",
    "explain_header": "Explanation (plain English)",
    "ideas_button": "Suggest analyses",
    "planner_button": "Create plan",
    "run_button": "Execute",
    "explain_button": "Explain results",
    "exp_columns": "Column types (dtypes)",
    "exp_missing": "Missing values (%)",
    "exp_health": "Project health check",
    "model_label": "Model",
    "temp_label": "Creativity (temperature)",
    "download_results": "Download results (CSV)",
    "download_chart": "Download chart (HTML)",
    "download_expl": "Download explanation (Markdown)",
    "code_preview": "Planned code",
    "chart_preview": "Planned chart spec",
    "show_error_details": "Show error details",
}

# Optional: light CSS polish
st.markdown("""
<style>
.block-container { padding-top: 1.25rem; padding-bottom: 1.75rem; }
h1, h2, h3 { letter-spacing: .2px; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
details summary { font-size: 0.95rem; }
[data-testid="stMetricValue"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title=UI["title"], layout="wide")
st.title(UI["title"])


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

def _guess_auto_chart_spec(df: pd.DataFrame) -> Optional[dict]:
    """
    If no chart spec provided, guess a simple bar chart when possible:
    - exactly 2 columns
    - one is non-numeric (treated as x), one is numeric (treated as y)
    """
    if df is None or df.empty or df.shape[1] != 2:
        return None
    cols = list(df.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1 and len(cat_cols) == 1:
        return {"type": "bar", "x": cat_cols[0], "y": num_cols[0]}
    return None

def _results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    sio = StringIO()
    df.to_csv(sio, index=False)
    return sio.getvalue().encode("utf-8")


# ----------------------------
# Sidebar: upload + model controls
# ----------------------------
st.sidebar.header(UI["sidebar_header"])
uploaded = st.sidebar.file_uploader(UI["uploader_label"], type=["csv", "xlsx", "xls"])

# Model & temperature controls
models = ["gpt-4.1-mini", "gpt-4.1"]
selected_model = st.sidebar.selectbox(UI["model_label"], models, index=0)
temperature = st.sidebar.slider(UI["temp_label"], min_value=0.0, max_value=1.0, value=0.2, step=0.05)

st.divider()

# ----------------------------
# Landing when no file yet
# ----------------------------
if not uploaded:
    st.markdown("""
**Features:**
1. Upload CSV/XLSX and view schema summary  
2. LLM reads schema and suggests analysis ideas  
3. Planner produces a structured JSON plan (SQL or Pandas)  
4. Execute the plan and render results (with charts)  
5. Explain results in plain English
""")
    with st.expander(UI["exp_health"], expanded=True):
        sample = Path("data/samples/sales.csv")
        st.write("✅ Streamlit is running.")
        st.write(f"📄 Sample data exists: {sample.exists()}  →  {sample}")
    st.stop()

# ----------------------------
# Step 2: Upload + schema summary
# ----------------------------
dest_path = save_upload(uploaded.name, uploaded.getvalue())
df = _cache_load_dataframe(str(dest_path))

schema = summarize_schema(df)
st.subheader(UI["schema_header"])
_schema_metrics(schema)
with st.expander(UI["exp_columns"], expanded=True):
    st.json(schema["dtypes"])
with st.expander(UI["exp_missing"], expanded=False):
    st.json(schema["missing_pct"])
st.subheader("Sample Rows (first 5)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ----------------------------
# Step 3: Suggestions (no code execution)
# ----------------------------
st.subheader(UI["ideas_header"])
if st.button(UI["ideas_button"]):
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
        client = get_client(selected_model)
        with st.spinner("Asking the model..."):
            answer = client.chat(messages, temperature=temperature)
        st.success("Ideas generated")
        st.write(answer)
    except Exception as e:
        st.error("LLM call failed")
        with st.expander(UI["show_error_details"]):
            st.exception(e)

st.divider()

# ----------------------------
# Step 4: Plan (structured JSON: SQL or Pandas)
# ----------------------------
st.subheader(UI["planner_header"])
question = st.text_input(
    "Enter your analysis question",
    "Which category has the highest total revenue?"
)

if st.button(UI["planner_button"]):
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
        client = get_client(selected_model)
        with st.spinner("Planning..."):
            plan_json = client.chat(messages, temperature=temperature)
        st.success("Plan generated")
        st.code(plan_json, language="json")

        # Show planned code & chart spec (previews)
        try:
            preview = json.loads(plan_json)
            code_lang = "sql" if preview.get("approach") == "sql" else "python"
            st.caption(UI["code_preview"])
            st.code(preview.get("code", ""), language=code_lang)
            if preview.get("chart"):
                st.caption(UI["chart_preview"])
                st.json(preview["chart"])
        except Exception:
            pass

        # Persist plan for Step 5
        st.session_state.plan_output = plan_json
        st.session_state.plan_question = question
        st.session_state.plan_schema = schema
        st.session_state.plan_model = selected_model
        st.session_state.plan_temp = temperature
    except Exception as e:
        st.error("Planner failed")
        with st.expander(UI["show_error_details"]):
            st.exception(e)

st.divider()

# ----------------------------
# Step 5 & 6: Results + Visualization (execute plan)
# ----------------------------
if "plan_output" in st.session_state:
    plan_data = _safe_loads(st.session_state.plan_output)
    if not plan_data or not isinstance(plan_data, dict) or "approach" not in plan_data or "code" not in plan_data:
        st.error("Planner returned invalid JSON. Please click **Create plan** again.")
    else:
        if st.button(UI["run_button"]):
            try:
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

                st.subheader(UI["results_header"])
                st.dataframe(result_df, use_container_width=True)

                # Downloads: results CSV
                st.download_button(
                    label=UI["download_results"],
                    data=_results_to_csv_bytes(result_df),
                    file_name="results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # Visualization (chart spec) OR auto-fallback
                chart_spec = plan_data.get("chart")
                auto_spec = None
                if not chart_spec:
                    auto_spec = _guess_auto_chart_spec(result_df)

                if chart_spec or auto_spec:
                    try:
                        st.subheader(UI["viz_header"])
                        spec = chart_spec or auto_spec
                        fig = render_chart(result_df, spec)
                        st.plotly_chart(fig, use_container_width=True)

                        # Download chart as standalone HTML (portable, no extra deps)
                        chart_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
                        st.download_button(
                            label=UI["download_chart"],
                            data=chart_html.encode("utf-8"),
                            file_name="chart.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                    except Exception as ce:
                        st.warning(f"Chart rendering skipped: {ce}")

                # Save context for explanations (Step 7)
                st.session_state.last_result = result_df
                st.session_state.last_plan = plan_data
                st.session_state.last_question = st.session_state.get("plan_question", question)
                st.session_state.last_schema = st.session_state.get("plan_schema", schema)
                st.session_state.last_model = st.session_state.get("plan_model", selected_model)
                st.session_state.last_temp = st.session_state.get("plan_temp", temperature)

            except Exception as e:
                st.error("Execution failed")
                with st.expander(UI["show_error_details"]):
                    st.write("**Approach**:", plan_data.get("approach"))
                    st.write("**Code that failed:**")
                    lang = "sql" if plan_data.get("approach") == "sql" else "python"
                    st.code(plan_data.get("code", ""), language=lang)
                    st.exception(e)

st.divider()

# ----------------------------
# Step 7: Explanation (plain English)
# ----------------------------
st.subheader(UI["explain_header"])

if (
    "last_result" in st.session_state
    and isinstance(st.session_state.last_result, pd.DataFrame)
    and not st.session_state.last_result.empty
):
    if st.button(UI["explain_button"]):
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
            client = get_client(st.session_state.get("last_model", models[0]))
            with st.spinner("Generating explanation..."):
                explanation = client.chat(messages, temperature=st.session_state.get("last_temp", 0.2))

            st.success("Explanation")
            st.markdown(explanation)  # markdown renders bullets better

            # Download explanation as markdown
            st.download_button(
                label=UI["download_expl"],
                data=explanation.encode("utf-8"),
                file_name="explanation.md",
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as e:
            st.error("Explanation failed")
            with st.expander(UI["show_error_details"]):
                st.exception(e)
else:
    st.caption("Run a plan first to enable explanations.")