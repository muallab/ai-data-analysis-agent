# agent/executor.py
from __future__ import annotations
import duckdb
import pandas as pd
import io
import plotly.express as px

def execute_sql(df: pd.DataFrame, sql_code: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("table_name", df)
    try:
        result = con.execute(sql_code).fetchdf()
    finally:
        con.close()
    return result

def execute_pandas(df: pd.DataFrame, pandas_code: str) -> pd.DataFrame:
    # Restrict globals and locals for safety
    safe_globals = {"pd": pd}
    safe_locals = {"df": df.copy()}
    exec(pandas_code, safe_globals, safe_locals)
    # Expect the final dataframe to be in safe_locals['result']
    result = safe_locals.get("result")
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Pandas code did not produce a DataFrame named 'result'.")
    return result

def render_chart(df: pd.DataFrame, chart_spec: dict):
    chart_type = chart_spec.get("type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")

    if chart_type == "bar":
        return px.bar(df, x=x, y=y)
    elif chart_type == "line":
        return px.line(df, x=x, y=y)
    elif chart_type == "scatter":
        return px.scatter(df, x=x, y=y)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
