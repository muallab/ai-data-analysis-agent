# agent/executor.py
from __future__ import annotations
import duckdb
import pandas as pd
import plotly.express as px

def execute_sql(df: pd.DataFrame, sql_code: str) -> pd.DataFrame:
    """
    Execute SQL against a registered in-memory table via DuckDB.
    The in-memory table is registered as 'table_name'.
    """
    con = duckdb.connect()
    con.register("table_name", df)
    try:
        result = con.execute(sql_code).fetchdf()
    finally:
        con.close()
    return result

def execute_pandas(df: pd.DataFrame, pandas_code: str) -> pd.DataFrame:
    """
    Execute *restricted* Pandas code.
    Require the code to set a variable named 'result' (DataFrame).
    """
    safe_globals = {"pd": pd}
    safe_locals = {"df": df.copy()}
    exec(pandas_code, safe_globals, safe_locals)  # guarded context
    result = safe_locals.get("result")
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Pandas code did not produce a DataFrame named 'result'.")
    return result

def render_chart(df: pd.DataFrame, chart_spec: dict):
    """
    Render a Plotly chart from a minimal spec like:
    {"type": "bar", "x": "category", "y": "total_revenue"}
    """
    if not isinstance(chart_spec, dict):
        raise ValueError("chart_spec must be a dict with keys like type/x/y")

    ctype = chart_spec.get("type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")

    if ctype == "bar":
        return px.bar(df, x=x, y=y)
    elif ctype == "line":
        return px.line(df, x=x, y=y)
    elif ctype == "scatter":
        return px.scatter(df, x=x, y=y)
    elif ctype == "pie":
        # for pie we expect names and values
        names = chart_spec.get("names", x)
        values = chart_spec.get("values", y)
        return px.pie(df, names=names, values=values)
    else:
        raise ValueError(f"Unsupported chart type: {ctype}")