# agent/dataio.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".csv", ".xlsx", ".xls"}

def save_upload(file_name: str, file_bytes: bytes) -> Path:
    """Persist the uploaded file to data/uploads and return its path."""
    ext = Path(file_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTS)}")
    dest = UPLOAD_DIR / file_name
    with open(dest, "wb") as f:
        f.write(file_bytes)
    return dest

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from CSV or Excel by file extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def summarize_schema(df: pd.DataFrame) -> dict:
    """Return a compact schema profile for the UI and the LLM planner."""
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing_pct = (df.isna().sum() / max(len(df), 1) * 100.0).round(2).to_dict()
    sample_rows = df.head(5).to_dict(orient="records")
    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "dtypes": dtypes,
        "missing_pct": missing_pct,
        "sample_rows": sample_rows,
        "columns": list(df.columns),
    }
