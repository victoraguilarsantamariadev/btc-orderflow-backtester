import pandas as pd
from pathlib import Path


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet file. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", index=False)


def load_parquet(path: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Load parquet file with optional date filtering.

    Expects a 'timestamp' column as UTC datetime for filtering.
    """
    df = pd.read_parquet(path, engine="pyarrow")

    if start and "timestamp" in df.columns:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end and "timestamp" in df.columns:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]

    return df


def append_parquet(df: pd.DataFrame, path: str) -> None:
    """Append rows to existing parquet file, or create if not exists."""
    p = Path(path)
    if p.exists():
        existing = pd.read_parquet(p, engine="pyarrow")
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        save_parquet(combined, path)
    else:
        save_parquet(df, path)
