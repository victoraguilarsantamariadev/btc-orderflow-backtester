import datetime as dt

import pandas as pd


def to_ms(date_str: str) -> int:
    """Convert date string 'YYYY-MM-DD' to UTC milliseconds timestamp."""
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def from_ms(ms: int) -> dt.datetime:
    """Convert UTC milliseconds timestamp to datetime."""
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc)


def ms_to_str(ms: int) -> str:
    """Convert UTC milliseconds to 'YYYY-MM-DD HH:MM:SS' string."""
    return from_ms(ms).strftime("%Y-%m-%d %H:%M:%S")


def date_range_to_ms(start: str, end: str) -> tuple[int, int]:
    """Convert start/end date strings to (start_ms, end_ms)."""
    return to_ms(start), to_ms(end)


def align_to_5min(ts_ms: int) -> int:
    """Round down a millisecond timestamp to the nearest 5-minute boundary."""
    five_min_ms = 5 * 60 * 1000
    return (ts_ms // five_min_ms) * five_min_ms


def ms_series_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a Series of millisecond timestamps to UTC datetime."""
    return pd.to_datetime(series, unit="ms", utc=True)
