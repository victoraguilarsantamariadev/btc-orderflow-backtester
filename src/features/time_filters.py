"""Time-based features: session labels, time-of-day filters."""

import pandas as pd
import numpy as np


def add_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label each bar with trading session.

    Sessions (UTC):
        asia:    00:00 - 08:00
        europe:  08:00 - 14:00
        us:      14:00 - 21:00
        off:     21:00 - 00:00

    Also flags US market open (14:30 UTC) and key event windows.
    """
    hour = df["timestamp"].dt.hour

    conditions = [
        (hour >= 0) & (hour < 8),
        (hour >= 8) & (hour < 14),
        (hour >= 14) & (hour < 21),
        (hour >= 21) | (hour < 0),
    ]
    choices = ["asia", "europe", "us", "off"]

    df["session"] = np.select(conditions, choices, default="off")

    # US market open window (14:00-15:00 UTC = 9:00-10:00 EST)
    df["us_open"] = ((hour >= 14) & (hour < 15)).astype(int)

    # London open window (08:00-09:00 UTC)
    df["london_open"] = ((hour >= 8) & (hour < 9)).astype(int)

    # Asian close / London pre-open (07:00-08:00 UTC)
    df["asia_close"] = ((hour >= 7) & (hour < 8)).astype(int)

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Weekend flag (lower liquidity in crypto)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df
