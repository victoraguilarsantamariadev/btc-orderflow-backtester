"""Funding rate features: z-scores and extreme detection."""

import pandas as pd
import numpy as np


def calc_funding_zscore(df: pd.DataFrame, lookback: int = 168) -> pd.Series:
    """Z-score of funding rate over rolling window.

    lookback=168 covers 7 days of 8h funding snapshots (forward-filled to 5min).
    Extreme positive (>2) = market overly long, contrarian short signal.
    Extreme negative (<-2) = market overly short, contrarian long signal.
    """
    if "funding_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    fr = df["funding_rate"]
    mean = fr.rolling(lookback, min_periods=lookback // 2).mean()
    std = fr.rolling(lookback, min_periods=lookback // 2).std()
    return (fr - mean) / std.replace(0, np.nan)


def calc_funding_extreme(df: pd.DataFrame, zscore: pd.Series,
                          threshold: float = 2.0) -> pd.Series:
    """Binary flag for extreme funding.

    Returns:
        +1: Extreme positive (contrarian short signal)
        -1: Extreme negative (contrarian long signal)
         0: Normal
    """
    result = pd.Series(0, index=df.index)
    result[zscore > threshold] = 1
    result[zscore < -threshold] = -1
    return result


def calc_funding_cumulative(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Cumulative funding over N 5-min bars.

    Shows total cost of holding a long position over the window.
    Very high cumulative = longs being drained.
    """
    if "funding_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return df["funding_rate"].rolling(window).sum()


def add_funding_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all funding rate features to the DataFrame."""
    lookback = config["features"]["funding_zscore_lookback"]
    threshold = config["features"]["funding_extreme_threshold"]

    df["funding_zscore"] = calc_funding_zscore(df, lookback)
    df["funding_extreme"] = calc_funding_extreme(df, df["funding_zscore"], threshold)
    df["funding_cumulative"] = calc_funding_cumulative(df)

    return df
