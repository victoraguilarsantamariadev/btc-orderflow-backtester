"""Orderflow features: Delta, CVD, absorption detection, imbalance.

These are calculated from aggregate trades data (buy_volume, sell_volume, delta).
"""

import pandas as pd
import numpy as np


def calc_delta(df: pd.DataFrame) -> pd.Series:
    """Volume delta: buy_volume - sell_volume.

    Already computed during aggTrades download, but this recalculates if needed.
    """
    if "delta" in df.columns:
        return df["delta"]
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        return df["buy_volume"] - df["sell_volume"]
    return pd.Series(np.nan, index=df.index)


def calc_cvd(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume Delta.

    Running sum of delta. Rising CVD = sustained buying pressure.
    """
    delta = calc_delta(df)
    return delta.cumsum()


def calc_cvd_divergence(df: pd.DataFrame, lookback: int = 12) -> pd.Series:
    """Detect CVD divergence with price.

    Returns:
        +1: Bullish divergence (price makes new low but CVD doesn't)
        -1: Bearish divergence (price makes new high but CVD doesn't)
         0: No divergence
    """
    if "cvd" not in df.columns:
        return pd.Series(0, index=df.index)

    close = df["close"]
    cvd = df["cvd"]

    result = pd.Series(0, index=df.index)

    for i in range(lookback, len(df)):
        window_close = close.iloc[i - lookback:i + 1]
        window_cvd = cvd.iloc[i - lookback:i + 1]

        # Bearish divergence: price at window high but CVD below its window high
        if close.iloc[i] >= window_close.max() * 0.999:
            if cvd.iloc[i] < window_cvd.max() * 0.95:
                result.iloc[i] = -1

        # Bullish divergence: price at window low but CVD above its window low
        if close.iloc[i] <= window_close.min() * 1.001:
            if cvd.iloc[i] > window_cvd.min() * 1.05:
                result.iloc[i] = 1

    return result


def calc_absorption(df: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """Detect absorption: strong delta in one direction but price doesn't follow.

    High absorption score = someone is absorbing aggressive orders.

    Returns:
        Positive: Bullish absorption (sell pressure absorbed, price holds)
        Negative: Bearish absorption (buy pressure absorbed, price holds)
    """
    if "delta" not in df.columns:
        return pd.Series(0.0, index=df.index)

    delta_sum = df["delta"].rolling(lookback).sum()
    price_change = df["close"] - df["close"].shift(lookback)
    atr = (df["high"] - df["low"]).rolling(lookback).mean()

    # Normalize price change by ATR
    price_move = price_change / atr.replace(0, np.nan)

    # Absorption: big delta but small price move
    # Positive delta_sum + negative/flat price = bearish absorption
    # Negative delta_sum + positive/flat price = bullish absorption
    absorption = pd.Series(0.0, index=df.index)

    # Bearish absorption: lots of buying but price doesn't go up
    mask_bear = (delta_sum > delta_sum.rolling(50).quantile(0.8)) & (price_move < 0.5)
    absorption[mask_bear] = -delta_sum[mask_bear] / delta_sum.rolling(50).std().replace(0, np.nan)

    # Bullish absorption: lots of selling but price doesn't go down
    mask_bull = (delta_sum < delta_sum.rolling(50).quantile(0.2)) & (price_move > -0.5)
    absorption[mask_bull] = -delta_sum[mask_bull] / delta_sum.rolling(50).std().replace(0, np.nan)

    return absorption


def calc_delta_zscore(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """Z-score of delta to identify extreme orderflow."""
    if "delta" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    delta = df["delta"]
    mean = delta.rolling(lookback).mean()
    std = delta.rolling(lookback).std()
    return (delta - mean) / std.replace(0, np.nan)


def add_orderflow_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all orderflow features to the DataFrame."""
    lookback = config["features"]["absorption_lookback"]
    cvd_lookback = config["features"]["cvd_divergence_lookback"]

    if "delta" not in df.columns:
        # No aggTrades data available — skip orderflow features
        df["delta"] = np.nan
        df["cvd"] = np.nan
        df["cvd_divergence"] = 0
        df["absorption"] = 0.0
        df["delta_zscore"] = np.nan
        return df

    if "cvd" not in df.columns:
        df["cvd"] = calc_cvd(df)

    df["cvd_divergence"] = calc_cvd_divergence(df, cvd_lookback)
    df["absorption"] = calc_absorption(df, lookback)
    df["delta_zscore"] = calc_delta_zscore(df)

    return df
