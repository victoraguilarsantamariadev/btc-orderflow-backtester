"""OHLCV-based features: ATR, Bollinger Bands, range metrics."""

import pandas as pd
import numpy as np


def calc_true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range from OHLCV data."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def calc_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Average True Range over N periods."""
    tr = calc_true_range(df)
    return tr.rolling(period).mean()


def calc_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, lower, bandwidth, %B.

    Returns DataFrame with columns: bb_upper, bb_lower, bb_bandwidth, bb_pctb
    """
    sma = df["close"].rolling(period).mean()
    rolling_std = df["close"].rolling(period).std()

    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    bandwidth = (upper - lower) / sma  # Normalized bandwidth
    pctb = (df["close"] - lower) / (upper - lower)  # %B: 0-1 range

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_pctb": pctb,
    }, index=df.index)


def calc_range_percentile(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current bar range as percentile of recent N-period ranges.

    Low values = compression, high values = expansion.
    """
    bar_range = df["high"] - df["low"]
    return bar_range.rolling(period).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100,
        raw=False,
    )


def calc_body_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of candle body to total range. High = strong directional bar."""
    body = (df["close"] - df["open"]).abs()
    total = df["high"] - df["low"]
    return body / total.replace(0, np.nan)


def calc_returns(df: pd.DataFrame) -> pd.Series:
    """Close-to-close log returns."""
    return np.log(df["close"] / df["close"].shift(1))


def add_ohlcv_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all OHLCV-based features to the DataFrame."""
    period = config["features"]["atr_period"]
    bb_period = config["features"]["bollinger_period"]
    bb_std = config["features"]["bollinger_std"]

    df["atr"] = calc_atr(df, period)
    df["returns"] = calc_returns(df)

    bb = calc_bollinger(df, bb_period, bb_std)
    for col in bb.columns:
        df[col] = bb[col].values

    df["range_pctile"] = calc_range_percentile(df, period)
    df["body_ratio"] = calc_body_ratio(df)

    return df
