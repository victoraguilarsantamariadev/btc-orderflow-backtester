"""Sentiment features: Long/Short ratio and Taker buy/sell ratio."""

import pandas as pd
import numpy as np


def calc_ls_ratio_zscore(df: pd.DataFrame, lookback: int = 168) -> pd.Series:
    """Z-score of long/short account ratio.

    Extreme positive = too many longs (contrarian short).
    Extreme negative = too many shorts (contrarian long).
    """
    if "ls_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    ls = df["ls_ratio"]
    mean = ls.rolling(lookback, min_periods=lookback // 2).mean()
    std = ls.rolling(lookback, min_periods=lookback // 2).std()
    return (ls - mean) / std.replace(0, np.nan)


def calc_ls_extreme(zscore: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Flag extreme long/short positioning.

    Returns:
        +1: Too many longs (contrarian short signal)
        -1: Too many shorts (contrarian long signal)
         0: Normal
    """
    result = pd.Series(0, index=zscore.index)
    result[zscore > threshold] = 1
    result[zscore < -threshold] = -1
    return result


def calc_taker_ratio_zscore(df: pd.DataFrame, lookback: int = 168) -> pd.Series:
    """Z-score of taker buy/sell volume ratio.

    High = aggressive buying dominance.
    Low = aggressive selling dominance.
    """
    if "taker_ratio" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    tr = df["taker_ratio"]
    mean = tr.rolling(lookback, min_periods=lookback // 2).mean()
    std = tr.rolling(lookback, min_periods=lookback // 2).std()
    return (tr - mean) / std.replace(0, np.nan)


def calc_taker_extreme(zscore: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Flag extreme taker volume imbalance.

    Returns:
        +1: Extreme buying (confirmation of bullish move)
        -1: Extreme selling (confirmation of bearish move)
         0: Normal
    """
    result = pd.Series(0, index=zscore.index)
    result[zscore > threshold] = 1
    result[zscore < -threshold] = -1
    return result


def add_sentiment_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all sentiment features to the DataFrame."""
    ls_lookback = config["features"]["ls_ratio_zscore_lookback"]
    taker_lookback = config["features"]["taker_ratio_zscore_lookback"]

    df["ls_ratio_zscore"] = calc_ls_ratio_zscore(df, ls_lookback)
    df["ls_extreme"] = calc_ls_extreme(df["ls_ratio_zscore"])

    df["taker_ratio_zscore"] = calc_taker_ratio_zscore(df, taker_lookback)
    df["taker_extreme"] = calc_taker_extreme(df["taker_ratio_zscore"])

    return df
