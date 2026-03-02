"""Volatility features: Bollinger squeeze, ATR compression, realized vol."""

import pandas as pd
import numpy as np


def calc_bb_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                     kc_period: int = 20, kc_mult: float = 1.5) -> pd.DataFrame:
    """Bollinger Band Squeeze: Bollinger inside Keltner Channel.

    When BBs are inside KC, volatility is compressed. When they break out,
    expect an expansion move.

    Returns DataFrame with:
        squeeze: bool flag (True = in squeeze)
        squeeze_duration: consecutive bars in squeeze
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Bollinger Bands
    bb_sma = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_sma + bb_std * bb_std_val
    bb_lower = bb_sma - bb_std * bb_std_val

    # Keltner Channel (using ATR)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_upper = bb_sma + kc_mult * atr
    kc_lower = bb_sma - kc_mult * atr

    # Squeeze: BB inside KC
    squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # Count consecutive squeeze bars
    squeeze_groups = (~squeeze).cumsum()
    squeeze_duration = squeeze.groupby(squeeze_groups).cumsum()

    return pd.DataFrame({
        "squeeze": squeeze.astype(int),
        "squeeze_duration": squeeze_duration,
    }, index=df.index)


def calc_atr_compression(df: pd.DataFrame, short_period: int = 5,
                          long_period: int = 20) -> pd.Series:
    """ATR compression ratio: ATR(short) / ATR(long).

    Low ratio (<0.7) = volatility compressing.
    High ratio (>1.3) = volatility expanding.
    """
    from src.features.ohlcv import calc_atr

    atr_short = calc_atr(df, short_period)
    atr_long = calc_atr(df, long_period)
    return atr_short / atr_long.replace(0, np.nan)


def calc_realized_vol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Annualized realized volatility from close-to-close returns.

    For 5-min bars: multiply by sqrt(288 * 365) to annualize.
    """
    log_returns = np.log(df["close"] / df["close"].shift(1))
    # 288 bars per day at 5min, 365 days/year
    return log_returns.rolling(period).std() * np.sqrt(288 * 365)


def calc_vol_regime(df: pd.DataFrame, period: int = 100) -> pd.Series:
    """Classify volatility regime based on ATR percentile.

    Returns:
        0: Low vol (below 25th percentile)
        1: Normal vol (25th-75th)
        2: High vol (above 75th percentile)
    """
    from src.features.ohlcv import calc_atr

    atr = calc_atr(df, 20)
    pctile = atr.rolling(period).rank(pct=True)

    regime = pd.Series(1, index=df.index)
    regime[pctile < 0.25] = 0
    regime[pctile > 0.75] = 2
    return regime


def add_volatility_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all volatility features to the DataFrame."""
    kc_mult = config["features"]["bb_squeeze_kc_mult"]
    atr_short = config["features"]["atr_compression_short"]
    atr_long = config["features"]["atr_compression_long"]
    bb_period = config["features"]["bollinger_period"]
    bb_std = config["features"]["bollinger_std"]

    sq = calc_bb_squeeze(df, bb_period, bb_std, bb_period, kc_mult)
    df["squeeze"] = sq["squeeze"].values
    df["squeeze_duration"] = sq["squeeze_duration"].values

    df["atr_compression"] = calc_atr_compression(df, atr_short, atr_long)
    df["realized_vol"] = calc_realized_vol(df)
    df["vol_regime"] = calc_vol_regime(df)

    return df
