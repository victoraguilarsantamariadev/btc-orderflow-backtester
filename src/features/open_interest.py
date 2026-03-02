"""Open Interest features: OI change, divergence, OI drop (liquidation proxy)."""

import pandas as pd
import numpy as np


def calc_oi_change(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """Percentage change in open interest over N bars."""
    if "open_interest" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return df["open_interest"].pct_change(period) * 100


def calc_oi_zscore(df: pd.DataFrame, lookback: int = 288) -> pd.Series:
    """Z-score of OI level (288 bars = 24h at 5min)."""
    if "open_interest" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    oi = df["open_interest"]
    mean = oi.rolling(lookback).mean()
    std = oi.rolling(lookback).std()
    return (oi - mean) / std.replace(0, np.nan)


def calc_oi_divergence(df: pd.DataFrame, lookback: int = 12) -> pd.Series:
    """Detect divergence between OI and price.

    OI rising + price dropping = shorts being opened as hedge (institutional exit).
    OI rising + price rising = longs accumulating.
    OI dropping + price moving = liquidations happening.

    Returns:
        +1: Bullish OI signal (OI drops while price drops = shorts closing)
        -1: Bearish OI signal (OI rises while price drops = hedge shorts)
         0: No divergence
    """
    if "open_interest" not in df.columns:
        return pd.Series(0, index=df.index)

    oi_change = df["open_interest"].pct_change(lookback)
    price_change = df["close"].pct_change(lookback)

    result = pd.Series(0, index=df.index)

    # Bearish: OI rises significantly while price drops
    # Someone opening shorts as hedge while dumping spot
    mask_bearish = (oi_change > 0.005) & (price_change < -0.002)
    result[mask_bearish] = -1

    # Bullish: OI drops while price drops then stabilizes
    # Shorts being liquidated / closed
    mask_bullish = (oi_change < -0.005) & (price_change < -0.002)
    result[mask_bullish] = 1

    return result


def calc_oi_drop(df: pd.DataFrame, threshold_pct: float = -5.0,
                  price_move_pct: float = 2.0) -> pd.Series:
    """Detect sharp OI drops as proxy for mass liquidations.

    When OI drops sharply (>threshold_pct) and price moves strongly,
    it means many positions were forcefully closed = liquidations.

    Returns:
        +1: Liquidation event on downside (longs liquidated, potential bounce)
        -1: Liquidation event on upside (shorts liquidated, potential reversal)
         0: No liquidation event
    """
    if "open_interest" not in df.columns:
        return pd.Series(0, index=df.index)

    # Use 6-bar OI change to catch cascading liquidations (30 min window)
    oi_change_pct = df["open_interest"].pct_change(6) * 100
    price_change_pct = df["close"].pct_change(6) * 100

    result = pd.Series(0, index=df.index)

    # Longs liquidated: OI drops + price drops = potential bounce
    mask_long_liq = (oi_change_pct < threshold_pct) & (price_change_pct < -price_move_pct)
    result[mask_long_liq] = 1

    # Shorts liquidated: OI drops + price rises = potential reversal down
    mask_short_liq = (oi_change_pct < threshold_pct) & (price_change_pct > price_move_pct)
    result[mask_short_liq] = -1

    return result


def calc_oi_cluster_distance(df: pd.DataFrame, lookback: int = 288) -> pd.Series:
    """Estimate distance to nearest OI cluster (high OI accumulation zone).

    Zones where OI accumulated the most are likely liquidation zones.
    Returns normalized distance (0 = at cluster, 1 = far from cluster).
    """
    if "open_interest" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Find price levels where OI was highest in the lookback window
    # This is a simplified proxy — real implementation would use order book data
    result = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        # Price level with max OI change (= most position building)
        oi_changes = window["open_interest"].diff().abs()
        if oi_changes.max() > 0:
            max_oi_idx = oi_changes.idxmax()
            cluster_price = window.loc[max_oi_idx, "close"]
            current_price = df["close"].iloc[i]
            atr = (window["high"] - window["low"]).mean()
            if atr > 0:
                result.iloc[i] = abs(current_price - cluster_price) / atr

    return result


def add_open_interest_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all open interest features to the DataFrame."""
    oi_lookback = config["features"]["oi_divergence_lookback"]
    oi_drop_thresh = config["features"]["oi_drop_threshold_pct"]

    df["oi_change_pct"] = calc_oi_change(df)
    df["oi_change_pct_12"] = calc_oi_change(df, period=12)
    df["oi_zscore"] = calc_oi_zscore(df)
    df["oi_divergence"] = calc_oi_divergence(df, oi_lookback)
    df["oi_drop"] = calc_oi_drop(df, threshold_pct=oi_drop_thresh, price_move_pct=0.5)

    return df
