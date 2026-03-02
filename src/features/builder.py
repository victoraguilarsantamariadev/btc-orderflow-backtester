"""Feature builder: orchestrates all feature modules and applies anti-bias shift.

CRITICAL: This module applies .shift(1) to all derived features at the end.
This ensures that the feature at row T represents information available BEFORE
bar T opens (i.e., computed from data up to and including bar T-1).
"""

import pandas as pd

from src.utils.config_loader import load_config
from src.features.ohlcv import add_ohlcv_features
from src.features.orderflow import add_orderflow_features
from src.features.funding import add_funding_features
from src.features.open_interest import add_open_interest_features
from src.features.volatility import add_volatility_features
from src.features.sentiment import add_sentiment_features
from src.features.time_filters import add_session_labels


# Columns that should NOT be shifted (they are raw data, not derived features)
RAW_COLUMNS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
}

# Columns that are categorical/labels (not shifted)
LABEL_COLUMNS = {
    "session", "day_of_week", "is_weekend", "us_open", "london_open", "asia_close",
}


def build_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Build all features from merged raw data.

    Steps:
        1. Add OHLCV features (ATR, Bollinger, etc.)
        2. Add orderflow features (delta, CVD, absorption)
        3. Add funding features (z-scores, extremes)
        4. Add open interest features (OI change, divergence, drop)
        5. Add volatility features (squeeze, compression)
        6. Add sentiment features (LS ratio, taker ratio)
        7. Add time features (session labels)
        8. Apply .shift(1) to ALL derived features (anti look-ahead bias)

    Args:
        df: Merged DataFrame from cleaner.merge_all_sources()
        config: Config dict. If None, loads from file.

    Returns:
        DataFrame with all features, shifted to prevent look-ahead bias.
    """
    if config is None:
        config = load_config()

    print("Building features...")
    df = df.copy()

    # Step 1: OHLCV
    print("  OHLCV features...")
    df = add_ohlcv_features(df, config)

    # Step 2: Orderflow
    print("  Orderflow features...")
    df = add_orderflow_features(df, config)

    # Step 3: Funding
    print("  Funding features...")
    df = add_funding_features(df, config)

    # Step 4: Open Interest
    print("  Open Interest features...")
    df = add_open_interest_features(df, config)

    # Step 5: Volatility
    print("  Volatility features...")
    df = add_volatility_features(df, config)

    # Step 6: Sentiment
    print("  Sentiment features...")
    df = add_sentiment_features(df, config)

    # Step 7: Time features
    print("  Time features...")
    df = add_session_labels(df)

    # Step 8: CRITICAL — shift all derived features by 1 bar
    # This prevents look-ahead bias: feature at row T is computed from data up to T-1
    feature_cols = [
        col for col in df.columns
        if col not in RAW_COLUMNS and col not in LABEL_COLUMNS
    ]

    print(f"  Shifting {len(feature_cols)} feature columns by 1 bar (anti look-ahead)...")
    for col in feature_cols:
        df[col] = df[col].shift(1)

    # Drop the first row (all features are NaN due to shift)
    df = df.iloc[1:].reset_index(drop=True)

    print(f"Done: {len(df)} rows, {len(df.columns)} columns")
    print(f"Feature columns: {len(feature_cols)}")

    return df
