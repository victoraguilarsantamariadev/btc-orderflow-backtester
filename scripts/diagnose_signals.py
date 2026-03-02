"""Diagnostic script: check individual signal firing rates and feature distributions."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.utils.config_loader import load_config
from src.data.cleaner import merge_all_sources
from src.features.builder import build_features
from src.signals.confluence import ConfluenceEngine

config = load_config()

# --- Step 1: Load and merge data ---
print("=" * 60)
print("STEP 1: Loading and merging data")
print("=" * 60)
merged = merge_all_sources(config)
print(f"Merged rows: {len(merged)}")
print(f"Columns: {list(merged.columns)}")
print()

# Check raw data coverage
for col in ["open_interest", "funding_rate", "delta", "ls_ratio", "buy_volume", "sell_volume"]:
    if col in merged.columns:
        valid = merged[col].notna().sum()
        pct = valid / len(merged) * 100
        print(f"  {col}: {valid}/{len(merged)} ({pct:.1f}%) non-null")
    else:
        print(f"  {col}: MISSING from merged data")

# --- Step 2: Build features ---
print()
print("=" * 60)
print("STEP 2: Building features")
print("=" * 60)
features = build_features(merged, config)
print(f"Feature rows: {len(features)}")
print()

# --- Step 3: Check key feature distributions ---
print("=" * 60)
print("STEP 3: Feature distributions (BEFORE signal calculation)")
print("=" * 60)

key_features = [
    "absorption", "cvd_divergence", "delta", "delta_zscore",
    "funding_zscore", "funding_extreme", "funding_rate",
    "oi_change_pct", "oi_change_pct_12", "oi_divergence", "oi_drop",
    "oi_zscore", "open_interest",
    "squeeze", "squeeze_duration", "atr_compression",
    "ls_ratio_zscore", "ls_extreme",
    "atr", "bb_bandwidth",
]

for col in key_features:
    if col not in features.columns:
        print(f"  {col}: NOT IN FEATURES")
        continue
    s = features[col]
    valid = s.notna().sum()
    nonzero = (s.fillna(0) != 0).sum()
    print(f"  {col}: valid={valid}, nonzero={nonzero}, "
          f"min={s.min():.6f}, max={s.max():.6f}, "
          f"mean={s.mean():.6f}, std={s.std():.6f}")

# --- Step 4: Check individual signal firing ---
print()
print("=" * 60)
print("STEP 4: Individual signal firing rates")
print("=" * 60)

engine = ConfluenceEngine(config)
for signal in engine.signals:
    vals = signal.calculate(features)
    total = len(vals)
    long_fires = (vals > 0).sum()
    short_fires = (vals < 0).sum()
    any_fires = (vals != 0).sum()
    print(f"  {signal.name} (weight={signal.weight}):")
    print(f"    Long:  {long_fires} ({long_fires/total*100:.3f}%)")
    print(f"    Short: {short_fires} ({short_fires/total*100:.3f}%)")
    print(f"    Total: {any_fires} ({any_fires/total*100:.3f}%)")

# --- Step 5: Confluence score distribution ---
print()
print("=" * 60)
print("STEP 5: Confluence score distribution")
print("=" * 60)

signal_df = engine.calculate_scores(features)
print(f"Raw score stats:")
print(f"  min={signal_df['raw_score'].min()}, max={signal_df['raw_score'].max()}")
print(f"  mean={signal_df['raw_score'].mean():.4f}, std={signal_df['raw_score'].std():.4f}")
print()

# Distribution of abs_score
for threshold in [1, 2, 3, 4, 5, 6, 7]:
    count = (signal_df["abs_score"] >= threshold).sum()
    pct = count / len(signal_df) * 100
    print(f"  abs_score >= {threshold}: {count} bars ({pct:.3f}%)")

# Show top scores
top = signal_df.nlargest(20, "abs_score")
print()
print("Top 20 highest abs_score bars:")
for idx, row in top.iterrows():
    bar_info = features.iloc[idx] if idx < len(features) else None
    ts = features.loc[idx, "timestamp"] if bar_info is not None and "timestamp" in features.columns else idx
    signal_cols = [c for c in signal_df.columns if c.startswith("signal_")]
    active = {c: row[c] for c in signal_cols if row[c] != 0}
    print(f"  Bar {idx} (ts={ts}): raw={row['raw_score']:.1f}, abs={row['abs_score']:.1f}, "
          f"signals={active}")
