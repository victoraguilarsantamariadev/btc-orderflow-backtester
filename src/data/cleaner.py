"""Data cleaning, validation, gap filling, and multi-source alignment.

Takes raw parquet files from different exchanges, validates them,
fills small gaps, and merges everything into a single aligned 5-min DataFrame.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config_loader import load_config
from src.data.storage import load_parquet, save_parquet


def validate_no_gaps(df: pd.DataFrame, freq: str = "5min") -> list[tuple]:
    """Check for gaps in timestamp column.

    Returns list of (gap_start, gap_end, gap_minutes) tuples.
    """
    if df.empty or "timestamp" not in df.columns:
        return []

    expected = pd.date_range(
        df["timestamp"].min(), df["timestamp"].max(), freq=freq
    )
    actual = set(df["timestamp"])
    missing = sorted(set(expected) - actual)

    if not missing:
        return []

    # Group consecutive missing timestamps into gap ranges
    gaps = []
    gap_start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if (ts - prev) > pd.Timedelta(freq):
            gap_minutes = (prev - gap_start).total_seconds() / 60 + 5
            gaps.append((gap_start, prev, gap_minutes))
            gap_start = ts
        prev = ts
    gap_minutes = (prev - gap_start).total_seconds() / 60 + 5
    gaps.append((gap_start, prev, gap_minutes))

    return gaps


def fill_gaps(df: pd.DataFrame, freq: str = "5min", max_gap_bars: int = 3) -> pd.DataFrame:
    """Forward-fill small gaps (up to max_gap_bars consecutive missing bars).

    Larger gaps are left as NaN and flagged.
    """
    if df.empty:
        return df

    # Create complete index
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=freq)
    df = df.set_index("timestamp").reindex(full_idx)
    df.index.name = "timestamp"

    # Forward-fill only small gaps
    df = df.ffill(limit=max_gap_bars)

    # Flag remaining NaN rows as gap
    df["is_gap"] = df.isnull().any(axis=1)

    return df.reset_index()


def align_to_5min_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Align timestamps to exact 5-minute boundaries."""
    if "timestamp" not in df.columns:
        return df

    df["timestamp"] = df["timestamp"].dt.floor("5min")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def merge_all_sources(config: dict = None) -> pd.DataFrame:
    """Load all raw data sources and merge into a single aligned DataFrame.

    The base is the OHLCV klines (5min), onto which we left-join:
    - Funding rate (8h → forward-filled to 5min)
    - Open interest (5min from Bybit)
    - Long/short ratio (5min from Bybit)
    - Delta/CVD (5min from aggTrades, if available)
    - Taker volume (5min, only 30 days)
    - Daily aggregated OI from Coinalyze (context)
    - Daily liquidations from Coinalyze (context)

    Returns:
        Merged DataFrame aligned to 5-minute grid
    """
    if config is None:
        config = load_config()

    symbol = config["symbol"].lower()
    raw_dir = Path(config["data"]["raw_dir"])

    # ── 1. Load OHLCV (base) ────────────────────────────────────────
    print("Loading OHLCV klines...")
    klines_path = raw_dir / "klines" / f"{symbol}_5m.parquet"
    df = load_parquet(str(klines_path))
    df = align_to_5min_grid(df)
    print(f"  {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")

    # ── 2. Merge Funding Rate ────────────────────────────────────────
    funding_path = raw_dir / "funding" / f"{symbol}_funding.parquet"
    if funding_path.exists():
        print("Merging funding rate...")
        funding = load_parquet(str(funding_path))
        funding = align_to_5min_grid(funding)
        df = df.merge(funding, on="timestamp", how="left")
        # Forward-fill funding rate (published every 8h)
        df["funding_rate"] = df["funding_rate"].ffill()
        print(f"  Funding: {funding['funding_rate'].notna().sum()} raw records")

    # ── 3. Merge Open Interest (Bybit) ───────────────────────────────
    oi_path = raw_dir / "oi" / f"{symbol}_oi_5m.parquet"
    if oi_path.exists():
        print("Merging open interest (Bybit)...")
        oi = load_parquet(str(oi_path))
        oi = align_to_5min_grid(oi)
        df = df.merge(oi, on="timestamp", how="left")
        df["open_interest"] = df["open_interest"].ffill(limit=3)
        print(f"  OI: {oi['open_interest'].notna().sum()} records")

    # ── 4. Merge Long/Short Ratio (Bybit) ────────────────────────────
    ls_path = raw_dir / "ls_ratio" / f"{symbol}_ls_5m.parquet"
    if ls_path.exists():
        print("Merging long/short ratio (Bybit)...")
        ls = load_parquet(str(ls_path))
        ls = align_to_5min_grid(ls)
        df = df.merge(ls, on="timestamp", how="left")
        df["ls_ratio"] = df["ls_ratio"].ffill(limit=3)
        print(f"  LS ratio: {ls['ls_ratio'].notna().sum()} records")

    # ── 5. Merge Delta/CVD (aggTrades) ───────────────────────────────
    delta_path = raw_dir / "aggtrades" / f"{symbol}_delta_5m.parquet"
    if delta_path.exists():
        print("Merging delta/CVD (aggTrades)...")
        delta = load_parquet(str(delta_path))
        delta = align_to_5min_grid(delta)
        df = df.merge(delta, on="timestamp", how="left")
        # Calculate CVD (cumulative volume delta)
        df["cvd"] = df["delta"].cumsum()
        print(f"  Delta: {delta['delta'].notna().sum()} records")

    # ── 6. Merge Taker Volume ────────────────────────────────────────
    taker_path = raw_dir / "taker_volume" / f"{symbol}_taker_5m.parquet"
    if taker_path.exists():
        print("Merging taker volume...")
        taker = load_parquet(str(taker_path))
        taker = align_to_5min_grid(taker)
        df = df.merge(taker, on="timestamp", how="left")
        print(f"  Taker: {taker['taker_ratio'].notna().sum()} records")

    # ── 7. Merge Coinalyze daily OI (context) ────────────────────────
    oi_daily_path = raw_dir / "coinalyze" / f"{symbol}_oi_daily.parquet"
    if oi_daily_path.exists():
        print("Merging Coinalyze daily OI...")
        oi_daily = load_parquet(str(oi_daily_path))
        oi_daily["date"] = oi_daily["timestamp"].dt.date
        oi_daily = oi_daily.rename(columns={"oi_aggregated": "oi_agg_daily"})
        df["date"] = df["timestamp"].dt.date
        df = df.merge(oi_daily[["date", "oi_agg_daily"]], on="date", how="left")
        df["oi_agg_daily"] = df["oi_agg_daily"].ffill()
        df = df.drop(columns=["date"])

    # ── 8. Merge Coinalyze daily liquidations (context) ──────────────
    liq_path = raw_dir / "coinalyze" / f"{symbol}_liq_daily.parquet"
    if liq_path.exists():
        print("Merging Coinalyze daily liquidations...")
        liq = load_parquet(str(liq_path))
        liq["date"] = liq["timestamp"].dt.date
        df["date"] = df["timestamp"].dt.date
        df = df.merge(liq[["date", "liq_long", "liq_short"]], on="date", how="left")
        df = df.drop(columns=["date"])

    # ── Final cleanup ────────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Report coverage
    print(f"\nMerged DataFrame: {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\nColumn coverage:")
    for col in df.columns:
        if col == "timestamp":
            continue
        pct = df[col].notna().sum() / len(df) * 100
        print(f"  {col}: {pct:.1f}%")

    return df


def run_cleaning_pipeline(config: dict = None) -> pd.DataFrame:
    """Full pipeline: merge all sources, fill gaps, save to processed."""
    if config is None:
        config = load_config()

    df = merge_all_sources(config)

    # Fill small gaps in OHLCV (up to 3 bars = 15 minutes)
    gaps = validate_no_gaps(df, freq="5min")
    if gaps:
        print(f"\nFound {len(gaps)} gaps:")
        for start, end, minutes in gaps[:10]:
            print(f"  {start} to {end} ({minutes:.0f} min)")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more")

    # Save processed
    output_path = Path(config["data"]["processed_dir"]) / "features.parquet"
    save_parquet(df, str(output_path))
    print(f"\nSaved merged data to {output_path}")

    return df
