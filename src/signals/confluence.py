"""Confluence scoring engine: combines all signals into weighted scores.

Only generates trade signals when the absolute score exceeds the minimum threshold.
Higher confluence = more confidence = optionally larger position size.
"""

import pandas as pd
import numpy as np

from src.signals.base import BaseSignal
from src.signals.liquidation_sweep import LiquidationSweepSignal
from src.signals.absorption import AbsorptionSignal
from src.signals.funding_reversion import FundingReversionSignal
from src.signals.oi_divergence import OIDivergenceSignal
from src.signals.oi_drop import OIDropSignal
from src.signals.vol_compression import VolCompressionSignal


def create_all_signals(config: dict) -> list[BaseSignal]:
    """Instantiate all signals with their configured weights."""
    weights = config["signals"]["weights"]

    signals = [
        LiquidationSweepSignal({"weight": weights["liquidation_sweep"]}),
        AbsorptionSignal({"weight": weights["absorption"]}),
        FundingReversionSignal({"weight": weights["funding_extreme"]}),
        OIDivergenceSignal({"weight": weights["oi_divergence"]}),
        OIDropSignal({"weight": weights.get("oi_drop", 1)}),
        VolCompressionSignal({"weight": weights["vol_compression"]}),
    ]

    return signals


class ConfluenceEngine:
    """Combines all signals into a single confluent trading signal."""

    def __init__(self, config: dict):
        self.config = config
        self.min_score = config["confluence"]["min_score"]
        self.enhanced_threshold = config["confluence"]["enhanced_threshold"]
        self.signals = create_all_signals(config)

    def calculate_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate confluence scores from features.

        Returns DataFrame with columns:
            - raw_score: Sum of all weighted signals (signed: +long, -short)
            - abs_score: Absolute value of raw_score
            - direction: +1 (long), -1 (short), 0 (no trade)
            - trade_signal: 1 where abs_score >= min_score, else 0
            - enhanced_size: 1 where abs_score >= enhanced_threshold, else 0
            - Per-signal columns (signal_<name>): individual signal values
        """
        signal_df = pd.DataFrame(index=features.index)

        # Calculate each individual signal
        for signal in self.signals:
            col_name = f"signal_{signal.name}"
            signal_df[col_name] = signal.calculate(features)

        # Sum all signals (they are already directionally signed)
        signal_cols = [c for c in signal_df.columns if c.startswith("signal_")]
        signal_df["raw_score"] = signal_df[signal_cols].sum(axis=1)
        signal_df["abs_score"] = signal_df["raw_score"].abs()

        # Direction
        signal_df["direction"] = 0
        signal_df.loc[signal_df["raw_score"] > 0, "direction"] = 1
        signal_df.loc[signal_df["raw_score"] < 0, "direction"] = -1

        # Trade signal: only when confluence is high enough
        signal_df["trade_signal"] = (signal_df["abs_score"] >= self.min_score).astype(int)

        # Enhanced sizing for very high confluence
        signal_df["enhanced_size"] = (signal_df["abs_score"] >= self.enhanced_threshold).astype(int)

        # Apply time filter: no trades during off-hours or weekends (optional)
        if "session" in features.columns:
            off_hours = features["session"] == "off"
            signal_df.loc[off_hours, "trade_signal"] = 0

        return signal_df

    def get_signal_summary(self, signal_df: pd.DataFrame) -> dict:
        """Get summary statistics of signal distribution."""
        total_bars = len(signal_df)
        trade_bars = signal_df["trade_signal"].sum()
        long_bars = ((signal_df["trade_signal"] == 1) & (signal_df["direction"] == 1)).sum()
        short_bars = ((signal_df["trade_signal"] == 1) & (signal_df["direction"] == -1)).sum()
        enhanced_bars = signal_df["enhanced_size"].sum()

        return {
            "total_bars": total_bars,
            "trade_signals": int(trade_bars),
            "long_signals": int(long_bars),
            "short_signals": int(short_bars),
            "enhanced_signals": int(enhanced_bars),
            "signal_rate": trade_bars / total_bars * 100 if total_bars > 0 else 0,
            "avg_score": signal_df.loc[signal_df["trade_signal"] == 1, "abs_score"].mean(),
        }
