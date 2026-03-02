"""C1: Volatility Compression → Expansion signal.

Logic: When volatility is compressed (BB squeeze, low ATR) and OI is building,
the market is coiling for a big move. Enter on the breakout with delta confirmation.
"""

import pandas as pd
from src.signals.base import BaseSignal


class VolCompressionSignal(BaseSignal):

    @property
    def name(self) -> str:
        return "vol_compression"

    def calculate(self, features: pd.DataFrame) -> pd.Series:
        result = pd.Series(0, index=features.index)

        has_squeeze = "squeeze" in features.columns
        has_compression = "atr_compression" in features.columns

        if not has_squeeze and not has_compression:
            return result

        # Detect squeeze release: was in squeeze, now breaking out
        if has_squeeze:
            # Squeeze just ended (previous bar was squeeze, current is not)
            squeeze_release = (features["squeeze"].shift(1) == 1) & (features["squeeze"] == 0)
        else:
            squeeze_release = pd.Series(False, index=features.index)

        # Or ATR compression with sudden expansion
        if has_compression:
            atr_expanding = (
                (features["atr_compression"].shift(1) < 0.7) &
                (features["atr_compression"] > 0.9)
            )
            squeeze_release = squeeze_release | atr_expanding

        # Direction from price action: breakout direction
        # If close > previous close = bullish breakout
        bullish_breakout = squeeze_release & (features["close"] > features["close"].shift(1))
        bearish_breakout = squeeze_release & (features["close"] < features["close"].shift(1))

        # Add delta confirmation if available
        if "delta" in features.columns and features["delta"].notna().any():
            bullish_breakout = bullish_breakout & (features["delta"] > 0)
            bearish_breakout = bearish_breakout & (features["delta"] < 0)

        result[bullish_breakout] = self.weight
        result[bearish_breakout] = -self.weight

        return result
