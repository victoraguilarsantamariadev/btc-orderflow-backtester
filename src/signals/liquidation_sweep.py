"""A1: Liquidation Sweep Reversal signal.

Logic: When price sweeps a zone with high OI (liquidation cluster),
OI drops sharply (liquidations occurred), and delta shows absorption,
enter counter to the sweep direction.
"""

import pandas as pd
from src.signals.base import BaseSignal


class LiquidationSweepSignal(BaseSignal):

    @property
    def name(self) -> str:
        return "liquidation_sweep"

    def calculate(self, features: pd.DataFrame) -> pd.Series:
        result = pd.Series(0, index=features.index)

        has_oi = "oi_drop" in features.columns and features["oi_drop"].notna().any()
        has_delta = "absorption" in features.columns and features["absorption"].notna().any()

        if not has_oi:
            return result

        # Long signal: longs were liquidated (OI drop + price drop),
        # AND absorption detected (buying pressure absorbing the dump)
        long_mask = (features["oi_drop"] == 1)  # Longs liquidated = potential bounce
        if has_delta:
            long_mask = long_mask & (features["absorption"] > 0.5)

        # Short signal: shorts were liquidated (OI drop + price spike),
        # AND absorption detected (selling pressure absorbing the pump)
        short_mask = (features["oi_drop"] == -1)  # Shorts liquidated = potential reversal
        if has_delta:
            short_mask = short_mask & (features["absorption"] < -0.5)

        result[long_mask] = self.weight
        result[short_mask] = -self.weight

        return result
