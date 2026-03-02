"""A2: Orderflow Imbalance / Absorption signal.

Logic: Strong delta in one direction but price doesn't move.
Someone is absorbing the aggressive flow = hidden institutional activity.
Combined with CVD divergence for confirmation.
"""

import pandas as pd
from src.signals.base import BaseSignal


class AbsorptionSignal(BaseSignal):

    @property
    def name(self) -> str:
        return "absorption"

    def calculate(self, features: pd.DataFrame) -> pd.Series:
        result = pd.Series(0, index=features.index)

        has_absorption = "absorption" in features.columns and features["absorption"].notna().any()
        has_cvd_div = "cvd_divergence" in features.columns

        if not has_absorption:
            return result

        # Bullish absorption: selling pressure absorbed, price holds
        # absorption > 1.0 means strong bullish absorption
        bull_mask = features["absorption"] > 1.0
        if has_cvd_div:
            # Extra confirmation: CVD divergence also bullish
            bull_confirmed = bull_mask & (features["cvd_divergence"] == 1)
            bull_unconfirmed = bull_mask & (features["cvd_divergence"] != 1)
            result[bull_confirmed] = self.weight
            result[bull_unconfirmed] = self.weight  # Still signal, just less confident
        else:
            result[bull_mask] = self.weight

        # Bearish absorption: buying pressure absorbed, price holds
        bear_mask = features["absorption"] < -1.0
        if has_cvd_div:
            bear_confirmed = bear_mask & (features["cvd_divergence"] == -1)
            bear_unconfirmed = bear_mask & (features["cvd_divergence"] != -1)
            result[bear_confirmed] = -self.weight
            result[bear_unconfirmed] = -self.weight
        else:
            result[bear_mask] = -self.weight

        return result
