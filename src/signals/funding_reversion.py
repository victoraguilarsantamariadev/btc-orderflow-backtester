"""B1: Funding Rate Mean Reversion signal.

Logic: Extreme funding rates mean the market is one-sided.
Extreme positive = everyone long with leverage = contrarian short.
Extreme negative = everyone short with leverage = contrarian long.
"""

import pandas as pd
from src.signals.base import BaseSignal


class FundingReversionSignal(BaseSignal):

    @property
    def name(self) -> str:
        return "funding_extreme"

    def calculate(self, features: pd.DataFrame) -> pd.Series:
        result = pd.Series(0, index=features.index)

        if "funding_extreme" not in features.columns:
            return result

        # Extreme positive funding = contrarian SHORT
        result[features["funding_extreme"] == 1] = -self.weight

        # Extreme negative funding = contrarian LONG
        result[features["funding_extreme"] == -1] = self.weight

        return result
