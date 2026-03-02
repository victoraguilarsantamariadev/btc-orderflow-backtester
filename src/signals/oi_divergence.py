"""B2: OI Divergence + Spot Selling signal.

Logic: When OI rises (shorts opened as hedge) while price drops,
it suggests institutional players are exiting spot positions while
hedging with shorts. Follow the institutional flow.
"""

import pandas as pd
from src.signals.base import BaseSignal


class OIDivergenceSignal(BaseSignal):

    @property
    def name(self) -> str:
        return "oi_divergence"

    def calculate(self, features: pd.DataFrame) -> pd.Series:
        result = pd.Series(0, index=features.index)

        if "oi_divergence" not in features.columns:
            return result

        # Bearish: OI rising + price dropping = institutional hedging/exit
        result[features["oi_divergence"] == -1] = -self.weight

        # Bullish: OI dropping + price dropping = shorts closing (squeeze potential)
        result[features["oi_divergence"] == 1] = self.weight

        return result
