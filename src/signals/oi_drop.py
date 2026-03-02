"""B3: OI Drop = Implicit Liquidations signal.

Logic: Sharp OI drop + strong price move = mass liquidations already happened.
After a cascade, the forced selling/buying is done and price tends to revert.
"""

import pandas as pd
from src.signals.base import BaseSignal


class OIDropSignal(BaseSignal):

    @property
    def name(self) -> str:
        return "oi_drop"

    def calculate(self, features: pd.DataFrame) -> pd.Series:
        result = pd.Series(0, index=features.index)

        if "oi_drop" not in features.columns:
            return result

        # Longs liquidated (OI drop + price drop) → potential bounce (long)
        result[features["oi_drop"] == 1] = self.weight

        # Shorts liquidated (OI drop + price up) → potential reversal (short)
        result[features["oi_drop"] == -1] = -self.weight

        return result
