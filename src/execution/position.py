"""Position sizing based on risk parameters."""


class PositionSizer:
    def __init__(self, config: dict):
        self.base_risk_pct = config["risk"]["per_trade"]           # 0.005
        self.enhanced_multiplier = config["risk"]["enhanced_multiplier"]  # 1.5
        self.max_drawdown = config["risk"]["max_drawdown"]         # 0.10

    def calculate_size(self, equity: float, stop_distance: float,
                       enhanced: bool = False) -> float:
        """Calculate position size in BTC.

        Args:
            equity: Current account equity in USD
            stop_distance: Distance to stop loss in USD per BTC
            enhanced: If True, use enhanced multiplier (high confluence)

        Returns:
            Position size in BTC
        """
        if stop_distance <= 0:
            return 0.0

        risk_pct = self.base_risk_pct
        if enhanced:
            risk_pct *= self.enhanced_multiplier

        risk_amount = equity * risk_pct  # USD to risk
        size = risk_amount / stop_distance  # BTC

        return size

    def calculate_risk_amount(self, equity: float, enhanced: bool = False) -> float:
        """Calculate max USD risk for this trade."""
        risk_pct = self.base_risk_pct
        if enhanced:
            risk_pct *= self.enhanced_multiplier
        return equity * risk_pct
