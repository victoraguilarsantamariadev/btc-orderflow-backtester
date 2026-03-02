"""Trade management: daily limits, consecutive loss tracking."""


class TradeManager:
    def __init__(self, config: dict):
        self.max_daily_trades = config["limits"]["max_daily_trades"]       # 3
        self.max_consecutive_losses = config["limits"]["max_consecutive_losses"]  # 2
        self.max_daily_loss_pct = config.get("prop_firm", {}).get("max_daily_loss_pct", 0.05)

        # State tracking
        self.daily_trades = {}       # date -> count
        self.consecutive_losses = 0
        self.daily_pnl = {}          # date -> cumulative pnl

    def reset(self):
        """Reset state for a new backtest run."""
        self.daily_trades = {}
        self.consecutive_losses = 0
        self.daily_pnl = {}

    def can_trade(self, date: str, equity: float) -> bool:
        """Check if we're allowed to take a new trade.

        Returns False if:
        - Max daily trades reached
        - Max consecutive losses reached
        - Daily loss limit approached
        """
        # Check daily trade count
        if self.daily_trades.get(date, 0) >= self.max_daily_trades:
            return False

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False

        # Check daily loss limit
        daily_pnl = self.daily_pnl.get(date, 0)
        daily_loss_limit = equity * self.max_daily_loss_pct * 0.8  # 80% of limit as buffer
        if daily_pnl < -daily_loss_limit:
            return False

        return True

    def record_trade(self, date: str, pnl: float):
        """Record a completed trade."""
        self.daily_trades[date] = self.daily_trades.get(date, 0) + 1
        self.daily_pnl[date] = self.daily_pnl.get(date, 0) + pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def new_day(self, date: str):
        """Reset daily counters for new trading day.

        consecutive_losses resets each day: the rule is
        'stop trading for the rest of the day after N losses'.
        """
        if date not in self.daily_trades:
            self.daily_trades[date] = 0
            self.daily_pnl[date] = 0.0
            self.consecutive_losses = 0
