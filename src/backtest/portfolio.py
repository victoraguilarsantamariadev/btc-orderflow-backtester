"""Portfolio state: equity tracking, drawdown calculation."""

import pandas as pd
import numpy as np


class Portfolio:
    def __init__(self, config: dict):
        self.initial_equity = config["backtest"]["initial_equity"]
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.equity_history = []
        self.timestamp_history = []

    def reset(self):
        """Reset for a new backtest run."""
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.equity_history = []
        self.timestamp_history = []

    def update(self, pnl: float, timestamp=None):
        """Update equity after a trade closes."""
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        if timestamp is not None:
            self.equity_history.append(self.equity)
            self.timestamp_history.append(timestamp)

    def record_bar(self, timestamp, unrealized_pnl: float = 0.0):
        """Record equity at each bar (including unrealized P&L)."""
        self.equity_history.append(self.equity + unrealized_pnl)
        self.timestamp_history.append(timestamp)

    def current_drawdown(self) -> float:
        """Current drawdown from peak as a fraction (0.05 = 5%)."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def max_drawdown_exceeded(self, max_dd: float) -> bool:
        """Check if max drawdown has been exceeded."""
        return self.current_drawdown() > max_dd

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        return pd.DataFrame({
            "timestamp": self.timestamp_history,
            "equity": self.equity_history,
        })
