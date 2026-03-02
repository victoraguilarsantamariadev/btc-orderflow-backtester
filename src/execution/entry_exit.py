"""Entry confirmation, stop loss, take profit, and trailing stop logic."""

import numpy as np


class EntryExitManager:
    def __init__(self, config: dict):
        self.sl_atr_mult = config["entry_exit"]["sl_atr_mult"]       # 1.5
        self.tp1_rr = config["entry_exit"]["tp1_rr"]                 # 1.5
        self.tp2_rr = config["entry_exit"]["tp2_rr"]                 # 3.0
        self.tp1_close_pct = config["entry_exit"]["tp1_close_pct"]   # 0.5
        self.time_stop = config["entry_exit"]["time_stop"]           # 30 bars
        self.confirm_window = config["entry_exit"]["confirm_window"] # 3 bars
        self.trail_atr = config["entry_exit"]["trail_atr"]           # 1.0
        self.commission_pct = config["backtest"]["commission_pct"]   # 0.0004
        self.slippage_pct = config["backtest"]["slippage_pct"]       # 0.0001

    def check_confirmation(self, features, signal_bar_idx: int,
                           direction: int) -> int | None:
        """Check if a confirmation candle appears within the confirm window.

        Args:
            features: DataFrame with OHLCV
            signal_bar_idx: Index of the signal bar
            direction: +1 (long) or -1 (short)

        Returns:
            Index of the confirmation bar, or None if not confirmed.
        """
        for i in range(1, self.confirm_window + 1):
            idx = signal_bar_idx + i
            if idx >= len(features):
                return None

            bar = features.iloc[idx]
            # Confirmation: candle closes in the direction of the trade
            if direction == 1 and bar["close"] > bar["open"]:
                return idx
            elif direction == -1 and bar["close"] < bar["open"]:
                return idx

        return None

    def calculate_levels(self, entry_price: float, atr: float,
                          direction: int) -> dict:
        """Calculate SL, TP1, TP2 levels.

        Args:
            entry_price: Price at entry
            atr: ATR value at signal time
            direction: +1 (long) or -1 (short)

        Returns:
            dict with stop_loss, tp1, tp2, stop_distance
        """
        stop_distance = atr * self.sl_atr_mult

        if direction == 1:  # Long
            stop_loss = entry_price - stop_distance
            tp1 = entry_price + stop_distance * self.tp1_rr
            tp2 = entry_price + stop_distance * self.tp2_rr
        else:  # Short
            stop_loss = entry_price + stop_distance
            tp1 = entry_price - stop_distance * self.tp1_rr
            tp2 = entry_price - stop_distance * self.tp2_rr

        return {
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "stop_distance": stop_distance,
        }

    def apply_slippage(self, price: float, direction: int, is_entry: bool) -> float:
        """Apply slippage to a price.

        Entry: worse price (buy higher, sell lower).
        Exit: worse price (sell lower, buy higher).
        """
        slip = price * self.slippage_pct
        if is_entry:
            return price + slip * direction  # Buy higher, sell lower
        else:
            return price - slip * direction  # Sell lower, buy higher

    def calculate_commission(self, price: float, size: float) -> float:
        """Calculate commission for a trade."""
        return abs(price * size * self.commission_pct)

    def update_trailing_stop(self, current_price: float, atr: float,
                              direction: int, current_stop: float,
                              tp1_hit: bool) -> float:
        """Update trailing stop after TP1 is hit.

        After TP1: move stop to breakeven, then trail at 1x ATR.
        Never move stop against the trade direction.
        """
        if not tp1_hit:
            return current_stop

        trail_distance = atr * self.trail_atr

        if direction == 1:  # Long
            new_stop = current_price - trail_distance
            return max(new_stop, current_stop)  # Never lower the stop
        else:  # Short
            new_stop = current_price + trail_distance
            return min(new_stop, current_stop)  # Never raise the stop
