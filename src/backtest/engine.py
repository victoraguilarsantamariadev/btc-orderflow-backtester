"""Core backtest engine: hybrid vectorized + bar-by-bar loop.

Features and signals are computed vectorized (fast).
Position management uses a bar-by-bar loop (necessary for state: partial exits,
trailing stops, time stops).

ANTI LOOK-AHEAD RULES:
1. Signal at bar T → entry checked from bar T+1 (confirmation)
2. Entry at bar T+k → use that bar's OPEN price
3. SL/TP checked against bar's HIGH and LOW
4. Check SL BEFORE TP (conservative: assume worst case)
5. ATR from features is already shifted by 1 bar in builder.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from src.backtest.portfolio import Portfolio
from src.execution.entry_exit import EntryExitManager
from src.execution.position import PositionSizer
from src.execution.trade_manager import TradeManager


@dataclass
class Trade:
    """Record of a completed trade."""
    entry_bar: int
    exit_bar: int
    entry_time: object
    exit_time: object
    direction: int           # +1 long, -1 short
    entry_price: float
    exit_price: float
    size: float              # BTC
    pnl: float               # USD after commissions
    pnl_pct: float           # As % of equity at entry
    stop_loss: float
    tp1: float
    tp2: float
    exit_reason: str         # 'sl', 'tp1', 'tp2', 'time_stop', 'trailing', 'dd_limit'
    confluence_score: float
    bars_held: int


@dataclass
class OpenPosition:
    """Active position being managed."""
    entry_bar: int
    entry_time: object
    direction: int
    entry_price: float
    size: float              # Current remaining size
    initial_size: float      # Original size
    stop_loss: float
    tp1: float
    tp2: float
    tp1_hit: bool = False
    trailing_stop: float = 0.0
    confluence_score: float = 0.0
    commissions: float = 0.0
    is_closed: bool = False
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_bar: int = 0


class BacktestEngine:
    """Main backtest engine supporting V1-V4 modes.

    Versions:
        v1: Simple SL + fixed TP (single exit)
        v2: Partial exits (50% at TP1, 50% at TP2)
        v3: Trailing stop after TP1
        v4: Confluence-modulated position sizing
    """

    def __init__(self, config: dict):
        self.config = config
        self.portfolio = Portfolio(config)
        self.entry_exit = EntryExitManager(config)
        self.sizer = PositionSizer(config)
        self.trade_mgr = TradeManager(config)
        self.max_dd = config["risk"]["max_drawdown"]

    def run(self, features: pd.DataFrame, signals: pd.DataFrame,
            version: str = "v1") -> dict:
        """Run backtest over the full feature + signal DataFrame.

        Args:
            features: DataFrame with OHLCV + all features
            signals: DataFrame from ConfluenceEngine.calculate_scores()
            version: 'v1', 'v2', 'v3', or 'v4'

        Returns:
            dict with 'trades', 'equity_curve', 'metrics'
        """
        self.portfolio.reset()
        self.trade_mgr.reset()

        trades: list[Trade] = []
        open_positions: list[OpenPosition] = []
        pending_signals: list[dict] = []  # Signals waiting for confirmation
        current_date = ""

        for i in range(len(features)):
            bar = features.iloc[i]
            sig = signals.iloc[i]
            timestamp = bar["timestamp"]
            date_str = str(timestamp.date()) if hasattr(timestamp, 'date') else str(timestamp)[:10]

            # New day: reset daily counters
            if date_str != current_date:
                current_date = date_str
                self.trade_mgr.new_day(date_str)

            # ── 1. Update open positions (check SL, TP, trailing, time stop) ──
            for pos in open_positions:
                if pos.is_closed:
                    continue
                self._update_position(pos, bar, i, version)

            # ── 2. Close completed positions, record trades ──
            newly_closed = [p for p in open_positions if p.is_closed]
            for pos in newly_closed:
                pnl = self._calc_pnl(pos)
                self.portfolio.update(pnl, timestamp)
                self.trade_mgr.record_trade(date_str, pnl)

                trades.append(Trade(
                    entry_bar=pos.entry_bar,
                    exit_bar=pos.exit_bar,
                    entry_time=pos.entry_time,
                    exit_time=timestamp,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=pos.exit_price,
                    size=pos.initial_size,
                    pnl=pnl,
                    pnl_pct=pnl / self.portfolio.equity * 100 if self.portfolio.equity > 0 else 0,
                    stop_loss=pos.stop_loss,
                    tp1=pos.tp1,
                    tp2=pos.tp2,
                    exit_reason=pos.exit_reason,
                    confluence_score=pos.confluence_score,
                    bars_held=pos.exit_bar - pos.entry_bar,
                ))

            open_positions = [p for p in open_positions if not p.is_closed]

            # ── 3. Check pending signals for confirmation ──
            confirmed = []
            still_pending = []
            for pending in pending_signals:
                confirm_idx = pending["signal_bar"] + pending["bars_waited"] + 1
                if confirm_idx != i:
                    still_pending.append(pending)
                    continue

                # Check confirmation candle
                if pending["direction"] == 1 and bar["close"] > bar["open"]:
                    confirmed.append(pending)
                elif pending["direction"] == -1 and bar["close"] < bar["open"]:
                    confirmed.append(pending)
                elif pending["bars_waited"] < self.entry_exit.confirm_window - 1:
                    pending["bars_waited"] += 1
                    still_pending.append(pending)
                # else: expired, discard

            pending_signals = still_pending

            # ── 4. Open confirmed positions ──
            for conf in confirmed:
                if open_positions:
                    continue  # Only one position at a time
                if not self.trade_mgr.can_trade(date_str, self.portfolio.equity):
                    continue
                if self.portfolio.max_drawdown_exceeded(self.max_dd):
                    continue

                entry_price = self.entry_exit.apply_slippage(
                    bar["open"], conf["direction"], is_entry=True
                )
                atr = conf["atr"]
                levels = self.entry_exit.calculate_levels(entry_price, atr, conf["direction"])

                enhanced = conf.get("enhanced", False)
                if version == "v4":
                    size = self.sizer.calculate_size(
                        self.portfolio.equity, levels["stop_distance"], enhanced
                    )
                else:
                    size = self.sizer.calculate_size(
                        self.portfolio.equity, levels["stop_distance"], False
                    )

                commission = self.entry_exit.calculate_commission(entry_price, size)

                pos = OpenPosition(
                    entry_bar=i,
                    entry_time=timestamp,
                    direction=conf["direction"],
                    entry_price=entry_price,
                    size=size,
                    initial_size=size,
                    stop_loss=levels["stop_loss"],
                    tp1=levels["tp1"],
                    tp2=levels["tp2"],
                    confluence_score=conf["score"],
                    commissions=commission,
                )
                open_positions.append(pos)

            # ── 5. Check for new signals ──
            if sig["trade_signal"] == 1 and not open_positions and not pending_signals:
                if self.trade_mgr.can_trade(date_str, self.portfolio.equity):
                    atr_val = bar.get("atr", np.nan)
                    if not np.isnan(atr_val) and atr_val > 0:
                        pending_signals.append({
                            "signal_bar": i,
                            "direction": int(sig["direction"]),
                            "score": float(sig["abs_score"]),
                            "enhanced": bool(sig["enhanced_size"]),
                            "atr": float(atr_val),
                            "bars_waited": 0,
                        })

            # ── 6. Record equity ──
            unrealized = sum(
                self._unrealized_pnl(p, bar["close"]) for p in open_positions
            )
            self.portfolio.record_bar(timestamp, unrealized)

        # Close any remaining positions at market
        for pos in open_positions:
            if not pos.is_closed:
                pos.exit_price = features.iloc[-1]["close"]
                pos.exit_reason = "end_of_data"
                pos.exit_bar = len(features) - 1
                pos.is_closed = True
                pnl = self._calc_pnl(pos)
                self.portfolio.update(pnl, features.iloc[-1]["timestamp"])
                trades.append(Trade(
                    entry_bar=pos.entry_bar,
                    exit_bar=pos.exit_bar,
                    entry_time=pos.entry_time,
                    exit_time=features.iloc[-1]["timestamp"],
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=pos.exit_price,
                    size=pos.initial_size,
                    pnl=pnl,
                    pnl_pct=pnl / self.portfolio.equity * 100 if self.portfolio.equity > 0 else 0,
                    stop_loss=pos.stop_loss,
                    tp1=pos.tp1,
                    tp2=pos.tp2,
                    exit_reason="end_of_data",
                    confluence_score=pos.confluence_score,
                    bars_held=pos.exit_bar - pos.entry_bar,
                ))

        equity_curve = self.portfolio.get_equity_curve()

        return {
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def _update_position(self, pos: OpenPosition, bar, bar_idx: int, version: str):
        """Update an open position against the current bar.

        Check order: SL first, then TP (conservative).
        """
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        # Time stop check
        bars_held = bar_idx - pos.entry_bar
        if bars_held >= self.entry_exit.time_stop and not pos.tp1_hit:
            exit_p = self.entry_exit.apply_slippage(close, pos.direction, is_entry=False)
            exit_p = max(low, min(high, exit_p))  # Clamp to bar range
            pos.exit_price = exit_p
            pos.exit_reason = "time_stop"
            pos.exit_bar = bar_idx
            pos.is_closed = True
            pos.commissions += self.entry_exit.calculate_commission(exit_p, pos.size)
            return

        # SL check (BEFORE TP — conservative)
        stop = pos.trailing_stop if pos.tp1_hit and pos.trailing_stop > 0 else pos.stop_loss

        if pos.direction == 1:  # Long
            if low <= stop:
                # Clamp exit price to bar range (gap protection)
                exit_p = max(stop, low)
                pos.exit_price = exit_p
                pos.exit_reason = "trailing" if pos.tp1_hit else "sl"
                pos.exit_bar = bar_idx
                pos.is_closed = True
                pos.commissions += self.entry_exit.calculate_commission(exit_p, pos.size)
                return
        else:  # Short
            if high >= stop:
                exit_p = min(stop, high)
                pos.exit_price = exit_p
                pos.exit_reason = "trailing" if pos.tp1_hit else "sl"
                pos.exit_bar = bar_idx
                pos.is_closed = True
                pos.commissions += self.entry_exit.calculate_commission(exit_p, pos.size)
                return

        # TP1 check
        if not pos.tp1_hit:
            if pos.direction == 1 and high >= pos.tp1:
                pos.tp1_hit = True
                if version in ("v2", "v3", "v4"):
                    # Partial close: reduce size by tp1_close_pct
                    close_size = pos.initial_size * self.entry_exit.tp1_close_pct
                    pos.size -= close_size
                    pos.commissions += self.entry_exit.calculate_commission(pos.tp1, close_size)

                    if version in ("v3", "v4"):
                        # Set trailing stop at breakeven
                        pos.trailing_stop = pos.entry_price
                else:
                    # V1: Full close at TP1
                    pos.exit_price = pos.tp1
                    pos.exit_reason = "tp1"
                    pos.exit_bar = bar_idx
                    pos.is_closed = True
                    pos.commissions += self.entry_exit.calculate_commission(pos.tp1, pos.size)
                    return

            elif pos.direction == -1 and low <= pos.tp1:
                pos.tp1_hit = True
                if version in ("v2", "v3", "v4"):
                    close_size = pos.initial_size * self.entry_exit.tp1_close_pct
                    pos.size -= close_size
                    pos.commissions += self.entry_exit.calculate_commission(pos.tp1, close_size)

                    if version in ("v3", "v4"):
                        pos.trailing_stop = pos.entry_price
                else:
                    pos.exit_price = pos.tp1
                    pos.exit_reason = "tp1"
                    pos.exit_bar = bar_idx
                    pos.is_closed = True
                    pos.commissions += self.entry_exit.calculate_commission(pos.tp1, pos.size)
                    return

        # TP2 check (only for v2+)
        if pos.tp1_hit and version in ("v2", "v3", "v4"):
            if pos.direction == 1 and high >= pos.tp2:
                pos.exit_price = pos.tp2
                pos.exit_reason = "tp2"
                pos.exit_bar = bar_idx
                pos.is_closed = True
                pos.commissions += self.entry_exit.calculate_commission(pos.tp2, pos.size)
                return
            elif pos.direction == -1 and low <= pos.tp2:
                pos.exit_price = pos.tp2
                pos.exit_reason = "tp2"
                pos.exit_bar = bar_idx
                pos.is_closed = True
                pos.commissions += self.entry_exit.calculate_commission(pos.tp2, pos.size)
                return

        # Update trailing stop (v3, v4)
        if pos.tp1_hit and version in ("v3", "v4"):
            atr = bar.get("atr", 0)
            if atr > 0:
                pos.trailing_stop = self.entry_exit.update_trailing_stop(
                    close, atr, pos.direction, pos.trailing_stop, True
                )

    def _calc_pnl(self, pos: OpenPosition) -> float:
        """Calculate realized P&L for a closed position."""
        if pos.tp1_hit and pos.exit_reason in ("tp2", "trailing", "time_stop", "sl", "end_of_data"):
            # Partial close at TP1 + remaining at exit price
            tp1_pnl = (pos.tp1 - pos.entry_price) * pos.direction * (
                pos.initial_size * self.entry_exit.tp1_close_pct
            )
            remaining_pnl = (pos.exit_price - pos.entry_price) * pos.direction * pos.size
            total_pnl = tp1_pnl + remaining_pnl - pos.commissions
        else:
            total_pnl = (pos.exit_price - pos.entry_price) * pos.direction * pos.initial_size
            total_pnl -= pos.commissions

        return total_pnl

    def _unrealized_pnl(self, pos: OpenPosition, current_price: float) -> float:
        """Calculate unrealized P&L for an open position."""
        return (current_price - pos.entry_price) * pos.direction * pos.size
