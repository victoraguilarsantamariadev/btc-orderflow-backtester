"""Backtest performance metrics: Sharpe, Sortino, drawdown, win rate, etc."""

import pandas as pd
import numpy as np


def calculate_metrics(trades: list, equity_curve: pd.DataFrame,
                       config: dict) -> dict:
    """Calculate comprehensive performance metrics.

    Args:
        trades: List of Trade dataclass instances
        equity_curve: DataFrame with timestamp, equity columns
        config: Config dict for prop firm parameters

    Returns:
        dict with all metrics
    """
    if not trades:
        return _empty_metrics()

    # Basic trade stats
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    largest_win = max(pnls) if pnls else 0
    largest_loss = min(pnls) if pnls else 0

    # Profit factor
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = np.mean(pnls) if pnls else 0

    # R-multiple stats
    r_multiples = [t.pnl / (t.size * abs(t.entry_price - t.stop_loss))
                   if t.size > 0 and t.entry_price != t.stop_loss else 0
                   for t in trades]
    avg_r = np.mean(r_multiples) if r_multiples else 0

    # Equity curve metrics
    equity = equity_curve["equity"].values
    initial_equity = config["backtest"]["initial_equity"]

    # Returns
    total_return_pct = (equity[-1] - initial_equity) / initial_equity * 100

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = drawdown.max()

    # Max drawdown duration (in bars)
    dd_duration = 0
    max_dd_duration = 0
    for dd in drawdown:
        if dd > 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # Sharpe Ratio (daily returns)
    if len(equity) > 288:  # At least 1 day of data
        # Resample to daily equity
        daily_equity = equity_curve.set_index("timestamp")["equity"].resample("1D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        else:
            sharpe = 0.0

        # Sortino (only downside deviation)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = daily_returns.mean() / downside.std() * np.sqrt(365)
        else:
            sortino = 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Calmar ratio
    calmar = (total_return_pct / 100 * 365 / max(len(equity) / 288, 1)) / max_drawdown \
        if max_drawdown > 0 else 0

    # Trade duration stats
    durations = [t.bars_held for t in trades]
    avg_duration = np.mean(durations) if durations else 0
    avg_duration_minutes = avg_duration * 5  # 5-min bars

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Consecutive losses
    max_consecutive_losses = 0
    current_streak = 0
    for pnl in pnls:
        if pnl <= 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # Prop firm check
    prop_config = config.get("prop_firm", {})
    profit_target = prop_config.get("profit_target_pct", 0.08)
    max_dd_limit = prop_config.get("max_drawdown_pct", 0.10)

    hit_profit_target = total_return_pct / 100 >= profit_target
    exceeded_drawdown = max_drawdown > max_dd_limit
    passed_challenge = hit_profit_target and not exceeded_drawdown

    # Trading days
    if hasattr(trades[0], "entry_time"):
        trading_days = len(set(
            str(t.entry_time)[:10] for t in trades if t.entry_time is not None
        ))
    else:
        trading_days = 0

    return {
        # Trade stats
        "total_trades": total_trades,
        "win_rate": win_rate,
        "win_count": win_count,
        "loss_count": loss_count,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_r_multiple": avg_r,

        # Returns
        "total_return_pct": total_return_pct,
        "total_pnl": sum(pnls),

        # Risk
        "max_drawdown_pct": max_drawdown * 100,
        "max_dd_duration_bars": max_dd_duration,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_consecutive_losses": max_consecutive_losses,

        # Duration
        "avg_bars_held": avg_duration,
        "avg_minutes_held": avg_duration_minutes,
        "trading_days": trading_days,

        # Exit reasons
        "exit_reasons": exit_reasons,

        # Prop firm
        "passed_challenge": passed_challenge,
        "hit_profit_target": hit_profit_target,
        "exceeded_drawdown": exceeded_drawdown,
    }


def _empty_metrics() -> dict:
    """Return metrics dict for zero trades."""
    return {
        "total_trades": 0,
        "win_rate": 0,
        "win_count": 0,
        "loss_count": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "largest_win": 0,
        "largest_loss": 0,
        "profit_factor": 0,
        "expectancy": 0,
        "avg_r_multiple": 0,
        "total_return_pct": 0,
        "total_pnl": 0,
        "max_drawdown_pct": 0,
        "max_dd_duration_bars": 0,
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "calmar_ratio": 0,
        "max_consecutive_losses": 0,
        "avg_bars_held": 0,
        "avg_minutes_held": 0,
        "trading_days": 0,
        "exit_reasons": {},
        "passed_challenge": False,
        "hit_profit_target": False,
        "exceeded_drawdown": False,
    }


def print_metrics(metrics: dict):
    """Pretty-print metrics to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n--- Trade Stats ---")
    print(f"Total trades:     {metrics['total_trades']}")
    print(f"Win rate:         {metrics['win_rate']:.1%}")
    print(f"Avg win:          ${metrics['avg_win']:.2f}")
    print(f"Avg loss:         ${metrics['avg_loss']:.2f}")
    print(f"Profit factor:    {metrics['profit_factor']:.2f}")
    print(f"Expectancy:       ${metrics['expectancy']:.2f}")
    print(f"Avg R-multiple:   {metrics['avg_r_multiple']:.2f}")

    print(f"\n--- Returns ---")
    print(f"Total return:     {metrics['total_return_pct']:.2f}%")
    print(f"Total P&L:        ${metrics['total_pnl']:.2f}")

    print(f"\n--- Risk ---")
    print(f"Max drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino ratio:    {metrics['sortino_ratio']:.2f}")
    print(f"Max consec. losses: {metrics['max_consecutive_losses']}")

    print(f"\n--- Duration ---")
    print(f"Avg bars held:    {metrics['avg_bars_held']:.1f} ({metrics['avg_minutes_held']:.0f} min)")
    print(f"Trading days:     {metrics['trading_days']}")

    print(f"\n--- Exit Reasons ---")
    for reason, count in sorted(metrics["exit_reasons"].items()):
        print(f"  {reason}: {count}")

    print(f"\n--- Prop Firm ---")
    print(f"Passed challenge: {'YES' if metrics['passed_challenge'] else 'NO'}")
    print(f"  Hit profit target: {'YES' if metrics['hit_profit_target'] else 'NO'}")
    print(f"  Exceeded drawdown: {'YES' if metrics['exceeded_drawdown'] else 'NO'}")
    print("=" * 60)
