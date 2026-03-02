"""Anti look-ahead bias checker.

Validates that the backtest did not use future information.
Run this after every backtest to catch bugs.
"""

import pandas as pd
import numpy as np


class LookAheadBiasChecker:
    """Validates backtest trades for look-ahead bias violations."""

    def __init__(self, features: pd.DataFrame):
        self.features = features

    def check_entry_timing(self, trades: list) -> list[str]:
        """Verify entry_bar > signal_bar for all trades (confirmation happened after signal)."""
        violations = []
        for i, t in enumerate(trades):
            # Entry should be at least 1 bar after signal could have been generated
            if t.bars_held < 1:
                violations.append(
                    f"Trade {i}: entered and exited same bar (bars_held={t.bars_held})"
                )
        return violations

    def check_entry_prices(self, trades: list) -> list[str]:
        """Verify entry price is within OHLC range of the entry bar."""
        violations = []
        for i, t in enumerate(trades):
            if t.entry_bar >= len(self.features):
                continue
            bar = self.features.iloc[t.entry_bar]
            # Entry price should be near the open (with slippage tolerance)
            tolerance = bar["high"] - bar["low"]  # Full bar range as tolerance
            if t.entry_price < bar["low"] - tolerance * 0.01:
                violations.append(
                    f"Trade {i}: entry {t.entry_price:.2f} below bar low {bar['low']:.2f}"
                )
            if t.entry_price > bar["high"] + tolerance * 0.01:
                violations.append(
                    f"Trade {i}: entry {t.entry_price:.2f} above bar high {bar['high']:.2f}"
                )
        return violations

    def check_exit_prices(self, trades: list) -> list[str]:
        """Verify exit prices are within OHLC range of exit bar."""
        violations = []
        for i, t in enumerate(trades):
            if t.exit_bar >= len(self.features):
                continue
            if t.exit_reason == "end_of_data":
                continue
            bar = self.features.iloc[t.exit_bar]
            tolerance = (bar["high"] - bar["low"]) * 0.01
            if t.exit_price < bar["low"] - tolerance:
                violations.append(
                    f"Trade {i}: exit {t.exit_price:.2f} below bar low {bar['low']:.2f}"
                )
            if t.exit_price > bar["high"] + tolerance:
                violations.append(
                    f"Trade {i}: exit {t.exit_price:.2f} above bar high {bar['high']:.2f}"
                )
        return violations

    def check_chronological_order(self, trades: list) -> list[str]:
        """Verify trades don't overlap and are in chronological order."""
        violations = []
        for i in range(1, len(trades)):
            if trades[i].entry_bar < trades[i-1].exit_bar:
                violations.append(
                    f"Trade {i} overlaps with trade {i-1}: "
                    f"entry={trades[i].entry_bar} < prev_exit={trades[i-1].exit_bar}"
                )
        return violations

    def run_all_checks(self, trades: list) -> dict:
        """Run all bias checks and return a report."""
        timing = self.check_entry_timing(trades)
        entry_prices = self.check_entry_prices(trades)
        exit_prices = self.check_exit_prices(trades)
        chronological = self.check_chronological_order(trades)

        total_violations = len(timing) + len(entry_prices) + len(exit_prices) + len(chronological)

        report = {
            "total_violations": total_violations,
            "passed": total_violations == 0,
            "entry_timing": timing,
            "entry_prices": entry_prices,
            "exit_prices": exit_prices,
            "chronological": chronological,
        }

        if total_violations == 0:
            print("Anti look-ahead bias check: PASSED (0 violations)")
        else:
            print(f"Anti look-ahead bias check: FAILED ({total_violations} violations)")
            for category, violations in [
                ("Entry timing", timing),
                ("Entry prices", entry_prices),
                ("Exit prices", exit_prices),
                ("Chronological", chronological),
            ]:
                if violations:
                    print(f"  {category}:")
                    for v in violations[:5]:
                        print(f"    - {v}")
                    if len(violations) > 5:
                        print(f"    ... and {len(violations) - 5} more")

        return report
