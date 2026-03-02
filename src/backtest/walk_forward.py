"""Walk-forward validation: train on block N, test on block N+1, advance.

Prevents overfitting by ensuring all performance metrics come from
out-of-sample periods that the system has never seen during optimization.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_metrics
from src.signals.confluence import ConfluenceEngine


@dataclass
class WalkForwardBlock:
    """A single train/test block."""
    block_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_bars: int = 0
    test_bars: int = 0
    is_metrics: dict = None
    oos_metrics: dict = None


class WalkForwardValidator:
    """Walk-forward validation engine."""

    def __init__(self, config: dict):
        self.config = config
        self.train_months = config["walk_forward"]["train_months"]  # 2
        self.test_months = config["walk_forward"]["test_months"]    # 1
        self.oos_months = config["walk_forward"]["oos_months"]      # 2

    def generate_splits(self, features: pd.DataFrame) -> list[WalkForwardBlock]:
        """Generate train/test splits, reserving last oos_months as final holdout.

        Example with 6 months, train=2, test=1, oos=2:
            Block 0: Train Sep-Oct, Test Nov
            Block 1: Train Oct-Nov, Test Dec
            (Jan-Feb reserved as final OOS)
        """
        ts = features["timestamp"]
        data_start = ts.min()
        data_end = ts.max()

        # Reserve OOS
        oos_start = data_end - pd.DateOffset(months=self.oos_months)

        blocks = []
        block_id = 0
        train_start = data_start

        while True:
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Stop if test period enters OOS
            if test_end > oos_start:
                break

            train_mask = (ts >= train_start) & (ts < train_end)
            test_mask = (ts >= test_start) & (ts < test_end)

            block = WalkForwardBlock(
                block_id=block_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_bars=train_mask.sum(),
                test_bars=test_mask.sum(),
            )
            blocks.append(block)

            # Advance by test_months
            train_start = train_start + pd.DateOffset(months=self.test_months)
            block_id += 1

        return blocks

    def run(self, features: pd.DataFrame, version: str = "v1") -> dict:
        """Run walk-forward validation.

        For each block:
        1. Calculate signals on train period (for validation only)
        2. Run backtest on test period using the same signal parameters
        3. Collect OOS metrics

        Returns:
            dict with blocks, aggregated OOS metrics, degradation ratio
        """
        blocks = self.generate_splits(features)

        if not blocks:
            print("Not enough data for walk-forward validation")
            return {"blocks": [], "oos_metrics": {}, "degradation": 0}

        print(f"Walk-forward: {len(blocks)} blocks")

        confluence = ConfluenceEngine(self.config)
        all_oos_trades = []

        for block in blocks:
            print(f"\n--- Block {block.block_id} ---")
            print(f"  Train: {block.train_start.date()} to {block.train_end.date()} ({block.train_bars} bars)")
            print(f"  Test:  {block.test_start.date()} to {block.test_end.date()} ({block.test_bars} bars)")

            ts = features["timestamp"]

            # In-sample (train)
            train_data = features[(ts >= block.train_start) & (ts < block.train_end)].reset_index(drop=True)
            train_signals = confluence.calculate_scores(train_data)
            engine_is = BacktestEngine(self.config)
            is_result = engine_is.run(train_data, train_signals, version)
            block.is_metrics = calculate_metrics(is_result["trades"], is_result["equity_curve"], self.config)
            print(f"  IS:  {block.is_metrics['total_trades']} trades, "
                  f"Sharpe={block.is_metrics['sharpe_ratio']:.2f}, "
                  f"Return={block.is_metrics['total_return_pct']:.2f}%")

            # Out-of-sample (test)
            test_data = features[(ts >= block.test_start) & (ts < block.test_end)].reset_index(drop=True)
            test_signals = confluence.calculate_scores(test_data)
            engine_oos = BacktestEngine(self.config)
            oos_result = engine_oos.run(test_data, test_signals, version)
            block.oos_metrics = calculate_metrics(oos_result["trades"], oos_result["equity_curve"], self.config)
            print(f"  OOS: {block.oos_metrics['total_trades']} trades, "
                  f"Sharpe={block.oos_metrics['sharpe_ratio']:.2f}, "
                  f"Return={block.oos_metrics['total_return_pct']:.2f}%")

            all_oos_trades.extend(oos_result["trades"])

        # Aggregate OOS metrics
        is_sharpes = [b.is_metrics["sharpe_ratio"] for b in blocks if b.is_metrics]
        oos_sharpes = [b.oos_metrics["sharpe_ratio"] for b in blocks if b.oos_metrics]

        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0

        degradation = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0

        print(f"\n=== Walk-Forward Summary ===")
        print(f"Avg IS Sharpe:  {avg_is_sharpe:.2f}")
        print(f"Avg OOS Sharpe: {avg_oos_sharpe:.2f}")
        print(f"Degradation ratio: {degradation:.2f} (>0.5 = robust, <0.5 = overfitted)")
        print(f"Total OOS trades: {len(all_oos_trades)}")

        return {
            "blocks": blocks,
            "all_oos_trades": all_oos_trades,
            "avg_is_sharpe": avg_is_sharpe,
            "avg_oos_sharpe": avg_oos_sharpe,
            "degradation": degradation,
        }
