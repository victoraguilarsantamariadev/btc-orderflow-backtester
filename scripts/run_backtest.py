"""Run backtest on downloaded data.

Usage:
    python -m scripts.run_backtest                    # V1 simple
    python -m scripts.run_backtest --version v3       # V3 with trailing
    python -m scripts.run_backtest --walk-forward     # Walk-forward validation
    python -m scripts.run_backtest --monte-carlo      # Monte Carlo simulation
    python -m scripts.run_backtest --all              # Everything
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.data.storage import load_parquet, save_parquet
from src.data.cleaner import run_cleaning_pipeline
from src.features.builder import build_features
from src.signals.confluence import ConfluenceEngine
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_metrics, print_metrics
from src.backtest.anti_bias import LookAheadBiasChecker
from src.backtest.walk_forward import WalkForwardValidator
from src.backtest.monte_carlo import MonteCarloSimulator


def main():
    parser = argparse.ArgumentParser(description="Run trading backtest")
    parser.add_argument("--version", default="v1", choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--monte-carlo", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use existing processed data")
    args = parser.parse_args()

    config = load_config()
    processed_path = Path(config["data"]["processed_dir"]) / "features.parquet"

    # ── 1. Load or build data ─────────────────────────────────────
    if processed_path.exists() and args.skip_download:
        print("Loading existing processed data...")
        merged = load_parquet(str(processed_path))
    else:
        print("Running data cleaning pipeline...")
        merged = run_cleaning_pipeline(config)

    # ── 2. Build features ─────────────────────────────────────────
    print("\nBuilding features...")
    features = build_features(merged, config)

    # ── 3. Calculate confluence signals ───────────────────────────
    print("\nCalculating confluence signals...")
    confluence = ConfluenceEngine(config)
    signals = confluence.calculate_scores(features)

    summary = confluence.get_signal_summary(signals)
    print(f"\nSignal summary:")
    print(f"  Total bars: {summary['total_bars']}")
    print(f"  Trade signals: {summary['trade_signals']} ({summary['signal_rate']:.2f}%)")
    print(f"  Long: {summary['long_signals']}, Short: {summary['short_signals']}")
    print(f"  Enhanced (7+): {summary['enhanced_signals']}")

    # ── 4. Run backtest ───────────────────────────────────────────
    versions = ["v1", "v2", "v3", "v4"] if args.all else [args.version]

    for version in versions:
        print(f"\n{'='*60}")
        print(f"BACKTEST VERSION: {version.upper()}")
        print(f"{'='*60}")

        engine = BacktestEngine(config)
        result = engine.run(features, signals, version)

        metrics = calculate_metrics(result["trades"], result["equity_curve"], config)
        print_metrics(metrics)

        # Anti-bias check
        print("\nRunning anti look-ahead bias checks...")
        checker = LookAheadBiasChecker(features)
        checker.run_all_checks(result["trades"])

    # ── 5. Walk-forward (if requested) ────────────────────────────
    if args.walk_forward or args.all:
        print(f"\n{'='*60}")
        print("WALK-FORWARD VALIDATION")
        print(f"{'='*60}")

        wf = WalkForwardValidator(config)
        wf_result = wf.run(features, version=args.version if not args.all else "v3")

    # ── 6. Monte Carlo (if requested) ─────────────────────────────
    if args.monte_carlo or args.all:
        print(f"\n{'='*60}")
        print("MONTE CARLO SIMULATION")
        print(f"{'='*60}")

        # Use trades from the last backtest run
        trade_pnls = [t.pnl for t in result["trades"]]
        mc = MonteCarloSimulator(config)
        mc_result = mc.simulate(trade_pnls, config)


if __name__ == "__main__":
    main()
