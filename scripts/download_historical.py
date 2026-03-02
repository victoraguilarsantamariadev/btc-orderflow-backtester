"""One-time historical data download script.

Downloads all available data from Binance, Bybit, and Coinalyze
and saves as parquet files in data/raw/.

Usage:
    python -m scripts.download_historical
    python -m scripts.download_historical --skip-aggtrades  (skip heavy aggTrades download)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.data.binance_client import BinanceClient
from src.data.bybit_client import BybitClient
from src.data.coinalyze_client import CoinalyzeClient
from src.data.storage import save_parquet


def main():
    parser = argparse.ArgumentParser(description="Download historical crypto data")
    parser.add_argument("--skip-aggtrades", action="store_true",
                        help="Skip aggregate trades download (very large, ~hours)")
    parser.add_argument("--aggtrades-only", action="store_true",
                        help="Only download aggregate trades")
    args = parser.parse_args()

    config = load_config()
    symbol = config["symbol"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    raw_dir = Path(config["data"]["raw_dir"])

    if args.aggtrades_only:
        print("\n=== Binance: Aggregate Trades (delta/CVD) ===")
        binance = BinanceClient(symbol)
        output = str(raw_dir / "aggtrades" / f"{symbol.lower()}_delta_5m.parquet")
        binance.download_aggtrades_range(start, end, output_path=output)
        print(f"Saved to {output}")
        return

    # ── 1. Binance: OHLCV 5min ──────────────────────────────────────
    print("\n=== Binance: OHLCV 5min ===")
    binance = BinanceClient(symbol)
    df_klines = binance.download_klines("5m", start, end)
    path = str(raw_dir / "klines" / f"{symbol.lower()}_5m.parquet")
    save_parquet(df_klines, path)
    print(f"Saved to {path}")

    # ── 2. Binance: Funding Rate ─────────────────────────────────────
    print("\n=== Binance: Funding Rate ===")
    df_funding = binance.download_funding_rate(start, end)
    path = str(raw_dir / "funding" / f"{symbol.lower()}_funding.parquet")
    save_parquet(df_funding, path)
    print(f"Saved to {path}")

    # ── 3. Binance: Taker Volume (last 30 days only) ─────────────────
    print("\n=== Binance: Taker Volume (30 days) ===")
    import datetime as dt
    taker_start = (dt.datetime.now() - dt.timedelta(days=29)).strftime("%Y-%m-%d")
    df_taker = binance.download_taker_volume("5m", taker_start, end)
    path = str(raw_dir / "taker_volume" / f"{symbol.lower()}_taker_5m.parquet")
    save_parquet(df_taker, path)
    print(f"Saved to {path}")

    # ── 4. Bybit: Open Interest 5min ─────────────────────────────────
    print("\n=== Bybit: Open Interest 5min ===")
    bybit = BybitClient(symbol)
    df_oi = bybit.download_open_interest("5min", start, end)
    path = str(raw_dir / "oi" / f"{symbol.lower()}_oi_5m.parquet")
    save_parquet(df_oi, path)
    print(f"Saved to {path}")

    # ── 5. Bybit: Long/Short Ratio 5min ──────────────────────────────
    print("\n=== Bybit: Long/Short Ratio 5min ===")
    df_ls = bybit.download_long_short_ratio("5min", start, end)
    path = str(raw_dir / "ls_ratio" / f"{symbol.lower()}_ls_5m.parquet")
    save_parquet(df_ls, path)
    print(f"Saved to {path}")

    # ── 6. Coinalyze: Daily aggregated OI + liquidations ─────────────
    print("\n=== Coinalyze: Daily aggregated data ===")
    coinalyze = CoinalyzeClient(symbol)

    df_oi_daily = coinalyze.download_open_interest_daily(start, end)
    path = str(raw_dir / "coinalyze" / f"{symbol.lower()}_oi_daily.parquet")
    save_parquet(df_oi_daily, path)
    print(f"Saved to {path}")

    df_liq = coinalyze.download_liquidations_daily(start, end)
    path = str(raw_dir / "coinalyze" / f"{symbol.lower()}_liq_daily.parquet")
    save_parquet(df_liq, path)
    print(f"Saved to {path}")

    # ── 7. Aggregate Trades ──────────────────────────────────────────
    if not args.skip_aggtrades:
        print("\n=== Binance: Aggregate Trades (delta/CVD) ===")
        print("WARNING: This downloads ~50-100 GB of raw data. Will take hours.")
        print("Use --skip-aggtrades to skip, or --aggtrades-only to run separately.")
        output = str(raw_dir / "aggtrades" / f"{symbol.lower()}_delta_5m.parquet")
        binance.download_aggtrades_range(start, end, output_path=output)
        print(f"Saved to {output}")
    else:
        print("\n=== Skipping aggTrades (use --aggtrades-only later) ===")

    print("\n✓ All downloads complete!")
    print("Next step: Run the data exploration notebook to verify data quality.")


if __name__ == "__main__":
    main()
