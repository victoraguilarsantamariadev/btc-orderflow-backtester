"""Binance Futures API client for historical data download.

Handles: OHLCV klines, funding rate, taker buy/sell volume,
and aggregate trades bulk download from data.binance.vision.
"""

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils.timestamps import to_ms, from_ms, ms_series_to_datetime


BASE_URL = "https://fapi.binance.com"
BULK_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

# Rate limiting: 1200 weight per minute. Most endpoints cost 5-10 weight.
# We use a simple sleep-based limiter.
REQUEST_DELAY = 0.15  # seconds between requests (~400 req/min, safe margin)


class BinanceClient:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: dict) -> list:
        """GET request with rate limiting and retry."""
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(3):
            time.sleep(REQUEST_DELAY)
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed after 3 attempts: {endpoint}")

    # ── OHLCV Klines ────────────────────────────────────────────────

    def download_klines(self, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download OHLCV klines by paginating through the time range.

        Args:
            interval: e.g. '5m', '1h', '1d'
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        start_ms = to_ms(start_date)
        end_ms = to_ms(end_date)
        all_rows = []

        print(f"Downloading {self.symbol} klines {interval} from {start_date} to {end_date}...")

        while start_ms < end_ms:
            data = self._get("/fapi/v1/klines", {
                "symbol": self.symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1500,
            })
            if not data:
                break

            all_rows.extend(data)
            # Move start to after the last candle
            start_ms = data[-1][0] + 1
            print(f"  {from_ms(data[-1][0]).strftime('%Y-%m-%d %H:%M')} — {len(all_rows)} rows")

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_vol",
            "taker_buy_quote_vol", "ignore",
        ])

        df["timestamp"] = ms_series_to_datetime(df["open_time"])
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].copy()
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} klines")
        return df

    # ── Funding Rate ────────────────────────────────────────────────

    def download_funding_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download funding rate history (every 8 hours)."""
        start_ms = to_ms(start_date)
        end_ms = to_ms(end_date)
        all_rows = []

        print(f"Downloading {self.symbol} funding rate from {start_date} to {end_date}...")

        while start_ms < end_ms:
            data = self._get("/fapi/v1/fundingRate", {
                "symbol": self.symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            })
            if not data:
                break

            all_rows.extend(data)
            start_ms = data[-1]["fundingTime"] + 1

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["timestamp"] = ms_series_to_datetime(df["fundingTime"])
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]].copy()
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} funding rate records")
        return df

    # ── Taker Buy/Sell Volume (30 days only) ────────────────────────

    def download_taker_volume(self, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download taker long/short ratio (buy/sell volume).

        Note: Binance only retains 30 days of this data.
        """
        start_ms = to_ms(start_date)
        end_ms = to_ms(end_date)
        all_rows = []

        print(f"Downloading {self.symbol} taker volume from {start_date} to {end_date}...")

        while start_ms < end_ms:
            data = self._get("/futures/data/takerlongshortRatio", {
                "symbol": self.symbol,
                "period": period,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 500,
            })
            if not data:
                break

            all_rows.extend(data)
            start_ms = data[-1]["timestamp"] + 1

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["timestamp_dt"] = ms_series_to_datetime(df["timestamp"])
        df["buy_vol"] = df["buyVol"].astype(float)
        df["sell_vol"] = df["sellVol"].astype(float)
        df["taker_ratio"] = df["buySellRatio"].astype(float)
        df = df.rename(columns={"timestamp_dt": "timestamp"})
        df = df[["timestamp", "buy_vol", "sell_vol", "taker_ratio"]].copy()
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} taker volume records")
        return df

    # ── Aggregate Trades (bulk download from data.binance.vision) ──

    def download_aggtrades_day(self, date: str) -> pd.DataFrame:
        """Download and aggregate one day of aggregate trades into 5-min delta.

        Args:
            date: 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: timestamp, buy_volume, sell_volume, delta, trade_count
        """
        url = f"{BULK_URL}/{self.symbol}/{self.symbol}-aggTrades-{date}.zip"
        resp = self.session.get(url, timeout=120)

        if resp.status_code == 404:
            print(f"  No aggTrades for {date}")
            return pd.DataFrame()

        resp.raise_for_status()

        # Extract CSV from zip
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=[
                    "agg_trade_id", "price", "quantity", "first_trade_id",
                    "last_trade_id", "timestamp", "is_buyer_maker", "best_match",
                ], dtype=str)

        # Drop header row if present and filter valid rows
        df = df[pd.to_numeric(df["timestamp"], errors="coerce").notna()].copy()
        df["timestamp"] = df["timestamp"].astype(int)
        df["timestamp_dt"] = ms_series_to_datetime(df["timestamp"])
        df["quantity"] = df["quantity"].astype(float)
        df["is_buyer_maker"] = df["is_buyer_maker"].str.strip().str.lower() == "true"

        # Resample to 5-minute bars
        df = df.set_index("timestamp_dt")
        buy_vol = df.loc[~df["is_buyer_maker"], "quantity"].resample("5min").sum()
        sell_vol = df.loc[df["is_buyer_maker"], "quantity"].resample("5min").sum()
        count = df["quantity"].resample("5min").count()

        result = pd.DataFrame({
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "trade_count": count,
        }).fillna(0)

        result["delta"] = result["buy_volume"] - result["sell_volume"]
        result = result.reset_index().rename(columns={"timestamp_dt": "timestamp"})
        return result

    def _download_and_aggregate_day(self, date_str: str) -> tuple[str, pd.DataFrame | None]:
        """Download and aggregate one day. Returns (date_str, df or None)."""
        try:
            df = self.download_aggtrades_day(date_str)
            if not df.empty:
                return (date_str, df)
            return (date_str, None)
        except Exception as e:
            return (date_str, None)

    def download_aggtrades_range(self, start_date: str, end_date: str,
                                  output_path: str = None,
                                  workers: int = 4) -> pd.DataFrame:
        """Download aggregate trades for a date range with parallel downloads.

        Args:
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            output_path: If provided, saves intermediate results and final parquet
            workers: Number of parallel download threads (default 4)

        Returns:
            DataFrame with 5-min aggregated delta data
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.data.storage import save_parquet, append_parquet

        dates = pd.date_range(start_date, end_date, freq="D")
        all_dfs = []
        completed = 0
        total = len(dates)

        print(f"Downloading {self.symbol} aggTrades from {start_date} to {end_date}")
        print(f"  {total} days, {workers} parallel workers")

        # Process in batches to save memory
        batch_size = 20
        for batch_start in range(0, total, batch_size):
            batch_dates = dates[batch_start:batch_start + batch_size]
            batch_dfs = []

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._download_and_aggregate_day, d.strftime("%Y-%m-%d")): d
                    for d in batch_dates
                }

                for future in as_completed(futures):
                    date_str, df = future.result()
                    completed += 1
                    if df is not None:
                        batch_dfs.append(df)
                        print(f"  [{completed}/{total}] {date_str}: {len(df)} bars")
                    else:
                        print(f"  [{completed}/{total}] {date_str}: skip")

            if batch_dfs:
                chunk = pd.concat(batch_dfs, ignore_index=True)
                all_dfs.append(chunk)
                if output_path:
                    append_parquet(chunk, output_path)
                    print(f"  Saved batch ({batch_start+1}-{min(batch_start+batch_size, total)})")

        # Return final result
        if output_path and Path(output_path).exists():
            return pd.read_parquet(output_path)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)

        return pd.DataFrame()
