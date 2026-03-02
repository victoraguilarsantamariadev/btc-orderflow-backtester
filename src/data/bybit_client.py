"""Bybit V5 API client for historical Open Interest and Long/Short Ratio.

Key advantage: Bybit retains OI history since symbol launch and LS ratio since July 2020,
both in 5-minute resolution. This solves Binance's 30-day retention limit.
"""

import time

import pandas as pd
import requests

from src.utils.timestamps import to_ms, from_ms, ms_series_to_datetime


BASE_URL = "https://api.bybit.com"
REQUEST_DELAY = 0.1  # ~10 req/sec, Bybit allows more but we stay safe


class BybitClient:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict) -> dict:
        """GET request with rate limiting and retry."""
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(3):
            time.sleep(REQUEST_DELAY)
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print("Rate limited, waiting 5s...")
                time.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")
            return data["result"]
        raise RuntimeError(f"Failed after 3 attempts: {endpoint}")

    # ── Open Interest ───────────────────────────────────────────────

    def download_open_interest(self, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical open interest.

        Args:
            interval: '5min', '15min', '30min', '1h', '4h', '1d'
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: timestamp, open_interest
        """
        start_ms = to_ms(start_date)
        end_ms = to_ms(end_date)
        all_rows = []

        print(f"Downloading {self.symbol} OI {interval} from {start_date} to {end_date}...")

        # Bybit returns data newest-first, so we paginate backwards using cursor
        cursor = ""
        while True:
            params = {
                "category": "linear",
                "symbol": self.symbol,
                "intervalTime": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            result = self._get("/v5/market/open-interest", params)
            rows = result.get("list", [])

            if not rows:
                break

            all_rows.extend(rows)
            cursor = result.get("nextPageCursor", "")

            # Log progress
            oldest = rows[-1]["timestamp"]
            print(f"  {from_ms(int(oldest)).strftime('%Y-%m-%d %H:%M')} — {len(all_rows)} rows")

            if not cursor:
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["timestamp"] = ms_series_to_datetime(df["timestamp"].astype(int))
        df["open_interest"] = df["openInterest"].astype(float)
        df = df[["timestamp", "open_interest"]].copy()
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} OI records")
        return df

    # ── Long/Short Account Ratio ────────────────────────────────────

    def download_long_short_ratio(self, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical long/short account ratio.

        Available since July 2020 for BTCUSDT.

        Args:
            period: '5min', '15min', '30min', '1h', '4h', '1d'
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: timestamp, buy_ratio, sell_ratio, ls_ratio
        """
        start_ms = to_ms(start_date)
        end_ms = to_ms(end_date)
        all_rows = []

        print(f"Downloading {self.symbol} LS ratio {period} from {start_date} to {end_date}...")

        # Bybit returns newest-first, paginate with cursor
        cursor = ""
        while True:
            params = {
                "category": "linear",
                "symbol": self.symbol,
                "period": period,
                "limit": 500,
            }
            if cursor:
                params["cursor"] = cursor

            result = self._get("/v5/market/account-ratio", params)
            rows = result.get("list", [])

            if not rows:
                break

            all_rows.extend(rows)
            cursor = result.get("nextPageCursor", "")

            oldest = rows[-1]["timestamp"]
            oldest_dt = from_ms(int(oldest))
            print(f"  {oldest_dt.strftime('%Y-%m-%d %H:%M')} — {len(all_rows)} rows")

            # Stop if we've gone past our start date
            if oldest_dt.timestamp() * 1000 <= start_ms:
                break

            if not cursor:
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["timestamp"] = ms_series_to_datetime(df["timestamp"].astype(int))
        df["buy_ratio"] = df["buyRatio"].astype(float)
        df["sell_ratio"] = df["sellRatio"].astype(float)
        # LS ratio: >1 means more longs, <1 means more shorts
        df["ls_ratio"] = df["buy_ratio"] / df["sell_ratio"].replace(0, float("nan"))
        df = df[["timestamp", "buy_ratio", "sell_ratio", "ls_ratio"]].copy()

        # Filter to our date range
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = pd.Timestamp(end_date, tz="UTC")
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} LS ratio records")
        return df
