"""Coinalyze API client for aggregated multi-exchange data.

Free API, no registration required. 40 requests/minute limit.
Daily timeframe has unlimited history. Intraday limited to ~2000 datapoints.
Used as supplementary context (aggregated OI across all exchanges).
"""

import time

import pandas as pd
import requests

from src.utils.timestamps import ms_series_to_datetime


BASE_URL = "https://api.coinalyze.net/v1"
REQUEST_DELAY = 1.6  # 40 req/min = 1 every 1.5s, with margin


class CoinalyzeClient:
    def __init__(self, symbol: str = "BTCUSDT"):
        # Coinalyze uses different symbol format for perps
        self.symbol = f"{symbol}_PERP.A"  # Aggregated across exchanges
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict) -> list:
        """GET request with rate limiting."""
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(3):
            time.sleep(REQUEST_DELAY)
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print("Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed after 3 attempts: {endpoint}")

    def download_open_interest_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download daily aggregated open interest across all exchanges.

        Daily data has no retention limit — full history available.

        Returns:
            DataFrame with columns: timestamp, oi_aggregated
        """
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
        end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp())

        print(f"Downloading aggregated daily OI from Coinalyze ({start_date} to {end_date})...")

        data = self._get("/open-interest-history", {
            "symbols": self.symbol,
            "interval": "daily",
            "from": start_ts,
            "to": end_ts,
        })

        if not data or not data[0].get("history"):
            print("No data returned")
            return pd.DataFrame()

        history = data[0]["history"]
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["oi_aggregated"] = df["o"].astype(float)
        df = df[["timestamp", "oi_aggregated"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} daily OI records")
        return df

    def download_funding_rate_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download daily aggregated funding rate."""
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
        end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp())

        print(f"Downloading aggregated daily funding from Coinalyze ({start_date} to {end_date})...")

        data = self._get("/funding-rate-history", {
            "symbols": self.symbol,
            "interval": "daily",
            "from": start_ts,
            "to": end_ts,
        })

        if not data or not data[0].get("history"):
            print("No data returned")
            return pd.DataFrame()

        history = data[0]["history"]
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["funding_avg"] = df["o"].astype(float)
        df = df[["timestamp", "funding_avg"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} daily funding records")
        return df

    def download_liquidations_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download daily aggregated liquidation data."""
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
        end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp())

        print(f"Downloading aggregated daily liquidations from Coinalyze ({start_date} to {end_date})...")

        data = self._get("/liquidation-history", {
            "symbols": self.symbol,
            "interval": "daily",
            "from": start_ts,
            "to": end_ts,
        })

        if not data or not data[0].get("history"):
            print("No data returned")
            return pd.DataFrame()

        history = data[0]["history"]
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        # Coinalyze returns long and short liquidation volumes
        df["liq_long"] = df.get("l", 0)
        df["liq_short"] = df.get("s", 0)
        df = df[["timestamp", "liq_long", "liq_short"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"Done: {len(df)} daily liquidation records")
        return df
