"""
Daily closing prices for Ethereum (ETH).

Uses CoinGecko public API (works where Binance returns HTTP 451, e.g. US).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import requests

# CoinGecko OHLC: for `days` in 180, 365, max — candles are daily (shorter `days` uses intraday candles).
COINGECKO_ETH_OHLC = "https://api.coingecko.com/api/v3/coins/ethereum/ohlc"


def get_eth_daily_closes(limit: int = 365) -> list[tuple[date, float]]:
    """
    Return (calendar date in UTC, daily close in USD) for each candle.

    Requests enough history from CoinGecko (daily granularity), then returns the
    last `limit` days. Cap is soft-limited by what `days=max` returns.
    """
    limit = max(1, limit)
    # Need daily OHLC: use 365 or max (not 30 — that returns 4h candles, not daily closes).
    cg_days = "max" if limit > 365 else "365"
    params = {"vs_currency": "usd", "days": cg_days}
    headers = {
        "Accept": "application/json",
        "User-Agent": "personalProject/Prices.py (educational)",
    }
    response = requests.get(
        COINGECKO_ETH_OHLC,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected CoinGecko response: {rows!r}")

    out: list[tuple[date, float]] = []
    for row in rows:
        # [timestamp_ms, open, high, low, close]
        ts_ms, _, _, _, close = row
        dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        out.append((dt_utc.date(), float(close)))

    return out[-limit:] if len(out) >= limit else out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETH daily closing prices in USD (CoinGecko)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of most recent daily closes to show (default 30)",
    )
    args = parser.parse_args()

    rows = get_eth_daily_closes(limit=args.days)
    print(f"ETH/USD daily closes — last {len(rows)} days (UTC dates):\n")
    for d, close in rows:
        print(f"{d.isoformat()}\t{close:.2f}")


if __name__ == "__main__":
    main()
