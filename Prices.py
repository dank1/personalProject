"""
Daily closing prices for Ethereum (ETH/USD).

Uses Coinbase Exchange public candles with 1-day granularity (86400s). CoinGecko's
`/ohlc` endpoint uses multi-day candles for long ranges (e.g. ~4-day), which is not
one close per calendar day.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import requests

# Coinbase Exchange: max 300 candles per request for daily granularity.
COINBASE_ETH_CANDLES = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
GRANULARITY_1_DAY_S = 86400
MAX_CANDLES_PER_REQUEST = 300


def get_eth_daily_closes(limit: int = 365) -> list[tuple[date, float]]:
    """
    Return (UTC calendar date, daily close in USD) — one row per day.

    Paginates backwards in chunks of up to 300 days until `limit` days are collected
    or history runs out.
    """
    limit = max(1, limit)
    by_day: dict[date, float, float, float, float, float] = {}
    end = datetime.now(timezone.utc)

    while len(by_day) < limit:
        start = end - timedelta(days=MAX_CANDLES_PER_REQUEST)
        params = {
            "granularity": GRANULARITY_1_DAY_S,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        response = requests.get(COINBASE_ETH_CANDLES, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        # Each row: [ time, low, high, open, close, volume ] — time = bucket start (UTC)
        for row in batch:
            ts, low, high, open_price, close, volume = row
            day = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            by_day[day] = (float(low), float(high), float(open_price), float(close), float(volume))
        oldest_ts = min(int(c[0]) for c in batch)
        end = datetime.fromtimestamp(oldest_ts, tz=timezone.utc) - timedelta(seconds=1)
        if len(batch) < 2:
            break

    ordered = sorted(by_day.items(), key=lambda x: x[0])
    return ordered[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETH/USD daily closing prices (Coinbase Exchange, 1d candles)",
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
    for d, (low, high, open_price, close, volume) in rows:
        print(f"{d.isoformat()}\t{low:.2f}\t{high:.2f}\t{open_price:.2f}\t{close:.2f}\t{volume:.2f}")


if __name__ == "__main__":
    main()
