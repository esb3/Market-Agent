"""Stooq client — daily bars, used only as a price cross-check against
Yahoo (see reconcile.py). Free, no API key, no documented rate limits,
but also no options data — this is strictly a secondary source that
exists to catch Yahoo's split/dividend adjustment errors, never a
primary feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Optional

import pandas as pd
import requests

from sources.base import SchemaError, SourceUnavailable, with_retry

STOOQ_CSV_URL = "https://stooq.com/q/d/l/"
DAILY_BAR_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
# Stooq's body for an unknown symbol or an outage, instead of a 4xx/5xx.
NO_DATA_MARKER = "N/D"


def stooq_symbol(ticker: str) -> str:
    """Stooq's US-listing convention: lowercase ticker + '.us'."""
    return f"{ticker.lower()}.us"


@dataclass(frozen=True)
class StooqConfig:
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_base_seconds: float = 2.0


class StooqClient:
    def __init__(self, config: StooqConfig):
        self._config = config

    def daily_bars(self, ticker: str) -> pd.DataFrame:
        """Full available daily history, indexed by Date. Stooq has no
        date-range param on this endpoint — trim to what's needed at
        the call site."""
        params = {"s": stooq_symbol(ticker), "i": "d"}

        def call() -> str:
            try:
                resp = requests.get(STOOQ_CSV_URL, params=params, timeout=self._config.timeout_seconds)
            except requests.RequestException as exc:
                raise SourceUnavailable(f"stooq {ticker}: {exc}") from exc
            if resp.status_code != 200:
                raise SourceUnavailable(f"stooq {ticker}: HTTP {resp.status_code}")
            return resp.text

        text = with_retry(
            call,
            max_retries=self._config.max_retries,
            backoff_base=self._config.backoff_base_seconds,
            retry_on=(SourceUnavailable,),
        )
        return _parse_daily_bars(text, ticker)

    def close_on(self, ticker: str, target_date: date) -> Optional[float]:
        """Close for one specific date, or None if Stooq has no bar
        that day (holiday, weekend, too-new listing)."""
        frame = self.daily_bars(ticker)
        ts = pd.Timestamp(target_date)
        if ts not in frame.index:
            return None
        return float(frame.loc[ts, "Close"])


def _parse_daily_bars(text: str, ticker: str) -> pd.DataFrame:
    stripped = text.strip()
    if not stripped or stripped == NO_DATA_MARKER:
        raise SchemaError(f"stooq {ticker}: no data returned (unknown symbol or stooq outage)")
    try:
        frame = pd.read_csv(StringIO(text))
    except Exception as exc:  # pandas raises several distinct error types here
        raise SchemaError(f"stooq {ticker}: unparseable CSV: {exc}") from exc
    missing = [c for c in DAILY_BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"stooq {ticker}: missing columns {missing}, got {list(frame.columns)}")
    if frame.empty:
        raise SchemaError(f"stooq {ticker}: CSV parsed but has zero rows")
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.set_index("Date")
