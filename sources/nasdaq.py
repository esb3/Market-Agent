"""Nasdaq public earnings calendar — cross-check for yfinance's scraped
earnings date, which is often off by a day or two.

Nasdaq's calendar endpoint returns everything reporting on ONE date, not
a per-ticker lookup. `earnings_date_on()` checks a single candidate
date; reconcile.py's earnings reconciliation calls it across a small
window of candidate dates around what yfinance reported, bounded by
config so this never turns into an unbounded day-by-day scan of one
ticker's whole future.

Note: this endpoint's exact response shape could not be live-verified
from this build environment (no network egress here to api.nasdaq.com)
— see the top-level build notes for how to smoke-test sources against
real network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import requests

from sources.base import SchemaError, SourceUnavailable, with_retry

NASDAQ_CALENDAR_URL = "https://api.nasdaq.com/api/calendar/earnings"

# Nasdaq's API 403s without a browser-like User-Agent on the request.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


@dataclass(frozen=True)
class NasdaqConfig:
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_base_seconds: float = 2.0


class NasdaqClient:
    def __init__(self, config: NasdaqConfig):
        self._config = config

    def _fetch(self, target_date: date) -> dict:
        def call() -> dict:
            try:
                resp = requests.get(
                    NASDAQ_CALENDAR_URL,
                    params={"date": target_date.isoformat()},
                    headers=REQUEST_HEADERS,
                    timeout=self._config.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise SourceUnavailable(f"nasdaq {target_date}: {exc}") from exc
            if resp.status_code != 200:
                raise SourceUnavailable(f"nasdaq {target_date}: HTTP {resp.status_code}")
            return resp.json()

        return with_retry(
            call,
            max_retries=self._config.max_retries,
            backoff_base=self._config.backoff_base_seconds,
            retry_on=(SourceUnavailable,),
        )

    def earnings_date_on(self, ticker: str, target_date: date) -> bool:
        """True if `ticker` is listed as reporting on `target_date`."""
        payload = self._fetch(target_date)
        if not isinstance(payload, dict) or "data" not in payload:
            raise SchemaError(f"nasdaq {target_date}: response missing 'data' — {payload}")

        data = payload["data"]
        if data is None:
            # Nasdaq returns data=None for dates with nothing scheduled
            # (far future, weekends) -- that's zero results, not broken.
            return False
        if not isinstance(data, dict):
            raise SchemaError(f"nasdaq {target_date}: 'data' is not an object: {data!r}")

        rows = data.get("rows")
        if rows is None:
            return False
        if not isinstance(rows, list):
            raise SchemaError(f"nasdaq {target_date}: 'rows' is not a list: {rows!r}")

        target = ticker.upper()
        for row in rows:
            if not isinstance(row, dict) or "symbol" not in row:
                raise SchemaError(f"nasdaq {target_date}: row missing 'symbol': {row!r}")
            if str(row["symbol"]).upper() == target:
                return True
        return False
