"""Yahoo Finance client — daily bars, option chains, earnings dates.

Wraps `yfinance_cache` (an on-disk cache in front of `yfinance`) rather
than calling yfinance directly, per the operational rules this project
runs under: yfinance scrapes an undocumented Yahoo endpoint that changes
without notice and rate-limits/blocks aggressively on request volume.

    - Never re-fetch what's already on disk. yfinance_cache keys its
      cache by ticker under the directory set via `configure_cache()`;
      `max_age` controls when a cached read is considered stale enough
      to bother refetching.
    - Requests are issued one ticker at a time by the caller (main.py),
      with a configurable delay between them. This module never fans
      out or parallelizes — that's what gets an IP blocked.
    - `yfinance.exceptions.YFRateLimitError` is caught explicitly and
      retried with backoff. If it persists past max_retries, one more
      attempt is made forcing yfinance_cache to serve whatever is on
      disk however old (`stale_fallback`), instead of failing the
      ticker outright — the caller sees this via `DailyBars.stale`.
    - Every response is validated against the exact schema confirmed
      against the installed yfinance/yfinance_cache versions (see the
      `_validate_*` functions below). An unexpected shape raises
      SchemaError and is never treated as "no data" / zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Protocol

import pandas as pd

from sources.base import SchemaError, SourceUnavailable, with_retry

try:
    import yfinance_cache as yfc
    from yfinance.exceptions import YFRateLimitError
except ImportError:  # pragma: no cover - tests inject a fake client instead
    yfc = None
    YFRateLimitError = Exception


# Confirmed against yfinance 1.7.0 / yfinance_cache's _options2df() and
# PriceHistory.history(): the exact columns Yahoo's endpoints return.
DAILY_BAR_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
OPTION_CHAIN_COLUMNS = [
    "contractSymbol", "strike", "lastPrice", "bid", "ask",
    "volume", "openInterest", "impliedVolatility", "inTheMoney",
]
EARNINGS_DATE_COLUMNS_ANY_OF = ("EPS Estimate", "Reported EPS", "Surprise(%)")


class TickerLike(Protocol):
    def history(self, **kwargs) -> pd.DataFrame: ...
    def option_chain(self, expiry_date, max_age=None): ...
    def get_earnings_dates(self, start): ...
    options: tuple


def configure_cache(path: Path) -> None:
    """Point yfinance_cache's on-disk store at `path`. Call once at
    startup, before any YFinanceClient is used — this is what makes
    "never re-fetch data already on disk" actually true."""
    if yfc is None:
        raise SourceUnavailable("yfinance_cache is not installed")
    path.mkdir(parents=True, exist_ok=True)
    yfc.yfc_cache_manager.SetCacheDirpath(str(path))


@dataclass(frozen=True)
class YFinanceConfig:
    max_retries: int = 3
    backoff_base_seconds: float = 2.0
    # On persistent rate-limiting, fall back to whatever is cached,
    # however old, rather than failing the ticker outright.
    stale_fallback: bool = True


@dataclass(frozen=True)
class DailyBars:
    frame: pd.DataFrame  # columns: Open, High, Low, Close, Volume; DatetimeIndex
    stale: bool  # True if served from cache after exhausting live retries


def _default_ticker_factory(ticker: str) -> TickerLike:
    if yfc is None:
        raise SourceUnavailable("yfinance_cache is not installed")
    return yfc.Ticker(ticker)


class YFinanceClient:
    """`ticker_factory` is injectable so tests exercise retry/backoff,
    the rate-limit stale-fallback, and schema validation against a fake
    ticker — never against real yfinance or the network."""

    def __init__(
        self,
        config: YFinanceConfig,
        ticker_factory: Callable[[str], TickerLike] = _default_ticker_factory,
    ):
        self._config = config
        self._ticker_factory = ticker_factory

    def daily_bars(self, ticker: str, lookback_days: int = 260) -> DailyBars:
        t = self._ticker_factory(ticker)
        # Pad the fetch window for weekends/holidays so `lookback_days`
        # trading days actually come back.
        start = date.today() - timedelta(days=int(lookback_days * 1.6) + 10)

        def call(max_age=None) -> pd.DataFrame:
            kwargs = {"start": start.isoformat(), "interval": "1d"}
            if max_age is not None:
                kwargs["max_age"] = max_age
            try:
                return t.history(**kwargs)
            except YFRateLimitError as exc:
                raise SourceUnavailable(f"yfinance {ticker}: rate limited: {exc}") from exc

        try:
            frame = with_retry(
                call,
                max_retries=self._config.max_retries,
                backoff_base=self._config.backoff_base_seconds,
                retry_on=(SourceUnavailable,),
            )
            stale = False
        except SourceUnavailable:
            if not self._config.stale_fallback:
                raise
            frame = call(max_age=timedelta(days=36500))  # cache-only, any age
            stale = True

        _validate_daily_bars(frame, ticker)
        return DailyBars(frame=frame.tail(lookback_days), stale=stale)

    def list_expirations(self, ticker: str) -> List[date]:
        t = self._ticker_factory(ticker)

        def call():
            try:
                return t.options
            except YFRateLimitError as exc:
                raise SourceUnavailable(f"yfinance {ticker}: rate limited: {exc}") from exc

        raw = with_retry(
            call,
            max_retries=self._config.max_retries,
            backoff_base=self._config.backoff_base_seconds,
            retry_on=(SourceUnavailable,),
        )
        return _validate_expirations(raw, ticker)

    def option_chain(self, ticker: str, expiry: date):
        t = self._ticker_factory(ticker)

        def call():
            try:
                return t.option_chain(expiry.isoformat())
            except YFRateLimitError as exc:
                raise SourceUnavailable(f"yfinance {ticker} {expiry}: rate limited: {exc}") from exc

        chain = with_retry(
            call,
            max_retries=self._config.max_retries,
            backoff_base=self._config.backoff_base_seconds,
            retry_on=(SourceUnavailable,),
        )
        _validate_option_chain(chain, ticker, expiry)
        return chain

    def next_earnings_date(self, ticker: str, as_of: date) -> Optional[date]:
        t = self._ticker_factory(ticker)

        def call():
            try:
                return t.get_earnings_dates(as_of.isoformat())
            except YFRateLimitError as exc:
                raise SourceUnavailable(f"yfinance {ticker}: rate limited: {exc}") from exc

        frame = with_retry(
            call,
            max_retries=self._config.max_retries,
            backoff_base=self._config.backoff_base_seconds,
            retry_on=(SourceUnavailable,),
        )
        return _next_earnings_date_from_frame(frame, ticker, as_of)


# ---------------------------------------------------------------------------
# Schema validation — pure functions, unit-tested without yfinance/network.
# ---------------------------------------------------------------------------


def _validate_daily_bars(frame: pd.DataFrame, ticker: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError(f"yfinance {ticker}: history() returned {type(frame)!r}, not a DataFrame")
    if frame.empty:
        raise SchemaError(f"yfinance {ticker}: history() returned an empty frame")
    missing = [c for c in DAILY_BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"yfinance {ticker}: history() missing columns {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise SchemaError(f"yfinance {ticker}: history() index is not a DatetimeIndex")


def _validate_expirations(raw, ticker: str) -> List[date]:
    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise SchemaError(f"yfinance {ticker}: options list is empty or malformed: {raw!r}")
    try:
        return sorted(date.fromisoformat(str(d)) for d in raw)
    except ValueError as exc:
        raise SchemaError(f"yfinance {ticker}: unparseable expiration date in {raw!r}") from exc


def _validate_option_chain(chain, ticker: str, expiry: date) -> None:
    calls, puts = getattr(chain, "calls", None), getattr(chain, "puts", None)
    if calls is None or puts is None:
        raise SchemaError(f"yfinance {ticker} {expiry}: option_chain() returned no calls/puts")
    for label, frame in (("calls", calls), ("puts", puts)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise SchemaError(f"yfinance {ticker} {expiry}: {label} frame is empty or not a DataFrame")
        missing = [c for c in OPTION_CHAIN_COLUMNS if c not in frame.columns]
        if missing:
            raise SchemaError(f"yfinance {ticker} {expiry}: {label} frame missing columns {missing}")


def _next_earnings_date_from_frame(frame: pd.DataFrame, ticker: str, as_of: date) -> Optional[date]:
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError(f"yfinance {ticker}: get_earnings_dates() returned {type(frame)!r}")
    if frame.empty:
        return None
    if not any(c in frame.columns for c in EARNINGS_DATE_COLUMNS_ANY_OF):
        raise SchemaError(
            f"yfinance {ticker}: earnings dates frame missing expected columns, got {list(frame.columns)}"
        )
    try:
        idx = pd.to_datetime(frame.index)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"yfinance {ticker}: earnings dates index is not date-like") from exc
    upcoming = sorted(d for d in idx.date if d >= as_of)
    return upcoming[0] if upcoming else None
