"""Cross-source reconciliation.

Every free source in this stack is unofficial or partial, so accuracy
comes from comparing sources against each other rather than trusting
any single feed:

    - Prices: Yahoo vs Stooq. A disagreement beyond config tolerance
      usually means Yahoo silently applied a wrong split/dividend
      adjustment — that bar is flagged as disputed and derived metrics
      must not be computed from it (see PriceComparison.trusted_close).

    - Earnings dates: yfinance vs Nasdaq's public calendar. These
      disagree often; per the honesty rule, an unconfirmed date must
      never be silently dropped, only labeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from sources.base import SourceUnavailable
from sources.nasdaq import NasdaqClient


@dataclass(frozen=True)
class PriceComparison:
    ticker: str
    as_of: date
    yahoo_close: Optional[float]
    stooq_close: Optional[float]
    tolerance_pct: float  # e.g. 0.01 for 1%

    @property
    def both_available(self) -> bool:
        return self.yahoo_close is not None and self.stooq_close is not None

    @property
    def disagreement_pct(self) -> Optional[float]:
        if not self.both_available or self.stooq_close == 0:
            return None
        return abs(self.yahoo_close - self.stooq_close) / self.stooq_close * 100

    @property
    def disputed(self) -> bool:
        d = self.disagreement_pct
        return d is not None and d > self.tolerance_pct * 100

    @property
    def trusted_close(self) -> Optional[float]:
        """The close derived metrics may use. None when disputed --
        metrics must skip this bar rather than compute off it. Falls
        back to whichever single source is available when only one
        returned data (a missing source, not a disagreement)."""
        if self.disputed:
            return None
        return self.yahoo_close if self.yahoo_close is not None else self.stooq_close


def compare_closes(
    *,
    ticker: str,
    as_of: date,
    yahoo_close: Optional[float],
    stooq_close: Optional[float],
    tolerance_pct: float,
) -> PriceComparison:
    return PriceComparison(
        ticker=ticker, as_of=as_of, yahoo_close=yahoo_close, stooq_close=stooq_close, tolerance_pct=tolerance_pct
    )


@dataclass(frozen=True)
class EarningsDateComparison:
    ticker: str
    yahoo_date: Optional[date]
    resolved_date: Optional[date]  # the date to actually use in the briefing
    unconfirmed: bool  # True: label this date as unconfirmed in output
    nasdaq_checked: bool  # True if the cross-check actually ran


def reconcile_earnings_date(
    ticker: str,
    yahoo_date: Optional[date],
    nasdaq_client: NasdaqClient,
    search_radius_days: int = 3,
) -> EarningsDateComparison:
    """Cross-check yfinance's earnings date against Nasdaq's calendar
    across a small window of candidate dates -- sources typically
    disagree by a day or two, not by weeks, so this stays a handful of
    requests rather than an unbounded scan.

    If Nasdaq confirms a date earlier than yfinance's within the
    window, use the earlier date (per spec) and mark it unconfirmed.
    If Nasdaq confirms the exact same date, it's fully confirmed. If
    the cross-check finds nothing nearby, or can't run at all (Nasdaq
    unreachable), yfinance's date is still used -- never silently
    dropped -- but always marked unconfirmed.
    """
    if yahoo_date is None:
        return EarningsDateComparison(
            ticker=ticker, yahoo_date=None, resolved_date=None, unconfirmed=False, nasdaq_checked=False
        )

    candidates = [yahoo_date + timedelta(days=offset) for offset in range(-search_radius_days, search_radius_days + 1)]

    try:
        confirmed = [d for d in candidates if nasdaq_client.earnings_date_on(ticker, d)]
    except SourceUnavailable:
        return EarningsDateComparison(
            ticker=ticker, yahoo_date=yahoo_date, resolved_date=yahoo_date, unconfirmed=True, nasdaq_checked=False
        )

    if not confirmed:
        return EarningsDateComparison(
            ticker=ticker, yahoo_date=yahoo_date, resolved_date=yahoo_date, unconfirmed=True, nasdaq_checked=True
        )

    earliest = min(confirmed)
    if earliest == yahoo_date:
        return EarningsDateComparison(
            ticker=ticker, yahoo_date=yahoo_date, resolved_date=yahoo_date, unconfirmed=False, nasdaq_checked=True
        )
    return EarningsDateComparison(
        ticker=ticker, yahoo_date=yahoo_date, resolved_date=earliest, unconfirmed=True, nasdaq_checked=True
    )
