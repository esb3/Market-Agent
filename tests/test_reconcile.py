from datetime import date
from unittest.mock import Mock

import pytest

from reconcile import compare_closes, reconcile_earnings_date
from sources.base import SourceUnavailable


# ---------------------------------------------------------------------------
# Price cross-check
# ---------------------------------------------------------------------------


def test_prices_agree_within_tolerance():
    cmp = compare_closes(ticker="AAPL", as_of=date(2026, 9, 5), yahoo_close=100.0, stooq_close=100.05, tolerance_pct=0.01)
    assert not cmp.disputed
    assert cmp.trusted_close == pytest.approx(100.0)


def test_prices_disagree_beyond_tolerance():
    cmp = compare_closes(ticker="AAPL", as_of=date(2026, 9, 5), yahoo_close=100.0, stooq_close=95.0, tolerance_pct=0.01)
    assert cmp.disputed
    assert cmp.trusted_close is None
    assert cmp.disagreement_pct == pytest.approx(5.263, abs=0.01)


def test_prices_exactly_at_tolerance_boundary_not_disputed():
    # disagreement is exactly 1% and tolerance is 1% -> boundary is NOT disputed (> not >=)
    cmp = compare_closes(ticker="AAPL", as_of=date(2026, 9, 5), yahoo_close=101.0, stooq_close=100.0, tolerance_pct=0.01)
    assert not cmp.disputed


def test_only_yahoo_available_uses_yahoo():
    cmp = compare_closes(ticker="AAPL", as_of=date(2026, 9, 5), yahoo_close=100.0, stooq_close=None, tolerance_pct=0.01)
    assert not cmp.disputed
    assert cmp.trusted_close == pytest.approx(100.0)


def test_only_stooq_available_uses_stooq():
    cmp = compare_closes(ticker="AAPL", as_of=date(2026, 9, 5), yahoo_close=None, stooq_close=98.0, tolerance_pct=0.01)
    assert not cmp.disputed
    assert cmp.trusted_close == pytest.approx(98.0)


def test_neither_available():
    cmp = compare_closes(ticker="AAPL", as_of=date(2026, 9, 5), yahoo_close=None, stooq_close=None, tolerance_pct=0.01)
    assert not cmp.disputed
    assert cmp.trusted_close is None
    assert cmp.disagreement_pct is None


# ---------------------------------------------------------------------------
# Earnings date reconciliation
# ---------------------------------------------------------------------------


def test_no_yahoo_date_short_circuits():
    nasdaq = Mock()
    result = reconcile_earnings_date("AAPL", None, nasdaq)
    assert result.resolved_date is None
    assert not result.unconfirmed
    assert not result.nasdaq_checked
    nasdaq.earnings_date_on.assert_not_called()


def test_nasdaq_confirms_exact_date():
    nasdaq = Mock()
    nasdaq.earnings_date_on.side_effect = lambda ticker, d: d == date(2026, 9, 24)
    result = reconcile_earnings_date("AAPL", date(2026, 9, 24), nasdaq, search_radius_days=3)
    assert result.resolved_date == date(2026, 9, 24)
    assert not result.unconfirmed
    assert result.nasdaq_checked


def test_nasdaq_confirms_earlier_date_wins():
    nasdaq = Mock()
    nasdaq.earnings_date_on.side_effect = lambda ticker, d: d == date(2026, 9, 22)
    result = reconcile_earnings_date("AAPL", date(2026, 9, 24), nasdaq, search_radius_days=3)
    assert result.resolved_date == date(2026, 9, 22)
    assert result.unconfirmed  # sources disagreed -- still flagged


def test_nasdaq_finds_nothing_nearby_keeps_yahoo_date_unconfirmed():
    nasdaq = Mock()
    nasdaq.earnings_date_on.return_value = False
    result = reconcile_earnings_date("AAPL", date(2026, 9, 24), nasdaq, search_radius_days=2)
    assert result.resolved_date == date(2026, 9, 24)
    assert result.unconfirmed
    assert result.nasdaq_checked


def test_nasdaq_unavailable_keeps_yahoo_date_unconfirmed_not_checked():
    nasdaq = Mock()
    nasdaq.earnings_date_on.side_effect = SourceUnavailable("nasdaq down")
    result = reconcile_earnings_date("AAPL", date(2026, 9, 24), nasdaq)
    assert result.resolved_date == date(2026, 9, 24)
    assert result.unconfirmed
    assert not result.nasdaq_checked


def test_search_radius_is_respected():
    nasdaq = Mock()
    # Only confirms a date 5 days later than yahoo's -- outside a radius of 2.
    nasdaq.earnings_date_on.side_effect = lambda ticker, d: d == date(2026, 9, 29)
    result = reconcile_earnings_date("AAPL", date(2026, 9, 24), nasdaq, search_radius_days=2)
    assert result.resolved_date == date(2026, 9, 24)
    assert result.unconfirmed
    # confirm the call count matches the window: 2*radius + 1
    assert nasdaq.earnings_date_on.call_count == 5
