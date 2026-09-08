from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import metrics as m

FIXTURE = pd.read_csv(Path(__file__).parent / "fixtures" / "AAPL_sample.csv", parse_dates=["Date"])


# ---------------------------------------------------------------------------
# Hand-computable cases: small, exact series so expected values are obvious.
# ---------------------------------------------------------------------------


def test_sma_exact():
    closes = pd.Series([10, 20, 30, 40, 50])
    assert m.sma(closes, 3) == pytest.approx((30 + 40 + 50) / 3)


def test_sma_insufficient_history():
    closes = pd.Series([10, 20])
    assert m.sma(closes, 5) is None


def test_distance_from_dma_pct():
    assert m.distance_from_dma_pct(110, 100) == pytest.approx(10.0)
    assert m.distance_from_dma_pct(90, 100) == pytest.approx(-10.0)
    assert m.distance_from_dma_pct(100, None) is None
    assert m.distance_from_dma_pct(100, 0) is None


def test_atr_flat_series_is_zero():
    # No day-to-day movement and high==low==close -> true range is 0 every day.
    closes = pd.Series([100.0] * 20)
    result = m.atr(closes, closes, closes, period=14)
    assert result == pytest.approx(0.0)


def test_atr_known_value():
    # 3 days, period=2: TR1 = max(high-low, |high-prevclose|, |low-prevclose|)
    highs = pd.Series([102.0, 104.0, 103.0])
    lows = pd.Series([98.0, 100.0, 99.0])
    closes = pd.Series([100.0, 103.0, 101.0])
    # day2 TR = max(104-100, |104-100|, |100-100|) = 4
    # day3 TR = max(103-99, |103-103|, |99-103|) = 4
    result = m.atr(highs, lows, closes, period=2)
    assert result == pytest.approx(4.0)


def test_atr_insufficient_history():
    closes = pd.Series([100.0, 101.0])
    assert m.atr(closes, closes, closes, period=14) is None


def test_overnight_gap_pct_and_atr_units():
    gap = m.overnight_gap(prev_close=100.0, today_open=102.0, atr_value=1.0)
    assert gap.pct == pytest.approx(2.0)
    assert gap.atr_units == pytest.approx(2.0)


def test_overnight_gap_no_atr():
    gap = m.overnight_gap(prev_close=100.0, today_open=99.0, atr_value=None)
    assert gap.pct == pytest.approx(-1.0)
    assert gap.atr_units is None


def test_range_percentile_at_high_and_low():
    closes = pd.Series([10, 12, 14, 16, 18, 20])
    at_high = m.range_percentile(closes, window=6)
    assert at_high.value == pytest.approx(100.0)
    assert at_high.sample_size == 6

    closes_low = pd.Series([20, 18, 16, 14, 12, 10])
    at_low = m.range_percentile(closes_low, window=6)
    assert at_low.value == pytest.approx(0.0)


def test_range_percentile_flat_series_is_50():
    closes = pd.Series([100.0] * 10)
    result = m.range_percentile(closes, window=10)
    assert result.value == pytest.approx(50.0)


def test_range_percentile_thin_sample_still_computes():
    closes = pd.Series([10, 20, 30])
    result = m.range_percentile(closes, window=30)
    assert result.sample_size == 3
    assert result.min_sample == 30
    assert not result.sufficient
    assert result.value == pytest.approx(100.0)


def test_realized_vol_constant_returns_is_zero():
    # Constant daily % return -> zero volatility of returns.
    closes = pd.Series([100 * (1.01 ** i) for i in range(25)])
    result = m.realized_vol(closes, window=20)
    assert result.value == pytest.approx(0.0, abs=1e-6)
    assert result.sample_size == 20


def test_realized_vol_insufficient_history():
    closes = pd.Series([100.0])
    result = m.realized_vol(closes, window=20)
    assert result.value is None
    assert result.sample_size == 0


def test_sampled_rank_and_percentile():
    history = [10, 20, 30, 40, 50]
    rank = m.sampled_rank(history, current=30, min_sample=5)
    # min-max over history+current == history itself here: (30-10)/(50-10)
    assert rank.value == pytest.approx(50.0)
    assert rank.sample_size == 5
    assert rank.sufficient

    pct = m.sampled_percentile(history, current=30, min_sample=5)
    # 3 of 5 historical values <= 30
    assert pct.value == pytest.approx(60.0)


def test_sampled_rank_empty_history():
    result = m.sampled_rank([], current=10, min_sample=5)
    assert result.value is None
    assert result.sample_size == 0


def test_sampled_rank_ignores_none_entries():
    history = [10, None, 30, None, 50]
    result = m.sampled_rank(history, current=30, min_sample=3)
    assert result.sample_size == 3


def test_sampled_value_thin_and_label():
    # Matches config.yaml's iv_history defaults: suppress below 20 days,
    # no longer "thin" once the sample reaches 126 days.
    thin = m.SampledValue(value=82.0, sample_size=41, min_sample=20, mature_sample=126)
    assert thin.sufficient
    assert thin.thin
    assert thin.label() == "41-day sample — thin"

    mature = m.SampledValue(value=82.0, sample_size=126, min_sample=20, mature_sample=126)
    assert mature.sufficient
    assert not mature.thin
    assert mature.label() == "126-day sample"

    unavailable = m.SampledValue(value=None, sample_size=0, min_sample=20, mature_sample=126)
    assert not unavailable.sufficient
    assert unavailable.label() == "unavailable"

    # Default mature_sample (2x min_sample) when the caller doesn't need
    # two separate thresholds, e.g. realized_vol / range_percentile.
    default_thin = m.SampledValue(value=15.0, sample_size=25, min_sample=20)
    assert default_thin.thin
    default_mature = m.SampledValue(value=15.0, sample_size=45, min_sample=20)
    assert not default_mature.thin


def test_iv_minus_rv():
    assert m.iv_minus_rv(30.0, 22.0) == pytest.approx(8.0)
    assert m.iv_minus_rv(None, 22.0) is None
    assert m.iv_minus_rv(30.0, None) is None


def test_term_structure_slope_contango():
    # back IV higher than front, 30 days further out -> positive slope of
    # exactly the IV point difference (scaled to a 30-day step).
    slope = m.term_structure_slope(front_iv=20.0, back_iv=24.0, front_dte=30, back_dte=60)
    assert slope == pytest.approx(4.0)


def test_term_structure_slope_inverted():
    slope = m.term_structure_slope(front_iv=30.0, back_iv=24.0, front_dte=30, back_dte=60)
    assert slope == pytest.approx(-6.0)


def test_term_structure_slope_same_dte_is_none():
    assert m.term_structure_slope(20.0, 24.0, 30, 30) is None


def test_expected_move():
    move = m.expected_move(atm_iv=0.20, spot=100.0, dte=30)
    expected_pct = 0.20 * np.sqrt(30 / 365) * 100
    assert move.pct == pytest.approx(expected_pct)
    assert move.dollars == pytest.approx(100.0 * 0.20 * np.sqrt(30 / 365))


def test_expected_move_missing_inputs():
    assert m.expected_move(None, 100.0, 30) is None
    assert m.expected_move(0.2, 100.0, 0) is None


def test_vix_term_structure_contango():
    ts = m.vix_term_structure(vix9d=14.0, vix=15.0, vix3m=16.0)
    assert ts.shape == "contango"
    assert ts.inverted is False


def test_vix_term_structure_backwardation():
    ts = m.vix_term_structure(vix9d=28.0, vix=24.0, vix3m=20.0)
    assert ts.shape == "backwardation"
    assert ts.inverted is True


def test_vix_term_structure_mixed_is_inverted():
    ts = m.vix_term_structure(vix9d=16.0, vix=14.0, vix3m=18.0)
    assert ts.shape == "mixed"
    assert ts.inverted is True


def test_vix_term_structure_flat():
    ts = m.vix_term_structure(vix9d=15.0, vix=15.0, vix3m=15.0)
    assert ts.shape == "flat"
    assert ts.inverted is False


def test_days_to():
    assert m.days_to(date(2026, 9, 15), date(2026, 9, 8)) == 7
    assert m.days_to(None, date(2026, 9, 8)) is None


# ---------------------------------------------------------------------------
# Fixture-driven: sanity checks against a full synthetic 60-day OHLC series
# (deterministic, seeded — see tests/fixtures/AAPL_sample.csv).
# ---------------------------------------------------------------------------


def test_fixture_shape():
    assert len(FIXTURE) == 60
    assert list(FIXTURE.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]


def test_fixture_sma_matches_pandas_rolling():
    closes = FIXTURE["Close"]
    for period in (20, 50):
        expected = closes.rolling(period).mean().iloc[-1]
        assert m.sma(closes, period) == pytest.approx(expected)


def test_fixture_atr_is_positive_and_reasonable():
    result = m.atr(FIXTURE["High"], FIXTURE["Low"], FIXTURE["Close"], period=14)
    assert result is not None
    # True range should be a small fraction of a ~$150 stock's price.
    assert 0 < result < FIXTURE["Close"].iloc[-1] * 0.2


def test_fixture_realized_vol_in_plausible_range():
    result = m.realized_vol(FIXTURE["Close"], window=20)
    assert result.sample_size == 20
    # Series was generated with ~1.2% daily vol -> annualized should be
    # in the double digits but nowhere near a market-crash number.
    assert 5.0 < result.value < 60.0


def test_fixture_range_percentile_between_0_and_100():
    result = m.range_percentile(FIXTURE["Close"], window=30)
    assert 0.0 <= result.value <= 100.0
    assert result.sample_size == 30


def test_fixture_gap_uses_prior_close_and_atr():
    closes = FIXTURE["Close"]
    prev_close = float(closes.iloc[-2])
    today_open = float(FIXTURE["Open"].iloc[-1])
    atr_value = m.atr(FIXTURE["High"], FIXTURE["Low"], closes, period=14)
    gap = m.overnight_gap(prev_close, today_open, atr_value)
    assert gap.pct == pytest.approx((today_open - prev_close) / prev_close * 100)
    assert gap.atr_units == pytest.approx((today_open - prev_close) / atr_value)
