import pandas as pd
import pytest

from chains import (
    ChainCleaningConfig,
    atm_iv_for_ticker,
    clean_contracts,
    implied_vol_by_strike,
    interpolate_atm_iv,
    mid_price,
    nearest_strike_straddle,
)

CONFIG = ChainCleaningConfig(max_spread_pct_of_mid=0.15, min_surviving_contracts=4, iv_sanity_bounds=(0.05, 3.0))


def contract(strike, bid, ask, volume, oi, iv, last=None):
    return {
        "contractSymbol": f"X{strike}",
        "strike": strike,
        "lastPrice": last if last is not None else (bid + ask) / 2,
        "bid": bid,
        "ask": ask,
        "volume": volume,
        "openInterest": oi,
        "impliedVolatility": iv,
        "inTheMoney": False,
    }


def frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# mid_price / clean_contracts
# ---------------------------------------------------------------------------


def test_mid_price():
    bid = pd.Series([1.0, 2.0])
    ask = pd.Series([1.2, 2.4])
    result = mid_price(bid, ask)
    assert result.tolist() == pytest.approx([1.1, 2.2])


def test_clean_contracts_drops_illiquid_zero_volume_and_oi():
    rows = frame([
        contract(100, 1.0, 1.1, volume=0, oi=0, iv=0.2),   # drop: illiquid
        contract(105, 1.0, 1.1, volume=10, oi=0, iv=0.2),  # keep: has volume
        contract(110, 1.0, 1.1, volume=0, oi=50, iv=0.2),  # keep: has OI
    ])
    result = clean_contracts(rows, CONFIG)
    assert result.total == 3
    assert result.survived == 2
    assert result.dropped_illiquid == 1


def test_clean_contracts_drops_zero_bid():
    rows = frame([
        contract(100, 0.0, 1.1, volume=10, oi=10, iv=0.2),
        contract(105, 1.0, 1.1, volume=10, oi=10, iv=0.2),
    ])
    result = clean_contracts(rows, CONFIG)
    assert result.survived == 1
    assert result.dropped_zero_bid == 1


def test_clean_contracts_drops_wide_spread():
    # mid = 1.0, spread = 0.5 -> 50% of mid, above 15% threshold
    rows = frame([
        contract(100, 0.75, 1.25, volume=10, oi=10, iv=0.2),
        contract(105, 0.98, 1.02, volume=10, oi=10, iv=0.2),  # 4% spread, keep
    ])
    result = clean_contracts(rows, CONFIG)
    assert result.survived == 1
    assert result.dropped_wide_spread == 1


def test_clean_contracts_empty_frame():
    result = clean_contracts(frame([]), CONFIG)
    assert result.total == 0
    assert result.survived == 0


def test_clean_contracts_adds_mid_column():
    rows = frame([contract(100, 1.05, 1.15, volume=10, oi=10, iv=0.2)])  # ~9% spread, survives
    result = clean_contracts(rows, CONFIG)
    assert result.frame.iloc[0]["mid"] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# implied_vol_by_strike / interpolate_atm_iv
# ---------------------------------------------------------------------------


def test_implied_vol_by_strike_averages_call_and_put():
    calls = frame([contract(100, 1.0, 1.1, 10, 10, iv=0.20)])
    puts = frame([contract(100, 1.0, 1.1, 10, 10, iv=0.24)])
    iv = implied_vol_by_strike(calls, puts)
    assert iv.loc[100] == pytest.approx(0.22)


def test_implied_vol_by_strike_falls_back_to_one_side():
    calls = frame([contract(100, 1.0, 1.1, 10, 10, iv=0.20)])
    puts = frame([])
    iv = implied_vol_by_strike(calls, puts)
    assert iv.loc[100] == pytest.approx(0.20)


def test_interpolate_atm_iv_between_two_strikes():
    iv_by_strike = pd.Series({95.0: 0.20, 100.0: 0.24, 105.0: 0.28})
    # spot exactly halfway between 100 and 105
    result = interpolate_atm_iv(iv_by_strike, spot=102.5)
    assert result == pytest.approx(0.26)


def test_interpolate_atm_iv_exact_strike_match():
    iv_by_strike = pd.Series({95.0: 0.20, 100.0: 0.24})
    assert interpolate_atm_iv(iv_by_strike, spot=100.0) == pytest.approx(0.24)


def test_interpolate_atm_iv_out_of_range_returns_none():
    iv_by_strike = pd.Series({95.0: 0.20, 100.0: 0.24})
    assert interpolate_atm_iv(iv_by_strike, spot=200.0) is None
    assert interpolate_atm_iv(iv_by_strike, spot=10.0) is None


def test_interpolate_atm_iv_needs_two_strikes():
    assert interpolate_atm_iv(pd.Series({100.0: 0.24}), spot=100.0) is None
    assert interpolate_atm_iv(pd.Series(dtype=float), spot=100.0) is None


# ---------------------------------------------------------------------------
# atm_iv_for_ticker (orchestration)
# ---------------------------------------------------------------------------


def _liquid_chain(strikes_ivs):
    calls = frame([contract(k, 1.0, 1.05, 50, 100, iv=iv) for k, iv in strikes_ivs])
    puts = frame([contract(k, 1.0, 1.05, 50, 100, iv=iv) for k, iv in strikes_ivs])
    return calls, puts


def test_atm_iv_for_ticker_happy_path():
    calls, puts = _liquid_chain([(95, 0.20), (100, 0.24), (105, 0.28)])
    result = atm_iv_for_ticker(calls, puts, spot=100.0, config=CONFIG)
    assert result.value == pytest.approx(0.24)
    assert result.reason is None
    assert result.survived_count == 6


def test_atm_iv_for_ticker_suppressed_below_min_contracts():
    calls, puts = _liquid_chain([(100, 0.24)])  # only 2 total survivors < min 4
    result = atm_iv_for_ticker(calls, puts, spot=100.0, config=CONFIG)
    assert result.value is None
    assert "min 4" in result.reason


def test_atm_iv_for_ticker_suppressed_outside_sanity_bounds():
    calls, puts = _liquid_chain([(95, 4.0), (100, 4.5), (105, 5.0)])  # way above 3.0 bound
    result = atm_iv_for_ticker(calls, puts, spot=100.0, config=CONFIG)
    assert result.value is None
    assert "sanity bounds" in result.reason


def test_atm_iv_for_ticker_suppressed_spot_out_of_range():
    calls, puts = _liquid_chain([(200, 0.24), (210, 0.24), (220, 0.24)])
    result = atm_iv_for_ticker(calls, puts, spot=100.0, config=CONFIG)
    assert result.value is None
    assert "outside the available strike range" in result.reason


def test_atm_iv_for_ticker_min_contracts_counts_after_filtering_not_before():
    # 6 raw contracts but only 3 survive cleaning (below min 4)
    rows = [
        contract(95, 1.0, 1.05, 50, 100, iv=0.2),
        contract(100, 1.0, 1.05, 50, 100, iv=0.24),
        contract(105, 0.0, 1.05, 50, 100, iv=0.28),  # zero bid, dropped
    ]
    calls = frame(rows)
    puts = frame(rows)
    result = atm_iv_for_ticker(calls, puts, spot=100.0, config=CONFIG)
    assert result.survived_count == 4  # 2 survive per side x2
    assert result.value is not None  # exactly at the min, not suppressed


# ---------------------------------------------------------------------------
# nearest_strike_straddle
# ---------------------------------------------------------------------------


def test_nearest_strike_straddle_picks_closest_common_strike():
    calls = clean_contracts(frame([contract(95, 4.0, 4.2, 10, 10, iv=0.2), contract(100, 2.0, 2.2, 10, 10, iv=0.2)]), CONFIG).frame
    puts = clean_contracts(frame([contract(95, 1.0, 1.2, 10, 10, iv=0.2), contract(100, 3.0, 3.2, 10, 10, iv=0.2)]), CONFIG).frame
    quote = nearest_strike_straddle(calls, puts, spot=101.0)
    assert quote.strike == 100
    assert quote.call_mid == pytest.approx(2.1)
    assert quote.put_mid == pytest.approx(3.1)
    assert quote.dollars == pytest.approx(5.2)
    assert quote.pct_of_spot(100.0) == pytest.approx(5.2)


def test_nearest_strike_straddle_no_common_strikes_returns_none():
    calls = clean_contracts(frame([contract(95, 4.0, 4.2, 10, 10, iv=0.2)]), CONFIG).frame
    puts = clean_contracts(frame([contract(105, 1.0, 1.2, 10, 10, iv=0.2)]), CONFIG).frame
    assert nearest_strike_straddle(calls, puts, spot=100.0) is None
