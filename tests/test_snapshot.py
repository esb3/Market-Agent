from datetime import date, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from chains import ChainCleaningConfig
from sources.base import SchemaError, SourceUnavailable
from snapshot import (
    IVSnapshot,
    _pick_target_expiration,
    _snapshot_one,
    open_store,
    read_history,
    run_daily_snapshot,
    sample_size,
    write_snapshot,
)

CONFIG = ChainCleaningConfig(max_spread_pct_of_mid=0.15, min_surviving_contracts=4, iv_sanity_bounds=(0.05, 3.0))


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


def test_write_and_read_history_roundtrip(tmp_path):
    conn = open_store(tmp_path / "iv.sqlite3")
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 1), 0.24, survived_count=10))
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 2), 0.26, survived_count=12))
    history = read_history(conn, "AAPL", before=date(2026, 9, 3))
    assert history == pytest.approx([0.24, 0.26])


def test_write_snapshot_is_idempotent_per_ticker_date(tmp_path):
    conn = open_store(tmp_path / "iv.sqlite3")
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 1), 0.24, survived_count=10))
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 1), 0.30, survived_count=15))  # re-run same day
    history = read_history(conn, "AAPL", before=date(2026, 9, 2))
    assert history == pytest.approx([0.30])  # overwritten, not duplicated


def test_read_history_excludes_suppressed_days(tmp_path):
    conn = open_store(tmp_path / "iv.sqlite3")
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 1), None, survived_count=1, reason="too few contracts"))
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 2), 0.26, survived_count=12))
    history = read_history(conn, "AAPL", before=date(2026, 9, 3))
    assert history == pytest.approx([0.26])


def test_read_history_excludes_on_or_after_before_date(tmp_path):
    conn = open_store(tmp_path / "iv.sqlite3")
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 5), 0.24, survived_count=10))
    history = read_history(conn, "AAPL", before=date(2026, 9, 5))
    assert history == []


def test_read_history_scoped_by_ticker(tmp_path):
    conn = open_store(tmp_path / "iv.sqlite3")
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 1), 0.24, survived_count=10))
    write_snapshot(conn, IVSnapshot("SPY", date(2026, 9, 1), 0.18, survived_count=10))
    assert read_history(conn, "AAPL", before=date(2026, 9, 2)) == pytest.approx([0.24])
    assert read_history(conn, "SPY", before=date(2026, 9, 2)) == pytest.approx([0.18])


def test_read_history_respects_limit_days(tmp_path):
    conn = open_store(tmp_path / "iv.sqlite3")
    for i in range(5):
        write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 1) + timedelta(days=i), 0.20 + i * 0.01, survived_count=10))
    history = read_history(conn, "AAPL", before=date(2026, 9, 10), limit_days=2)
    assert history == pytest.approx([0.23, 0.24])  # most recent 2, oldest-first


def test_sample_size_matches_history_length(tmp_path):
    conn = open_store(tmp_path / "iv.sqlite3")
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 1), 0.24, survived_count=10))
    write_snapshot(conn, IVSnapshot("AAPL", date(2026, 9, 2), None, survived_count=1, reason="thin"))
    assert sample_size(conn, "AAPL", before=date(2026, 9, 3)) == 1


# ---------------------------------------------------------------------------
# _pick_target_expiration
# ---------------------------------------------------------------------------


def test_pick_target_expiration_closest_to_30_dte():
    as_of = date(2026, 9, 8)
    expirations = [as_of + timedelta(days=d) for d in (7, 14, 28, 35, 60)]
    assert _pick_target_expiration(expirations, as_of) == as_of + timedelta(days=28)


# ---------------------------------------------------------------------------
# _snapshot_one / run_daily_snapshot
# ---------------------------------------------------------------------------


def _mock_yf_client(spot=100.0, iv=0.24):
    client = Mock()
    as_of = date(2026, 9, 8)
    client.list_expirations.return_value = [as_of + timedelta(days=30)]

    def contract_row(strike):
        return {
            "strike": strike, "bid": 1.0, "ask": 1.05, "lastPrice": 1.02,
            "volume": 50, "openInterest": 100, "impliedVolatility": iv, "inTheMoney": False,
            "contractSymbol": f"X{strike}",
        }

    frame = pd.DataFrame([contract_row(95), contract_row(100), contract_row(105)])
    client.option_chain.return_value = Mock(calls=frame, puts=frame)

    bars_frame = pd.DataFrame({"Close": [spot]}, index=pd.bdate_range("2026-09-08", periods=1))
    client.daily_bars.return_value = Mock(frame=bars_frame)
    return client


def test_snapshot_one_happy_path():
    client = _mock_yf_client(spot=100.0, iv=0.24)
    result = _snapshot_one("AAPL", client, CONFIG, as_of=date(2026, 9, 8))
    assert result.atm_iv == pytest.approx(0.24)
    assert result.reason is None


def test_snapshot_one_records_source_error_as_unavailable():
    client = Mock()
    client.list_expirations.side_effect = SourceUnavailable("rate limited")
    result = _snapshot_one("AAPL", client, CONFIG, as_of=date(2026, 9, 8))
    assert result.atm_iv is None
    assert "snapshot failed" in result.reason


def test_snapshot_one_schema_error_recorded_not_raised():
    client = Mock()
    client.list_expirations.side_effect = SchemaError("options list malformed")
    result = _snapshot_one("AAPL", client, CONFIG, as_of=date(2026, 9, 8))
    assert result.atm_iv is None
    assert "snapshot failed" in result.reason


def test_run_daily_snapshot_writes_all_tickers_and_continues_past_failure(tmp_path):
    good_client = _mock_yf_client(spot=100.0, iv=0.24)
    bad_ticker_calls = []

    def list_expirations_side_effect(ticker):
        if ticker == "BROKEN":
            raise SourceUnavailable("down")
        return [date(2026, 9, 8) + timedelta(days=30)]

    good_client.list_expirations.side_effect = list_expirations_side_effect

    store_path = tmp_path / "iv.sqlite3"
    results = run_daily_snapshot(
        tickers=["AAPL", "BROKEN", "SPY"],
        yf_client=good_client,
        chain_config=CONFIG,
        store_path=store_path,
        inter_request_delay_seconds=0,
        as_of=date(2026, 9, 8),
    )

    by_ticker = {r.ticker: r for r in results}
    assert by_ticker["AAPL"].atm_iv == pytest.approx(0.24)
    assert by_ticker["SPY"].atm_iv == pytest.approx(0.24)
    assert by_ticker["BROKEN"].atm_iv is None

    conn = open_store(store_path)
    assert sample_size(conn, "AAPL", before=date(2026, 9, 9)) == 1
    assert sample_size(conn, "BROKEN", before=date(2026, 9, 9)) == 0
