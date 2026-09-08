import json
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import main
import positions as positions_mod
from chains import ChainCleaningConfig
from main import Deps
from sources.base import SchemaError, SourceUnavailable

CONFIG = {
    "tickers": ["AAPL"],
    "requests": {"inter_request_delay_seconds": 0, "timeout_seconds": 1, "max_retries": 1, "backoff_base_seconds": 0.001},
    "underlying": {"atr_period": 14, "dma_periods": [20, 50, 200], "range_window_days": 30, "realized_vol_windows": [20, 30], "notable_gap_atr": 0.75},
    "options": {"max_spread_pct_of_mid": 0.15, "min_surviving_contracts": 4, "iv_sanity_bounds": [0.05, 3.0]},
    "iv_history": {"suppress_below_days": 20, "mature_sample_days": 126},
    "iv_rank_bands": {"low": 25, "high": 75},
    "price_cross_check": {"tolerance_pct": 0.01},
    "vix": {"flag_inversion": True, "percentile_window_days": 252, "percentile_min_sample": 60, "percentile_bands": {"low": 25, "high": 75}},
    "earnings": {"blackout_days": 5, "nasdaq_search_radius_days": 3},
    "positions": {"max_concurrent_positions": 10, "dte_window": [0, 7], "staleness_days": 3},
    "calendar": {"macro_dates": [], "blackout_days": 3},
    "dividends": {"blackout_days": 3},
    "delta_thresholds": {"dma_distance_pct": 3.0, "realized_vol_pts": 5.0, "iv_minus_rv_pts": 5.0, "term_structure_slope_pts": 2.0, "range_percentile_pts": 25.0, "vix_level_pts": 2.0, "expected_move_pct_pts": 2.0},
    "briefing": {"max_characters": 600, "max_flags": 8, "llm_model": "claude-sonnet-4-6", "llm_temperature": 0},
    "logging": {"dir": "logs", "keep_days": 90},
}
CHAIN_CONFIG = ChainCleaningConfig(max_spread_pct_of_mid=0.15, min_surviving_contracts=4, iv_sanity_bounds=(0.05, 3.0))
TODAY = date(2026, 9, 8)


def _bars_frame(n=260, spot=150.0):
    idx = pd.bdate_range(end=TODAY.isoformat(), periods=n)
    closes = [spot] * n
    return pd.DataFrame({"Open": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes], "Close": closes, "Volume": [1_000_000] * n}, index=idx)


def _option_frame(strikes, iv=0.24):
    return pd.DataFrame(
        [{"strike": k, "bid": 1.0, "ask": 1.05, "lastPrice": 1.02, "volume": 50, "openInterest": 100, "impliedVolatility": iv, "inTheMoney": False, "contractSymbol": f"X{k}"} for k in strikes]
    )


def make_yf_client(spot=150.0, iv=0.24, rate_limited=False):
    client = Mock()
    if rate_limited:
        client.daily_bars.side_effect = SourceUnavailable("rate limited")
    else:
        client.daily_bars.return_value = SimpleNamespace(frame=_bars_frame(spot=spot), stale=False)
    client.list_expirations.return_value = [TODAY + timedelta(days=30), TODAY + timedelta(days=60)]
    frame = _option_frame([spot - 5, spot, spot + 5], iv=iv)
    client.option_chain.return_value = SimpleNamespace(calls=frame, puts=frame)
    client.next_earnings_date.return_value = None
    client.next_ex_dividend_date.return_value = None
    return client


def make_stooq_client(close=150.0):
    client = Mock()
    client.close_on.return_value = close
    return client


def make_nasdaq_client():
    client = Mock()
    client.earnings_date_on.return_value = False
    return client


def make_fred_client(complete=True, history_length=300):
    client = Mock()
    obs = lambda v: SimpleNamespace(value=v)
    if complete:
        client.vix_complex.return_value = {"vix": obs(15.0), "vix9d": obs(14.0), "vix3m": obs(16.0), "tbill_3m": obs(5.0)}
    else:
        client.vix_complex.return_value = {"vix": obs(15.0)}
    client.history.return_value = [obs(14.0 + (i % 5)) for i in range(history_length)]
    return client


def make_anthropic_client(text="AAPL nominal\n1 tickers nominal"):
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return client


# ---------------------------------------------------------------------------
# is_trading_day
# ---------------------------------------------------------------------------


def test_is_trading_day_true_for_weekday():
    assert main.is_trading_day(date(2026, 9, 4))  # Friday


def test_is_trading_day_false_for_weekend():
    assert not main.is_trading_day(date(2026, 9, 6))  # Sunday


def test_is_trading_day_false_for_holiday():
    assert not main.is_trading_day(date(2026, 9, 7))  # Labor Day 2026


# ---------------------------------------------------------------------------
# fetch_ticker_data
# ---------------------------------------------------------------------------


def _deps(**overrides):
    defaults = dict(
        fred_client=make_fred_client(),
        yf_client=make_yf_client(),
        stooq_client=make_stooq_client(),
        nasdaq_client=make_nasdaq_client(),
        anthropic_client=make_anthropic_client(),
    )
    defaults.update(overrides)
    return Deps(**defaults)


def test_fetch_ticker_data_happy_path(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    deps = _deps()
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert "dma20_distance_pct" in m
    assert "realized_vol_20d" in m
    assert m["atm_iv"] == pytest.approx(0.24)
    assert not disagreements
    assert raw["yahoo_close"] == pytest.approx(150.0)


def test_fetch_ticker_data_iv_rank_and_percentile_with_sufficient_history(tmp_path):
    # Neither path had ever actually been exercised with seeded history
    # in a test before -- caught on a spec re-read alongside the
    # iv_percentile gap itself (rank was computed and tested via a fresh,
    # empty store, which trivially suppresses both).
    snapshot_mod = __import__("snapshot")
    conn = snapshot_mod.open_store(tmp_path / "iv.sqlite3")
    for i in range(25):
        conn_as_of = TODAY - timedelta(days=25 - i)
        snapshot_mod.write_snapshot(
            conn, snapshot_mod.IVSnapshot(ticker="AAPL", as_of=conn_as_of, atm_iv=0.20 + (i % 5) * 0.01, survived_count=10)
        )
    deps = _deps()
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert m["iv_rank"]["sufficient"] is True
    assert 0 <= m["iv_rank"]["value"] <= 100
    assert m["iv_percentile"]["sufficient"] is True
    assert 0 <= m["iv_percentile"]["value"] <= 100


def test_fetch_ticker_data_yfinance_unavailable_returns_empty_metrics(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    deps = _deps(yf_client=make_yf_client(rate_limited=True))
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert m == {}
    assert any("yfinance unavailable" in n for n in notes)


def test_fetch_ticker_data_price_disputed_skips_metrics(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    deps = _deps(yf_client=make_yf_client(spot=150.0), stooq_client=make_stooq_client(close=100.0))
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert m == {}
    assert len(disagreements) == 1
    assert any("disagree" in n for n in notes)


def test_fetch_ticker_data_iv_suppressed_when_too_few_contracts(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    yf_client = make_yf_client()
    thin_frame = _option_frame([150.0])  # only 1 strike -> 2 contracts total, below min 4
    yf_client.option_chain.return_value = SimpleNamespace(calls=thin_frame, puts=thin_frame)
    deps = _deps(yf_client=yf_client)
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert "atm_iv" not in m
    assert any("IV unavailable" in n for n in notes)


def test_fetch_ticker_data_earnings_flows_through(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    yf_client = make_yf_client()
    yf_client.next_earnings_date.return_value = TODAY + timedelta(days=3)
    deps = _deps(yf_client=yf_client)
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert m["days_to_earnings"] == 3
    assert m["earnings_unconfirmed"] is True  # nasdaq mock returns False for every date -> unconfirmed


def test_fetch_ticker_data_ex_dividend_flows_through(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    yf_client = make_yf_client()
    yf_client.next_ex_dividend_date.return_value = TODAY + timedelta(days=2)
    deps = _deps(yf_client=yf_client)
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert m["days_to_ex_dividend"] == 2


def test_fetch_ticker_data_no_ex_dividend_is_not_an_error(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    deps = _deps()  # make_yf_client defaults next_ex_dividend_date to None
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert "days_to_ex_dividend" not in m
    assert not any("ex-dividend" in n for n in notes)


def test_fetch_ticker_data_ex_dividend_source_error_noted(tmp_path):
    conn = __import__("snapshot").open_store(tmp_path / "iv.sqlite3")
    yf_client = make_yf_client()
    yf_client.next_ex_dividend_date.side_effect = SourceUnavailable("rate limited")
    deps = _deps(yf_client=yf_client)
    m, raw, notes, disagreements = main.fetch_ticker_data(
        "AAPL", deps=deps, iv_conn=conn, chain_config=CHAIN_CONFIG, config=CONFIG, today=TODAY
    )
    assert "days_to_ex_dividend" not in m
    assert any("ex-dividend date unavailable" in n for n in notes)


# ---------------------------------------------------------------------------
# build_macro_dates
# ---------------------------------------------------------------------------


def test_build_macro_dates_computes_days_to():
    config = dict(CONFIG, calendar={"macro_dates": [{"date": "2026-09-17", "label": "FOMC decision"}], "blackout_days": 3})
    result = main.build_macro_dates(config, today=TODAY)
    assert result == [{"label": "FOMC decision", "days_to": 9}]


def test_build_macro_dates_drops_past_dates():
    config = dict(CONFIG, calendar={"macro_dates": [{"date": "2026-01-01", "label": "old news"}], "blackout_days": 3})
    result = main.build_macro_dates(config, today=TODAY)
    assert result == []


def test_build_macro_dates_skips_malformed_entries_without_crashing():
    config = dict(
        CONFIG,
        calendar={"macro_dates": [{"date": "not-a-date", "label": "bad"}, {"label": "missing date"}], "blackout_days": 3},
    )
    result = main.build_macro_dates(config, today=TODAY)
    assert result == []


def test_build_macro_dates_empty_list_default():
    config = dict(CONFIG, calendar={"macro_dates": [], "blackout_days": 3})
    assert main.build_macro_dates(config, today=TODAY) == []


# ---------------------------------------------------------------------------
# build_market_metrics
# ---------------------------------------------------------------------------


def test_build_market_metrics_complete():
    deps = _deps(fred_client=make_fred_client(complete=True))
    market, notes = main.build_market_metrics(deps, CONFIG, TODAY)
    assert market["vix"] == pytest.approx(15.0)
    assert "vix_term_structure" in market
    assert market["risk_free_rate"] == pytest.approx(0.05)
    assert market["macro_dates"] == []  # CONFIG's fixture has none configured
    assert notes == []


def test_build_market_metrics_includes_macro_dates():
    config = dict(CONFIG, calendar={"macro_dates": [{"date": "2026-09-10", "label": "CPI"}], "blackout_days": 3})
    deps = _deps(fred_client=make_fred_client(complete=True))
    market, notes = main.build_market_metrics(deps, config, TODAY)
    assert market["macro_dates"] == [{"label": "CPI", "days_to": 2}]


def test_build_market_metrics_vix_percentile_computed_when_sufficient_history():
    deps = _deps(fred_client=make_fred_client(complete=True, history_length=300))
    market, notes = main.build_market_metrics(deps, CONFIG, TODAY)
    assert market["vix_percentile"]["sufficient"] is True
    assert 0 <= market["vix_percentile"]["value"] <= 100
    assert not any("percentile suppressed" in n for n in notes)


def test_build_market_metrics_vix_percentile_suppressed_on_thin_history():
    config = dict(CONFIG, vix=dict(CONFIG["vix"], percentile_min_sample=100))
    deps = _deps(fred_client=make_fred_client(complete=True, history_length=10))
    market, notes = main.build_market_metrics(deps, config, TODAY)
    assert "vix_percentile" not in market
    assert any("VIX percentile suppressed" in n for n in notes)


def test_build_market_metrics_vix_percentile_skipped_without_vix_level():
    deps = _deps(fred_client=make_fred_client(complete=True))
    deps.fred_client.vix_complex.return_value = {}  # FRED gave us nothing at all this run
    market, notes = main.build_market_metrics(deps, CONFIG, TODAY)
    assert "vix" not in market
    assert "vix_percentile" not in market
    deps.fred_client.history.assert_not_called()


def test_build_market_metrics_vix_history_source_error_noted():
    deps = _deps(fred_client=make_fred_client(complete=True))
    deps.fred_client.history.side_effect = SourceUnavailable("rate limited")
    market, notes = main.build_market_metrics(deps, CONFIG, TODAY)
    assert "vix_percentile" not in market
    assert any("VIX history unavailable" in n for n in notes)


def test_build_market_metrics_partial_notes_incompleteness():
    deps = _deps(fred_client=make_fred_client(complete=False))
    market, notes = main.build_market_metrics(deps, CONFIG, TODAY)
    assert "vix_term_structure" not in market
    assert any("incomplete" in n for n in notes)


def test_build_market_metrics_schema_error_is_noted_not_raised():
    deps = _deps()
    deps.fred_client.vix_complex.side_effect = SchemaError("bad series id")
    market, notes = main.build_market_metrics(deps, CONFIG, TODAY)
    assert any("SCHEMA ERROR" in n for n in notes)


# ---------------------------------------------------------------------------
# build_positions_context
# ---------------------------------------------------------------------------


def test_build_positions_context_none_when_no_snapshot():
    ctx, notes = main.build_positions_context(None, ["AAPL"], TODAY, CONFIG)
    assert ctx is None
    assert notes == []


def test_build_positions_context_expiring_and_concentration(tmp_path):
    csv_path = tmp_path / "positions_2026-09-08.csv"
    csv_path.write_text(
        '"Symbol","Description","Quantity","Security Type"\n'
        f'"AAPL {(TODAY + timedelta(days=5)).strftime("%m/%d/%Y")} 190.00 C","CALL","-2","Option"\n'
    )
    snap = positions_mod.load_positions(csv_path)
    ctx, notes = main.build_positions_context(snap, ["AAPL"], TODAY, CONFIG)
    assert "AAPL" in ctx["expiring_by_ticker"]
    assert ctx["concurrent_option_position_count"] == 1
    assert ctx["tickers_with_exposure"] == {"AAPL"}
    assert notes == []


def test_build_positions_context_exposure_excludes_tickers_with_no_position(tmp_path):
    csv_path = tmp_path / "positions_2026-09-08.csv"
    csv_path.write_text(
        '"Symbol","Description","Quantity","Security Type"\n'
        f'"AAPL {(TODAY + timedelta(days=30)).strftime("%m/%d/%Y")} 190.00 C","CALL","-2","Option"\n'
    )
    snap = positions_mod.load_positions(csv_path)
    ctx, notes = main.build_positions_context(snap, ["AAPL", "SPY", "QQQ"], TODAY, CONFIG)
    assert ctx["tickers_with_exposure"] == {"AAPL"}  # not SPY or QQQ


def test_build_positions_context_stale_note(tmp_path):
    csv_path = tmp_path / "positions_2026-08-01.csv"
    csv_path.write_text('"Symbol","Description","Quantity","Security Type"\n"AAPL","APPLE","100","Equity"\n')
    snap = positions_mod.load_positions(csv_path)
    ctx, notes = main.build_positions_context(snap, ["AAPL"], TODAY, CONFIG)
    assert any("stale" in n for n in notes)


# ---------------------------------------------------------------------------
# write_run_log / load_yesterday_metrics
# ---------------------------------------------------------------------------


def test_write_and_load_yesterday_metrics_roundtrip(tmp_path):
    main.write_run_log(tmp_path, date(2026, 9, 7), {"metrics": {"AAPL": {"gap_pct": 1.0}}})
    result = main.load_yesterday_metrics(tmp_path, date(2026, 9, 8))
    assert result == {"AAPL": {"gap_pct": 1.0}}


def test_load_yesterday_metrics_none_when_no_logs(tmp_path):
    assert main.load_yesterday_metrics(tmp_path, TODAY) is None


def test_load_yesterday_metrics_picks_most_recent_prior_day(tmp_path):
    main.write_run_log(tmp_path, date(2026, 9, 5), {"metrics": {"AAPL": {"gap_pct": 1.0}}})
    main.write_run_log(tmp_path, date(2026, 9, 7), {"metrics": {"AAPL": {"gap_pct": 2.0}}})
    result = main.load_yesterday_metrics(tmp_path, TODAY)
    assert result == {"AAPL": {"gap_pct": 2.0}}


def test_load_yesterday_metrics_ignores_same_day_and_future(tmp_path):
    main.write_run_log(tmp_path, TODAY, {"metrics": {"AAPL": {"gap_pct": 9.0}}})
    result = main.load_yesterday_metrics(tmp_path, TODAY)
    assert result is None


def test_load_yesterday_market_metrics_returns_dict(tmp_path):
    main.write_run_log(tmp_path, TODAY - timedelta(days=1), {"metrics": {}, "market_metrics": {"vix": 15.0}})
    result = main.load_yesterday_market_metrics(tmp_path, TODAY)
    assert result == {"vix": 15.0}


def test_load_yesterday_market_metrics_empty_dict_when_no_prior_log(tmp_path):
    # Always a dict, never None -- callers do .get("vix") on it directly.
    assert main.load_yesterday_market_metrics(tmp_path, TODAY) == {}


def test_load_yesterday_market_metrics_empty_dict_when_key_absent(tmp_path):
    main.write_run_log(tmp_path, TODAY - timedelta(days=1), {"metrics": {}})  # no market_metrics key at all
    assert main.load_yesterday_market_metrics(tmp_path, TODAY) == {}


# ---------------------------------------------------------------------------
# prune_old_logs
# ---------------------------------------------------------------------------


def test_prune_old_logs_deletes_only_past_cutoff(tmp_path):
    old = tmp_path / f"{(TODAY - timedelta(days=100)).isoformat()}.json"
    boundary = tmp_path / f"{(TODAY - timedelta(days=90)).isoformat()}.json"
    recent = tmp_path / f"{(TODAY - timedelta(days=1)).isoformat()}.json"
    for p in (old, boundary, recent):
        p.write_text("{}")

    main.prune_old_logs(tmp_path, keep_days=90, today=TODAY)

    assert not old.exists()
    assert boundary.exists()  # exactly at the cutoff -- kept, not deleted
    assert recent.exists()


def test_prune_old_logs_ignores_non_date_filenames(tmp_path):
    stray = tmp_path / "not-a-date.json"
    stray.write_text("{}")
    main.prune_old_logs(tmp_path, keep_days=1, today=TODAY)
    assert stray.exists()


def test_prune_old_logs_missing_directory_is_a_noop(tmp_path):
    main.prune_old_logs(tmp_path / "does_not_exist", keep_days=1, today=TODAY)  # must not raise


# ---------------------------------------------------------------------------
# run_briefing (full orchestration, all deps mocked)
# ---------------------------------------------------------------------------


def test_run_briefing_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    deps = _deps()
    result = main.run_briefing(CONFIG, deps, today=TODAY)
    assert isinstance(result.text, str) and result.text
    assert "<html" in result.html
    assert result.subject == f"Options briefing -- {TODAY.isoformat()}"
    log_path = tmp_path / "logs" / f"{TODAY.isoformat()}.json"
    assert log_path.exists()
    logged = json.loads(log_path.read_text())
    assert "AAPL" in logged["metrics"]


def test_run_briefing_wires_yesterdays_vix_into_market_metrics(tmp_path, monkeypatch):
    # End-to-end regression test for the vix_yesterday wiring bug: an
    # earlier version of main.py never populated this key at all, so
    # brief.py's VIX-level delta flag was unreachable in a real run
    # despite passing every existing test (none of which ran main.py
    # against a real logs/ directory with a prior day's log in it).
    monkeypatch.setattr(main, "ROOT", tmp_path)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    main.write_run_log(logs_dir, TODAY - timedelta(days=1), {"metrics": {}, "market_metrics": {"vix": 15.0}})

    deps = _deps(fred_client=make_fred_client(complete=True))  # today's vix_complex reports vix=15.0 by default...
    # ...override to a level that trips the vix_level_pts=2.0 threshold vs yesterday's logged 15.0
    obs = lambda v: SimpleNamespace(value=v)
    deps.fred_client.vix_complex.return_value = {"vix": obs(22.0), "vix9d": obs(20.0), "vix3m": obs(24.0), "tbill_3m": obs(5.0)}

    main.run_briefing(CONFIG, deps, today=TODAY)

    today_log = json.loads((logs_dir / f"{TODAY.isoformat()}.json").read_text())
    assert today_log["market_metrics"]["vix_yesterday"] == 15.0
    assert today_log["market_metrics"]["vix"] == 22.0


def test_run_briefing_falls_back_when_llm_output_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    bad_client = make_anthropic_client(text="AAPL looks bullish here")
    deps = _deps(anthropic_client=bad_client)
    result = main.run_briefing(CONFIG, deps, today=TODAY)
    assert "bullish" not in result.text.lower()
    assert "bullish" not in result.html.lower()
    log_path = tmp_path / "logs" / f"{TODAY.isoformat()}.json"
    logged = json.loads(log_path.read_text())
    assert any("LLM briefing generation failed" in n for n in logged["data_quality_notes"])


def test_run_briefing_falls_back_when_llm_call_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    broken_client = Mock()
    broken_client.messages.create.side_effect = RuntimeError("API down")
    deps = _deps(anthropic_client=broken_client)
    result = main.run_briefing(CONFIG, deps, today=TODAY)
    assert isinstance(result.text, str)  # fell back instead of raising


def test_run_briefing_continues_past_one_ticker_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    config = dict(CONFIG, tickers=["AAPL", "BROKEN"])
    yf_client = make_yf_client()

    def daily_bars_side_effect(ticker, lookback_days=260):
        if ticker == "BROKEN":
            raise SourceUnavailable("down")
        return SimpleNamespace(frame=_bars_frame(), stale=False)

    yf_client.daily_bars.side_effect = daily_bars_side_effect
    deps = _deps(yf_client=yf_client)
    result = main.run_briefing(config, deps, today=TODAY)
    assert isinstance(result.text, str) and result.text  # run completed despite BROKEN failing


def test_run_briefing_prunes_logs_older_than_keep_days(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    config = dict(CONFIG, logging=dict(CONFIG["logging"], keep_days=5))
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    old_log = logs_dir / f"{(TODAY - timedelta(days=30)).isoformat()}.json"
    old_log.write_text("{}")
    recent_log = logs_dir / f"{(TODAY - timedelta(days=1)).isoformat()}.json"
    recent_log.write_text("{}")

    deps = _deps()
    main.run_briefing(config, deps, today=TODAY)

    assert not old_log.exists()
    assert recent_log.exists()


# ---------------------------------------------------------------------------
# validate_config / load_config
# ---------------------------------------------------------------------------

FULL_CONFIG = dict(
    CONFIG,
    run={"time": "08:30", "timezone": "America/New_York"},
    delivery={"smtp_host": "smtp.gmail.com", "smtp_port": 587, "smtp_use_tls": True, "from_address": "", "to_address": ""},
)


def test_validate_config_passes_for_complete_config():
    main.validate_config(FULL_CONFIG)  # must not raise


def test_validate_config_raises_naming_missing_section():
    broken = dict(FULL_CONFIG)
    del broken["earnings"]
    with pytest.raises(main.ConfigError) as exc_info:
        main.validate_config(broken)
    assert "earnings" in str(exc_info.value)


def test_validate_config_raises_on_non_mapping():
    with pytest.raises(main.ConfigError):
        main.validate_config(["not", "a", "dict"])


def test_validate_config_raises_on_empty_tickers():
    broken = dict(FULL_CONFIG, tickers=[])
    with pytest.raises(main.ConfigError) as exc_info:
        main.validate_config(broken)
    assert "tickers" in str(exc_info.value)


def test_load_config_valid_file(tmp_path):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(FULL_CONFIG))
    loaded = main.load_config(path)
    assert loaded["tickers"] == ["AAPL"]


def test_load_config_missing_section_raises_config_error(tmp_path):
    import yaml

    broken = dict(FULL_CONFIG)
    del broken["briefing"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(broken))
    with pytest.raises(main.ConfigError) as exc_info:
        main.load_config(path)
    assert "briefing" in str(exc_info.value)
