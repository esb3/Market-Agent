from datetime import date, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from sources.base import SchemaError, SourceUnavailable
from sources.yfinance import YFinanceClient, YFinanceConfig, configure_cache


def make_client(mock_ticker, **overrides):
    config = YFinanceConfig(max_retries=2, backoff_base_seconds=0.001, **overrides)
    return YFinanceClient(config, ticker_factory=lambda ticker: mock_ticker)


def valid_bars_frame(n=30) -> pd.DataFrame:
    idx = pd.bdate_range("2026-07-01", periods=n)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


def valid_option_frame(n=5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contractSymbol": [f"AAPL26X{i}C" for i in range(n)],
            "strike": [100.0 + i * 5 for i in range(n)],
            "lastPrice": [5.0] * n,
            "bid": [4.8] * n,
            "ask": [5.2] * n,
            "volume": [100] * n,
            "openInterest": [500] * n,
            "impliedVolatility": [0.25] * n,
            "inTheMoney": [False] * n,
        }
    )


def valid_earnings_frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-09-24", "2026-06-25"])
    return pd.DataFrame(
        {"EPS Estimate": [1.5, None], "Reported EPS": [None, 1.42], "Surprise(%)": [None, -3.1]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# daily_bars
# ---------------------------------------------------------------------------


def test_daily_bars_happy_path():
    mock_ticker = Mock()
    mock_ticker.history.return_value = valid_bars_frame()
    client = make_client(mock_ticker)

    result = client.daily_bars("AAPL", lookback_days=20)

    assert not result.stale
    assert len(result.frame) == 20
    assert list(result.frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_daily_bars_empty_frame_is_schema_error():
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame()
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.daily_bars("AAPL")


def test_daily_bars_missing_columns_is_schema_error():
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.bdate_range("2026-07-01", periods=3))
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.daily_bars("AAPL")


def test_daily_bars_non_datetime_index_is_schema_error():
    mock_ticker = Mock()
    frame = valid_bars_frame()
    frame.index = list(range(len(frame)))
    mock_ticker.history.return_value = frame
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.daily_bars("AAPL")


def test_daily_bars_recovers_after_transient_rate_limit():
    mock_ticker = Mock()
    mock_ticker.history.side_effect = [YFRateLimitError(), valid_bars_frame()]
    client = make_client(mock_ticker)

    result = client.daily_bars("AAPL", lookback_days=10)
    assert not result.stale
    assert mock_ticker.history.call_count == 2


def test_daily_bars_degrades_gracefully_on_non_rate_limit_network_error():
    # yfinance_cache's internal calls can raise other libraries' own
    # connection errors (curl_cffi, requests), not just YFRateLimitError
    # -- these must degrade the same way, not crash the run.
    mock_ticker = Mock()
    mock_ticker.history.side_effect = [ConnectionError("curl: (7) CONNECT tunnel failed"), valid_bars_frame()]
    client = make_client(mock_ticker)

    result = client.daily_bars("AAPL", lookback_days=10)
    assert not result.stale
    assert mock_ticker.history.call_count == 2


def test_daily_bars_falls_back_to_stale_cache_on_persistent_non_rate_limit_error():
    mock_ticker = Mock()
    mock_ticker.history.side_effect = [
        ConnectionError("network down"),
        ConnectionError("network down"),
        ConnectionError("network down"),
        valid_bars_frame(),
    ]
    client = make_client(mock_ticker, stale_fallback=True)

    result = client.daily_bars("AAPL", lookback_days=10)
    assert result.stale


def test_daily_bars_falls_back_to_stale_cache_after_persistent_rate_limit():
    mock_ticker = Mock()
    # 1 initial + 2 retries all rate-limited, then the stale-fallback
    # call (forcing max_age) succeeds.
    mock_ticker.history.side_effect = [
        YFRateLimitError(),
        YFRateLimitError(),
        YFRateLimitError(),
        valid_bars_frame(),
    ]
    client = make_client(mock_ticker, stale_fallback=True)

    result = client.daily_bars("AAPL", lookback_days=10)
    assert result.stale
    # last call should have requested a cache-only (huge max_age) read
    last_kwargs = mock_ticker.history.call_args.kwargs
    assert last_kwargs["max_age"] == timedelta(days=36500)


def test_daily_bars_raises_when_stale_fallback_disabled():
    mock_ticker = Mock()
    mock_ticker.history.side_effect = YFRateLimitError()
    client = make_client(mock_ticker, stale_fallback=False)
    with pytest.raises(SourceUnavailable):
        client.daily_bars("AAPL")


def test_daily_bars_raises_when_even_stale_fallback_rate_limited():
    mock_ticker = Mock()
    mock_ticker.history.side_effect = YFRateLimitError()
    client = make_client(mock_ticker, stale_fallback=True)
    with pytest.raises(SourceUnavailable):
        client.daily_bars("AAPL")


# ---------------------------------------------------------------------------
# list_expirations
# ---------------------------------------------------------------------------


def test_list_expirations_happy_path_sorted():
    mock_ticker = Mock()
    mock_ticker.options = ("2026-10-16", "2026-09-18")
    client = make_client(mock_ticker)
    result = client.list_expirations("AAPL")
    assert result == [date(2026, 9, 18), date(2026, 10, 16)]


def test_list_expirations_empty_is_schema_error():
    mock_ticker = Mock()
    mock_ticker.options = ()
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.list_expirations("AAPL")


# ---------------------------------------------------------------------------
# option_chain
# ---------------------------------------------------------------------------


def test_option_chain_happy_path():
    mock_ticker = Mock()
    mock_ticker.option_chain.return_value = Mock(calls=valid_option_frame(), puts=valid_option_frame())
    client = make_client(mock_ticker)
    chain = client.option_chain("AAPL", date(2026, 10, 16))
    assert len(chain.calls) == 5
    assert len(chain.puts) == 5


def test_option_chain_none_calls_is_schema_error():
    mock_ticker = Mock()
    mock_ticker.option_chain.return_value = Mock(calls=None, puts=None)
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.option_chain("AAPL", date(2026, 10, 16))


def test_option_chain_missing_columns_is_schema_error():
    mock_ticker = Mock()
    broken = valid_option_frame().drop(columns=["impliedVolatility"])
    mock_ticker.option_chain.return_value = Mock(calls=broken, puts=valid_option_frame())
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.option_chain("AAPL", date(2026, 10, 16))


def test_option_chain_rate_limit_retries():
    mock_ticker = Mock()
    good = Mock(calls=valid_option_frame(), puts=valid_option_frame())
    mock_ticker.option_chain.side_effect = [YFRateLimitError(), good]
    client = make_client(mock_ticker)
    chain = client.option_chain("AAPL", date(2026, 10, 16))
    assert chain is good


# ---------------------------------------------------------------------------
# next_earnings_date
# ---------------------------------------------------------------------------


def test_next_earnings_date_picks_nearest_upcoming():
    mock_ticker = Mock()
    mock_ticker.get_earnings_dates.return_value = valid_earnings_frame()
    client = make_client(mock_ticker)
    result = client.next_earnings_date("AAPL", as_of=date(2026, 9, 8))
    assert result == date(2026, 9, 24)


def test_next_earnings_date_no_upcoming_rows_returns_none():
    mock_ticker = Mock()
    mock_ticker.get_earnings_dates.return_value = valid_earnings_frame()
    client = make_client(mock_ticker)
    result = client.next_earnings_date("AAPL", as_of=date(2027, 1, 1))
    assert result is None


def test_next_earnings_date_empty_frame_returns_none():
    mock_ticker = Mock()
    mock_ticker.get_earnings_dates.return_value = pd.DataFrame()
    client = make_client(mock_ticker)
    assert client.next_earnings_date("AAPL", as_of=date(2026, 9, 8)) is None


def test_next_earnings_date_missing_columns_is_schema_error():
    mock_ticker = Mock()
    idx = pd.to_datetime(["2026-09-24"])
    mock_ticker.get_earnings_dates.return_value = pd.DataFrame({"Nonsense": [1]}, index=idx)
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.next_earnings_date("AAPL", as_of=date(2026, 9, 8))


def test_next_earnings_date_non_dataframe_is_schema_error():
    mock_ticker = Mock()
    mock_ticker.get_earnings_dates.return_value = None
    client = make_client(mock_ticker)
    with pytest.raises(SchemaError):
        client.next_earnings_date("AAPL", as_of=date(2026, 9, 8))


# ---------------------------------------------------------------------------
# configure_cache
# ---------------------------------------------------------------------------


def test_configure_cache_creates_directory(tmp_path):
    cache_dir = tmp_path / "yf_cache"
    configure_cache(cache_dir)
    assert cache_dir.exists()
