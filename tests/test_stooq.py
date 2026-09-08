from datetime import date
from unittest.mock import patch

import pytest
import requests

from sources.base import SchemaError, SourceUnavailable
from sources.stooq import StooqClient, StooqConfig, stooq_symbol


class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def make_client(**overrides) -> StooqClient:
    config = StooqConfig(timeout_seconds=1, max_retries=2, backoff_base_seconds=0.001, **overrides)
    return StooqClient(config)


VALID_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2026-08-19,159.62,160.02,158.69,159.38,3047848\n"
    "2026-08-20,158.35,158.91,157.18,157.77,6985964\n"
    "2026-08-21,159.36,160.21,158.92,159.65,2320291\n"
)


def test_stooq_symbol_format():
    assert stooq_symbol("AAPL") == "aapl.us"
    assert stooq_symbol("spy") == "spy.us"


def test_daily_bars_happy_path():
    client = make_client()
    with patch("sources.stooq.requests.get", return_value=FakeResponse(200, VALID_CSV)):
        frame = client.daily_bars("AAPL")
    assert len(frame) == 3
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert float(frame.iloc[-1]["Close"]) == pytest.approx(159.65)


def test_daily_bars_no_data_marker_is_schema_error():
    client = make_client()
    with patch("sources.stooq.requests.get", return_value=FakeResponse(200, "N/D")):
        with pytest.raises(SchemaError):
            client.daily_bars("NOTREAL")


def test_daily_bars_empty_body_is_schema_error():
    client = make_client()
    with patch("sources.stooq.requests.get", return_value=FakeResponse(200, "")):
        with pytest.raises(SchemaError):
            client.daily_bars("AAPL")


def test_daily_bars_missing_columns_is_schema_error():
    client = make_client()
    bad_csv = "Date,Close\n2026-08-21,159.65\n"
    with patch("sources.stooq.requests.get", return_value=FakeResponse(200, bad_csv)):
        with pytest.raises(SchemaError):
            client.daily_bars("AAPL")


def test_daily_bars_unparseable_csv_is_schema_error():
    client = make_client()
    with patch("sources.stooq.requests.get", return_value=FakeResponse(200, "\x00\x01garbage\x02")):
        with pytest.raises(SchemaError):
            client.daily_bars("AAPL")


def test_http_error_retries_then_raises_source_unavailable():
    client = make_client()
    with patch("sources.stooq.requests.get", return_value=FakeResponse(500, "server error")) as mock_get:
        with pytest.raises(SourceUnavailable):
            client.daily_bars("AAPL")
    assert mock_get.call_count == 3  # 1 + max_retries(2)


def test_network_exception_retries_then_raises_source_unavailable():
    client = make_client()
    with patch("sources.stooq.requests.get", side_effect=requests.ConnectionError("boom")) as mock_get:
        with pytest.raises(SourceUnavailable):
            client.daily_bars("AAPL")
    assert mock_get.call_count == 3


def test_close_on_returns_matching_date():
    client = make_client()
    with patch("sources.stooq.requests.get", return_value=FakeResponse(200, VALID_CSV)):
        close = client.close_on("AAPL", date(2026, 8, 20))
    assert close == pytest.approx(157.77)


def test_close_on_returns_none_when_date_missing():
    client = make_client()
    with patch("sources.stooq.requests.get", return_value=FakeResponse(200, VALID_CSV)):
        close = client.close_on("AAPL", date(2026, 1, 1))
    assert close is None
