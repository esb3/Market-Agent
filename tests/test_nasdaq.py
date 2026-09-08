from datetime import date
from unittest.mock import patch

import pytest
import requests

from sources.base import SchemaError, SourceUnavailable
from sources.nasdaq import NasdaqClient, NasdaqConfig


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


def make_client(**overrides) -> NasdaqClient:
    config = NasdaqConfig(timeout_seconds=1, max_retries=2, backoff_base_seconds=0.001, **overrides)
    return NasdaqClient(config)


def test_earnings_date_on_found():
    client = make_client()
    payload = {"data": {"rows": [{"symbol": "AAPL", "time": "time-after-hours"}, {"symbol": "MSFT"}]}}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        assert client.earnings_date_on("AAPL", date(2026, 9, 24)) is True


def test_earnings_date_on_not_found():
    client = make_client()
    payload = {"data": {"rows": [{"symbol": "MSFT"}]}}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        assert client.earnings_date_on("AAPL", date(2026, 9, 24)) is False


def test_earnings_date_on_case_insensitive():
    client = make_client()
    payload = {"data": {"rows": [{"symbol": "aapl"}]}}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        assert client.earnings_date_on("AAPL", date(2026, 9, 24)) is True


def test_earnings_date_on_null_data_is_no_results_not_error():
    client = make_client()
    payload = {"data": None}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        assert client.earnings_date_on("AAPL", date(2026, 9, 24)) is False


def test_earnings_date_on_null_rows_is_no_results():
    client = make_client()
    payload = {"data": {"rows": None}}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        assert client.earnings_date_on("AAPL", date(2026, 9, 24)) is False


def test_missing_data_key_is_schema_error():
    client = make_client()
    payload = {"unexpected": True}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        with pytest.raises(SchemaError):
            client.earnings_date_on("AAPL", date(2026, 9, 24))


def test_rows_not_a_list_is_schema_error():
    client = make_client()
    payload = {"data": {"rows": "oops"}}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        with pytest.raises(SchemaError):
            client.earnings_date_on("AAPL", date(2026, 9, 24))


def test_row_missing_symbol_is_schema_error():
    client = make_client()
    payload = {"data": {"rows": [{"time": "before-open"}]}}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)):
        with pytest.raises(SchemaError):
            client.earnings_date_on("AAPL", date(2026, 9, 24))


def test_http_error_retries_then_raises_source_unavailable():
    client = make_client()
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(403, None)) as mock_get:
        with pytest.raises(SourceUnavailable):
            client.earnings_date_on("AAPL", date(2026, 9, 24))
    assert mock_get.call_count == 3


def test_network_exception_retries_then_raises_source_unavailable():
    client = make_client()
    with patch("sources.nasdaq.requests.get", side_effect=requests.ConnectionError("boom")) as mock_get:
        with pytest.raises(SourceUnavailable):
            client.earnings_date_on("AAPL", date(2026, 9, 24))
    assert mock_get.call_count == 3


def test_request_uses_browser_user_agent():
    client = make_client()
    payload = {"data": {"rows": []}}
    with patch("sources.nasdaq.requests.get", return_value=FakeResponse(200, payload)) as mock_get:
        client.earnings_date_on("AAPL", date(2026, 9, 24))
    _, kwargs = mock_get.call_args
    assert "User-Agent" in kwargs["headers"]
