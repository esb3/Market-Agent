from datetime import date
from unittest.mock import patch

import pytest
import requests

from sources.base import SchemaError, SourceUnavailable
from sources.fred import FredClient, FredConfig


def make_client(**overrides) -> FredClient:
    config = FredConfig(api_key="test-key", timeout_seconds=1, max_retries=2, backoff_base_seconds=0.001, **overrides)
    return FredClient(config)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


def test_requires_api_key():
    with pytest.raises(SourceUnavailable):
        FredClient(FredConfig(api_key=""))


def test_latest_observation_happy_path():
    client = make_client()
    payload = {
        "observations": [
            {"date": "2026-09-04", "value": "14.5"},
            {"date": "2026-09-05", "value": "15.1"},
        ]
    }
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        obs = client.latest_observation("vix")
    assert obs.value == pytest.approx(15.1)
    assert obs.as_of == date(2026, 9, 5)
    assert obs.source == "fred"
    assert obs.series == "VIXCLS"


def test_latest_observation_skips_missing_marker():
    client = make_client()
    payload = {
        "observations": [
            {"date": "2026-09-04", "value": "14.5"},
            {"date": "2026-09-05", "value": "."},  # e.g. a holiday
        ]
    }
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        obs = client.latest_observation("vix")
    assert obs.value == pytest.approx(14.5)
    assert obs.as_of == date(2026, 9, 4)


def test_latest_observation_all_missing_returns_none():
    client = make_client()
    payload = {"observations": [{"date": "2026-09-05", "value": "."}]}
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        assert client.latest_observation("vix") is None


def test_unknown_series_key_raises_value_error():
    client = make_client()
    with pytest.raises(ValueError):
        client.latest_observation("not_a_real_series")


def test_missing_observations_key_is_schema_error():
    client = make_client()
    payload = {"error_message": "The series does not exist."}
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        with pytest.raises(SchemaError):
            client.latest_observation("vix9d")


def test_non_numeric_value_is_schema_error():
    client = make_client()
    payload = {"observations": [{"date": "2026-09-05", "value": "not-a-number"}]}
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        with pytest.raises(SchemaError):
            client.latest_observation("vix")


def test_observations_not_a_list_is_schema_error():
    client = make_client()
    payload = {"observations": "oops"}
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        with pytest.raises(SchemaError):
            client.latest_observation("vix")


def test_http_error_retries_then_raises_source_unavailable():
    client = make_client()
    with patch("sources.fred.requests.get", return_value=FakeResponse(500, {}, "server error")) as mock_get:
        with pytest.raises(SourceUnavailable):
            client.latest_observation("vix")
    # 1 initial + max_retries(2) = 3 calls
    assert mock_get.call_count == 3


def test_network_exception_retries_then_raises_source_unavailable():
    client = make_client()
    with patch("sources.fred.requests.get", side_effect=requests.ConnectionError("boom")) as mock_get:
        with pytest.raises(SourceUnavailable):
            client.latest_observation("vix")
    assert mock_get.call_count == 3


def test_transient_failure_then_success_within_retry_budget():
    client = make_client()
    good_payload = {"observations": [{"date": "2026-09-05", "value": "15.1"}]}
    responses = [requests.ConnectionError("boom"), FakeResponse(200, good_payload)]

    def side_effect(*args, **kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch("sources.fred.requests.get", side_effect=side_effect):
        obs = client.latest_observation("vix")
    assert obs.value == pytest.approx(15.1)


def test_vix_complex_skips_unavailable_series_but_keeps_others():
    client = make_client()
    good_payload = {"observations": [{"date": "2026-09-05", "value": "15.1"}]}

    def side_effect(url, params, timeout):
        if params["series_id"] == "VXSTCLS":
            return FakeResponse(500, {}, "down")
        return FakeResponse(200, good_payload)

    with patch("sources.fred.requests.get", side_effect=side_effect):
        result = client.vix_complex()

    assert "vix" in result
    assert "tbill_3m" in result
    assert "vix9d" not in result  # the broken one is omitted, not raised


def test_vix_complex_raises_on_schema_error_rather_than_hiding_it():
    client = make_client()
    bad_payload = {"error_message": "The series does not exist."}
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, bad_payload)):
        with pytest.raises(SchemaError):
            client.vix_complex()


def test_risk_free_rate_decimal_converts_from_percent():
    client = make_client()
    payload = {"observations": [{"date": "2026-09-05", "value": "5.25"}]}
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        obs = client.risk_free_rate_decimal()
    assert obs.value == pytest.approx(0.0525)


def test_risk_free_rate_decimal_none_when_no_data():
    client = make_client()
    payload = {"observations": []}
    with patch("sources.fred.requests.get", return_value=FakeResponse(200, payload)):
        assert client.risk_free_rate_decimal() is None
