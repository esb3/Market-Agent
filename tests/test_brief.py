from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import brief
from brief import (
    BriefingValidationError,
    Flag,
    build_data_quality_notes,
    build_flags,
    build_payload,
    format_fallback,
    generate_and_validate,
    generate_briefing,
    nominal_tickers,
    validate_briefing_text,
)

CONFIG = {
    "underlying": {"notable_gap_atr": 0.75},
    "iv_rank_bands": {"low": 25, "high": 75},
    "earnings": {"blackout_days": 5},
    "positions": {"max_concurrent_positions": 10, "dte_window": [0, 7]},
    "vix": {"flag_inversion": True},
    "delta_thresholds": {
        "dma_distance_pct": 3.0,
        "realized_vol_pts": 5.0,
        "iv_minus_rv_pts": 5.0,
        "term_structure_slope_pts": 2.0,
        "range_percentile_pts": 25.0,
        "vix_level_pts": 2.0,
    },
    "briefing": {"max_characters": 600, "max_flags": 8, "llm_model": "claude-sonnet-4-6", "llm_temperature": 0},
}


# ---------------------------------------------------------------------------
# validate_briefing_text
# ---------------------------------------------------------------------------


def test_validate_clean_text_passes():
    validate_briefing_text("AAPL gap +2.1% (1.1 ATR)\n3 tickers nominal", max_characters=600)


@pytest.mark.parametrize(
    "text",
    [
        "AAPL looks bullish here",
        "SPY is bearish going into CPI",
        "we expect a move higher",
        "this is likely to break out",
        "the setup suggests a rally",
        "good entry near the 50DMA",
        "consider buying the dip",
        "AAPL is poised for a breakout",
        "the chart is setting up for a move",
        "should rally into earnings",
        "about to break resistance",
        "breakout imminent on SPY",
        "time to buy calls",
        "good time to add exposure",
        "we recommend trimming",
        "price target of 200",
        "will rally into the print",
        "70% probability of a move",
        "odds favor a bounce",
        "RSI shows overbought conditions",
        "stock is oversold",
    ],
)
def test_validate_catches_forbidden_language(text):
    with pytest.raises(BriefingValidationError) as exc_info:
        validate_briefing_text(text, max_characters=600)
    assert exc_info.value.failures[0].reason == "forbidden_language"


def test_validate_does_not_flag_neutral_options_vocabulary():
    # "long", "short", "call", "put", "strike" are ordinary vocabulary
    # for describing existing exposure -- must not trip the validator.
    text = "AAPL: 3 short puts exp 5d. SPY: long 2 calls strike 560."
    validate_briefing_text(text, max_characters=600)  # should not raise


def test_validate_over_character_limit():
    with pytest.raises(BriefingValidationError) as exc_info:
        validate_briefing_text("x" * 700, max_characters=600)
    assert exc_info.value.failures[0].reason == "over_character_limit"


def test_validate_collects_multiple_failures():
    text = "bullish " * 200  # both forbidden language AND over cap
    with pytest.raises(BriefingValidationError) as exc_info:
        validate_briefing_text(text, max_characters=600)
    reasons = {f.reason for f in exc_info.value.failures}
    assert reasons == {"forbidden_language", "over_character_limit"}


# ---------------------------------------------------------------------------
# build_flags
# ---------------------------------------------------------------------------


def test_gap_flag_triggers_above_threshold():
    metrics = {"AAPL": {"gap_pct": 2.0, "gap_atr_units": 1.2}}
    flags = build_flags(metrics, {}, None, CONFIG)
    assert any(f.category == "gap" and f.ticker == "AAPL" for f in flags)


def test_gap_flag_silent_below_threshold():
    metrics = {"AAPL": {"gap_pct": 0.3, "gap_atr_units": 0.2}}
    flags = build_flags(metrics, {}, None, CONFIG)
    assert not any(f.category == "gap" for f in flags)


def test_iv_rank_flag_only_when_sufficient():
    metrics = {
        "AAPL": {"iv_rank": {"value": 82, "sufficient": True, "label": "41-day sample — thin"}},
    }
    flags = build_flags(metrics, {}, None, CONFIG)
    assert any(f.category == "iv_rank" for f in flags)


def test_iv_rank_flag_suppressed_when_insufficient_sample():
    metrics = {"AAPL": {"iv_rank": {"value": 82, "sufficient": False, "label": "unavailable"}}}
    flags = build_flags(metrics, {}, None, CONFIG)
    assert not any(f.category == "iv_rank" for f in flags)


def test_iv_rank_flag_silent_in_middle_band():
    metrics = {"AAPL": {"iv_rank": {"value": 50, "sufficient": True, "label": "200-day sample"}}}
    flags = build_flags(metrics, {}, None, CONFIG)
    assert not any(f.category == "iv_rank" for f in flags)


def test_earnings_blackout_flag():
    metrics = {"AAPL": {"days_to_earnings": 3}}
    flags = build_flags(metrics, {}, None, CONFIG)
    earnings_flags = [f for f in flags if f.category == "earnings"]
    assert len(earnings_flags) == 1
    assert "3d" in earnings_flags[0].message
    assert "unconfirmed" not in earnings_flags[0].message


def test_earnings_flag_marks_unconfirmed():
    metrics = {"AAPL": {"days_to_earnings": 3, "earnings_unconfirmed": True}}
    flags = build_flags(metrics, {}, None, CONFIG)
    earnings_flags = [f for f in flags if f.category == "earnings"]
    assert "unconfirmed" in earnings_flags[0].message


def test_earnings_flag_silent_outside_blackout_window():
    metrics = {"AAPL": {"days_to_earnings": 30}}
    flags = build_flags(metrics, {}, None, CONFIG)
    assert not any(f.category == "earnings" for f in flags)


def test_expiration_flags_from_positions_context():
    metrics = {"AAPL": {}}
    positions_ctx = {"expiring_by_ticker": {"AAPL": [{"option_type": "C", "strike": 190.0, "dte": 5}]}}
    flags = build_flags(metrics, {}, positions_ctx, CONFIG)
    exp_flags = [f for f in flags if f.category == "expiration"]
    assert len(exp_flags) == 1
    assert "AAPL C 190.0 exp in 5d" == exp_flags[0].message


def test_concentration_flag_fires_once_not_per_ticker():
    metrics = {"AAPL": {}, "SPY": {}}
    positions_ctx = {"concurrent_option_position_count": 12}
    flags = build_flags(metrics, {}, positions_ctx, CONFIG)
    concentration_flags = [f for f in flags if f.category == "concentration"]
    assert len(concentration_flags) == 1
    assert "12" in concentration_flags[0].message


def test_concentration_flag_silent_under_max():
    metrics = {"AAPL": {}}
    positions_ctx = {"concurrent_option_position_count": 5}
    flags = build_flags(metrics, {}, positions_ctx, CONFIG)
    assert not any(f.category == "concentration" for f in flags)


def test_vix_inversion_flag():
    market = {"vix_term_structure": {"shape": "backwardation", "inverted": True}, "vix9d": 28, "vix": 24, "vix3m": 20}
    flags = build_flags({}, market, None, CONFIG)
    assert any(f.category == "vix_term_structure" for f in flags)


def test_vix_inversion_flag_silent_when_normal():
    market = {"vix_term_structure": {"shape": "contango", "inverted": False}}
    flags = build_flags({}, market, None, CONFIG)
    assert not any(f.category == "vix_term_structure" for f in flags)


def test_delta_flag_fires_above_threshold():
    metrics = {"AAPL": {"dma50_distance_pct": 8.0}}
    yesterday = {"AAPL": {"dma50_distance_pct": 3.0}}  # delta = 5.0 >= threshold 3.0
    flags = build_flags(metrics, {}, None, CONFIG, yesterday=yesterday)
    assert any(f.category == "delta" and "50DMA" in f.message for f in flags)


def test_delta_flag_silent_below_threshold():
    metrics = {"AAPL": {"dma50_distance_pct": 4.0}}
    yesterday = {"AAPL": {"dma50_distance_pct": 3.0}}  # delta = 1.0 < threshold 3.0
    flags = build_flags(metrics, {}, None, CONFIG, yesterday=yesterday)
    assert not any(f.category == "delta" for f in flags)


def test_delta_flag_silent_with_no_yesterday_data():
    metrics = {"AAPL": {"dma50_distance_pct": 8.0}}
    flags = build_flags(metrics, {}, None, CONFIG)  # no yesterday at all
    assert not any(f.category == "delta" for f in flags)


def test_flags_ranked_by_severity_descending():
    metrics = {
        "AAPL": {"gap_pct": 2.0, "gap_atr_units": 1.2, "days_to_earnings": 2},  # gap ~78, earnings 80
    }
    flags = build_flags(metrics, {}, None, CONFIG)
    severities = [f.severity for f in flags]
    assert severities == sorted(severities, reverse=True)


# ---------------------------------------------------------------------------
# nominal_tickers / build_payload
# ---------------------------------------------------------------------------


def test_nominal_tickers_excludes_flagged():
    flags = [Flag(severity=70, category="gap", ticker="AAPL", message="x")]
    result = nominal_tickers(["AAPL", "SPY", "QQQ"], flags)
    assert result == ["SPY", "QQQ"]


def test_build_payload_shape():
    metrics = {"AAPL": {"gap_pct": 2.0, "gap_atr_units": 1.2}, "SPY": {}}
    payload = build_payload(metrics, {}, None, CONFIG, data_quality_notes=["note1"])
    assert payload["nominal"] == ["SPY"]
    assert len(payload["flags"]) == 1
    assert payload["data_quality_notes"] == ["note1"]


def test_build_payload_truncates_to_max_flags():
    config = dict(CONFIG, briefing=dict(CONFIG["briefing"], max_flags=1))
    metrics = {
        "AAPL": {"gap_pct": 3.0, "gap_atr_units": 2.0},
        "SPY": {"gap_pct": 2.0, "gap_atr_units": 1.0},
    }
    payload = build_payload(metrics, {}, None, config)
    assert len(payload["flags"]) == 1


# ---------------------------------------------------------------------------
# build_data_quality_notes
# ---------------------------------------------------------------------------


def test_data_quality_notes_price_disagreement():
    cmp = SimpleNamespace(ticker="AAPL", disputed=True, disagreement_pct=2.5)
    notes = build_data_quality_notes(price_comparisons=[cmp])
    assert "AAPL" in notes[0] and "2.5%" in notes[0]


def test_data_quality_notes_earnings_unconfirmed():
    from datetime import date

    ec = SimpleNamespace(ticker="AAPL", unconfirmed=True, resolved_date=date(2026, 9, 24))
    notes = build_data_quality_notes(earnings_comparisons=[ec])
    assert "AAPL" in notes[0] and "unconfirmed" in notes[0]


def test_data_quality_notes_stale_banner_goes_first():
    notes = build_data_quality_notes(
        iv_suppressed={"AAPL": "only 2 contracts survived"},
        stale_data_banner="AAPL prices from cache, 2 days old",
    )
    assert notes[0].startswith("STALE DATA:")


# ---------------------------------------------------------------------------
# format_fallback
# ---------------------------------------------------------------------------


def test_format_fallback_never_calls_llm():
    payload = {
        "flags": [{"message": "AAPL gap +2.0%"}],
        "nominal": ["SPY", "QQQ"],
        "data_quality_notes": ["AAPL: IV unavailable"],
    }
    text = format_fallback(payload, CONFIG)
    assert "AAPL gap +2.0%" in text
    assert "2 tickers nominal" in text
    assert "AAPL: IV unavailable" in text


def test_format_fallback_respects_character_cap_by_dropping_lines():
    config = dict(CONFIG, briefing=dict(CONFIG["briefing"], max_characters=30))
    payload = {
        "flags": [{"message": "x" * 20}, {"message": "y" * 20}, {"message": "z" * 20}],
        "nominal": [],
        "data_quality_notes": [],
    }
    text = format_fallback(payload, config)
    assert len(text) <= 30


# ---------------------------------------------------------------------------
# generate_briefing / generate_and_validate (fake Anthropic client)
# ---------------------------------------------------------------------------


def _fake_client(response_text: str):
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=response_text)]
    )
    return client


def test_generate_briefing_extracts_text_blocks():
    client = _fake_client("AAPL gap +2.1%\n3 tickers nominal")
    payload = {"flags": [], "nominal": [], "data_quality_notes": []}
    text = generate_briefing(payload, CONFIG, client=client)
    assert text == "AAPL gap +2.1%\n3 tickers nominal"


def test_generate_briefing_passes_model_and_temperature():
    client = _fake_client("ok")
    payload = {"flags": [], "nominal": [], "data_quality_notes": []}
    generate_briefing(payload, CONFIG, client=client)
    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["temperature"] == 0


def test_generate_and_validate_happy_path():
    client = _fake_client("AAPL gap +2.1%\n3 tickers nominal")
    payload = {"flags": [], "nominal": [], "data_quality_notes": []}
    text = generate_and_validate(payload, CONFIG, client=client)
    assert text == "AAPL gap +2.1%\n3 tickers nominal"


def test_generate_and_validate_raises_on_forbidden_language():
    client = _fake_client("AAPL looks bullish into earnings")
    payload = {"flags": [], "nominal": [], "data_quality_notes": []}
    with pytest.raises(BriefingValidationError):
        generate_and_validate(payload, CONFIG, client=client)


def test_generate_and_validate_raises_on_over_cap_response():
    client = _fake_client("x" * 700)
    payload = {"flags": [], "nominal": [], "data_quality_notes": []}
    with pytest.raises(BriefingValidationError):
        generate_and_validate(payload, CONFIG, client=client)
