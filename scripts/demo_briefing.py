"""Runs the real brief.py pipeline (build_flags -> build_payload ->
validator) against realistic synthetic data, and prints the
deterministic fallback text -- no network, no LLM key required. Useful
to sanity-check formatting/terseness without waiting for a live run.

    .venv/bin/python scripts/demo_briefing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brief

CONFIG = {
    "underlying": {"notable_gap_atr": 0.75},
    "iv_rank_bands": {"low": 25, "high": 75},
    "earnings": {"blackout_days": 5},
    "positions": {"max_concurrent_positions": 10, "dte_window": [0, 7]},
    "vix": {"flag_inversion": True},
    "delta_thresholds": {
        "dma_distance_pct": 3.0, "realized_vol_pts": 5.0, "iv_minus_rv_pts": 5.0,
        "term_structure_slope_pts": 2.0, "range_percentile_pts": 25.0, "vix_level_pts": 2.0,
    },
    "briefing": {"max_characters": 600, "max_flags": 8, "llm_model": "claude-sonnet-4-6", "llm_temperature": 0},
}

BUSY_DAY_METRICS = {
    "SPY": {
        "gap_pct": 0.2, "gap_atr_units": 0.15,
        "dma20_distance_pct": 1.1, "realized_vol_20d": 11.2, "range_percentile": 58.0,
    },
    "QQQ": {
        "gap_pct": -1.8, "gap_atr_units": -1.05,
        "dma20_distance_pct": -2.4, "realized_vol_20d": 17.8, "range_percentile": 12.0,
    },
    "AAPL": {
        "gap_pct": 0.4, "gap_atr_units": 0.3,
        "iv_rank": {"value": 82, "sufficient": True, "label": "41-day sample — thin"},
        "days_to_earnings": 3, "earnings_unconfirmed": False,
        "realized_vol_20d": 24.5,
    },
}
BUSY_DAY_MARKET = {
    "vix": 24.1, "vix9d": 27.8, "vix3m": 21.0,
    "vix_term_structure": {"shape": "backwardation", "inverted": True},
}
BUSY_DAY_POSITIONS_CTX = {
    "expiring_by_ticker": {"AAPL": [{"option_type": "C", "strike": 190.0, "dte": 5}]},
    "concurrent_option_position_count": 4,
}
BUSY_DAY_DQ_NOTES = ["QQQ: IV unavailable — only 3 contracts survived filtering (min 4)"]

QUIET_DAY_METRICS = {
    "SPY": {"gap_pct": 0.1, "gap_atr_units": 0.08, "realized_vol_20d": 10.5},
    "QQQ": {"gap_pct": -0.2, "gap_atr_units": -0.12, "realized_vol_20d": 13.1},
    "AAPL": {"gap_pct": 0.3, "gap_atr_units": 0.2, "realized_vol_20d": 18.9},
}
QUIET_DAY_MARKET = {"vix": 14.8, "vix9d": 13.9, "vix3m": 15.6, "vix_term_structure": {"shape": "contango", "inverted": False}}


def show(label: str, metrics: dict, market: dict, positions_ctx, dq_notes) -> dict:
    payload = brief.build_payload(metrics, market, positions_ctx, CONFIG, data_quality_notes=dq_notes)
    text = brief.format_fallback(payload, CONFIG)
    print(f"=== {label} ===")
    print(json.dumps(payload, indent=2))
    print("--- rendered (deterministic, no LLM) ---")
    print(text)
    print(f"[{len(text)} characters]")
    print()
    return payload


if __name__ == "__main__":
    show("BUSY DAY", BUSY_DAY_METRICS, BUSY_DAY_MARKET, BUSY_DAY_POSITIONS_CTX, BUSY_DAY_DQ_NOTES)
    show("QUIET DAY", QUIET_DAY_METRICS, QUIET_DAY_MARKET, None, [])
