"""Orchestration: config -> sources -> reconcile -> chains -> metrics ->
positions -> brief -> deliver, with failure handling.

Design principles this file exists to enforce:

    - Weekdays only, skipping US market holidays (pandas_market_calendars).
    - Every external call already has its own timeout + retry/backoff
      (built into each sources/ client) -- this file's job is to make
      sure one ticker or one source failing degrades that ticker's
      metrics and is reported in data quality notes, without killing
      the run for the other tickers or the market-wide read.
    - If the LLM call fails OR its output fails validation, fall back
      to brief.format_fallback() (raw, code-only flags) rather than
      sending nothing.
    - logs/<date>.json records raw inputs, computed metrics, the final
      output, and source disagreements -- this is what tomorrow's run
      reads back for the "changed materially since yesterday" deltas.

Run it directly for today's briefing:

    .venv/bin/python main.py             # sends the email
    .venv/bin/python main.py --dry-run   # prints instead of sending
    .venv/bin/python main.py --force     # runs even on a non-trading day
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas_market_calendars as mcal
import yaml
from dotenv import load_dotenv

import brief
import chains
import deliver
import metrics
import positions as positions_mod
import snapshot
from reconcile import compare_closes, reconcile_earnings_date
from sources.base import SchemaError, SourceError, SourceUnavailable
from sources.fred import FredClient
from sources.nasdaq import NasdaqClient
from sources.stooq import StooqClient
from sources.yfinance import YFinanceClient

ROOT = Path(__file__).resolve().parent


@dataclass
class Deps:
    """Every external client the run needs, bundled so tests can inject
    fakes for a real end-to-end test of the orchestration logic without
    touching the network, FRED, Yahoo, Stooq, Nasdaq, Anthropic, or SMTP."""

    fred_client: Any
    yf_client: Any
    stooq_client: Any
    nasdaq_client: Any
    anthropic_client: Any = None  # None -> brief.py builds a real one
    smtp_client_factory: Any = None  # None -> deliver.py builds a real one


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_trading_day(d: date, calendar_name: str = "NYSE") -> bool:
    cal = mcal.get_calendar(calendar_name)
    sched = cal.schedule(start_date=d.isoformat(), end_date=d.isoformat())
    return not sched.empty


def _note(prefix: str, exc: Exception) -> str:
    if isinstance(exc, SchemaError):
        return f"{prefix}: SCHEMA ERROR (needs a code fix) -- {exc}"
    return f"{prefix}: {exc}"


# ---------------------------------------------------------------------------
# Per-ticker fetch + metrics computation
# ---------------------------------------------------------------------------


def fetch_ticker_data(
    ticker: str,
    *,
    deps: Deps,
    iv_conn,
    chain_config: chains.ChainCleaningConfig,
    config: dict,
    today: date,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[dict]]:
    """Returns (metrics, raw_inputs, data_quality_notes, disagreements)
    for one ticker. Never raises for an ordinary source failure --
    every external call is individually wrapped so one bad source
    degrades this ticker's metrics, not the whole run."""
    m: Dict[str, Any] = {}
    raw: Dict[str, Any] = {}
    notes: List[str] = []
    disagreements: List[dict] = []

    try:
        bars = deps.yf_client.daily_bars(ticker, lookback_days=260)
    except SourceError as exc:
        notes.append(_note(f"{ticker}: yfinance unavailable", exc))
        return m, raw, notes, disagreements
    if bars.stale:
        notes.append(f"{ticker}: yfinance rate-limited, using cached data")

    closes, highs, lows = bars.frame["Close"], bars.frame["High"], bars.frame["Low"]
    yahoo_close = float(closes.iloc[-1])
    bar_date = bars.frame.index[-1].date()

    stooq_close = None
    try:
        stooq_close = deps.stooq_client.close_on(ticker, bar_date)
    except SourceError as exc:
        notes.append(_note(f"{ticker}: stooq unavailable", exc))

    cmp = compare_closes(
        ticker=ticker,
        as_of=bar_date,
        yahoo_close=yahoo_close,
        stooq_close=stooq_close,
        tolerance_pct=config["price_cross_check"]["tolerance_pct"],
    )
    raw["yahoo_close"] = yahoo_close
    raw["stooq_close"] = stooq_close
    if cmp.disputed:
        notes.append(
            f"{ticker}: Yahoo/Stooq closes disagree {cmp.disagreement_pct:.2f}% -- price metrics skipped"
        )
        disagreements.append(
            {"ticker": ticker, "yahoo_close": yahoo_close, "stooq_close": stooq_close, "pct": cmp.disagreement_pct}
        )
        return m, raw, notes, disagreements

    spot = cmp.trusted_close
    raw["trusted_close"] = spot

    atr14 = metrics.atr(highs, lows, closes, period=config["underlying"]["atr_period"])
    if len(closes) >= 2 and atr14 is not None:
        prev_close = float(closes.iloc[-2])
        today_open = float(bars.frame["Open"].iloc[-1])
        gap = metrics.overnight_gap(prev_close, today_open, atr14)
        m["gap_pct"] = gap.pct
        m["gap_atr_units"] = gap.atr_units

    for period in config["underlying"]["dma_periods"]:
        dma = metrics.sma(closes, period)
        dist = metrics.distance_from_dma_pct(spot, dma)
        if dist is not None:
            m[f"dma{period}_distance_pct"] = dist

    for window in config["underlying"]["realized_vol_windows"]:
        rv = metrics.realized_vol(closes, window=window)
        if rv.value is not None:
            m[f"realized_vol_{window}d"] = rv.value

    rp = metrics.range_percentile(closes, window=config["underlying"]["range_window_days"])
    if rp.value is not None:
        m["range_percentile"] = rp.value

    _fetch_options_metrics(ticker, m, raw, notes, spot, deps, iv_conn, chain_config, config, today)
    _fetch_earnings_metrics(ticker, m, notes, deps, config, today)

    return m, raw, notes, disagreements


def _fetch_options_metrics(
    ticker: str,
    m: dict,
    raw: dict,
    notes: List[str],
    spot: float,
    deps: Deps,
    iv_conn,
    chain_config: chains.ChainCleaningConfig,
    config: dict,
    today: date,
) -> None:
    try:
        expirations = deps.yf_client.list_expirations(ticker)
    except SourceError as exc:
        notes.append(_note(f"{ticker}: option expirations unavailable", exc))
        return

    front_expiry = min(expirations, key=lambda d: abs((d - today).days - snapshot.TARGET_DTE_DAYS))
    try:
        front_chain = deps.yf_client.option_chain(ticker, front_expiry)
    except SourceError as exc:
        notes.append(_note(f"{ticker}: option chain unavailable", exc))
        return

    front_iv = chains.atm_iv_for_ticker(front_chain.calls, front_chain.puts, spot, chain_config)
    raw["atm_iv_survived_count"] = front_iv.survived_count
    snapshot.write_snapshot(
        iv_conn,
        snapshot.IVSnapshot(
            ticker=ticker, as_of=today, atm_iv=front_iv.value, survived_count=front_iv.survived_count, reason=front_iv.reason
        ),
    )

    front_dte = (front_expiry - today).days
    if front_iv.value is None:
        notes.append(f"{ticker}: IV unavailable -- {front_iv.reason}")
    else:
        m["atm_iv"] = front_iv.value
        history = snapshot.read_history(iv_conn, ticker, before=today)
        iv_history_cfg = config["iv_history"]
        rank = metrics.sampled_rank(
            [v * 100 for v in history],
            front_iv.value * 100,
            min_sample=iv_history_cfg["suppress_below_days"],
            mature_sample=iv_history_cfg["mature_sample_days"],
        )
        if rank.sufficient:
            m["iv_rank"] = {"value": rank.value, "sufficient": True, "label": rank.label()}
        else:
            notes.append(f"{ticker}: IV rank suppressed ({rank.label()}, need {iv_history_cfg['suppress_below_days']}+)")

        rv30 = m.get("realized_vol_30d")
        if rv30 is not None:
            m["iv_minus_rv"] = metrics.iv_minus_rv(front_iv.value * 100, rv30)

        cleaned_calls = chains.clean_contracts(front_chain.calls, chain_config).frame
        cleaned_puts = chains.clean_contracts(front_chain.puts, chain_config).frame
        straddle = chains.nearest_strike_straddle(cleaned_calls, cleaned_puts, spot)
        if straddle is not None:
            m["expected_move_dollars"] = straddle.dollars
            m["expected_move_pct"] = straddle.pct_of_spot(spot)

    # Back-month leg for the term structure slope -- best-effort, never
    # blocks the rest of this ticker's metrics if it fails.
    back_candidates = [e for e in expirations if e != front_expiry]
    if back_candidates and front_iv.value is not None:
        back_expiry = min(back_candidates, key=lambda d: abs((d - today).days - snapshot.TARGET_DTE_DAYS * 2))
        try:
            back_chain = deps.yf_client.option_chain(ticker, back_expiry)
            back_iv = chains.atm_iv_for_ticker(back_chain.calls, back_chain.puts, spot, chain_config)
        except SourceError as exc:
            notes.append(_note(f"{ticker}: back-month chain unavailable", exc))
        else:
            if back_iv.value is not None:
                back_dte = (back_expiry - today).days
                slope = metrics.term_structure_slope(front_iv.value * 100, back_iv.value * 100, front_dte, back_dte)
                if slope is not None:
                    m["term_structure_slope"] = slope


def _fetch_earnings_metrics(ticker: str, m: dict, notes: List[str], deps: Deps, config: dict, today: date) -> None:
    try:
        yahoo_date = deps.yf_client.next_earnings_date(ticker, as_of=today)
    except SourceError as exc:
        notes.append(_note(f"{ticker}: earnings date unavailable", exc))
        return
    if yahoo_date is None:
        return

    try:
        comparison = reconcile_earnings_date(
            ticker, yahoo_date, deps.nasdaq_client, search_radius_days=config["earnings"]["nasdaq_search_radius_days"]
        )
    except SchemaError as exc:
        notes.append(_note(f"{ticker}: nasdaq earnings cross-check", exc))
        comparison = None

    resolved = comparison.resolved_date if comparison else yahoo_date
    unconfirmed = comparison.unconfirmed if comparison else True
    m["days_to_earnings"] = metrics.days_to(resolved, today)
    m["earnings_unconfirmed"] = unconfirmed
    if unconfirmed:
        notes.append(f"{ticker}: earnings date {resolved.isoformat()} unconfirmed")


def build_market_metrics(deps: Deps, config: dict) -> Tuple[dict, List[str]]:
    market: Dict[str, Any] = {}
    notes: List[str] = []
    try:
        complex_ = deps.fred_client.vix_complex()
    except SourceError as exc:
        notes.append(_note("FRED VIX complex unavailable", exc))
        complex_ = {}

    for key in ("vix", "vix9d", "vix3m"):
        obs = complex_.get(key)
        if obs is not None:
            market[key] = obs.value

    if all(k in market for k in ("vix", "vix9d", "vix3m")):
        ts = metrics.vix_term_structure(market["vix9d"], market["vix"], market["vix3m"])
        market["vix_term_structure"] = {"shape": ts.shape, "inverted": ts.inverted}
    else:
        notes.append("VIX term structure incomplete -- one or more of VIX/VIX9D/VIX3M unavailable from FRED")

    tbill = complex_.get("tbill_3m")
    if tbill is not None:
        market["risk_free_rate"] = tbill.value / 100

    return market, notes


def build_positions_context(
    snap: Optional[positions_mod.PositionsSnapshot], tickers: List[str], today: date, config: dict
) -> Tuple[Optional[dict], List[str]]:
    if snap is None:
        return None, []

    notes: List[str] = []
    if positions_mod.is_stale(snap, today, config["positions"]["staleness_days"]):
        age = (today - snap.as_of).days
        notes.append(f"positions file is {age}d old (stale, source: {snap.source_path.name})")

    dte_window = tuple(config["positions"]["dte_window"])
    expiring_by_ticker = {
        ticker: [
            {"option_type": p.option_type, "strike": p.strike, "dte": (p.expiration - today).days}
            for p in positions_mod.positions_expiring_within(snap, ticker, today, dte_window)
        ]
        for ticker in tickers
    }
    expiring_by_ticker = {k: v for k, v in expiring_by_ticker.items() if v}

    ctx = {
        "expiring_by_ticker": expiring_by_ticker,
        "concurrent_option_position_count": positions_mod.concurrent_option_position_count(snap),
    }
    return ctx, notes


# ---------------------------------------------------------------------------
# Run log (logs/<date>.json) and yesterday's-metrics lookup
# ---------------------------------------------------------------------------


def write_run_log(logs_dir: Path, today: date, payload: dict) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{today.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_yesterday_metrics(logs_dir: Path, today: date) -> Optional[Dict[str, dict]]:
    if not logs_dir.exists():
        return None
    prior_dates: List[date] = []
    for p in logs_dir.glob("*.json"):
        try:
            d = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d < today:
            prior_dates.append(d)
    if not prior_dates:
        return None
    latest = max(prior_dates)
    try:
        data = json.loads((logs_dir / f"{latest.isoformat()}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("metrics")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_briefing(config: dict, deps: Deps, today: Optional[date] = None) -> str:
    """Runs the full pipeline and returns the final briefing text.
    Never raises for a single ticker/source failure -- those become
    data quality notes. Falls back to brief.format_fallback() if the
    LLM call or its output validation fails."""
    today = today or date.today()

    iv_store_path = ROOT / "data" / "iv_history.sqlite3"
    iv_conn = snapshot.open_store(iv_store_path)
    chain_config = chains.ChainCleaningConfig(
        max_spread_pct_of_mid=config["options"]["max_spread_pct_of_mid"],
        min_surviving_contracts=config["options"]["min_surviving_contracts"],
        iv_sanity_bounds=tuple(config["options"]["iv_sanity_bounds"]),
    )

    tickers = config["tickers"]
    positions_snapshot = positions_mod.load_latest_positions(ROOT / "data" / "positions")
    positions_ctx, positions_notes = build_positions_context(positions_snapshot, tickers, today, config)

    tickers_metrics: Dict[str, dict] = {}
    raw_inputs: Dict[str, dict] = {}
    all_notes: List[str] = list(positions_notes)
    all_disagreements: List[dict] = []

    delay = config["requests"]["inter_request_delay_seconds"]
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(delay)
        m, raw, notes, disagreements = fetch_ticker_data(
            ticker, deps=deps, iv_conn=iv_conn, chain_config=chain_config, config=config, today=today
        )
        tickers_metrics[ticker] = m
        raw_inputs[ticker] = raw
        all_notes.extend(notes)
        all_disagreements.extend(disagreements)

    market_metrics, market_notes = build_market_metrics(deps, config)
    all_notes.extend(market_notes)
    iv_conn.close()

    yesterday = load_yesterday_metrics(ROOT / config["logging"]["dir"], today)
    payload = brief.build_payload(
        tickers_metrics, market_metrics, positions_ctx, config, yesterday=yesterday, data_quality_notes=all_notes
    )

    log_notes = all_notes
    try:
        text = brief.generate_and_validate(payload, config, client=deps.anthropic_client)
    except Exception as exc:  # LLM call failed, or its output failed validation -- fall back, never send nothing
        # The exception detail (e.g. BriefingValidationError) may quote
        # the offending LLM text verbatim for debuggability -- that must
        # never reach the delivered briefing, so it goes in the run log
        # only. The delivered note stays generic.
        log_notes = all_notes + [f"LLM briefing generation failed, sent raw flags instead: {exc}"]
        all_notes.append("LLM briefing generation failed, sent raw flags instead (see run log for detail)")
        payload["data_quality_notes"] = all_notes
        text = brief.format_fallback(payload, config)

    write_run_log(
        ROOT / config["logging"]["dir"],
        today,
        {
            "date": today.isoformat(),
            "metrics": tickers_metrics,
            "market_metrics": market_metrics,
            "output": text,
            "data_quality_notes": log_notes,
            "disagreements": all_disagreements,
            "raw_inputs": raw_inputs,
        },
    )

    return text


def build_deps(config: dict) -> Deps:
    from sources.fred import FredConfig
    from sources.nasdaq import NasdaqConfig
    from sources.stooq import StooqConfig
    from sources.yfinance import YFinanceConfig, configure_cache

    req = config["requests"]
    configure_cache(ROOT / ".yfinance_cache")
    return Deps(
        fred_client=FredClient(FredConfig(
            api_key=os.environ.get("FRED_API_KEY", ""),
            timeout_seconds=req["timeout_seconds"], max_retries=req["max_retries"], backoff_base_seconds=req["backoff_base_seconds"],
        )),
        yf_client=YFinanceClient(YFinanceConfig(
            max_retries=req["max_retries"], backoff_base_seconds=req["backoff_base_seconds"],
        )),
        stooq_client=StooqClient(StooqConfig(
            timeout_seconds=req["timeout_seconds"], max_retries=req["max_retries"], backoff_base_seconds=req["backoff_base_seconds"],
        )),
        nasdaq_client=NasdaqClient(NasdaqConfig(
            timeout_seconds=req["timeout_seconds"], max_retries=req["max_retries"], backoff_base_seconds=req["backoff_base_seconds"],
        )),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Options morning briefing")
    parser.add_argument("--dry-run", action="store_true", help="print the briefing instead of emailing it")
    parser.add_argument("--force", action="store_true", help="run even on a non-trading day")
    parser.add_argument("--date", type=str, default=None, help="override today's date (YYYY-MM-DD), for testing")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config()
    today = date.fromisoformat(args.date) if args.date else date.today()

    if not args.force and not is_trading_day(today):
        print(f"{today.isoformat()} is not a trading day, skipping (use --force to override)")
        return

    deps = build_deps(config)
    text = run_briefing(config, deps, today=today)

    if args.dry_run:
        print(text)
        print(f"\n[{len(text)} characters]")
        return

    smtp_config = deliver.load_smtp_config(config, dict(os.environ))
    subject = f"Options briefing -- {today.isoformat()}"
    deliver.send_briefing(subject, text, smtp_config)


if __name__ == "__main__":
    main()
