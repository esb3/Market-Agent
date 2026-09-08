"""Daily ATM IV snapshot writer — the fix for the IV-history sample-size
problem.

No free source gives historical implied volatility, so IV rank and IV
percentile cannot be computed on day one. This module is what makes
them eventually mean something: it writes cleaned 30-day ATM IV per
ticker to a local SQLite store once a day, every day, starting
immediately, independent of whether the morning briefing itself runs.
metrics.SampledValue is what later reports the honest, growing sample
size (e.g. "IV rank 82 (41-day sample — thin)") once brief.py reads
this history back.

Run standalone (its own cron/launchd entry, separate from the
briefing's schedule):

    .venv/bin/python snapshot.py

Writes are idempotent per (ticker, date): re-running the same day
overwrites that day's row instead of duplicating it, so a retry or a
manual re-run is always safe.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from chains import ChainCleaningConfig, atm_iv_for_ticker
from sources.base import SourceError

# Fixed by definition, not a config knob: changing it would break
# comparability of the historical series IV rank/percentile depend on.
TARGET_DTE_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS iv_snapshots (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,       -- ISO date the IV reading applies to
    atm_iv REAL,               -- NULL when IV was unavailable/suppressed that day
    survived_count INTEGER NOT NULL,
    reason TEXT,               -- why atm_iv is NULL, if it is
    written_at TEXT NOT NULL,  -- ISO timestamp this row was written
    PRIMARY KEY (ticker, as_of)
);
"""


@dataclass(frozen=True)
class IVSnapshot:
    ticker: str
    as_of: date
    atm_iv: Optional[float]
    survived_count: int
    reason: Optional[str] = None


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def write_snapshot(conn: sqlite3.Connection, snapshot: IVSnapshot, written_at: Optional[str] = None) -> None:
    conn.execute(
        """
        INSERT INTO iv_snapshots (ticker, as_of, atm_iv, survived_count, reason, written_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, as_of) DO UPDATE SET
            atm_iv=excluded.atm_iv,
            survived_count=excluded.survived_count,
            reason=excluded.reason,
            written_at=excluded.written_at
        """,
        (
            snapshot.ticker,
            snapshot.as_of.isoformat(),
            snapshot.atm_iv,
            snapshot.survived_count,
            snapshot.reason,
            written_at or datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def read_history(conn: sqlite3.Connection, ticker: str, before: date, limit_days: Optional[int] = None) -> List[float]:
    """Historical non-null ATM IV values strictly before `before`,
    oldest first, capped to the most recent `limit_days` if given.
    Suppressed-day rows (atm_iv NULL) are excluded — IV rank must never
    be computed with a silently-substituted value for a day IV was
    unavailable."""
    rows = conn.execute(
        """
        SELECT atm_iv FROM iv_snapshots
        WHERE ticker = ? AND as_of < ? AND atm_iv IS NOT NULL
        ORDER BY as_of ASC
        """,
        (ticker, before.isoformat()),
    ).fetchall()
    values = [r[0] for r in rows]
    if limit_days is not None:
        values = values[-limit_days:]
    return values


def sample_size(conn: sqlite3.Connection, ticker: str, before: date) -> int:
    """True count of non-null historical IV readings before `before` —
    the number metrics.SampledValue needs, distinct from how many
    calendar days have merely passed since this project started."""
    return len(read_history(conn, ticker, before))


def _pick_target_expiration(expirations: Sequence[date], as_of: date) -> date:
    return min(expirations, key=lambda d: abs((d - as_of).days - TARGET_DTE_DAYS))


def run_daily_snapshot(
    tickers: Sequence[str],
    yf_client,
    chain_config: ChainCleaningConfig,
    store_path: Path,
    inter_request_delay_seconds: float = 1.5,
    as_of: Optional[date] = None,
) -> List[IVSnapshot]:
    """One pass: for each ticker, pull the chain nearest 30 DTE, compute
    cleaned ATM IV, and write one row. Serialized with a delay between
    tickers like every other yfinance-touching path in this project.
    One ticker's failure (rate limit exhausted, schema error) is
    recorded as an unavailable reading and does NOT abort the rest of
    the run — a single bad ticker must not stop IV history from
    accumulating for the others.
    """
    as_of = as_of or date.today()
    conn = open_store(store_path)
    results: List[IVSnapshot] = []
    try:
        for i, ticker in enumerate(tickers):
            if i > 0:
                time.sleep(inter_request_delay_seconds)
            snapshot = _snapshot_one(ticker, yf_client, chain_config, as_of)
            write_snapshot(conn, snapshot)
            results.append(snapshot)
    finally:
        conn.close()
    return results


def _snapshot_one(ticker: str, yf_client, chain_config: ChainCleaningConfig, as_of: date) -> IVSnapshot:
    try:
        expirations = yf_client.list_expirations(ticker)
        target = _pick_target_expiration(expirations, as_of)
        chain = yf_client.option_chain(ticker, target)
        bars = yf_client.daily_bars(ticker, lookback_days=5)
        spot = float(bars.frame["Close"].iloc[-1])
        result = atm_iv_for_ticker(chain.calls, chain.puts, spot, chain_config)
        return IVSnapshot(ticker=ticker, as_of=as_of, atm_iv=result.value, survived_count=result.survived_count, reason=result.reason)
    except SourceError as exc:
        return IVSnapshot(ticker=ticker, as_of=as_of, atm_iv=None, survived_count=0, reason=f"snapshot failed: {exc}")


if __name__ == "__main__":
    import sys

    import yaml
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from sources.yfinance import YFinanceClient, YFinanceConfig, configure_cache  # noqa: E402

    load_dotenv(root / ".env")
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    configure_cache(root / ".yfinance_cache")
    yf_client = YFinanceClient(
        YFinanceConfig(
            max_retries=cfg["requests"]["max_retries"],
            backoff_base_seconds=cfg["requests"]["backoff_base_seconds"],
        )
    )
    chain_config = ChainCleaningConfig(
        max_spread_pct_of_mid=cfg["options"]["max_spread_pct_of_mid"],
        min_surviving_contracts=cfg["options"]["min_surviving_contracts"],
        iv_sanity_bounds=tuple(cfg["options"]["iv_sanity_bounds"]),
    )

    results = run_daily_snapshot(
        tickers=cfg["tickers"],
        yf_client=yf_client,
        chain_config=chain_config,
        store_path=root / "data" / "iv_history.sqlite3",
        inter_request_delay_seconds=cfg["requests"]["inter_request_delay_seconds"],
    )
    for r in results:
        status = f"{r.atm_iv:.2%}" if r.atm_iv is not None else f"unavailable ({r.reason})"
        print(f"{r.ticker}: {status}")
