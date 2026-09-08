"""Pure metric calculations for the morning briefing.

No I/O, no network, no config parsing — every function here takes plain
data (prices, IV history, dates) and returns a number or a small typed
result. Fetching lives in sources/, thresholds live in config.yaml,
wiring lives in main.py.

Honesty rule: any metric whose reliability depends on how much history
backs it (realized vol, range percentile, IV rank/percentile, VIX
percentile) returns a `SampledValue` carrying its own sample size, so
callers can label or suppress it instead of presenting it bare.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Sample-size-aware values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledValue:
    """A derived number plus how much history backs it.

    `min_sample` is the threshold below which the value should be
    suppressed entirely (not just labeled). `mature_sample` is the size
    at or above which it's no longer "thin"; if omitted it defaults to
    twice `min_sample`. Between the two, the value is shown but flagged.
    """

    value: Optional[float]
    sample_size: int
    min_sample: int
    mature_sample: Optional[int] = None

    @property
    def sufficient(self) -> bool:
        return self.value is not None and self.sample_size >= self.min_sample

    @property
    def thin(self) -> bool:
        mature = self.mature_sample if self.mature_sample is not None else self.min_sample * 2
        return self.sufficient and self.sample_size < mature

    def label(self, unit: str = "day") -> str:
        if self.value is None:
            return "unavailable"
        qualifier = " — thin" if self.thin else ""
        return f"{self.sample_size}-{unit} sample{qualifier}"


# ---------------------------------------------------------------------------
# Underlying: moving averages, ATR, gap, range
# ---------------------------------------------------------------------------


def sma(closes: pd.Series, period: int) -> Optional[float]:
    """Simple moving average of the last `period` closes, or None if
    there isn't enough history yet."""
    if len(closes) < period:
        return None
    return float(closes.iloc[-period:].mean())


def distance_from_dma_pct(close: float, dma: Optional[float]) -> Optional[float]:
    """% distance of `close` above/below a moving average."""
    if dma is None or dma == 0:
        return None
    return (close - dma) / dma * 100


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Optional[float]:
    """Average True Range over `period` days (simple average of TR, not
    Wilder-smoothed)."""
    if len(closes) < period + 1:
        return None
    prev_close = closes.shift(1)
    tr = true_range(highs, lows, prev_close).iloc[1:]
    if len(tr) < period:
        return None
    return float(tr.iloc[-period:].mean())


@dataclass(frozen=True)
class GapResult:
    pct: float
    atr_units: Optional[float]


def overnight_gap(prev_close: float, today_open: float, atr_value: Optional[float]) -> GapResult:
    """Gap between yesterday's close and today's open, in % and in ATR
    units (None if ATR isn't available)."""
    pct = (today_open - prev_close) / prev_close * 100
    atr_units = (today_open - prev_close) / atr_value if atr_value else None
    return GapResult(pct=pct, atr_units=atr_units)


def range_percentile(closes: pd.Series, window: int = 30) -> SampledValue:
    """Where the latest close sits within the [min, max] of the last
    `window` closes, as a 0-100 percentile."""
    n = min(len(closes), window)
    if n == 0:
        return SampledValue(None, 0, window)
    recent = closes.iloc[-n:]
    lo, hi = float(recent.min()), float(recent.max())
    current = float(closes.iloc[-1])
    pct = 50.0 if hi == lo else (current - lo) / (hi - lo) * 100
    # sample_size is capped at `window` by construction, so a full window
    # is already "mature" — there's no such thing as a thin-but-sufficient
    # reading here (unlike IV/VIX rank, whose history keeps growing).
    return SampledValue(pct, n, window, mature_sample=window)


def realized_vol(closes: pd.Series, window: int, annualize: int = TRADING_DAYS_PER_YEAR) -> SampledValue:
    """Annualized realized volatility (%) from the last `window` daily
    log returns."""
    log_returns = np.log(closes / closes.shift(1)).dropna()
    n = min(len(log_returns), window)
    if n < 2:
        return SampledValue(None, n, window)
    recent = log_returns.iloc[-n:]
    vol = float(recent.std(ddof=1) * np.sqrt(annualize) * 100)
    # Same reasoning as range_percentile: sample_size is capped at
    # `window`, so a full window is already mature, not thin.
    return SampledValue(vol, n, window, mature_sample=window)


# ---------------------------------------------------------------------------
# Rank / percentile against a history (shared by IV and VIX)
# ---------------------------------------------------------------------------


def _clean(history: Sequence[Optional[float]]) -> list:
    return [v for v in history if v is not None]


def sampled_rank(
    history: Sequence[Optional[float]],
    current: float,
    min_sample: int,
    mature_sample: Optional[int] = None,
) -> SampledValue:
    """0-100 rank of `current` within [min(history+current), max(...)].
    This is "IV rank" / "VIX rank" style min-max scaling."""
    hist = _clean(history)
    n = len(hist)
    if n == 0 or current is None:
        return SampledValue(None, n, min_sample, mature_sample)
    lo, hi = min(hist + [current]), max(hist + [current])
    rank = 50.0 if hi == lo else (current - lo) / (hi - lo) * 100
    return SampledValue(rank, n, min_sample, mature_sample)


def sampled_percentile(
    history: Sequence[Optional[float]],
    current: float,
    min_sample: int,
    mature_sample: Optional[int] = None,
) -> SampledValue:
    """% of historical observations at or below `current`. This is "IV
    percentile" / "VIX percentile" style empirical CDF."""
    hist = _clean(history)
    n = len(hist)
    if n == 0 or current is None:
        return SampledValue(None, n, min_sample, mature_sample)
    below = sum(1 for v in hist if v <= current)
    pct = below / n * 100
    return SampledValue(pct, n, min_sample, mature_sample)


# ---------------------------------------------------------------------------
# Options: IV vs RV, term structure, expected move
# ---------------------------------------------------------------------------


def iv_minus_rv(iv: Optional[float], rv: Optional[float]) -> Optional[float]:
    """ATM IV minus realized vol, both as annualized % points."""
    if iv is None or rv is None:
        return None
    return iv - rv


def term_structure_slope(front_iv: float, back_iv: float, front_dte: int, back_dte: int) -> Optional[float]:
    """Slope of the IV term structure between two monthlies, expressed
    as IV points per 30 additional days of tenor. Positive = upward
    sloping (contango); negative = inverted."""
    if front_iv is None or back_iv is None or back_dte == front_dte:
        return None
    return (back_iv - front_iv) / (back_dte - front_dte) * 30


@dataclass(frozen=True)
class ExpectedMove:
    dollars: float
    pct: float


def expected_move(atm_iv: Optional[float], spot: Optional[float], dte: int) -> Optional[ExpectedMove]:
    """Expected move to the nearest monthly expiration from the ATM
    straddle approximation: spot * IV * sqrt(dte / 365)."""
    if atm_iv is None or spot is None or dte is None or dte <= 0:
        return None
    factor = atm_iv * np.sqrt(dte / 365)
    return ExpectedMove(dollars=float(spot * factor), pct=float(factor * 100))


# ---------------------------------------------------------------------------
# Market: VIX term structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VixTermStructure:
    shape: str  # "contango" | "backwardation" | "flat" | "mixed"
    inverted: bool  # True whenever the curve isn't vix9d <= vix <= vix3m


def vix_term_structure(vix9d: float, vix: float, vix3m: float) -> VixTermStructure:
    normal = vix9d <= vix <= vix3m
    full_backwardation = vix9d >= vix >= vix3m
    if normal:
        shape = "flat" if vix9d == vix == vix3m else "contango"
    elif full_backwardation:
        shape = "backwardation"
    else:
        shape = "mixed"
    return VixTermStructure(shape=shape, inverted=not normal)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def days_to(target: Optional[date], as_of: date) -> Optional[int]:
    if target is None:
        return None
    return (target - as_of).days
