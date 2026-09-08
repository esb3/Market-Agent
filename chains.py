"""Option chain cleaning and ATM IV derivation.

Yahoo's option chain includes an impliedVolatility field, but it is
unreliable on illiquid contracts and must not be consumed naively. This
is the cleaning layer the project brief calls for:

    1. Drop contracts with zero volume AND zero open interest.
    2. Drop contracts with a zero bid, or a bid-ask spread wider than a
       configured fraction of the mid.
    3. Never use lastPrice (goes stale for hours) -- always the mid.
    4. Derive ATM IV by interpolating across the two strikes bracketing
       spot, not by reading one contract's IV field.
    5. Sanity-bound the interpolated result; anything outside a
       plausible range is a data error, not a real reading.
    6. If too few contracts survive filtering, suppress IV output for
       that ticker entirely and say why -- never report a number built
       on a handful of illiquid quotes.

Pure functions operating on the calls/puts DataFrames sources/yfinance
returns (already schema-validated there) -- no I/O here, fully
unit-testable against constructed fixture frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChainCleaningConfig:
    max_spread_pct_of_mid: float  # e.g. 0.15 -- config.yaml options.max_spread_pct_of_mid
    min_surviving_contracts: int  # combined calls+puts -- options.min_surviving_contracts
    iv_sanity_bounds: Tuple[float, float]  # e.g. (0.05, 3.0) -- options.iv_sanity_bounds


@dataclass(frozen=True)
class CleaningResult:
    frame: pd.DataFrame  # survivors, with a 'mid' column added
    total: int
    survived: int
    dropped_illiquid: int  # zero volume AND zero open interest
    dropped_zero_bid: int
    dropped_wide_spread: int


@dataclass(frozen=True)
class ATMIVResult:
    value: Optional[float]  # None if suppressed
    survived_count: int
    reason: Optional[str]  # why suppressed, when value is None


@dataclass(frozen=True)
class StraddleQuote:
    strike: float
    call_mid: float
    put_mid: float

    @property
    def dollars(self) -> float:
        """Expected move to this expiration, in dollars, read directly
        off the market's ATM straddle price -- not backed out of IV."""
        return self.call_mid + self.put_mid

    def pct_of_spot(self, spot: float) -> Optional[float]:
        if not spot:
            return None
        return self.dollars / spot * 100


def mid_price(bid: pd.Series, ask: pd.Series) -> pd.Series:
    return (bid + ask) / 2


def clean_contracts(frame: pd.DataFrame, config: ChainCleaningConfig) -> CleaningResult:
    total = len(frame)
    if total == 0:
        return CleaningResult(frame=frame, total=0, survived=0, dropped_illiquid=0, dropped_zero_bid=0, dropped_wide_spread=0)

    working = frame.copy()
    working["mid"] = mid_price(working["bid"], working["ask"])

    illiquid = (working["volume"].fillna(0) <= 0) & (working["openInterest"].fillna(0) <= 0)
    zero_bid = working["bid"].fillna(0) <= 0
    mid = working["mid"]
    # A non-positive mid can't have a meaningful spread ratio -- treat
    # as maximally wide (dropped) rather than dividing by zero.
    spread_pct = pd.Series(
        np.where(mid > 0, (working["ask"] - working["bid"]) / mid, np.inf),
        index=working.index,
    )
    wide_spread = spread_pct > config.max_spread_pct_of_mid

    keep = ~illiquid & ~zero_bid & ~wide_spread
    survivors = working[keep]

    return CleaningResult(
        frame=survivors,
        total=total,
        survived=int(keep.sum()),
        dropped_illiquid=int(illiquid.sum()),
        dropped_zero_bid=int(zero_bid.sum()),
        dropped_wide_spread=int(wide_spread.sum()),
    )


def implied_vol_by_strike(cleaned_calls: pd.DataFrame, cleaned_puts: pd.DataFrame) -> pd.Series:
    """Per-strike IV, averaging call and put IV where both survived
    filtering at that strike; falls back to whichever side survived
    alone. Indexed by strike, ascending."""
    call_iv = cleaned_calls.groupby("strike")["impliedVolatility"].mean() if len(cleaned_calls) else pd.Series(dtype=float)
    put_iv = cleaned_puts.groupby("strike")["impliedVolatility"].mean() if len(cleaned_puts) else pd.Series(dtype=float)
    combined = pd.concat([call_iv, put_iv], axis=1, keys=["call", "put"])
    if combined.empty:
        return pd.Series(dtype=float)
    return combined.mean(axis=1, skipna=True).sort_index()


def interpolate_atm_iv(iv_by_strike: pd.Series, spot: float) -> Optional[float]:
    """Linear interpolation between the two strikes bracketing spot.
    None if fewer than 2 strikes are available, or spot falls outside
    the surviving strike range -- extrapolation isn't attempted; an
    out-of-range ATM read is worse than none."""
    strikes = iv_by_strike.dropna()
    if len(strikes) < 2:
        return None
    idx = strikes.index.to_numpy(dtype=float)
    if spot in strikes.index:
        return float(strikes.loc[spot])
    if spot < idx.min() or spot > idx.max():
        return None
    lower = idx[idx <= spot].max()
    upper = idx[idx >= spot].min()
    lower_iv, upper_iv = float(strikes.loc[lower]), float(strikes.loc[upper])
    weight = (spot - lower) / (upper - lower)
    return lower_iv + weight * (upper_iv - lower_iv)


def atm_iv_for_ticker(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, config: ChainCleaningConfig) -> ATMIVResult:
    calls_clean = clean_contracts(calls, config)
    puts_clean = clean_contracts(puts, config)
    survived = calls_clean.survived + puts_clean.survived

    if survived < config.min_surviving_contracts:
        return ATMIVResult(
            value=None,
            survived_count=survived,
            reason=f"only {survived} contracts survived filtering (min {config.min_surviving_contracts})",
        )

    iv_by_strike = implied_vol_by_strike(calls_clean.frame, puts_clean.frame)
    interpolated = interpolate_atm_iv(iv_by_strike, spot)
    if interpolated is None:
        return ATMIVResult(value=None, survived_count=survived, reason="spot is outside the available strike range after cleaning")

    lo, hi = config.iv_sanity_bounds
    if not (lo <= interpolated <= hi):
        return ATMIVResult(
            value=None,
            survived_count=survived,
            reason=f"interpolated IV {interpolated:.2f} outside sanity bounds [{lo}, {hi}]",
        )

    return ATMIVResult(value=interpolated, survived_count=survived, reason=None)


def nearest_strike_straddle(cleaned_calls: pd.DataFrame, cleaned_puts: pd.DataFrame, spot: float) -> Optional[StraddleQuote]:
    """Call+put mid at whichever surviving strike is closest to spot
    and quoted on both sides -- a straddle needs both legs."""
    common_strikes = set(cleaned_calls["strike"]).intersection(set(cleaned_puts["strike"]))
    if not common_strikes:
        return None
    nearest = min(common_strikes, key=lambda k: abs(k - spot))
    call_mid = float(cleaned_calls.loc[cleaned_calls["strike"] == nearest, "mid"].mean())
    put_mid = float(cleaned_puts.loc[cleaned_puts["strike"] == nearest, "mid"].mean())
    return StraddleQuote(strike=float(nearest), call_mid=call_mid, put_mid=put_mid)
