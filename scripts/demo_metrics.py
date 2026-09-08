"""Print every metrics.py calculation for one fixture ticker.

Not part of the pipeline — a standalone check so the math can be
eyeballed before sources/, chains.py, or brief.py get wired up. Run:

    .venv/bin/python scripts/demo_metrics.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics as m

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "AAPL_sample.csv"


def main() -> None:
    df = pd.read_csv(FIXTURE_PATH, parse_dates=["Date"])
    closes, highs, lows = df["Close"], df["High"], df["Low"]
    spot = float(closes.iloc[-1])
    as_of = df["Date"].iloc[-1].date()

    print(f"=== AAPL (fixture) as of {as_of} — {len(df)} bars ===\n")

    print(f"last close: {spot:.2f}")

    atr14 = m.atr(highs, lows, closes, period=14)
    print(f"ATR(14): {atr14:.3f}")

    gap = m.overnight_gap(prev_close=float(closes.iloc[-2]), today_open=float(df['Open'].iloc[-1]), atr_value=atr14)
    print(f"overnight gap: {gap.pct:+.2f}%  ({gap.atr_units:+.2f} ATR)")

    for period in (20, 50, 200):
        dma = m.sma(closes, period)
        dist = m.distance_from_dma_pct(spot, dma)
        if dma is None:
            print(f"{period}DMA: unavailable ({len(closes)} bars < {period})")
        else:
            print(f"{period}DMA: {dma:.2f}  (spot {dist:+.2f}% away)")

    rp = m.range_percentile(closes, window=30)
    print(f"30d range percentile: {rp.value:.1f}  [{rp.label()}]")

    for window in (20, 30):
        rv = m.realized_vol(closes, window=window)
        print(f"realized vol ({window}d): {rv.value:.2f}%  [{rv.label()}]")

    print()
    print("--- options (synthetic inputs — chains.py not wired up yet) ---")
    atm_iv = 0.28  # placeholder, stands in for chains.py's cleaned ATM IV
    rv30 = m.realized_vol(closes, window=30).value
    print(f"ATM IV (placeholder): {atm_iv * 100:.1f}%")
    print(f"IV - RV(30): {m.iv_minus_rv(atm_iv * 100, rv30):+.2f} pts")

    iv_history = [0.22, 0.24, 0.19, 0.31, 0.27, 0.25, 0.30, 0.21]  # placeholder
    rank = m.sampled_rank([v * 100 for v in iv_history], atm_iv * 100, min_sample=20, mature_sample=126)
    print(f"IV rank: {'suppressed' if not rank.sufficient else f'{rank.value:.0f}'}  [{rank.label()}] "
          f"(min_sample=20 not met yet in real snapshot history)")

    slope = m.term_structure_slope(front_iv=28.0, back_iv=26.5, front_dte=30, back_dte=58)
    print(f"term structure slope (front->back, /30d): {slope:+.2f} pts")

    move = m.expected_move(atm_iv=atm_iv, spot=spot, dte=30)
    print(f"expected move to 30d monthly: ±${move.dollars:.2f}  (±{move.pct:.2f}%)")

    print()
    print("--- market (placeholder VIX complex — fred.py not wired up yet) ---")
    ts = m.vix_term_structure(vix9d=14.2, vix=15.1, vix3m=16.4)
    print(f"VIX9D/VIX/VIX3M shape: {ts.shape}  (inverted={ts.inverted})")

    print()
    print("--- calendar ---")
    earnings = date(2026, 9, 24)
    print(f"days to earnings: {m.days_to(earnings, as_of)}")


if __name__ == "__main__":
    main()
