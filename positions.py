"""Manual positions CSV parser — Schwab "Positions" export.

Positions and trade history come from CSV files exported by hand from
Schwab's web UI (Positions page -> Export) and dropped into
data/positions/. This module never fetches them over the network.

Schwab's export has a few quirks this parser works around:
    - The file opens with a title/metadata line (e.g. "Positions for
      account ...as of...") and a blank line before the real header
      row appears -- the header is located by content ("Symbol" +
      "Quantity" on the same line), not assumed to be line 1.
    - After the position rows there's usually a "Cash & Cash
      Investments" summary row (no parseable Quantity) and a
      disclaimer paragraph -- both are skipped as non-data rather than
      raising, since they aren't a schema problem.
    - An option row's Symbol is Schwab's human-readable option symbol,
      e.g. "AAPL 09/18/2026 150.00 C" (ticker, expiration, strike,
      C/P). A Symbol that doesn't match this pattern is treated as a
      non-option holding (equity, ETF, etc.) rather than an error --
      most accounts hold a mix.

This project was built without access to a real Schwab export to
verify column names and the option-symbol string format against (see
the top-level build notes on this sandbox's network restrictions).
Run it against your real file the first time before trusting it:

    .venv/bin/python -c "from pathlib import Path; import positions as p; s = p.load_latest_positions(Path('data/positions')); print(s.positions if s else 'no file found')"

A schema mismatch fails loudly with PositionsSchemaError naming the
missing column, rather than silently producing wrong flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

REQUIRED_COLUMNS = ["Symbol", "Quantity", "Security Type"]

_OPTION_SYMBOL_RE = re.compile(r"^([A-Z][A-Z.]{0,5})\s+(\d{2})/(\d{2})/(\d{4})\s+([\d.]+)\s+([CP])$")
_FILENAME_DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


class PositionsSchemaError(Exception):
    """The CSV doesn't look like a Schwab positions export (missing
    required column, or no header row found). Never treat this as
    zero/empty positions -- surface it so a format change gets noticed
    instead of silently disabling position-aware flags."""


@dataclass(frozen=True)
class ParsedOptionSymbol:
    underlying: str
    expiration: date
    strike: float
    option_type: str  # "C" or "P"


@dataclass(frozen=True)
class Position:
    raw_symbol: str
    underlying: str
    security_type: str  # as Schwab reports it, e.g. "Equity", "Option"
    quantity: float  # negative for short
    option_type: Optional[str]  # "C" / "P", None for non-options
    strike: Optional[float]
    expiration: Optional[date]
    description: Optional[str] = None

    @property
    def is_option(self) -> bool:
        return self.option_type is not None


@dataclass(frozen=True)
class PositionsSnapshot:
    positions: List[Position]
    source_path: Path
    as_of: date  # date the export was taken (from filename, else file mtime)
    skipped_rows: int  # rows present but not parseable as a position (cash line, etc.)


def find_positions_file(directory: Path) -> Optional[Path]:
    """Most recent CSV in `directory`, by filename date if present else
    file mtime. None if the directory has no CSV files -- callers must
    run fine with no positions file, skipping position-aware flags."""
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*.csv"))
    if not candidates:
        return None

    def sort_key(p: Path):
        return (_date_from_filename(p.name) or date.min, p.stat().st_mtime)

    return max(candidates, key=sort_key)


def load_positions(path: Path) -> PositionsSnapshot:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_idx = _find_header_row(lines)
    if header_idx is None:
        raise PositionsSchemaError(
            f"{path.name}: couldn't find a header row containing 'Symbol' and 'Quantity' -- "
            "is this a Schwab Positions export?"
        )

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip() for c in df.columns]
    _validate_columns(df, path)

    positions: List[Position] = []
    skipped = 0
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol or symbol.lower() == "nan":
            skipped += 1
            continue
        qty_raw = row.get("Quantity")
        if pd.isna(qty_raw):
            # e.g. the "Cash & Cash Investments" row, or a trailing
            # disclaimer line pandas read as a ragged single-cell row.
            skipped += 1
            continue
        try:
            quantity = float(str(qty_raw).replace(",", "").replace("$", ""))
        except (TypeError, ValueError):
            skipped += 1
            continue

        sec_type = str(row.get("Security Type", "")).strip()
        parsed = _parse_option_symbol(symbol)
        description = row.get("Description")
        positions.append(
            Position(
                raw_symbol=symbol,
                underlying=parsed.underlying if parsed else symbol,
                security_type=sec_type,
                quantity=quantity,
                option_type=parsed.option_type if parsed else None,
                strike=parsed.strike if parsed else None,
                expiration=parsed.expiration if parsed else None,
                description=None if description is None or str(description) == "nan" else str(description),
            )
        )

    as_of = _date_from_filename(path.name) or date.fromtimestamp(path.stat().st_mtime)
    return PositionsSnapshot(positions=positions, source_path=path, as_of=as_of, skipped_rows=skipped)


def load_latest_positions(directory: Path) -> Optional[PositionsSnapshot]:
    """None when no positions file exists -- the tool must run fine
    with no file present, skipping position-aware flags entirely."""
    path = find_positions_file(directory)
    if path is None:
        return None
    return load_positions(path)


def is_stale(snapshot: PositionsSnapshot, today: date, staleness_days: int) -> bool:
    return (today - snapshot.as_of).days > staleness_days


def option_positions_for_ticker(snapshot: PositionsSnapshot, ticker: str) -> List[Position]:
    return [p for p in snapshot.positions if p.is_option and p.underlying == ticker]


def concurrent_option_position_count(snapshot: PositionsSnapshot) -> int:
    """Count of open option position rows, for the max-concurrent-
    positions concentration rule."""
    return sum(1 for p in snapshot.positions if p.is_option)


def positions_expiring_within(
    snapshot: PositionsSnapshot, ticker: str, today: date, dte_window: Tuple[int, int]
) -> List[Position]:
    lo, hi = dte_window
    result = []
    for p in option_positions_for_ticker(snapshot, ticker):
        if p.expiration is None:
            continue
        dte = (p.expiration - today).days
        if lo <= dte <= hi:
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _find_header_row(lines: Sequence[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        if "Symbol" in line and "Quantity" in line:
            return i
    return None


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PositionsSchemaError(
            f"{path.name}: missing expected column(s) {missing} -- Schwab's export format may "
            f"have changed. Found columns: {list(df.columns)}"
        )


def _parse_option_symbol(raw: str) -> Optional[ParsedOptionSymbol]:
    match = _OPTION_SYMBOL_RE.match(raw.strip())
    if not match:
        return None
    ticker, month, day, year, strike, cp = match.groups()
    try:
        expiration = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return ParsedOptionSymbol(underlying=ticker, expiration=expiration, strike=float(strike), option_type=cp)


def _date_from_filename(name: str) -> Optional[date]:
    match = _FILENAME_DATE_RE.search(name)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None
