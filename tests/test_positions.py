from datetime import date
from pathlib import Path

import pytest

from positions import (
    PositionsSchemaError,
    concurrent_option_position_count,
    find_positions_file,
    is_stale,
    load_latest_positions,
    load_positions,
    option_positions_for_ticker,
    positions_expiring_within,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURE_DIR / "schwab_positions_sample.csv"


# ---------------------------------------------------------------------------
# load_positions against the synthetic Schwab-format fixture
# ---------------------------------------------------------------------------


def test_load_positions_parses_all_real_rows():
    snapshot = load_positions(SAMPLE_CSV)
    assert len(snapshot.positions) == 4  # AAPL equity + 2 AAPL options + 1 SPY option
    assert snapshot.skipped_rows == 2  # the Cash & Cash Investments row + disclaimer footer line


def test_load_positions_as_of_from_filename_date(tmp_path):
    dated = tmp_path / "positions_2026-09-05.csv"
    dated.write_text(SAMPLE_CSV.read_text())
    snapshot = load_positions(dated)
    assert snapshot.as_of == date(2026, 9, 5)


def test_load_positions_equity_row_has_no_option_fields():
    snapshot = load_positions(SAMPLE_CSV)
    equity = next(p for p in snapshot.positions if p.security_type == "Equity")
    assert equity.underlying == "AAPL"
    assert equity.quantity == 100
    assert not equity.is_option
    assert equity.option_type is None
    assert equity.strike is None
    assert equity.expiration is None


def test_load_positions_option_row_parses_symbol_fields():
    snapshot = load_positions(SAMPLE_CSV)
    call = next(p for p in snapshot.positions if p.raw_symbol == "AAPL 09/18/2026 190.00 C")
    assert call.underlying == "AAPL"
    assert call.is_option
    assert call.option_type == "C"
    assert call.strike == pytest.approx(190.00)
    assert call.expiration == date(2026, 9, 18)
    assert call.quantity == -2  # short


def test_load_positions_missing_required_column_names_it(tmp_path):
    # Has Symbol+Quantity (so the header is locatable) but no Security
    # Type -- this is the realistic "Schwab changed a column" case.
    broken = tmp_path / "broken.csv"
    broken.write_text('"Symbol","Description","Quantity","Price"\n"AAPL","APPLE INC","100","$182.50"\n')
    with pytest.raises(PositionsSchemaError) as exc_info:
        load_positions(broken)
    assert "Security Type" in str(exc_info.value)


def test_load_positions_no_header_row_raises(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text("just,some,random,csv\n1,2,3,4\n")
    with pytest.raises(PositionsSchemaError):
        load_positions(broken)


def test_unrecognized_option_looking_symbol_treated_as_non_option(tmp_path):
    # Doesn't match the Schwab option pattern -- should be treated as a
    # plain holding, not crash the load.
    csv_text = (
        '"Symbol","Description","Quantity","Security Type"\n'
        '"WEIRD-TICKER-123","Something","5","Equity"\n'
    )
    f = tmp_path / "weird.csv"
    f.write_text(csv_text)
    snapshot = load_positions(f)
    assert len(snapshot.positions) == 1
    assert not snapshot.positions[0].is_option
    assert snapshot.positions[0].underlying == "WEIRD-TICKER-123"


# ---------------------------------------------------------------------------
# find_positions_file
# ---------------------------------------------------------------------------


def test_find_positions_file_none_when_empty(tmp_path):
    assert find_positions_file(tmp_path) is None


def test_find_positions_file_none_when_directory_missing(tmp_path):
    assert find_positions_file(tmp_path / "does_not_exist") is None


def test_find_positions_file_picks_latest_by_filename_date(tmp_path):
    older = tmp_path / "positions_2026-08-01.csv"
    newer = tmp_path / "positions_2026-09-05.csv"
    older.write_text(SAMPLE_CSV.read_text())
    newer.write_text(SAMPLE_CSV.read_text())
    assert find_positions_file(tmp_path) == newer


def test_find_positions_file_falls_back_to_mtime_without_dated_names(tmp_path):
    import os
    import time

    first = tmp_path / "export.csv"
    first.write_text(SAMPLE_CSV.read_text())
    time.sleep(0.01)
    second = tmp_path / "export_final.csv"
    second.write_text(SAMPLE_CSV.read_text())
    # bump mtime explicitly to avoid filesystem timestamp resolution flakiness
    now = time.time()
    os.utime(first, (now - 100, now - 100))
    os.utime(second, (now, now))
    assert find_positions_file(tmp_path) == second


# ---------------------------------------------------------------------------
# load_latest_positions
# ---------------------------------------------------------------------------


def test_load_latest_positions_none_when_no_file(tmp_path):
    assert load_latest_positions(tmp_path) is None


def test_load_latest_positions_loads_the_latest_file(tmp_path):
    dated = tmp_path / "positions_2026-09-05.csv"
    dated.write_text(SAMPLE_CSV.read_text())
    snapshot = load_latest_positions(tmp_path)
    assert snapshot is not None
    assert len(snapshot.positions) == 4


# ---------------------------------------------------------------------------
# staleness / flag-support helpers
# ---------------------------------------------------------------------------


def test_is_stale_boundary_and_over(tmp_path):
    dated = tmp_path / "positions_2026-09-01.csv"
    dated.write_text(SAMPLE_CSV.read_text())
    snapshot = load_positions(dated)
    assert not is_stale(snapshot, today=date(2026, 9, 4), staleness_days=3)  # exactly at threshold
    assert is_stale(snapshot, today=date(2026, 9, 5), staleness_days=3)  # over


def test_option_positions_for_ticker():
    snapshot = load_positions(SAMPLE_CSV)
    aapl_options = option_positions_for_ticker(snapshot, "AAPL")
    assert len(aapl_options) == 2
    assert all(p.underlying == "AAPL" and p.is_option for p in aapl_options)


def test_concurrent_option_position_count_excludes_equity():
    snapshot = load_positions(SAMPLE_CSV)
    assert concurrent_option_position_count(snapshot) == 3  # 2 AAPL options + 1 SPY option


def test_positions_expiring_within_window():
    snapshot = load_positions(SAMPLE_CSV)
    # AAPL 09/18/2026 call is DTE=10 from 2026-09-08
    expiring = positions_expiring_within(snapshot, "AAPL", today=date(2026, 9, 8), dte_window=(0, 10))
    assert len(expiring) == 1
    assert expiring[0].raw_symbol == "AAPL 09/18/2026 190.00 C"


def test_positions_expiring_within_window_excludes_out_of_range():
    snapshot = load_positions(SAMPLE_CSV)
    expiring = positions_expiring_within(snapshot, "AAPL", today=date(2026, 9, 8), dte_window=(0, 5))
    assert expiring == []
