"""One-shot setup check: config.yaml, .env, FRED series IDs, and a live
SMTP login (no email sent). Run this once after setup, and again any
time .env or config.yaml changes -- catches a bad App Password or a
wrong FRED series ID before the first real 8:30am run, not during it.

    .venv/bin/python scripts/doctor.py

Config, the FRED key, FRED series resolution, and SMTP are treated as
critical (exit 1 if any fails, since these would break a real run
outright). A missing ANTHROPIC_API_KEY and a missing/stale positions
file are advisory only -- both have a working fallback (raw flags
without LLM phrasing; skipped position-aware flags), so they're
reported but don't fail the check.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import deliver  # noqa: E402
import main as main_mod  # noqa: E402
import positions as positions_mod  # noqa: E402
from sources.fred import SERIES_IDS  # noqa: E402


def _run(label: str, fn, critical: bool) -> bool:
    try:
        detail = fn()
    except Exception as exc:
        tag = "FAIL" if critical else "WARN"
        print(f"{tag}  {label}: {exc}")
        return not critical
    print(f"OK    {label}" + (f"  ({detail})" if detail else ""))
    return True


def check_fred_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY not set in .env")
    return f"{len(key)}-char key set"


def check_fred_series() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    bad = []
    for name, series_id in SERIES_IDS.items():
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series",
                params={"series_id": series_id, "api_key": key, "file_type": "json"},
                timeout=10,
            )
            ok = resp.status_code == 200 and bool(resp.json().get("seriess"))
        except requests.RequestException:
            ok = False
        if not ok:
            bad.append(f"{name} ({series_id})")
    if bad:
        raise RuntimeError(f"don't resolve: {', '.join(bad)} -- fix SERIES_IDS in sources/fred.py")
    return f"all {len(SERIES_IDS)} series resolve"


def check_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set -- briefings will use the raw-flags fallback every day, never LLM-phrased text")
    return f"{len(key)}-char key set (not tested live -- would cost a real API call)"


def check_smtp(cfg: dict) -> str:
    smtp_config = deliver.load_smtp_config(cfg, dict(os.environ))
    deliver.check_connection(smtp_config)
    return f"logged in as {smtp_config.username} @ {smtp_config.host}:{smtp_config.port}"


def check_positions(cfg: dict) -> str:
    snap = positions_mod.load_latest_positions(ROOT / "data" / "positions")
    if snap is None:
        return "no positions file -- position-aware flags will be skipped (this is fine)"
    stale = positions_mod.is_stale(snap, date.today(), cfg["positions"]["staleness_days"])
    note = " -- STALE, drop a fresh export in data/positions/" if stale else ""
    return f"{snap.source_path.name}, {len(snap.positions)} row(s){note}"


def main() -> None:
    load_dotenv(ROOT / ".env")
    all_ok = True

    cfg_holder: dict = {}

    def load_cfg() -> str:
        cfg_holder["cfg"] = main_mod.load_config()
        tickers = cfg_holder["cfg"]["tickers"]
        return f"{len(tickers)} ticker(s): {', '.join(tickers)}"

    all_ok &= _run("config.yaml", load_cfg, critical=True)
    if "cfg" not in cfg_holder:
        print("\nFix config.yaml before continuing -- every other check depends on it.")
        raise SystemExit(1)
    cfg = cfg_holder["cfg"]

    all_ok &= _run("FRED_API_KEY", check_fred_key, critical=True)
    all_ok &= _run("FRED series IDs", check_fred_series, critical=True)
    all_ok &= _run("ANTHROPIC_API_KEY", check_anthropic_key, critical=False)
    all_ok &= _run("SMTP login", lambda: check_smtp(cfg), critical=True)
    all_ok &= _run("positions file", lambda: check_positions(cfg), critical=False)

    print()
    if all_ok:
        print("All critical checks passed. Try a --dry-run next:")
        print("  .venv\\Scripts\\python.exe main.py --dry-run --force")
    else:
        print("One or more critical checks failed -- fix those before scheduling this.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
