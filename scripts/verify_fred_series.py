"""One-time local check: confirm the FRED series IDs in sources/fred.py
actually resolve, before the first live run.

This project was built in a sandbox with no network egress to
api.stlouisfed.org, so SERIES_IDS in sources/fred.py (particularly
VIX9D and VIX3M) were never live-tested. Run this once on a machine
with real internet access:

    .venv/bin/python scripts/verify_fred_series.py

If any line prints FAIL, fix SERIES_IDS in sources/fred.py — search
https://fred.stlouisfed.org for the index name to find the right one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from dotenv import load_dotenv

from sources.fred import SERIES_IDS

load_dotenv()


def main() -> None:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("FRED_API_KEY not set (check .env)")
        raise SystemExit(1)

    all_ok = True
    for key, series_id in SERIES_IDS.items():
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series",
                params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
                timeout=10,
            )
        except requests.RequestException as exc:
            all_ok = False
            print(f"FAIL  {key:10s} {series_id:10s} request error: {exc}")
            continue

        if resp.status_code != 200:
            all_ok = False
            print(f"FAIL  {key:10s} {series_id:10s} HTTP {resp.status_code}: {resp.text[:150]}")
            continue

        payload = resp.json()
        series = payload.get("seriess") or []
        if not series:
            all_ok = False
            print(f"FAIL  {key:10s} {series_id:10s} no series returned")
            continue

        title = series[0].get("title", "?")
        last_updated = series[0].get("last_updated", "?")
        print(f"OK    {key:10s} {series_id:10s} {title}  (updated {last_updated})")

    print()
    if not all_ok:
        print("One or more series IDs are wrong — fix SERIES_IDS in sources/fred.py.")
        raise SystemExit(1)
    print("All FRED series IDs resolve.")


if __name__ == "__main__":
    main()
