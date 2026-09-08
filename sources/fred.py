"""FRED client — VIX complex and the risk-free rate.

FRED is the most reliable source in this stack: an official API, a
stable schema, no scraping. Treat it as authoritative wherever it
overlaps another source (see reconcile.py). The VIX9D/VIX/VIX3M term
structure requires no options chain and never breaks the way a scraped
chain can.

Series ID note: FRED often keeps a legacy series ID after CBOE renames
an index's public ticker (VIX9D was VXST, VIX3M was VXV). VIXCLS and
DTB3 below are long-standing, high-confidence IDs. VXVCLS and VXSTCLS
are best-effort and were NOT live-verified against FRED — this codebase
was built in a sandbox with no network egress to api.stlouisfed.org.
Run this once on a machine with real internet access, before the first
live briefing:

    python scripts/verify_fred_series.py

If a series ID is wrong, FRED returns an error body rather than empty
data, and latest_observation() surfaces that as SchemaError rather than
returning None — a wrong ID fails loudly, not silently as zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional

import requests

from sources.base import Observation, SchemaError, SourceUnavailable, with_retry

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# key -> FRED series ID. See module docstring on VXVCLS/VXSTCLS confidence.
SERIES_IDS = {
    "vix": "VIXCLS",
    "vix9d": "VXSTCLS",
    "vix3m": "VXVCLS",
    "tbill_3m": "DTB3",
}

# FRED's marker for "no observation this period" (holidays, gaps).
FRED_MISSING_VALUE = "."


@dataclass(frozen=True)
class FredConfig:
    api_key: str
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_base_seconds: float = 2.0


class FredClient:
    def __init__(self, config: FredConfig):
        if not config.api_key:
            raise SourceUnavailable("FRED_API_KEY is not set (check .env)")
        self._config = config

    def _fetch(self, series_id: str, start: date, end: date) -> dict:
        params = {
            "series_id": series_id,
            "api_key": self._config.api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }

        def call() -> dict:
            try:
                resp = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=self._config.timeout_seconds)
            except requests.RequestException as exc:
                raise SourceUnavailable(f"FRED {series_id}: {exc}") from exc
            if resp.status_code != 200:
                raise SourceUnavailable(f"FRED {series_id}: HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()

        return with_retry(
            call,
            max_retries=self._config.max_retries,
            backoff_base=self._config.backoff_base_seconds,
            retry_on=(SourceUnavailable,),
        )

    def _parse_observations(self, payload: dict, series_id: str) -> List[Observation]:
        if not isinstance(payload, dict) or "observations" not in payload:
            raise SchemaError(
                f"FRED {series_id}: response missing 'observations' "
                f"(possibly a bad series ID) — {payload.get('error_message', payload) if isinstance(payload, dict) else payload}"
            )
        obs = payload["observations"]
        if not isinstance(obs, list):
            raise SchemaError(f"FRED {series_id}: 'observations' is not a list")

        result = []
        for row in obs:
            if not isinstance(row, dict) or "date" not in row or "value" not in row:
                raise SchemaError(f"FRED {series_id}: observation missing date/value: {row}")
            if row["value"] == FRED_MISSING_VALUE:
                continue
            try:
                value = float(row["value"])
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"FRED {series_id}: non-numeric value {row['value']!r}") from exc
            try:
                as_of = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"FRED {series_id}: unparseable date {row['date']!r}") from exc
            result.append(Observation(value=value, as_of=as_of, source="fred", series=series_id, fetched_at=date.today()))
        return result

    def latest_observation(self, series_key: str, lookback_days: int = 10) -> Optional[Observation]:
        """Most recent non-missing observation for a tracked series (see
        SERIES_IDS). Returns None only when FRED legitimately has no
        data in the lookback window — a malformed response is always a
        SchemaError, never None."""
        if series_key not in SERIES_IDS:
            raise ValueError(f"unknown FRED series key: {series_key!r}")
        series_id = SERIES_IDS[series_key]
        end = date.today()
        start = end - timedelta(days=lookback_days)
        payload = self._fetch(series_id, start, end)
        observations = self._parse_observations(payload, series_id)
        return observations[-1] if observations else None

    def history(self, series_key: str, start: date, end: date) -> List[Observation]:
        """All non-missing observations for a tracked series between
        `start` and `end` inclusive, oldest first. Unlike
        latest_observation(), this exists for percentile/rank
        calculations that need real history — FRED (unlike IV) has
        decades of VIX data available for free from day one, so there's
        no sample-size problem to work around here."""
        if series_key not in SERIES_IDS:
            raise ValueError(f"unknown FRED series key: {series_key!r}")
        series_id = SERIES_IDS[series_key]
        payload = self._fetch(series_id, start, end)
        return self._parse_observations(payload, series_id)

    def vix_complex(self) -> dict:
        """VIX, VIX9D, VIX3M, and the 3-month T-bill rate as of the most
        recent available date for each. A series that's merely
        unreachable (rate limit, timeout) is omitted from the result —
        the caller logs that as a data-quality note, not a run failure.
        A SchemaError (wrong series ID, broken response shape) is NOT
        caught here: that's a code bug, not routine flakiness, and
        should surface immediately rather than quietly missing a flag.
        """
        result: dict[str, Observation] = {}
        for key in SERIES_IDS:
            try:
                obs = self.latest_observation(key)
            except SourceUnavailable:
                continue
            if obs is not None:
                result[key] = obs
        return result

    def risk_free_rate_decimal(self) -> Optional[Observation]:
        """3-month T-bill rate as a decimal (FRED quotes DTB3 in percent,
        e.g. 5.25 for 5.25%) for use anywhere a greeks calc needs r."""
        obs = self.latest_observation("tbill_3m")
        if obs is None:
            return None
        return Observation(
            value=obs.value / 100,
            as_of=obs.as_of,
            source=obs.source,
            series=obs.series,
            fetched_at=obs.fetched_at,
        )


def from_env() -> FredClient:
    return FredClient(FredConfig(api_key=os.environ.get("FRED_API_KEY", "")))
