"""Common interface every data source implements.

Every source returns `Observation`s tagged with provenance and freshness,
so reconcile.py and the honesty-rule labeling in brief.py never have to
guess where a number came from or how old it is.

Exception hierarchy matters here: `SourceUnavailable` means "couldn't
reach it, try again later or run on cached data" — always safe to catch
and degrade gracefully. `SchemaError` means "reached it, but the shape
was wrong" — usually a sign the source changed its API/HTML, or that a
series ID / column mapping in this codebase is wrong. Never silently
treat a SchemaError as zero or missing; it should surface loudly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Callable, TypeVar

T = TypeVar("T")


class SourceError(Exception):
    """Base class for all source failures."""


class SourceUnavailable(SourceError):
    """Could not reach the source (network, timeout, rate limit) after
    retries were exhausted. Safe to catch and degrade gracefully."""


class SchemaError(SourceError):
    """The source responded, but not in the expected shape. Never catch
    this and treat it as zero/missing — it means the parsing code or a
    series/column mapping needs a human to look at it."""


@dataclass(frozen=True)
class Observation:
    """One data point plus where it came from and how fresh it is."""

    value: float
    as_of: date  # the date the value applies to (e.g. the bar's date)
    source: str  # "fred", "yfinance", "stooq", "nasdaq"
    series: str  # series ID, ticker symbol, etc.
    fetched_at: date  # when this run pulled it, for staleness checks


def with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int,
    backoff_base: float,
    retry_on: tuple = (Exception,),
) -> T:
    """Call `fn`, retrying with exponential backoff on the given
    exception types. Raises the last exception once retries run out."""
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(backoff_base * (2 ** (attempt - 1)))
