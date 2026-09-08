"""metrics -> flags -> LLM -> validator -> final text.

The two hard constraints from the project brief are enforced here, and
enforced twice each, on purpose:

    1. CONDITIONS, NOT RECOMMENDATIONS. The LLM is told this in its
       system prompt (soft enforcement) AND every LLM response is
       scanned by `validate_briefing_text()` for forbidden directional/
       recommendation language (hard enforcement) — a match raises
       loudly rather than being silently stripped. But the deeper
       enforcement is architectural: the LLM never decides what counts
       as notable. `build_flags()` is pure, deterministic, code-only —
       it compares metrics against config thresholds and against
       yesterday's logged metrics, and only crossings get surfaced. The
       LLM receives an already-ranked, already-filtered list of facts
       and is only asked to phrase and order them, never to judge
       materiality or state a number that isn't already in its input.

    2. TERSE BY DEFAULT. `build_flags()` only surfaces a metric that
       crossed a config threshold or moved materially since yesterday.
       Flags are ranked by severity and truncated to config's max_flags
       before the LLM ever sees them. The character cap is stated in
       the system prompt AND enforced in code after the response comes
       back — exceeding it fails the run exactly like forbidden
       language does.

On either failure, callers should fall back to `format_fallback()`,
which formats the same payload without an LLM at all — deterministic,
so it can never contain a recommendation, and safe to truncate because
it drops whole lines (lowest severity first) rather than cutting a
sentence mid-word.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Flags — deterministic, code-computed, no LLM involved.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Flag:
    severity: int  # higher = more severe; used for ranking and truncation
    category: str
    message: str  # fully-formed, code-generated fact — never LLM text
    ticker: Optional[str] = None


def build_flags(
    tickers_metrics: Dict[str, dict],
    market_metrics: dict,
    positions_ctx: Optional[dict],
    config: dict,
    yesterday: Optional[Dict[str, dict]] = None,
) -> List[Flag]:
    """`tickers_metrics`: {ticker: {...metric keys...}} for this run.
    `yesterday`: same shape, from the previous logged run (see
    logs/), used for the "changed materially" comparisons. Missing on
    day one — delta flags simply don't fire yet, which is correct, not
    a bug: there's nothing to compare against."""
    yesterday = yesterday or {}
    flags: List[Flag] = []

    for ticker, m in tickers_metrics.items():
        y = yesterday.get(ticker, {})
        flags.extend(_ticker_flags(ticker, m, y, positions_ctx, config))

    flags.extend(_market_flags(market_metrics, config))

    flags.sort(key=lambda f: f.severity, reverse=True)
    return flags


def _ticker_flags(ticker: str, m: dict, y: dict, positions_ctx: Optional[dict], config: dict) -> List[Flag]:
    out: List[Flag] = []

    gap_atr = m.get("gap_atr_units")
    if gap_atr is not None and abs(gap_atr) >= config["underlying"]["notable_gap_atr"]:
        severity = 70 + min(30, int(abs(gap_atr) * 10))
        out.append(
            Flag(
                severity=severity,
                category="gap",
                ticker=ticker,
                message=f"{ticker} gap {m.get('gap_pct', 0):+.1f}% ({gap_atr:+.2f} ATR)",
            )
        )

    iv_rank = m.get("iv_rank")  # {"value":.., "sufficient":bool, "label":..}
    if iv_rank and iv_rank.get("sufficient") and iv_rank.get("value") is not None:
        bands = config["iv_rank_bands"]
        val = iv_rank["value"]
        if val <= bands["low"] or val >= bands["high"]:
            out.append(
                Flag(
                    severity=60,
                    category="iv_rank",
                    ticker=ticker,
                    message=f"{ticker} IV rank {val:.0f} ({iv_rank.get('label', '')})",
                )
            )

    days_to_earnings = m.get("days_to_earnings")
    if days_to_earnings is not None and 0 <= days_to_earnings <= config["earnings"]["blackout_days"]:
        tag = " (unconfirmed)" if m.get("earnings_unconfirmed") else ""
        out.append(
            Flag(
                severity=80,
                category="earnings",
                ticker=ticker,
                message=f"{ticker} earnings in {days_to_earnings}d{tag}",
            )
        )

    if positions_ctx:
        for exp in positions_ctx.get("expiring_by_ticker", {}).get(ticker, []):
            out.append(
                Flag(
                    severity=75,
                    category="expiration",
                    ticker=ticker,
                    message=f"{ticker} {exp['option_type']} {exp['strike']} exp in {exp['dte']}d",
                )
            )
        count = positions_ctx.get("concurrent_option_position_count")
        max_positions = config["positions"]["max_concurrent_positions"]
        if count is not None and count > max_positions and not positions_ctx.get("_concentration_flagged"):
            positions_ctx["_concentration_flagged"] = True  # emit once per run, not once per ticker
            out.append(
                Flag(
                    severity=85,
                    category="concentration",
                    ticker=None,
                    message=f"{count} concurrent option positions (max {max_positions})",
                )
            )

    out.extend(_delta_flags(ticker, m, y, config))
    return out


# (metric key in payload, label, config threshold key, value format)
DELTA_METRICS = [
    ("dma50_distance_pct", "50DMA distance", "dma_distance_pct", "{:+.1f}%"),
    ("realized_vol_20d", "20d realized vol", "realized_vol_pts", "{:.1f}%"),
    ("iv_minus_rv", "IV-RV", "iv_minus_rv_pts", "{:+.1f}pts"),
    ("term_structure_slope", "term structure slope", "term_structure_slope_pts", "{:+.2f}"),
    ("range_percentile", "30d range pct", "range_percentile_pts", "{:.0f}"),
]


def _delta_flags(ticker: str, m: dict, y: dict, config: dict) -> List[Flag]:
    out: List[Flag] = []
    thresholds = config.get("delta_thresholds", {})
    for key, label, threshold_key, fmt in DELTA_METRICS:
        current, previous = m.get(key), y.get(key)
        threshold = thresholds.get(threshold_key)
        if current is None or previous is None or threshold is None:
            continue
        delta = current - previous
        if abs(delta) >= threshold:
            out.append(
                Flag(
                    severity=40,
                    category="delta",
                    ticker=ticker,
                    message=f"{ticker} {label} {fmt.format(current)} (Δ{delta:+.1f} vs yesterday)",
                )
            )
    return out


def _market_flags(market_metrics: dict, config: dict) -> List[Flag]:
    out: List[Flag] = []
    ts = market_metrics.get("vix_term_structure")  # {"shape":.., "inverted":bool}
    if ts and ts.get("inverted") and config.get("vix", {}).get("flag_inversion", True):
        out.append(
            Flag(
                severity=90,
                category="vix_term_structure",
                ticker=None,
                message=(
                    f"VIX term structure {ts.get('shape')} "
                    f"(VIX9D {market_metrics.get('vix9d')}, VIX {market_metrics.get('vix')}, "
                    f"VIX3M {market_metrics.get('vix3m')})"
                ),
            )
        )

    yesterday_vix = market_metrics.get("vix_yesterday")
    vix = market_metrics.get("vix")
    threshold = config.get("delta_thresholds", {}).get("vix_level_pts")
    if vix is not None and yesterday_vix is not None and threshold is not None:
        delta = vix - yesterday_vix
        if abs(delta) >= threshold:
            out.append(
                Flag(
                    severity=50,
                    category="delta",
                    ticker=None,
                    message=f"VIX {vix:.1f} (Δ{delta:+.1f} vs yesterday)",
                )
            )
    return out


def nominal_tickers(all_tickers: Sequence[str], flags: Sequence[Flag]) -> List[str]:
    flagged = {f.ticker for f in flags if f.ticker}
    return [t for t in all_tickers if t not in flagged]


def build_data_quality_notes(
    price_comparisons: Sequence = (),
    earnings_comparisons: Sequence = (),
    iv_suppressed: Optional[Dict[str, str]] = None,
    fetch_failures: Optional[Dict[str, str]] = None,
    positions_stale_note: Optional[str] = None,
    stale_data_banner: Optional[str] = None,
) -> List[str]:
    """Every derived number's honesty-rule caveats, collected into one
    list. Order: staleness banner first (most important), then
    per-ticker issues."""
    notes: List[str] = []
    for cmp in price_comparisons:
        if cmp.disputed:
            notes.append(f"{cmp.ticker}: Yahoo/Stooq closes disagree {cmp.disagreement_pct:.1f}% — price metrics skipped")
    for ec in earnings_comparisons:
        if ec.unconfirmed and ec.resolved_date is not None:
            notes.append(f"{ec.ticker}: earnings date {ec.resolved_date.isoformat()} unconfirmed")
    for ticker, reason in (iv_suppressed or {}).items():
        notes.append(f"{ticker}: IV unavailable — {reason}")
    for ticker, reason in (fetch_failures or {}).items():
        notes.append(f"{ticker}: {reason}")
    if positions_stale_note:
        notes.append(positions_stale_note)
    if stale_data_banner:
        notes.insert(0, f"STALE DATA: {stale_data_banner}")
    return notes


def build_payload(
    tickers_metrics: Dict[str, dict],
    market_metrics: dict,
    positions_ctx: Optional[dict],
    config: dict,
    yesterday: Optional[Dict[str, dict]] = None,
    data_quality_notes: Optional[Sequence[str]] = None,
) -> dict:
    flags = build_flags(tickers_metrics, market_metrics, positions_ctx, config, yesterday)
    top_flags = flags[: config["briefing"]["max_flags"]]
    nominal = nominal_tickers(list(tickers_metrics.keys()), top_flags)
    return {
        "flags": [asdict(f) for f in top_flags],
        "nominal": nominal,
        "data_quality_notes": list(data_quality_notes or []),
    }


# ---------------------------------------------------------------------------
# Validator — hard enforcement of both constraints on the LLM's output.
# ---------------------------------------------------------------------------

# Starting list, same philosophy as config.yaml's thresholds: not
# exhaustive, meant to be extended as real output review turns up
# phrasing that slips through. Bias toward over-triggering — a false
# positive costs one fallback-to-raw-flags run; a false negative costs
# a recommendation reaching the reader, which this tool exists to
# prevent. Deliberately NOT banning bare "long"/"short"/"buy"/"sell"/
# "call"/"put" — those are ordinary options vocabulary needed to
# describe existing exposure neutrally.
FORBIDDEN_PATTERNS = [
    r"\bbullish\b",
    r"\bbearish\b",
    r"\bexpect(?:s|ed|ing)?\b",
    r"\blikely to\b",
    r"\bunlikely to\b",
    r"\bsuggests? (?:a|an) (?:move|rally|selloff|breakout|reversal|bounce|drop|rise)\b",
    r"\bgood (?:entry|exit)\b",
    r"\b(?:entry|exit) point\b",
    r"\bconsider (?:buying|selling|going|adding|trimming|rolling)\b",
    r"\bpoised\b",
    r"\bsetting up\b",
    r"\bset up for\b",
    r"\bshould (?:rally|fall|drop|rise|bounce|break(?:out)?|pull ?back|correct)\b",
    r"\babout to break\b",
    r"\bbreakout imminent\b",
    r"\bbuy the dip\b",
    r"\bsell the rally\b",
    r"\btime to (?:buy|sell)\b",
    r"\bgood (?:time|opportunity) to\b",
    r"\bwatch for a (?:breakout|reversal|move)\b",
    r"\blook(?:s|ing)? for a (?:breakout|move|reversal)\b",
    r"\brecommend(?:s|ed|ation)?\b",
    r"\bprice target\b",
    r"\b(?:will|going to) (?:rally|fall|drop|rise|break|move|bounce)\b",
    r"\bprobability of\b",
    r"\bodds (?:of|favor)\b",
    r"\boverbought\b",
    r"\boversold\b",
]


@dataclass(frozen=True)
class ValidationFailure:
    reason: str  # "forbidden_language" | "over_character_limit"
    detail: str


class BriefingValidationError(Exception):
    """Raised when generated text fails validation. Callers must NOT
    attempt to fix the text (strip a phrase, truncate) and use it
    anyway — treat this as a run failure and call format_fallback()
    instead."""

    def __init__(self, failures: List[ValidationFailure]):
        self.failures = failures
        super().__init__("; ".join(f"{f.reason}: {f.detail}" for f in failures))


def validate_briefing_text(text: str, max_characters: int) -> None:
    failures: List[ValidationFailure] = []
    for pattern in FORBIDDEN_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start, end = max(0, match.start() - 20), min(len(text), match.end() + 20)
            failures.append(
                ValidationFailure("forbidden_language", f'matched {pattern!r} in "...{text[start:end]}..."')
            )
    if len(text) > max_characters:
        failures.append(ValidationFailure("over_character_limit", f"{len(text)} chars > {max_characters} limit"))
    if failures:
        raise BriefingValidationError(failures)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You write a pre-market options briefing for a single trader who \
already knows how to read these numbers. Two rules, absolute:

1. CONDITIONS, NOT RECOMMENDATIONS. Describe where things stand and which \
configured rules were breached. NEVER output a directional call, a \
buy/sell verdict, a price target, or a probability of a move. Forbidden \
words/phrases include (not exhaustive): bullish, bearish, expect, likely \
to, suggests a move, good entry, consider buying, poised, setting up, \
overbought, oversold, price target, recommend. If you catch yourself \
writing any of these, rephrase as a plain description of the number \
instead.

2. TERSE. Every flag in the input JSON was already selected by code \
because it crossed a configured threshold or changed materially since \
yesterday, and is already ranked most-severe-first — your only job is to \
phrase these as short lines in the given order, not to add commentary, \
preamble, or a sign-off. A quiet day is three lines.

Hard constraints:
- Output MUST be {max_characters} characters or fewer, total. This is \
enforced in code after you respond; exceeding it fails the run.
- Use ONLY numbers present in the JSON input. Never compute, estimate, \
or state a number that isn't already there.
- Sections, in order, only if they have content:
  1. Flags, in the order given, one line each.
  2. Notable per-ticker one-liners, only for anything not already \
covered by a flag above.
  3. One line: "N tickers nominal" (name them only if it fits).
  4. Data quality notes, if any.
- No preamble ("Good morning", "Here's your briefing"). No sign-off \
("Trade carefully"). No restating a number already shown in the flags \
section."""


def generate_briefing(payload: dict, config: dict, client: Any = None) -> str:
    """`client`: an injected Anthropic-SDK-shaped client for testing
    (must expose .messages.create(...) -> response with .content, each
    item having .type and .text). Defaults to a real
    anthropic.Anthropic() reading ANTHROPIC_API_KEY from the
    environment."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    max_characters = config["briefing"]["max_characters"]
    response = client.messages.create(
        model=config["briefing"]["llm_model"],
        max_tokens=1024,
        temperature=config["briefing"]["llm_temperature"],
        system=SYSTEM_PROMPT_TEMPLATE.format(max_characters=max_characters),
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()


def generate_and_validate(payload: dict, config: dict, client: Any = None) -> str:
    """Raises BriefingValidationError on forbidden language or an
    over-cap response; raises whatever the client raises on an API
    failure. Either way, callers should fall back to
    format_fallback()."""
    text = generate_briefing(payload, config, client=client)
    validate_briefing_text(text, config["briefing"]["max_characters"])
    return text


# ---------------------------------------------------------------------------
# Deterministic fallback — no LLM, used when generate_and_validate() raises.
# ---------------------------------------------------------------------------


def format_fallback(payload: dict, config: dict) -> str:
    """Formats the same payload without an LLM. Can never contain a
    recommendation since no LLM touches it. Truncation here drops
    whole lines from the end (already lowest-severity-first) rather
    than cutting mid-sentence, so it's always safe to hard-cap."""
    lines: List[str] = [f["message"] for f in payload["flags"]]
    if payload.get("nominal"):
        lines.append(f"{len(payload['nominal'])} tickers nominal")
    lines.extend(payload.get("data_quality_notes", []))

    max_characters = config["briefing"]["max_characters"]
    text = "\n".join(lines)
    while len(text) > max_characters and lines:
        lines.pop()
        text = "\n".join(lines)
    if len(text) > max_characters:
        text = text[:max_characters]
    return text
