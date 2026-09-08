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

import html
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

    days_to_ex_div = m.get("days_to_ex_dividend")
    if days_to_ex_div is not None and 0 <= days_to_ex_div <= config["dividends"]["blackout_days"]:
        out.append(
            Flag(
                severity=55,
                category="ex_dividend",
                ticker=ticker,
                message=f"{ticker} ex-dividend in {days_to_ex_div}d",
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

    vix_pctl = market_metrics.get("vix_percentile")  # {"value":.., "sufficient":bool, "label":..}
    if vix_pctl and vix_pctl.get("sufficient") and vix_pctl.get("value") is not None:
        bands = config.get("vix", {}).get("percentile_bands")
        if bands:
            val = vix_pctl["value"]
            if val <= bands["low"] or val >= bands["high"]:
                out.append(
                    Flag(
                        severity=65,
                        category="vix_percentile",
                        ticker=None,
                        message=f"VIX percentile {val:.0f} ({vix_pctl.get('label', '')})",
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

    blackout_days = config.get("calendar", {}).get("blackout_days")
    for entry in market_metrics.get("macro_dates", []):
        days_to = entry.get("days_to")
        if days_to is None or blackout_days is None or days_to > blackout_days:
            continue
        out.append(
            Flag(
                severity=70,
                category="macro",
                ticker=None,
                message=f"{entry.get('label', 'macro event')} in {days_to}d",
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
- Your JSON input contains ONLY flags, a nominal-ticker list, and data \
quality notes -- no raw per-ticker metrics. There is nothing else to \
report: never add commentary, context, or a per-ticker note about \
anything not already present in this JSON, even if you recognize the \
ticker or have other knowledge about the company. If it isn't in the \
input, it doesn't exist for this briefing.
- Output MUST be {max_characters} characters or fewer, total. This is \
enforced in code after you respond; exceeding it fails the run.
- Use ONLY numbers present in the JSON input. Never compute, estimate, \
or state a number that isn't already there.
- Sections, in order, only if they have content:
  1. Flags, in the order given, one line each.
  2. One line: "N tickers nominal" (name them only if it fits).
  3. Data quality notes, if any.
- No preamble ("Good morning", "Here's your briefing"). No sign-off \
("Trade carefully"). No restating a number already shown in the flags \
section."""


def _resolve_client(client: Any) -> Any:
    if client is not None:
        return client
    import anthropic

    return anthropic.Anthropic()


def _extract_text(response: Any) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()


def generate_briefing(payload: dict, config: dict, client: Any = None) -> str:
    """One LLM call, no retry. `client`: an injected Anthropic-SDK-shaped
    client for testing (must expose .messages.create(...) -> response
    with .content, each item having .type and .text). Defaults to a
    real anthropic.Anthropic() reading ANTHROPIC_API_KEY from the
    environment."""
    client = _resolve_client(client)
    max_characters = config["briefing"]["max_characters"]
    response = client.messages.create(
        model=config["briefing"]["llm_model"],
        max_tokens=1024,
        temperature=config["briefing"]["llm_temperature"],
        system=SYSTEM_PROMPT_TEMPLATE.format(max_characters=max_characters),
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
    )
    return _extract_text(response)


def generate_and_validate(payload: dict, config: dict, client: Any = None, max_attempts: int = 2) -> str:
    """Calls the LLM and validates its response, retrying with a
    corrective follow-up turn (not a blind re-send — at temperature 0 a
    plain retry would just reproduce the same violation) up to
    `max_attempts` times before giving up. Raises BriefingValidationError
    if every attempt fails validation, or whatever the client raises on
    an API failure. Either way, callers should fall back to
    format_fallback()."""
    client = _resolve_client(client)
    max_characters = config["briefing"]["max_characters"]
    messages: List[dict] = [{"role": "user", "content": json.dumps(payload, default=str)}]

    for attempt in range(1, max_attempts + 1):
        response = client.messages.create(
            model=config["briefing"]["llm_model"],
            max_tokens=1024,
            temperature=config["briefing"]["llm_temperature"],
            system=SYSTEM_PROMPT_TEMPLATE.format(max_characters=max_characters),
            messages=messages,
        )
        text = _extract_text(response)
        try:
            validate_briefing_text(text, max_characters)
            return text
        except BriefingValidationError as exc:
            if attempt >= max_attempts:
                raise
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That response violated the rules in the system prompt: {exc}. "
                        "Regenerate from scratch, following the system prompt exactly — "
                        "no directional or recommendation language, "
                        f"{max_characters} characters or fewer total."
                    ),
                }
            )

    raise AssertionError("unreachable")  # loop always returns or raises


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


# ---------------------------------------------------------------------------
# HTML rendering — a second, always-safe view of the same payload.
# ---------------------------------------------------------------------------

_SEVERITY_HIGH = 80
_SEVERITY_MED = 50

_HTML_COLORS = {
    "bg": "#F6F1E7",
    "surface": "#FFFFFF",
    "border": "#E2D9C4",
    "text": "#221D14",
    "text_dim": "#6E6552",
    "accent": "#A9781A",
    "high": "#B14A3B",
    "med": "#A9781A",
    "low": "#4C7A63",
}
_MONO_STACK = "Menlo,Consolas,'SF Mono',monospace"


def _severity_color(severity: int) -> str:
    if severity >= _SEVERITY_HIGH:
        return _HTML_COLORS["high"]
    if severity >= _SEVERITY_MED:
        return _HTML_COLORS["med"]
    return _HTML_COLORS["low"]


def build_html_briefing(payload: dict, subject: str = "") -> str:
    """Deterministic HTML rendering of `payload`, built directly from
    each Flag's code-generated `message` field — never LLM text. This
    half of the delivered email carries the same safety guarantee as
    format_fallback(): it cannot contain a recommendation, because no
    LLM ever touches it, regardless of whether the plain-text part
    (sent alongside it, see deliver.py) came from a successful LLM call
    or the fallback. Table-based, inline-styled throughout — plain CSS
    and modern layout aren't reliably supported across mail clients.
    """
    c = _HTML_COLORS
    flag_rows = []
    for f in payload["flags"]:
        color = _severity_color(f["severity"])
        message = html.escape(f["message"])
        flag_rows.append(
            f'<tr><td style="width:4px;background:{color};font-size:0;line-height:0;">&nbsp;</td>'
            f'<td style="padding:9px 0 9px 12px;font-family:{_MONO_STACK};'
            f'font-size:13.5px;line-height:1.5;color:{c["text"]};">{message}</td></tr>'
        )
    if payload.get("nominal"):
        note = html.escape(f"{len(payload['nominal'])} tickers nominal")
        flag_rows.append(
            '<tr><td style="width:4px;"></td>'
            f'<td style="padding:9px 0 9px 12px;font-family:{_MONO_STACK};'
            f'font-size:13.5px;line-height:1.5;color:{c["text_dim"]};">{note}</td></tr>'
        )
    flags_table = "".join(flag_rows) or (
        f'<tr><td style="padding:9px 0;font-family:{_MONO_STACK};font-size:13.5px;'
        f'color:{c["text_dim"]};">nothing to report</td></tr>'
    )

    dq_section = ""
    dq_notes = payload.get("data_quality_notes", [])
    if dq_notes:
        dq_rows = "".join(
            f'<tr><td style="padding:4px 0;font-family:{_MONO_STACK};font-size:12px;'
            f'line-height:1.5;color:{c["text_dim"]};">{html.escape(note)}</td></tr>'
            for note in dq_notes
        )
        dq_section = (
            f'<tr><td style="padding:14px 18px 18px;border-top:1px solid {c["border"]};">'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{dq_rows}</table>'
            "</td></tr>"
        )

    subject_html = html.escape(subject)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="margin:0;padding:0;background:{c['bg']};">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{c['bg']};">
<tr><td align="center" style="padding:24px 16px;">
<table role="presentation" width="480" cellpadding="0" cellspacing="0" style="max-width:480px;width:100%;background:{c['surface']};border:1px solid {c['border']};border-radius:8px;">
<tr><td style="padding:16px 18px 4px;font-family:{_MONO_STACK};font-size:11px;letter-spacing:.06em;color:{c['accent']};text-transform:uppercase;">{subject_html}</td></tr>
<tr><td style="padding:6px 14px 14px 18px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{flags_table}</table>
</td></tr>
{dq_section}
</table>
</td></tr>
</table>
</body>
</html>"""
