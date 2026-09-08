# Options Morning Briefing

A local tool that gathers pre-market conditions for a fixed set of
underlyings and emails a short briefing before the open. No brokerage
connection — it never authenticates against a broker and cannot place
an order. All market data comes from free public sources
(yfinance/Yahoo, Stooq, FRED, Nasdaq's public earnings calendar).

Two rules the whole design serves: **conditions, not recommendations**
(enforced in the LLM prompt and by a post-generation validator that
fails the run loudly on any directional/recommendation language — see
`brief.py`), and **terse by default** (a hard character cap, only
metrics that cross a threshold or moved materially since yesterday get
surfaced, everything else rolls into one "N tickers nominal" line).

## One-time setup

```bash
cd /path/to/Market-Agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

- `FRED_API_KEY` — free key from https://fredaccount.stlouisfed.org/apikeys
- `ANTHROPIC_API_KEY` — from https://console.anthropic.com
- `BRIEFING_FROM_ADDRESS` / `BRIEFING_TO_ADDRESS` — usually the same address
- `BRIEFING_SMTP_USERNAME` / `BRIEFING_SMTP_PASSWORD` — for Gmail, this is an
  [App Password](https://myaccount.google.com/apppasswords), not your normal
  password (2FA must be on). Adjust `delivery.smtp_host`/`smtp_port` in
  `config.yaml` if you're not using Gmail.

Edit `config.yaml`:

- `tickers` — replace the placeholder list with your actual underlyings.
- Everything else already has a sensible default with a comment
  explaining what raising/lowering it does. Don't tune these before
  running for a week on real output — see the file's own header comment.

Drop your broker's positions export (see `positions.py`'s docstring —
currently built for Schwab's format) into `data/positions/` whenever you
want position-aware flags (existing exposure, concentration, expirations
approaching). The tool runs fine with no file there; it just skips those
flags.

**Verify FRED series IDs once**, on this machine (not verifiable during
this project's build — see `sources/fred.py`'s docstring):

```bash
.venv/bin/python scripts/verify_fred_series.py
```

Fix `SERIES_IDS` in `sources/fred.py` if anything prints `FAIL`.

## Running it

```bash
.venv/bin/python main.py --dry-run          # prints the briefing, sends nothing
.venv/bin/python main.py                    # sends the email
.venv/bin/python main.py --force            # runs even on a non-trading day
.venv/bin/python main.py --date 2026-09-08  # override "today", for testing
```

IV history (needed for IV rank/percentile to mean anything — see
`snapshot.py`'s docstring on the sample-size problem) accumulates
automatically as a side effect of `main.py`'s run, but also has its own
independent entry point so it keeps accumulating even on a day the
briefing itself fails:

```bash
.venv/bin/python snapshot.py
```

Run the test suite any time:

```bash
.venv/bin/python -m pytest
```

## Scheduling (macOS launchd)

Two plists in `launchd/`: the briefing (08:30 local, Mon–Fri) and the
independent IV snapshot (07:00 local, Mon–Fri — see that plist's comment
for why it's separate from the briefing run). Both fire every weekday;
`main.py` itself checks `pandas_market_calendars` and exits quietly on
US market holidays, so there's no holiday logic needed in launchd.

**1. Edit the paths.** Both plists have `/Users/YOUR_USERNAME/Market-Agent`
placeholders — replace with your actual project path (4 places per file:
the two `ProgramArguments` entries, `WorkingDirectory`, and the two log
paths):

```bash
sed -i '' "s|/Users/YOUR_USERNAME/Market-Agent|$(pwd)|g" launchd/com.marketagent.briefing.plist
sed -i '' "s|/Users/YOUR_USERNAME/Market-Agent|$(pwd)|g" launchd/com.marketagent.snapshot.plist
```

**2. Install and load them:**

```bash
cp launchd/com.marketagent.briefing.plist launchd/com.marketagent.snapshot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.marketagent.briefing.plist
launchctl load ~/Library/LaunchAgents/com.marketagent.snapshot.plist
```

**3. Verify they're loaded:**

```bash
launchctl list | grep marketagent
```

Two rows should print, each with a PID column (usually `-` when idle,
meaning loaded but not currently running) and a last-exit-status column
(`0` after it has fired successfully at least once).

**4. Test a job immediately** without waiting for its scheduled time:

```bash
launchctl start com.marketagent.briefing
tail -f logs/launchd_briefing.log logs/launchd_briefing.err.log
```

**To stop/reload** after editing a plist:

```bash
launchctl unload ~/Library/LaunchAgents/com.marketagent.briefing.plist
# ...edit, then...
launchctl load ~/Library/LaunchAgents/com.marketagent.briefing.plist
```

**Sleeping Mac:** launchd does not wake a sleeping machine by itself. If
this Mac sleeps overnight, either wake it on a schedule:

```bash
sudo pmset repeat wakeorpoweron MTWRF 08:15:00
```

or accept that the job runs late (whenever the Mac next wakes) — `main.py`
still checks the actual date, not "time since scheduled," so a late run
still produces today's briefing rather than a stale one.

## Architecture

```
config.yaml      tickers, thresholds, rules, delivery, verbosity caps
sources/         fred.py, yfinance.py, stooq.py, nasdaq.py — common interface (sources/base.py)
reconcile.py     cross-source comparison, disagreement flagging
chains.py        option chain cleaning and ATM IV derivation
snapshot.py      daily IV history writer, runs independently
positions.py     manual CSV parser with column validation
metrics.py       pure functions, no I/O, fully unit tested
brief.py         metrics -> flags -> LLM -> validator -> final text
deliver.py       SMTP email
main.py          orchestration, failure handling
launchd/         macOS scheduling
data/positions/  your manual CSV exports (gitignored)
logs/            per-run JSON: metrics, output, source disagreements (gitignored)
```

## Design notes worth knowing before you rely on this

- **Every free data source here is unofficial or partial.** Yahoo
  (via `yfinance`) scrapes an undocumented endpoint that breaks and
  rate-limits without notice. FRED is the one official, reliable
  source and is treated as authoritative wherever it overlaps. Stooq
  cross-checks Yahoo's closes; a disagreement beyond
  `price_cross_check.tolerance_pct` withholds that day's derived
  metrics rather than computing off a disputed bar. Nasdaq's earnings
  calendar cross-checks yfinance's earnings date; disagreements use the
  earlier date and mark it unconfirmed, never silently.
- **IV rank/percentile need real history.** They're suppressed
  entirely below `iv_history.suppress_below_days` of local snapshots,
  and labeled "thin" below `mature_sample_days`. This can't be
  shortcut — there's no free source of historical IV to backfill from.
- **This project was built in a sandbox with no live network access**
  to Yahoo, Stooq, FRED, or Nasdaq (only PyPI and GitHub). Every
  `sources/` client was built against the real installed library's
  actual API/response shapes (inspected via source, not guessed from
  memory) and is fully unit-tested against mocked responses, but
  nothing here has been smoke-tested against the real internet. Run
  `scripts/verify_fred_series.py` and a `--dry-run` briefing on your
  own machine before trusting live output, and watch the data quality
  notes section on the first few real runs.
