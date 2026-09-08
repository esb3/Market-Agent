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

## One-time setup (Windows)

In PowerShell:

```powershell
cd C:\path\to\Market-Agent
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env
```

(macOS/Linux equivalent, if this ever runs there instead:
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cp .env.example .env`
— swap `.venv\Scripts\python.exe` for `.venv/bin/python` everywhere below too.)

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

```powershell
.venv\Scripts\python.exe scripts\verify_fred_series.py
```

Fix `SERIES_IDS` in `sources/fred.py` if anything prints `FAIL`.

## Running it

```powershell
.venv\Scripts\python.exe main.py --dry-run          # prints the briefing + writes logs\dry_run_preview.html, sends nothing
.venv\Scripts\python.exe main.py                    # sends the email
.venv\Scripts\python.exe main.py --force            # runs even on a non-trading day
.venv\Scripts\python.exe main.py --date 2026-09-08  # override "today", for testing
```

IV history (needed for IV rank/percentile to mean anything — see
`snapshot.py`'s docstring on the sample-size problem) accumulates
automatically as a side effect of `main.py`'s run, but also has its own
independent entry point so it keeps accumulating even on a day the
briefing itself fails:

```powershell
.venv\Scripts\python.exe snapshot.py
```

Run the test suite any time:

```powershell
.venv\Scripts\python.exe -m pytest
```

## Scheduling (Windows Task Scheduler)

Two wrapper scripts in `scripts/`: `run_briefing.bat` and
`run_snapshot.bat`. Each `cd`s to the project root and calls the venv's
own `python.exe` directly, so Task Scheduler's working directory and
whatever `python` happens to resolve to on PATH don't matter. `main.py`
itself checks `pandas_market_calendars` and exits quietly on US market
holidays, so both tasks can just fire every weekday with no holiday
logic in Task Scheduler.

**1. Create the two tasks**, from an ordinary (non-admin) PowerShell or
Command Prompt — replace `C:\path\to\Market-Agent` with your actual
project path in both commands:

```cmd
schtasks /create /tn "MarketAgent Briefing" /tr "\"C:\path\to\Market-Agent\scripts\run_briefing.bat\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:30
schtasks /create /tn "MarketAgent Snapshot" /tr "\"C:\path\to\Market-Agent\scripts\run_snapshot.bat\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 07:00
```

(The snapshot job runs earlier and separately from the briefing on
purpose — IV history needs to keep accumulating even on a day the
briefing itself fails, e.g. an LLM or SMTP outage. Both writers are
idempotent per ticker/date, so it's harmless that `main.py` also writes
today's snapshot itself while fetching option chains for the briefing.)

This creates each task to run only while you're logged on, with no
password stored — the simplest option for a personal machine. If you
need it to run even when logged off, use `/ru` + `/rp` instead (stores
your Windows password with the task) or configure "Run whether user is
logged on or not" in the GUI (`taskschd.msc`).

**2. Verify they're registered:**

```cmd
schtasks /query /tn "MarketAgent Briefing" /v /fo LIST
schtasks /query /tn "MarketAgent Snapshot" /v /fo LIST
```

Check `Scheduled Task State: Enabled` and `Last Result: 0` after it's
fired at least once (`267011` means "hasn't run yet," not a failure).

**3. Test a task immediately** without waiting for its scheduled time:

```cmd
schtasks /run /tn "MarketAgent Briefing"
```

Nothing under this project writes its own Task-Scheduler-specific log
file (unlike the launchd setup below) — check `logs\<date>.json` for
the run's output and data quality notes, since that's written
regardless of how the run was triggered.

**4. Wake from sleep.** Task Scheduler won't run a task while the PC is
asleep unless you enable it: open `taskschd.msc` → find the task →
Properties → **Conditions** tab → check "Wake the computer to run this
task." Without this, a run during sleep is simply skipped that day
rather than deferred — `main.py` has no way to run retroactively for a
missed morning.

**To update or remove** a task after editing a `.bat` file (no need to
recreate it — the `.bat` file is read fresh each run) or to delete one
entirely:

```cmd
schtasks /delete /tn "MarketAgent Briefing" /f
```

### macOS (if this ever runs there instead)

Two plists live in `launchd/` for reference — `com.marketagent.briefing.plist`
(08:30 local) and `com.marketagent.snapshot.plist` (07:00 local), both
Mon–Fri. Replace the `/Users/YOUR_USERNAME/Market-Agent` placeholders
with the real project path, then:

```bash
cp launchd/*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.marketagent.briefing.plist
launchctl load ~/Library/LaunchAgents/com.marketagent.snapshot.plist
launchctl list | grep marketagent   # verify
launchctl start com.marketagent.briefing   # test immediately
```

launchd doesn't wake a sleeping Mac by itself either — `sudo pmset
repeat wakeorpoweron MTWRF 08:15:00` is the equivalent of the Windows
"wake to run" checkbox above.

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
scripts/         Task Scheduler wrapper .bat files, FRED verification, demo
launchd/         macOS scheduling (reference only -- see Scheduling section)
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
