@echo off
REM Wrapper for Task Scheduler: cd's to the project root (so config.yaml,
REM .env, and relative paths resolve correctly regardless of Task
REM Scheduler's "Start in" setting) and runs the briefing through the
REM venv's own python.exe, not whatever "python" resolves to on PATH.
cd /d "%~dp0\.."
".venv\Scripts\python.exe" main.py %*
