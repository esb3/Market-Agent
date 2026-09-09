@echo off
REM Wrapper for Task Scheduler -- see run_briefing.bat for why this cd's
REM first and calls the venv's python.exe directly.
cd /d "%~dp0.."
".venv\Scripts\python.exe" snapshot.py %*
