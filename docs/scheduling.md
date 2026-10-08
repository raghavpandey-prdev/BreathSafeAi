# Scheduling the Data Collection

You have two options. Start with Option A today; move to Option B once you're confident
things are working, since it survives reboots and closed terminals.

## Option A — Simple, works right now

Open a terminal, activate your venv, and run:
```bash
python pipelines\ingestion\run_scheduler.py
```
Leave this terminal window open in the background while you work on other things. It
polls immediately, then every 60 minutes after that. Stop with Ctrl+C.

**Limitation:** if you close the terminal, restart your PC, or it sleeps, collection stops.

## Option B — Windows Task Scheduler (persistent)

Once Option A is working reliably, set this up so data collection continues even if
you're not actively working on the project:

1. Open Start menu → search "Task Scheduler" → open it
2. Click "Create Basic Task" (right panel)
3. Name: `BreatheSafeAI AQI Collection`
4. Trigger: Daily, then set "Repeat task every: 1 hour, for a duration of: Indefinitely"
   (you may need to check "Repeat task" options after initial creation — right-click
   the task → Properties → Triggers → Edit → check "Repeat task every")
5. Action: "Start a program"
   - Program/script: full path to your venv's python.exe, e.g.
     `A:\BreatheSafeAI\breathesafe_env\Scripts\python.exe`
   - Add arguments: `pipelines\ingestion\poll_aqi.py`
   - Start in: `A:\BreatheSafeAI`
6. Finish, then right-click the task → Run, to test it fires correctly once
7. Check `data/raw/aqi_history.csv` afterward to confirm a new row count

This runs `poll_aqi.py` directly (a single poll) on the hour, every hour, regardless of
whether any terminal is open — Task Scheduler handles the repetition, not the Python
`schedule` library.
