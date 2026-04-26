# Daily Calendar Summary & Planner

An agentic example built on [agentic-boilerplate](../../README.md).  
The agent manages an in-memory calendar, summarises the day, finds free slots,
and helps plan meetings.

---

## What it demonstrates

| Boilerplate feature | How it's used |
|---|---|
| `ToolSpec` / `ToolResult` | Five calendar tools with typed JSON-schema parameters |
| `requires_approval=True` | `delete_event` requires approval — destructive write operation |
| `AgentLoop` subclass | `CalendarAgentLoop._call_llm()` parses scheduling intent |
| `ContextManager` | Full conversation history lets the agent reference previous turns |
| `ApprovalPolicy` / YOLO mode | Pass `--yolo` to skip the delete confirmation prompt |
| Config YAML | Custom system prompt and context window size in `config.yaml` |

---

## Tools

| Tool | Description | Approval? |
|---|---|---|
| `add_event` | Create a new calendar event | No |
| `list_events` | List events for a given date | No |
| `get_day_summary` | Render a human-readable schedule summary with priorities | No |
| `suggest_free_slots` | Find available time gaps that fit a desired meeting duration | No |
| `delete_event` | Delete an event by ID | **Yes** |

> **Stub vs production:** Events are stored in an in-memory dict pre-seeded with
> sample data for today.  To connect to a real calendar, replace the tool
> implementations with calls to the
> [Google Calendar API](https://developers.google.com/calendar) or
> [Microsoft Graph](https://learn.microsoft.com/en-us/graph/api/resources/calendar).

---

## Quick start

```bash
pip install -e ".[dev]"

cd examples/daily_calendar
python agent.py

# Headless examples
python agent.py --headless "what's on my calendar today?"
python agent.py --headless "summarise my day"
python agent.py --headless "when am I free for 30 minutes today?"
python agent.py --headless "add meeting: Sprint Review at 15:00-16:00"
python agent.py --headless "show tomorrow's schedule"
python agent.py --yolo --headless "delete event evt_0001"
```

---

## Sample session

```
you> summarise my day
⚙ TOOL CALL  get_day_summary(date='2025-01-15')
⚙ TOOL RESULT  get_day_summary →
  📅 Schedule for 2025-01-15
     5 events  |  4h 30m scheduled

  🔴 09:00–09:30  Morning standup  📍 Zoom
     ↳ Daily team sync
  🔴 10:00–12:00  Deep work block
     ↳ Focus time for project work — no interruptions
  🟢 12:00–13:00  Lunch
  🔴 14:00–15:00  Product review  📍 Conference Room B
     ↳ Sprint review with stakeholders
  🟡 16:00–16:30  1:1 with manager  📍 Zoom

  🔴 High-priority items:
     • Morning standup at 09:00
     • Deep work block at 10:00
     • Product review at 14:00

you> when am I free for 60 minutes?
⚙ TOOL CALL  suggest_free_slots(date='2025-01-15', duration_minutes=60)
⚙ TOOL RESULT  suggest_free_slots →
  Free slots on 2025-01-15 (≥60 min):
    • 13:00–14:00
    • 16:30–18:00
```

---

## Adapting to production

1. **Swap the LLM** — replace `_call_llm()` in `agent.py` with a real LLM call for
   full natural-language scheduling (e.g. "find a 2-hour block next Tuesday afternoon").
2. **Connect to Google Calendar:**
   ```python
   from googleapiclient.discovery import build
   
   def list_events(date: str = "today") -> ToolResult:
       service = build("calendar", "v3", credentials=creds)
       events = service.events().list(
           calendarId="primary",
           timeMin=f"{date}T00:00:00Z",
           timeMax=f"{date}T23:59:59Z",
           singleEvents=True,
           orderBy="startTime",
       ).execute()
       return ToolResult.ok(json.dumps(events.get("items", []), indent=2))
   ```
3. **Add reminders** — emit a `STATUS` event 5 minutes before each high-priority event.
4. **Daily briefing** — run `python agent.py --headless "summarise my day"` each morning
   via cron to receive a schedule digest in Slack or email.
