"""Tools for the daily calendar agent.

All data is stored in a module-level in-memory dict so the example runs
without any external dependency.  To adapt for production, replace the
stub implementations with calls to Google Calendar API, Microsoft Graph,
or any CalDAV-compatible service.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

from agentic_boilerplate.tools.base import ToolResult, ToolSpec
from agentic_boilerplate.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# In-memory calendar store
# ---------------------------------------------------------------------------

@dataclass
class CalendarEvent:
    id: str
    title: str
    date: str          # ISO date: YYYY-MM-DD
    start_time: str    # HH:MM (24-hour)
    end_time: str      # HH:MM (24-hour)
    description: str = ""
    location: str = ""
    priority: str = "normal"  # low | normal | high


_STORE: dict[str, CalendarEvent] = {}
_NEXT_ID = 1

_TODAY = date.today().isoformat()


def _next_id() -> str:
    global _NEXT_ID
    event_id = f"evt_{_NEXT_ID:04d}"
    _NEXT_ID += 1
    return event_id


def _seed_sample_events() -> None:
    """Populate the store with sample events for today so demos work out-of-the-box."""
    today = date.today()
    samples = [
        CalendarEvent(
            id=_next_id(),
            title="Morning standup",
            date=today.isoformat(),
            start_time="09:00",
            end_time="09:30",
            description="Daily team sync",
            location="Zoom",
            priority="high",
        ),
        CalendarEvent(
            id=_next_id(),
            title="Deep work block",
            date=today.isoformat(),
            start_time="10:00",
            end_time="12:00",
            description="Focus time for project work — no interruptions",
            priority="high",
        ),
        CalendarEvent(
            id=_next_id(),
            title="Lunch",
            date=today.isoformat(),
            start_time="12:00",
            end_time="13:00",
            priority="low",
        ),
        CalendarEvent(
            id=_next_id(),
            title="Product review",
            date=today.isoformat(),
            start_time="14:00",
            end_time="15:00",
            description="Sprint review with stakeholders",
            location="Conference Room B",
            priority="high",
        ),
        CalendarEvent(
            id=_next_id(),
            title="1:1 with manager",
            date=today.isoformat(),
            start_time="16:00",
            end_time="16:30",
            location="Zoom",
            priority="normal",
        ),
        # Tomorrow's events
        CalendarEvent(
            id=_next_id(),
            title="Quarterly planning",
            date=(today + timedelta(days=1)).isoformat(),
            start_time="10:00",
            end_time="12:00",
            description="Q3 goals and OKR review",
            location="Main boardroom",
            priority="high",
        ),
    ]
    for ev in samples:
        _STORE[ev.id] = ev


_seed_sample_events()

# ---------------------------------------------------------------------------
# Tool: add_event
# ---------------------------------------------------------------------------

ADD_EVENT_SPEC = ToolSpec(
    name="add_event",
    description="Add a new event to the calendar.",
    parameters={
        "title": {"type": "string", "description": "Event title."},
        "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
        "start_time": {"type": "string", "description": "Start time in HH:MM (24-hour) format."},
        "end_time": {"type": "string", "description": "End time in HH:MM (24-hour) format."},
        "description": {"type": "string", "description": "Optional description."},
        "location": {"type": "string", "description": "Optional location."},
        "priority": {"type": "string", "description": "Priority: low, normal, or high (default: normal)."},
    },
    requires_approval=False,
    tags=["calendar", "write"],
)


def add_event(
    title: str,
    date: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    priority: str = "normal",
) -> ToolResult:
    """Create a new calendar event."""
    # Basic validation
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return ToolResult.fail("date must be in YYYY-MM-DD format.")
    for t in (start_time, end_time):
        if not re.match(r"^\d{2}:\d{2}$", t):
            return ToolResult.fail("start_time and end_time must be in HH:MM format.")
    if priority not in ("low", "normal", "high"):
        priority = "normal"

    event = CalendarEvent(
        id=_next_id(),
        title=title,
        date=date,
        start_time=start_time,
        end_time=end_time,
        description=description,
        location=location,
        priority=priority,
    )
    _STORE[event.id] = event
    return ToolResult.ok(f"Event '{title}' added with id {event.id}.")


# ---------------------------------------------------------------------------
# Tool: list_events
# ---------------------------------------------------------------------------

LIST_EVENTS_SPEC = ToolSpec(
    name="list_events",
    description="List calendar events for a given date. Defaults to today.",
    parameters={
        "date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format. Omit or pass 'today' to use today's date.",
        }
    },
    requires_approval=False,
    tags=["calendar", "read"],
)


def list_events(date: str = "today") -> ToolResult:
    """Return all events for *date* as a JSON array sorted by start_time."""
    if not date or date.lower() == "today":
        target = _TODAY
    else:
        target = date

    events = sorted(
        [ev for ev in _STORE.values() if ev.date == target],
        key=lambda e: e.start_time,
    )
    if not events:
        return ToolResult.ok(f"No events found for {target}.")
    return ToolResult.ok(json.dumps([asdict(e) for e in events], indent=2))


# ---------------------------------------------------------------------------
# Tool: get_day_summary
# ---------------------------------------------------------------------------

DAY_SUMMARY_SPEC = ToolSpec(
    name="get_day_summary",
    description="Generate a human-readable summary of the day's schedule. Defaults to today.",
    parameters={
        "date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format. Omit or pass 'today' to use today's date.",
        }
    },
    requires_approval=False,
    tags=["calendar", "summary"],
)


def get_day_summary(date: str = "today") -> ToolResult:
    """Render a plain-text schedule summary for *date*."""
    if not date or date.lower() == "today":
        target = _TODAY
    else:
        target = date

    events = sorted(
        [ev for ev in _STORE.values() if ev.date == target],
        key=lambda e: e.start_time,
    )
    if not events:
        return ToolResult.ok(f"Your calendar is clear on {target}. No events scheduled.")

    high = [e for e in events if e.priority == "high"]
    def _event_duration_minutes(ev: CalendarEvent) -> int:
        start = datetime.strptime(ev.start_time, "%H:%M")
        end = datetime.strptime(ev.end_time, "%H:%M")
        return int((end - start).seconds // 60)

    total_minutes = sum(_event_duration_minutes(e) for e in events)

    lines = [
        f"📅 Schedule for {target}",
        f"   {len(events)} events  |  {total_minutes // 60}h {total_minutes % 60}m scheduled",
        "",
    ]
    for ev in events:
        flag = "🔴" if ev.priority == "high" else ("🟡" if ev.priority == "normal" else "🟢")
        loc = f"  📍 {ev.location}" if ev.location else ""
        lines.append(f"{flag} {ev.start_time}–{ev.end_time}  {ev.title}{loc}")
        if ev.description:
            lines.append(f"   ↳ {ev.description}")

    if high:
        lines += ["", "🔴 High-priority items:"]
        for ev in high:
            lines.append(f"   • {ev.title} at {ev.start_time}")

    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool: suggest_free_slots
# ---------------------------------------------------------------------------

SUGGEST_FREE_SLOTS_SPEC = ToolSpec(
    name="suggest_free_slots",
    description=(
        "Suggest available time slots for a new meeting on a given date. "
        "Looks for gaps in the existing schedule between a configurable working-day window."
    ),
    parameters={
        "date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format. Defaults to today.",
        },
        "duration_minutes": {
            "type": "integer",
            "description": "Desired meeting duration in minutes (default: 30).",
        },
        "day_start": {
            "type": "string",
            "description": "Start of working day in HH:MM format (default: 09:00).",
        },
        "day_end": {
            "type": "string",
            "description": "End of working day in HH:MM format (default: 18:00).",
        },
    },
    requires_approval=False,
    tags=["calendar", "planning"],
)


def suggest_free_slots(
    date: str = "today",
    duration_minutes: int = 30,
    day_start: str = "09:00",
    day_end: str = "18:00",
) -> ToolResult:
    """Return a list of free time slots on *date* that fit *duration_minutes*."""
    if not date or date.lower() == "today":
        target = _TODAY
    else:
        target = date

    events = sorted(
        [ev for ev in _STORE.values() if ev.date == target],
        key=lambda e: e.start_time,
    )

    def to_dt(t: str) -> datetime:
        return datetime.strptime(t, "%H:%M")

    cursor = to_dt(day_start)
    end_of_day = to_dt(day_end)
    duration = timedelta(minutes=duration_minutes)
    free_slots: list[str] = []

    for ev in events:
        ev_start = to_dt(ev.start_time)
        ev_end = to_dt(ev.end_time)
        if cursor + duration <= ev_start:
            free_slots.append(f"{cursor.strftime('%H:%M')}–{ev_start.strftime('%H:%M')}")
        cursor = max(cursor, ev_end)

    if cursor + duration <= end_of_day:
        free_slots.append(f"{cursor.strftime('%H:%M')}–{end_of_day.strftime('%H:%M')}")

    if not free_slots:
        return ToolResult.ok(f"No free slots of {duration_minutes} minutes found on {target}.")

    lines = [f"Free slots on {target} (≥{duration_minutes} min):"]
    for slot in free_slots:
        lines.append(f"  • {slot}")
    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool: delete_event
# ---------------------------------------------------------------------------

DELETE_EVENT_SPEC = ToolSpec(
    name="delete_event",
    description="Delete a calendar event by its ID.",
    parameters={
        "event_id": {
            "type": "string",
            "description": "The event ID to delete (as returned by add_event or list_events).",
        }
    },
    requires_approval=True,
    tags=["calendar", "write"],
)


def delete_event(event_id: str) -> ToolResult:
    """Remove *event_id* from the calendar store."""
    if event_id not in _STORE:
        return ToolResult.fail(f"Event '{event_id}' not found.")
    title = _STORE[event_id].title
    del _STORE[event_id]
    return ToolResult.ok(f"Event '{title}' ({event_id}) deleted.")


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_calendar_tools(registry: ToolRegistry) -> None:
    """Register all calendar tools into *registry*."""
    registry.register(ADD_EVENT_SPEC, add_event)
    registry.register(LIST_EVENTS_SPEC, list_events)
    registry.register(DAY_SUMMARY_SPEC, get_day_summary)
    registry.register(SUGGEST_FREE_SLOTS_SPEC, suggest_free_slots)
    registry.register(DELETE_EVENT_SPEC, delete_event)
