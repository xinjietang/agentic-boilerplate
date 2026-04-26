"""Daily Calendar Summary & Planner — example agent built on agentic-boilerplate.

This agent manages an in-memory calendar and helps the user plan their day.
It extends AgentLoop with a stub LLM that routes natural-language scheduling
requests to the five calendar tools.  Replace _call_llm() with a real LLM
call to unlock full natural-language scheduling.

Usage
-----
# Interactive REPL
python agent.py

# Headless single-shot
python agent.py --headless "what's on my calendar today?"
python agent.py --headless "summarise my day"
python agent.py --headless "when am I free for 30 minutes today?"
python agent.py --headless "add meeting: Sprint Review at 15:00-16:00 today"
python agent.py --headless "show tomorrow's schedule"

# With a custom config
python agent.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agentic_boilerplate.config.settings import AgentConfig
from agentic_boilerplate.context.manager import ContextManager
from agentic_boilerplate.core.agent_loop import AgentLoop
from agentic_boilerplate.core.events import EventBus, EventType
from agentic_boilerplate.core.session import Session
from agentic_boilerplate.core.submission import SubmissionQueue
from agentic_boilerplate.policies.approval import ApprovalPolicy
from agentic_boilerplate.tools.registry import ToolRegistry
from agentic_boilerplate.ui.cli import CLIRenderer, InteractiveCLI

from tools.calendar_tools import register_calendar_tools

_TODAY = date.today().isoformat()
_TOMORROW = (date.today() + timedelta(days=1)).isoformat()

# ---------------------------------------------------------------------------
# Extended agent loop with calendar-aware stub LLM
# ---------------------------------------------------------------------------

class CalendarAgentLoop(AgentLoop):
    """AgentLoop subclass with a stub LLM that routes calendar requests."""

    async def _call_llm(self, messages: list[dict]) -> dict:  # type: ignore[override]
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        lower = user_msg.lower()

        # ── add event ────────────────────────────────────────────────────────
        if any(w in lower for w in ("add", "schedule", "create", "book", "new meeting", "new event")):
            # Try to parse time range like "15:00-16:00" or "15:00 to 16:00"
            time_match = re.search(r"(\d{2}:\d{2})[^\d]*(?:to|[-–])[^\d]*(\d{2}:\d{2})", lower)
            start_time = time_match.group(1) if time_match else "10:00"
            end_time = time_match.group(2) if time_match else "11:00"

            target_date = _TOMORROW if "tomorrow" in lower else _TODAY

            # Extract title: text after "add"/"schedule" up to "at"/"from"
            title_match = re.search(
                r"(?:add|schedule|create|book)\s+(?:meeting[:\s]*|event[:\s]*)?(.+?)(?:\s+at\s+|\s+from\s+|\s+\d{2}:\d{2}|$)",
                lower,
            )
            title = title_match.group(1).strip().title() if title_match else "New Meeting"

            return {
                "content": f"Adding '{title}' to your calendar…",
                "tool_calls": [
                    {
                        "id": "call_add_1",
                        "name": "add_event",
                        "parameters": {
                            "title": title,
                            "date": target_date,
                            "start_time": start_time,
                            "end_time": end_time,
                        },
                    }
                ],
            }

        # ── day summary / what's on ──────────────────────────────────────────
        if any(w in lower for w in ("summarise", "summarize", "summary", "overview", "what's on", "whats on")):
            target_date = _TOMORROW if "tomorrow" in lower else _TODAY
            return {
                "content": f"Generating schedule summary for {target_date}…",
                "tool_calls": [
                    {
                        "id": "call_summary_1",
                        "name": "get_day_summary",
                        "parameters": {"date": target_date},
                    }
                ],
            }

        # ── list events ──────────────────────────────────────────────────────
        if any(w in lower for w in ("list", "show", "calendar", "schedule", "events", "meetings")):
            target_date = _TOMORROW if "tomorrow" in lower else _TODAY
            return {
                "content": f"Listing events for {target_date}…",
                "tool_calls": [
                    {
                        "id": "call_list_1",
                        "name": "list_events",
                        "parameters": {"date": target_date},
                    }
                ],
            }

        # ── free slots ───────────────────────────────────────────────────────
        if any(w in lower for w in ("free", "available", "slot", "gap", "when can", "find time")):
            duration_match = re.search(r"(\d+)\s*(?:min|minute)", lower)
            duration = int(duration_match.group(1)) if duration_match else 30
            target_date = _TOMORROW if "tomorrow" in lower else _TODAY
            return {
                "content": f"Finding free {duration}-minute slots on {target_date}…",
                "tool_calls": [
                    {
                        "id": "call_free_1",
                        "name": "suggest_free_slots",
                        "parameters": {"date": target_date, "duration_minutes": duration},
                    }
                ],
            }

        # ── help ─────────────────────────────────────────────────────────────
        if any(w in lower for w in ("help", "what can you", "commands")):
            return {
                "content": (
                    "I'm your Daily Calendar Assistant! Try:\n"
                    "  • 'What's on my calendar today?'\n"
                    "  • 'Summarise my day'\n"
                    "  • 'When am I free for 30 minutes?'\n"
                    "  • 'Add meeting: Design Review at 14:00-15:00'\n"
                    "  • 'Show tomorrow\\'s schedule'\n"
                ),
                "tool_calls": [],
            }

        # ── fallback ─────────────────────────────────────────────────────────
        return {
            "content": (
                f"I received: '{user_msg}'\n\n"
                "Try asking about your schedule, e.g.:\n"
                "  'What's on today?'\n"
                "  'When am I free?'\n"
                "  'Add a meeting at 15:00-16:00'"
            ),
            "tool_calls": [],
        }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(config_path: str | None = None, yolo: bool = False):
    if config_path:
        if config_path.endswith(".json"):
            config = AgentConfig.from_json(config_path)
        else:
            config = AgentConfig.from_yaml(config_path)
    else:
        config = AgentConfig()
        config.system_prompt = (
            "You are a Daily Calendar Assistant.  Help the user understand their "
            "schedule, find free time, add events, and plan their day effectively."
        )

    if yolo:
        config.yolo_mode = True

    event_bus = EventBus()
    context = ContextManager(
        max_messages=config.max_context_messages,
        compact_threshold=config.compact_threshold,
    )
    context.add_message("system", config.system_prompt)

    registry = ToolRegistry()
    register_calendar_tools(registry)

    approval = ApprovalPolicy(yolo_mode=config.yolo_mode)
    session = Session(config=config, yolo_mode=config.yolo_mode)
    loop = CalendarAgentLoop(config, context, registry, approval, event_bus)
    queue = SubmissionQueue()
    return loop, queue, session


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calendar-agent",
        description="Daily Calendar & Planner — agentic-boilerplate example",
    )
    parser.add_argument("--headless", metavar="PROMPT", help="Run a single prompt and exit")
    parser.add_argument("--config", metavar="PATH", help="Path to YAML/JSON config file")
    parser.add_argument("--yolo", action="store_true", help="Skip approval prompts")
    args = parser.parse_args()

    agent_loop, submission_queue, session = build_agent(args.config, args.yolo)
    renderer = CLIRenderer()
    agent_loop.event_bus.subscribe(renderer.handle_event)

    if args.headless:
        async def _run():
            await agent_loop.event_bus.emit_type(EventType.READY, {})
            await agent_loop.run_turn(args.headless)
        asyncio.run(_run())
    else:
        cli = InteractiveCLI(renderer=renderer)
        asyncio.run(cli.run(agent_loop, submission_queue, session))


if __name__ == "__main__":
    main()
