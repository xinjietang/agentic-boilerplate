"""Research Assistant — example agent built on agentic-boilerplate.

This agent helps the user research topics, take notes, and compile reports.
It extends AgentLoop with a stub LLM that routes natural-language research
requests to the four research tools.  Replace _call_llm() with a real LLM
call (OpenAI, Anthropic, etc.) and a real search API for production use.

Usage
-----
# Interactive REPL
python agent.py

# Headless single-shot
python agent.py --headless "search for RAG"
python agent.py --headless "search LLM reasoning"
python agent.py --headless "take note: Key finding about transformers"
python agent.py --headless "list my notes"
python agent.py --headless "generate report"

# With a custom config
python agent.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

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

from tools.research_tools import register_research_tools

# ---------------------------------------------------------------------------
# Extended agent loop with research-aware stub LLM
# ---------------------------------------------------------------------------

class ResearchAgentLoop(AgentLoop):
    """AgentLoop subclass with a stub LLM that routes research requests."""

    async def _call_llm(self, messages: list[dict]) -> dict:  # type: ignore[override]
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        lower = user_msg.lower()

        # ── search ───────────────────────────────────────────────────────────
        if any(w in lower for w in ("search", "look up", "find", "research", "what is", "tell me about")):
            # Extract query: everything after the trigger word
            m = re.search(
                r"(?:search(?:\s+for)?|look\s+up|find|research|what\s+is|tell\s+me\s+about)\s+(.+)",
                lower,
            )
            query = m.group(1).strip() if m else user_msg
            return {
                "content": f"Searching for '{query}'…",
                "tool_calls": [
                    {
                        "id": "call_search_1",
                        "name": "search_topic",
                        "parameters": {"query": query, "limit": 3},
                    }
                ],
            }

        # ── list notes (must come before take-note to avoid false matches) ──
        if any(w in lower for w in ("list notes", "show notes", "my notes", "all notes")):
            tag_m = re.search(r"(?:tag|tagged|about)\s+(\w+)", lower)
            tag = tag_m.group(1) if tag_m else ""
            return {
                "content": "Listing your research notes…",
                "tool_calls": [
                    {
                        "id": "call_list_1",
                        "name": "list_notes",
                        "parameters": {"tag": tag},
                    }
                ],
            }

        # ── take note ────────────────────────────────────────────────────────
        if any(w in lower for w in ("note", "save", "record", "remember", "write down")):
            # Format: "take note: <title> — <content>" or "note: <content>"
            m = re.search(r"(?:note|save|record|remember|write down)[:\s]+(.+)", user_msg, re.IGNORECASE)
            raw = m.group(1).strip() if m else user_msg
            if " — " in raw:
                title, content = raw.split(" — ", 1)
            elif " - " in raw:
                title, content = raw.split(" - ", 1)
            else:
                title = raw[:50] + ("…" if len(raw) > 50 else "")
                content = raw
            return {
                "content": f"Saving note '{title}'…",
                "tool_calls": [
                    {
                        "id": "call_note_1",
                        "name": "take_note",
                        "parameters": {"title": title.strip(), "content": content.strip()},
                    }
                ],
            }

        # ── format report ────────────────────────────────────────────────────
        if any(w in lower for w in ("report", "compile", "summarise notes", "summarize notes", "generate report")):
            title_m = re.search(r"(?:report|titled?|called?)\s+(?:on\s+)?['\"]?(.+?)['\"]?$", lower)
            report_title = title_m.group(1).strip().title() if title_m else "Research Report"
            return {
                "content": f"Compiling report '{report_title}'…",
                "tool_calls": [
                    {
                        "id": "call_report_1",
                        "name": "format_report",
                        "parameters": {"report_title": report_title},
                    }
                ],
            }

        # ── help ─────────────────────────────────────────────────────────────
        if any(w in lower for w in ("help", "what can you", "commands")):
            return {
                "content": (
                    "I'm your Research Assistant! Try:\n"
                    "  • 'Search for RAG'\n"
                    "  • 'What is the Transformer architecture?'\n"
                    "  • 'Take note: Key insight — transformers use self-attention'\n"
                    "  • 'List my notes'\n"
                    "  • 'Generate report'\n"
                ),
                "tool_calls": [],
            }

        # ── fallback ─────────────────────────────────────────────────────────
        return {
            "content": (
                f"I received: '{user_msg}'\n\n"
                "Try asking me to search, take notes, or compile a report:\n"
                "  'Search for chain-of-thought prompting'\n"
                "  'Take note: Important finding'\n"
                "  'Generate report'"
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
            "You are a Research Assistant.  Help the user explore topics, "
            "gather information, take structured notes, and compile research reports."
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
    register_research_tools(registry)

    approval = ApprovalPolicy(yolo_mode=config.yolo_mode)
    session = Session(config=config, yolo_mode=config.yolo_mode)
    loop = ResearchAgentLoop(config, context, registry, approval, event_bus)
    queue = SubmissionQueue()
    return loop, queue, session


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Research Assistant — agentic-boilerplate example",
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
