"""Daily News Curator — example agent built on agentic-boilerplate.

This agent fetches, filters, and formats a personalised news digest.
It extends AgentLoop with a stub LLM that routes common natural-language
requests to the three news tools.  Swap _call_llm() for a real OpenAI /
Anthropic call to get the full agentic experience.

Usage
-----
# Interactive REPL
python agent.py

# Headless single-shot
python agent.py --headless "fetch me tech news"
python agent.py --headless "get health headlines and filter by CRISPR"
python agent.py --headless "show me a digest for science"

# With a custom config
python agent.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow running from the examples/ sub-directory without installing
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

from tools.news_tools import register_news_tools

# ---------------------------------------------------------------------------
# Extended agent loop with news-aware stub LLM
# ---------------------------------------------------------------------------

class NewsCuratorLoop(AgentLoop):
    """AgentLoop subclass with a stub LLM that routes news requests to tools."""

    async def _call_llm(self, messages: list[dict]) -> dict:  # type: ignore[override]
        # Find the last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        lower = user_msg.lower()

        # ── fetch headlines ──────────────────────────────────────────────────
        _TOPICS = ("technology", "tech", "finance", "health", "science")
        for topic_kw in _TOPICS:
            if topic_kw in lower and any(w in lower for w in ("fetch", "get", "news", "headlines", "show")):
                topic = "technology" if topic_kw == "tech" else topic_kw
                return {
                    "content": f"Fetching {topic} headlines for you…",
                    "tool_calls": [
                        {
                            "id": "call_fetch_1",
                            "name": "fetch_headlines",
                            "parameters": {"topic": topic, "limit": 5},
                        }
                    ],
                }

        # ── filter by keyword ────────────────────────────────────────────────
        if "filter" in lower and "json" not in lower:
            # Extract keyword after "filter by" or "filter for"
            import re
            m = re.search(r"filter\s+(?:by|for)\s+(\w+)", lower)
            keyword = m.group(1) if m else "AI"
            return {
                "content": f"Filtering headlines for keyword '{keyword}'…",
                "tool_calls": [
                    {
                        "id": "call_filter_1",
                        "name": "filter_by_keyword",
                        "parameters": {
                            "headlines_json": "[]",  # user should chain tools in real usage
                            "keyword": keyword,
                        },
                    }
                ],
            }

        # ── format digest ────────────────────────────────────────────────────
        if any(w in lower for w in ("digest", "format", "briefing", "summary")):
            for topic_kw in _TOPICS:
                if topic_kw in lower:
                    topic = "technology" if topic_kw == "tech" else topic_kw
                    return {
                        "content": f"Generating a {topic} digest…",
                        "tool_calls": [
                            {
                                "id": "call_fetch_2",
                                "name": "fetch_headlines",
                                "parameters": {"topic": topic, "limit": 5},
                            }
                        ],
                    }
            # Generic digest — default to technology
            return {
                "content": "Generating your morning digest…",
                "tool_calls": [
                    {
                        "id": "call_fetch_3",
                        "name": "fetch_headlines",
                        "parameters": {"topic": "technology", "limit": 3},
                    }
                ],
            }

        # ── help / topics ────────────────────────────────────────────────────
        if any(w in lower for w in ("help", "topics", "what can you")):
            return {
                "content": (
                    "I'm your Daily News Curator! Try:\n"
                    "  • 'fetch me technology news'\n"
                    "  • 'get health headlines'\n"
                    "  • 'show me a finance digest'\n"
                    "  • 'science briefing'\n"
                    "\nAvailable topics: technology, finance, health, science"
                ),
                "tool_calls": [],
            }

        # ── fallback ─────────────────────────────────────────────────────────
        return {
            "content": (
                f"I received: '{user_msg}'\n\n"
                "Try asking for news by topic, e.g.:\n"
                "  'fetch technology news'\n"
                "  'show me a health digest'\n"
                "Available topics: technology, finance, health, science"
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
            "You are a Daily News Curator.  Your job is to fetch, filter, and "
            "summarise the latest news from reliable sources, tailored to the "
            "user's interests."
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
    register_news_tools(registry)

    approval = ApprovalPolicy(yolo_mode=config.yolo_mode)
    session = Session(config=config, yolo_mode=config.yolo_mode)
    loop = NewsCuratorLoop(config, context, registry, approval, event_bus)
    queue = SubmissionQueue()
    return loop, queue, session


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="news-curator",
        description="Daily News Curator — agentic-boilerplate example",
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
