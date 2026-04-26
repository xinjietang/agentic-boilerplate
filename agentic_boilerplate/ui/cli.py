"""CLI renderer and interactive/headless entry points."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from agentic_boilerplate.core.events import AgentEvent, EventType

if TYPE_CHECKING:
    from agentic_boilerplate.core.agent_loop import AgentLoop
    from agentic_boilerplate.core.session import Session
    from agentic_boilerplate.core.submission import SubmissionQueue

# ANSI colour codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"

_EVENT_STYLES: dict[EventType, tuple[str, str]] = {
    EventType.READY: (_GREEN, "✓ READY"),
    EventType.USER_MESSAGE: (_CYAN, "▶ YOU"),
    EventType.ASSISTANT_MESSAGE: (_GREEN, "◀ AGENT"),
    EventType.TOOL_CALL: (_YELLOW, "⚙ TOOL CALL"),
    EventType.TOOL_RESULT: (_BLUE, "⚙ TOOL RESULT"),
    EventType.ERROR: (_RED, "✗ ERROR"),
    EventType.TURN_COMPLETE: (_DIM, "─ TURN COMPLETE"),
    EventType.COMPACT: (_MAGENTA, "⊙ COMPACT"),
    EventType.STATUS: (_DIM, "ℹ STATUS"),
}


class CLIRenderer:
    """Subscribes to an EventBus and prints formatted output to stdout."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def handle_event(self, event: AgentEvent) -> None:
        """Handle an incoming AgentEvent and render it to stdout."""
        color, label = _EVENT_STYLES.get(event.type, (_RESET, event.type.value.upper()))

        if event.type == EventType.USER_MESSAGE:
            return  # User input is already shown in the prompt

        if event.type == EventType.TURN_COMPLETE:
            if self.verbose:
                print(f"{_DIM}{label}{_RESET}")
            return

        if event.type == EventType.ASSISTANT_MESSAGE:
            content = event.data.get("content", "")
            print(f"\n{color}{_BOLD}{label}{_RESET}  {content}\n")
            return

        if event.type == EventType.TOOL_CALL:
            name = event.data.get("name", "?")
            params = event.data.get("parameters", {})
            params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            print(f"{color}{label}{_RESET}  {_BOLD}{name}{_RESET}({params_str})")
            return

        if event.type == EventType.TOOL_RESULT:
            name = event.data.get("name", "?")
            result = event.data.get("result", "")
            print(f"{color}{label}{_RESET}  {_BOLD}{name}{_RESET} → {result}")
            return

        if event.type == EventType.ERROR:
            msg = event.data.get("message", str(event.data))
            print(f"{color}{_BOLD}{label}{_RESET}  {msg}", file=sys.stderr)
            return

        if event.type == EventType.COMPACT:
            removed = event.data.get("removed", 0)
            print(f"{color}{label}{_RESET}  Removed {removed} messages from context.")
            return

        if event.type == EventType.STATUS:
            msg = event.data.get("message", str(event.data))
            print(f"{color}{label}{_RESET}  {msg}")
            return

        if event.type == EventType.READY:
            print(f"{color}{_BOLD}{label}{_RESET}  Agent is ready.")
            return

        # Fallback for any future event types
        print(f"{color}{label}{_RESET}  {event.data}")


_HELP_TEXT = """
Available commands:
  /help     Show this help message
  /status   Show session status
  /compact  Compact the context window
  /yolo     Toggle YOLO mode (skip approval prompts)
  /quit     Exit the agent
"""


class InteractiveCLI:
    """Interactive REPL-style CLI for the agent."""

    def __init__(self, renderer: CLIRenderer | None = None) -> None:
        self.renderer = renderer or CLIRenderer()

    async def run(
        self,
        agent_loop: AgentLoop,
        submission_queue: SubmissionQueue,
        session: Session,
    ) -> None:
        """Run the interactive agent loop until the user quits."""
        agent_loop.event_bus.subscribe(self.renderer.handle_event)

        await agent_loop.event_bus.emit_type(EventType.READY, {})
        print(
            f"\n{_BOLD}Agentic Boilerplate{_RESET}  "
            f"(session {session.id[:8]}...)  "
            f"Type /help for commands.\n"
        )

        try:
            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input(f"{_CYAN}you>{_RESET} ")
                    )
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    if cmd == "/quit" or cmd == "/exit":
                        print("Goodbye!")
                        break
                    elif cmd == "/help":
                        print(_HELP_TEXT)
                    elif cmd == "/status":
                        status = session.status()
                        for k, v in status.items():
                            print(f"  {k}: {v}")
                    elif cmd == "/compact":
                        removed = agent_loop.context.compact()
                        await agent_loop.event_bus.emit_type(
                            EventType.COMPACT, {"removed": removed}
                        )
                    elif cmd == "/yolo":
                        new_state = not agent_loop.approval.is_yolo
                        agent_loop.approval.set_yolo(new_state)
                        session.yolo_mode = new_state
                        state_str = "ON" if new_state else "OFF"
                        print(f"YOLO mode: {_BOLD}{state_str}{_RESET}")
                    else:
                        print(f"Unknown command: {user_input}  (type /help for help)")
                    continue

                await submission_queue.submit(user_input)
                message = await submission_queue.get()
                session.increment_turn()
                await agent_loop.run_turn(message)
                submission_queue.task_done()

        finally:
            agent_loop.event_bus.unsubscribe(self.renderer.handle_event)


def _build_agent(config_path: str | None, yolo: bool):
    """Construct all agent components and return them."""
    from agentic_boilerplate.config.settings import AgentConfig
    from agentic_boilerplate.context.manager import ContextManager
    from agentic_boilerplate.core.agent_loop import AgentLoop
    from agentic_boilerplate.core.events import EventBus
    from agentic_boilerplate.core.session import Session
    from agentic_boilerplate.core.submission import SubmissionQueue
    from agentic_boilerplate.policies.approval import ApprovalPolicy
    from agentic_boilerplate.tools.calculator import register_calculator
    from agentic_boilerplate.tools.echo import register_echo_tools
    from agentic_boilerplate.tools.registry import ToolRegistry

    # Load config
    if config_path:
        if config_path.endswith(".json"):
            config = AgentConfig.from_json(config_path)
        else:
            config = AgentConfig.from_yaml(config_path)
    else:
        config = AgentConfig()

    if yolo:
        config.yolo_mode = True

    event_bus = EventBus()
    context = ContextManager(
        max_messages=config.max_context_messages,
        compact_threshold=config.compact_threshold,
    )
    context.add_message("system", config.system_prompt)

    registry = ToolRegistry()
    register_calculator(registry)
    register_echo_tools(registry)

    approval = ApprovalPolicy(yolo_mode=config.yolo_mode)
    session = Session(config=config, yolo_mode=config.yolo_mode)
    loop = AgentLoop(config, context, registry, approval, event_bus)
    queue = SubmissionQueue()

    return loop, queue, session


def main() -> None:
    """Console-scripts entry point for the agent CLI."""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agentic Boilerplate CLI",
    )
    parser.add_argument(
        "--headless",
        metavar="PROMPT",
        help="Run a single prompt in headless mode and exit",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a YAML or JSON configuration file",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Enable YOLO mode (skip approval prompts)",
    )
    args = parser.parse_args()

    agent_loop, submission_queue, session = _build_agent(args.config, args.yolo)
    renderer = CLIRenderer()
    agent_loop.event_bus.subscribe(renderer.handle_event)

    if args.headless:
        async def _headless():
            await agent_loop.event_bus.emit_type(EventType.READY, {})
            await agent_loop.run_turn(args.headless)

        asyncio.run(_headless())
    else:
        cli = InteractiveCLI(renderer=renderer)
        asyncio.run(cli.run(agent_loop, submission_queue, session))


if __name__ == "__main__":
    main()
