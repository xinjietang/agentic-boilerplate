"""Typed event dataclasses and EventBus for the agentic framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class EventType(Enum):
    READY = "ready"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SKILL_CALL = "skill_call"
    SKILL_RESULT = "skill_result"
    ERROR = "error"
    TURN_COMPLETE = "turn_complete"
    COMPACT = "compact"
    STATUS = "status"


@dataclass
class AgentEvent:
    type: EventType
    data: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"AgentEvent(type={self.type.value}, id={self.id[:8]}...)"


class EventBus:
    """Simple synchronous-compatible async event bus."""

    def __init__(self) -> None:
        self._handlers: list[Callable] = []

    def subscribe(self, handler: Callable) -> None:
        """Register an event handler."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    def unsubscribe(self, handler: Callable) -> None:
        """Remove a registered event handler."""
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    async def emit(self, event: AgentEvent) -> None:
        """Deliver an event to all registered handlers."""
        import asyncio

        for handler in list(self._handlers):
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

    async def emit_type(self, event_type: EventType, data: dict | None = None) -> None:
        """Convenience wrapper to create and emit an event by type."""
        event = AgentEvent(type=event_type, data=data or {})
        await self.emit(event)
