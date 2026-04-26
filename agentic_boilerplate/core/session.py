"""Session state management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_boilerplate.config.settings import AgentConfig


@dataclass
class Session:
    """Holds runtime state for a single agent session."""

    config: AgentConfig
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    yolo_mode: bool = False
    turn_count: int = 0

    def increment_turn(self) -> None:
        """Increment the turn counter."""
        self.turn_count += 1

    def status(self) -> dict:
        """Return a snapshot of the current session state."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "yolo_mode": self.yolo_mode,
            "turn_count": self.turn_count,
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model,
            "system_prompt_preview": self.config.system_prompt[:60] + "..."
            if len(self.config.system_prompt) > 60
            else self.config.system_prompt,
        }
