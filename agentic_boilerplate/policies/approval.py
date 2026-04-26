"""Approval policy for controlling which tools require human confirmation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_boilerplate.tools.base import ToolSpec


class ApprovalPolicy:
    """Determines whether a tool call requires explicit human approval."""

    def __init__(
        self,
        yolo_mode: bool = False,
        auto_approve_tags: list[str] | None = None,
    ) -> None:
        self._yolo = yolo_mode
        self._auto_approve_tags: set[str] = set(auto_approve_tags or [])

    def requires_approval(self, spec: ToolSpec, params: dict) -> bool:
        """Return True if the tool call needs human approval before execution.

        Rules (in priority order):
        1. YOLO mode → never requires approval.
        2. Tool has a tag in auto_approve_tags → no approval needed.
        3. Otherwise, defer to spec.requires_approval.
        """
        if self._yolo:
            return False
        if self._auto_approve_tags.intersection(set(spec.tags)):
            return False
        return spec.requires_approval

    def set_yolo(self, enabled: bool) -> None:
        """Enable or disable YOLO (no-approval) mode."""
        self._yolo = enabled

    @property
    def is_yolo(self) -> bool:
        """True when YOLO mode is active."""
        return self._yolo
