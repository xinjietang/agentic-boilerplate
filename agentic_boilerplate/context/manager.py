"""Context manager: message history with optional compaction."""

from __future__ import annotations


class ContextManager:
    """Maintains the conversation message history and handles compaction."""

    def __init__(self, max_messages: int = 100, compact_threshold: int = 80) -> None:
        self._max_messages = max_messages
        self._compact_threshold = compact_threshold
        self._messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the history.

        Args:
            role: One of "system", "user", "assistant", or "tool".
            content: The message text.
        """
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return a copy of the current message list."""
        return list(self._messages)

    def should_compact(self) -> bool:
        """Return True when the message count has reached the compact threshold."""
        return len(self._messages) >= self._compact_threshold

    def compact(self, summary: str | None = None) -> int:
        """Reduce history size by keeping system messages and the last 20 messages.

        Args:
            summary: Optional summary text inserted as a system message after compaction.

        Returns:
            The number of messages that were removed.
        """
        system_messages = [m for m in self._messages if m["role"] == "system"]
        non_system = [m for m in self._messages if m["role"] != "system"]

        keep_recent = non_system[-20:] if len(non_system) > 20 else non_system
        removed_count = len(self._messages) - len(system_messages) - len(keep_recent)

        new_messages = list(system_messages)
        if summary:
            new_messages.append({"role": "system", "content": f"[Summary] {summary}"})
        new_messages.extend(keep_recent)

        self._messages = new_messages
        return removed_count

    def clear(self) -> None:
        """Remove all messages from history."""
        self._messages = []

    def message_count(self) -> int:
        """Return the current number of messages in history."""
        return len(self._messages)
