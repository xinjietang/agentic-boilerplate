"""Submission queue for user messages."""

from __future__ import annotations

import asyncio


class SubmissionQueue:
    """asyncio.Queue-backed queue for submitting user messages to the agent."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def submit(self, message: str) -> None:
        """Enqueue a user message for processing."""
        await self._queue.put(message)

    async def get(self) -> str:
        """Dequeue and return the next user message (blocks until available)."""
        return await self._queue.get()

    def empty(self) -> bool:
        """Return True if the queue is currently empty."""
        return self._queue.empty()

    def task_done(self) -> None:
        """Signal that a previously dequeued message has been processed."""
        self._queue.task_done()
