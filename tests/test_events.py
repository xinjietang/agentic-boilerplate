"""Tests for EventBus and AgentEvent."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from agentic_boilerplate.core.events import AgentEvent, EventBus, EventType


# ---------------------------------------------------------------------------
# AgentEvent creation
# ---------------------------------------------------------------------------

def test_agent_event_defaults():
    event = AgentEvent(type=EventType.READY)
    assert event.type == EventType.READY
    assert isinstance(event.id, str)
    assert len(event.id) == 36  # UUID format
    assert isinstance(event.timestamp, datetime)
    assert event.data == {}


def test_agent_event_with_data():
    event = AgentEvent(type=EventType.USER_MESSAGE, data={"content": "hello"})
    assert event.data["content"] == "hello"


def test_agent_event_unique_ids():
    e1 = AgentEvent(type=EventType.READY)
    e2 = AgentEvent(type=EventType.READY)
    assert e1.id != e2.id


def test_agent_event_timestamp_is_utc():
    event = AgentEvent(type=EventType.READY)
    assert event.timestamp.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# EventBus subscribe / unsubscribe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscribe_and_emit():
    bus = EventBus()
    received: list[AgentEvent] = []

    def handler(event: AgentEvent):
        received.append(event)

    bus.subscribe(handler)
    await bus.emit(AgentEvent(type=EventType.READY))
    assert len(received) == 1
    assert received[0].type == EventType.READY


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received: list[AgentEvent] = []

    def handler(event: AgentEvent):
        received.append(event)

    bus.subscribe(handler)
    bus.unsubscribe(handler)
    await bus.emit(AgentEvent(type=EventType.READY))
    assert len(received) == 0


@pytest.mark.asyncio
async def test_unsubscribe_unknown_handler_is_safe():
    bus = EventBus()

    def orphan(event):
        pass

    bus.unsubscribe(orphan)  # Should not raise


@pytest.mark.asyncio
async def test_multiple_handlers_all_receive():
    bus = EventBus()
    counts: list[int] = [0, 0, 0]

    def h1(e): counts[0] += 1
    def h2(e): counts[1] += 1
    def h3(e): counts[2] += 1

    bus.subscribe(h1)
    bus.subscribe(h2)
    bus.subscribe(h3)

    await bus.emit(AgentEvent(type=EventType.STATUS))
    assert counts == [1, 1, 1]


@pytest.mark.asyncio
async def test_async_handler_is_awaited():
    bus = EventBus()
    received: list[AgentEvent] = []

    async def async_handler(event: AgentEvent):
        await asyncio.sleep(0)  # yield briefly
        received.append(event)

    bus.subscribe(async_handler)
    await bus.emit(AgentEvent(type=EventType.TOOL_CALL, data={"name": "calc"}))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_emit_type_convenience():
    bus = EventBus()
    received: list[AgentEvent] = []

    def handler(event: AgentEvent):
        received.append(event)

    bus.subscribe(handler)
    await bus.emit_type(EventType.ASSISTANT_MESSAGE, {"content": "hello"})

    assert len(received) == 1
    assert received[0].type == EventType.ASSISTANT_MESSAGE
    assert received[0].data["content"] == "hello"


@pytest.mark.asyncio
async def test_emit_type_no_data():
    bus = EventBus()
    received: list[AgentEvent] = []

    def handler(event: AgentEvent):
        received.append(event)

    bus.subscribe(handler)
    await bus.emit_type(EventType.TURN_COMPLETE)
    assert len(received) == 1
    assert received[0].data == {}


@pytest.mark.asyncio
async def test_duplicate_subscribe_ignored():
    bus = EventBus()
    count = [0]

    def handler(e): count[0] += 1

    bus.subscribe(handler)
    bus.subscribe(handler)  # duplicate
    await bus.emit(AgentEvent(type=EventType.READY))
    assert count[0] == 1  # handler called only once


@pytest.mark.asyncio
async def test_event_order_preserved():
    bus = EventBus()
    order: list[str] = []

    async def h1(e): order.append("h1")
    async def h2(e): order.append("h2")

    bus.subscribe(h1)
    bus.subscribe(h2)
    await bus.emit(AgentEvent(type=EventType.READY))
    assert order == ["h1", "h2"]
