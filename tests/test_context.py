"""Tests for ContextManager."""

from __future__ import annotations

import pytest

from agentic_boilerplate.context.manager import ContextManager


def test_initial_state():
    cm = ContextManager()
    assert cm.message_count() == 0
    assert cm.get_messages() == []


def test_add_and_get_messages():
    cm = ContextManager()
    cm.add_message("user", "hello")
    cm.add_message("assistant", "hi there")
    msgs = cm.get_messages()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": "hi there"}


def test_get_messages_returns_copy():
    cm = ContextManager()
    cm.add_message("user", "test")
    copy = cm.get_messages()
    copy.append({"role": "user", "content": "injected"})
    assert cm.message_count() == 1  # original unaffected


def test_should_compact_below_threshold():
    cm = ContextManager(max_messages=100, compact_threshold=5)
    for i in range(4):
        cm.add_message("user", f"msg {i}")
    assert cm.should_compact() is False


def test_should_compact_at_threshold():
    cm = ContextManager(compact_threshold=5)
    for i in range(5):
        cm.add_message("user", f"msg {i}")
    assert cm.should_compact() is True


def test_should_compact_above_threshold():
    cm = ContextManager(compact_threshold=5)
    for i in range(10):
        cm.add_message("user", f"msg {i}")
    assert cm.should_compact() is True


def test_compact_keeps_system_and_recent():
    cm = ContextManager(compact_threshold=5)
    cm.add_message("system", "you are helpful")
    for i in range(30):
        cm.add_message("user" if i % 2 == 0 else "assistant", f"msg {i}")

    removed = cm.compact()
    msgs = cm.get_messages()

    # System message must be preserved
    system_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(system_msgs) >= 1
    assert system_msgs[0]["content"] == "you are helpful"

    # Should keep at most 20 non-system messages + system
    non_system = [m for m in msgs if m["role"] != "system"]
    assert len(non_system) <= 20

    # Removed count should be positive
    assert removed > 0


def test_compact_with_summary():
    cm = ContextManager()
    cm.add_message("system", "original system prompt")
    for i in range(25):
        cm.add_message("user", f"msg {i}")

    cm.compact(summary="The user asked 25 questions.")
    msgs = cm.get_messages()
    system_msgs = [m for m in msgs if m["role"] == "system"]
    summary_msgs = [m for m in system_msgs if "Summary" in m["content"]]
    assert len(summary_msgs) == 1
    assert "25 questions" in summary_msgs[0]["content"]


def test_compact_returns_removed_count():
    cm = ContextManager()
    # 1 system + 30 user messages = 31 total
    cm.add_message("system", "sys")
    for i in range(30):
        cm.add_message("user", f"msg {i}")

    removed = cm.compact()
    # Should keep 1 system + 20 recent = 21; removed = 31 - 21 = 10
    assert removed == 10


def test_clear_resets():
    cm = ContextManager()
    cm.add_message("user", "hello")
    cm.add_message("assistant", "hi")
    cm.clear()
    assert cm.message_count() == 0
    assert cm.get_messages() == []


def test_message_count():
    cm = ContextManager()
    assert cm.message_count() == 0
    cm.add_message("user", "one")
    assert cm.message_count() == 1
    cm.add_message("user", "two")
    assert cm.message_count() == 2
