# Agentic Boilerplate

A reusable, modular Python boilerplate for building production-ready agentic applications. It gives you a clean event-driven architecture, a pluggable tool system, context window management, approval policies, and both interactive and headless CLI modes — all with zero mandatory external dependencies beyond PyYAML and anyio.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (ui/cli.py)                   │
│           Interactive REPL  │  Headless (--headless)     │
└──────────────────┬──────────┴──────────────┬─────────────┘
                   │ user messages            │ events
                   ▼                          ▼
┌──────────────────────────────────────────────────────────┐
│                    AgentLoop (core/agent_loop.py)         │
│  run_turn()  ──►  _call_llm()  ──►  _execute_tool_call() │
└────┬─────────────────┬──────────────────────┬────────────┘
     │                 │                       │
     ▼                 ▼                       ▼
ContextManager    ToolRegistry          ApprovalPolicy
(context/)        (tools/)              (policies/)
     │                 │
     ▼                 ▼
  EventBus         Tool Handlers
  (core/events)    calculator, echo, …
```

---

## Features

- **EventBus** — async pub/sub for all agent events (user messages, tool calls, errors, …)
- **AgentLoop** — structured turn loop: LLM call → tool dispatch → context update
- **ToolRegistry** — register any callable as a tool with a JSON-schema spec
- **ContextManager** — message history with automatic or manual compaction
- **ApprovalPolicy** — per-tool approval gates; YOLO mode disables all gates
- **Config** — YAML or JSON config files, or pure Python dataclasses
- **Stub LLM** — zero-dependency stub for local development and testing
- **Interactive & Headless CLI** — REPL with slash commands, or single-shot `--headless`

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd agentic-boilerplate

# Install (editable)
pip install -e ".[dev]"
```

---

## Quick Start

### Interactive mode

```bash
agent
```

### Headless mode

```bash
agent --headless "calc 2 + 2"
agent --headless "echo Hello, world!"
```

### With a config file

```bash
agent --config config.yaml
```

Example `config.yaml`:

```yaml
llm:
  provider: stub
  model: stub-model
  max_tokens: 4096
  temperature: 0.7

max_context_messages: 100
compact_threshold: 80
yolo_mode: false
system_prompt: "You are a helpful assistant."
```

### YOLO mode (skip all approval prompts)

```bash
agent --yolo
```

---

## Interactive CLI Commands

| Command    | Description                                      |
|------------|--------------------------------------------------|
| `/help`    | Show available commands                          |
| `/status`  | Display current session status                   |
| `/compact` | Manually compact the context window              |
| `/yolo`    | Toggle YOLO mode (disable all approval prompts)  |
| `/quit`    | Exit the agent                                   |

---

## How the Agent Loop Works

1. **User message** arrives via `SubmissionQueue`.
2. `AgentLoop.run_turn()` emits a `USER_MESSAGE` event and adds the message to `ContextManager`.
3. `_call_llm()` is called with the current message history; it returns `{"content": "...", "tool_calls": [...]}`.
4. For each tool call:
   - `ApprovalPolicy.requires_approval()` is checked.
   - A `TOOL_CALL` event is emitted.
   - `ToolRegistry.execute()` runs the handler.
   - A `TOOL_RESULT` event is emitted.
5. The assistant message (with tool results) is added to `ContextManager`.
6. An `ASSISTANT_MESSAGE` event is emitted.
7. A `TURN_COMPLETE` event is emitted.
8. If `context.should_compact()` is True, compaction runs automatically and emits a `COMPACT` event.

---

## Adding a New Tool

1. **Create the handler** (can be sync or async):

```python
# agentic_boilerplate/tools/my_tool.py
from agentic_boilerplate.tools.base import ToolResult, ToolSpec

MY_SPEC = ToolSpec(
    name="my_tool",
    description="Does something useful",
    parameters={"input": {"type": "string", "description": "The input"}},
    requires_approval=False,
    tags=["utility"],
)

def my_tool_handler(input: str) -> ToolResult:
    return ToolResult.ok(f"Processed: {input}")
```

2. **Register it** in `ui/cli.py` (inside `_build_agent`):

```python
from agentic_boilerplate.tools.my_tool import MY_SPEC, my_tool_handler
registry.register(MY_SPEC, my_tool_handler)
```

3. **Done.** The stub LLM won't call it automatically — to test, either extend the stub pattern matching in `agent_loop.py` or replace it with a real LLM.

---

## Configuration Reference

| Field                  | Type    | Default                         | Description                          |
|------------------------|---------|---------------------------------|--------------------------------------|
| `llm.provider`         | str     | `"stub"`                        | LLM backend: `stub`, `openai`, etc.  |
| `llm.model`            | str     | `"stub-model"`                  | Model identifier                     |
| `llm.api_key`          | str     | `null`                          | API key (use env vars in production)  |
| `llm.max_tokens`       | int     | `4096`                          | Maximum tokens per completion        |
| `llm.temperature`      | float   | `0.7`                           | Sampling temperature                 |
| `max_context_messages` | int     | `100`                           | Hard limit on message history size   |
| `compact_threshold`    | int     | `80`                            | Auto-compact when history reaches N  |
| `yolo_mode`            | bool    | `false`                         | Skip all tool approval prompts       |
| `system_prompt`        | str     | `"You are a helpful assistant."` | System prompt injected at turn start |

---

## How to Extend

### Swap the LLM provider

Replace `AgentLoop._call_llm()` with a real implementation:

```python
async def _call_llm(self, messages: list[dict]) -> dict:
    import openai
    client = openai.AsyncOpenAI(api_key=self.config.llm.api_key)
    response = await client.chat.completions.create(
        model=self.config.llm.model,
        messages=messages,
        tools=self.tools.to_llm_schema(),
    )
    # parse and return {"content": ..., "tool_calls": [...]}
```

### Add MCP (Model Context Protocol) tools

Implement an MCP client adapter that converts MCP tool definitions to `ToolSpec` objects and registers them with `ToolRegistry`. The `to_llm_schema()` method already produces the OpenAI-compatible function-calling format used by most MCP-aware LLMs.

---

## Examples

The [`examples/`](examples/) directory contains three complete, runnable agents
that show how to build real-world applications on top of this boilerplate:

| Example | Description |
|---|---|
| [`examples/daily_news_curator/`](examples/daily_news_curator/) | Fetches, filters, and formats a morning news digest by topic |
| [`examples/daily_calendar/`](examples/daily_calendar/) | Manages an in-memory calendar, summarises the day, and finds free time slots |
| [`examples/research_assistant/`](examples/research_assistant/) | Searches topics, takes notes, and compiles structured research reports |

Each example is self-contained (no API key required) and includes:
- Custom tools in a `tools/` sub-package
- An `AgentLoop` subclass with domain-specific stub LLM routing
- A `config.yaml` with a tailored system prompt
- A `README.md` with production adaptation guidance

```bash
# Run any example interactively
cd examples/daily_news_curator
python agent.py

# Or headless
python agent.py --headless "fetch technology news"
```

---

## Running Tests

```bash
pytest
```

---

## Project Layout

```
agentic_boilerplate/
├── core/           # EventBus, AgentLoop, Session, SubmissionQueue
├── tools/          # ToolSpec, ToolResult, ToolRegistry, built-in tools
├── context/        # ContextManager (history + compaction)
├── policies/       # ApprovalPolicy (YOLO mode)
├── ui/             # CLI renderer + interactive/headless entry points
├── config/         # AgentConfig + LLMConfig (YAML/JSON loading)
└── prompts/        # System prompt templates
tests/              # pytest test suite
```
