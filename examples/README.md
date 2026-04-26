# Examples

This directory contains self-contained, runnable agents built on
[agentic-boilerplate](../README.md).  Each example demonstrates a different
real-world use-case while showcasing the core boilerplate features.

---

## Examples at a glance

| Example | Use-case | Key tools | Highlights |
|---|---|---|---|
| [`daily_news_curator/`](daily_news_curator/) | Morning news briefing by topic | `fetch_headlines`, `filter_by_keyword`, `format_digest` | Read-only tools, custom stub LLM routing, format pipeline |
| [`daily_calendar/`](daily_calendar/) | Schedule management & day planning | `add_event`, `list_events`, `get_day_summary`, `suggest_free_slots`, `delete_event` | `requires_approval=True` for destructive ops, YOLO mode demo |
| [`research_assistant/`](research_assistant/) | Topic research, note-taking & reports | `search_topic`, `take_note`, `list_notes`, `format_report` | Multi-step workflow, in-memory note store, report generation |

---

## Prerequisites

Install the boilerplate from the repo root:

```bash
pip install -e ".[dev]"
```

No external API keys are needed — all examples ship with in-process stub data
so you can run and explore them immediately.

---

## Running an example

Each example has the same interface:

```bash
cd examples/<example-name>

# Interactive REPL
python agent.py

# Headless single-shot
python agent.py --headless "your prompt here"

# With a custom config file
python agent.py --config config.yaml

# Skip approval prompts (YOLO mode)
python agent.py --yolo
```

---

## How each example is structured

```
examples/<name>/
├── agent.py          # Entry point: AgentLoop subclass + CLI wiring
├── config.yaml       # Custom system prompt and LLM/context settings
├── README.md         # Example-specific docs and production guide
└── tools/
    ├── __init__.py
    └── <name>_tools.py   # ToolSpec definitions, handlers, register_*() helper
```

### Key patterns

**1. Subclass `AgentLoop` to add stub LLM routing**

```python
class MyAgentLoop(AgentLoop):
    async def _call_llm(self, messages):
        user_msg = messages[-1]["content"].lower()
        if "fetch" in user_msg:
            return {"content": "", "tool_calls": [{"name": "my_tool", "parameters": {...}}]}
        return {"content": "I don't know how to handle that yet.", "tool_calls": []}
```

Replace `_call_llm()` with a real OpenAI / Anthropic call to move from stub to production.

**2. Define tools with `ToolSpec` + handler function**

```python
MY_SPEC = ToolSpec(
    name="my_tool",
    description="Does something useful",
    parameters={"input": {"type": "string", "description": "The input"}},
    requires_approval=False,
    tags=["utility"],
)

def my_tool_handler(input: str) -> ToolResult:
    return ToolResult.ok(f"Result: {input}")
```

**3. Register in a helper and call from `build_agent()`**

```python
def register_my_tools(registry: ToolRegistry) -> None:
    registry.register(MY_SPEC, my_tool_handler)

# In agent.py
registry = ToolRegistry()
register_my_tools(registry)
```

---

## Suggested extensions

These examples are intentionally minimal.  Here are pragmatic next steps:

| Idea | How |
|---|---|
| **Real LLM** | Replace `_call_llm()` with `openai.AsyncOpenAI().chat.completions.create()` |
| **Scheduled runs** | Add a `cron` / GitHub Actions trigger calling `--headless` mode |
| **Slack / email output** | Subscribe a custom handler to `EventType.ASSISTANT_MESSAGE` and POST to a webhook |
| **Persistent storage** | Replace in-memory stores with SQLite (`aiosqlite`) or Postgres (`asyncpg`) |
| **MCP tools** | Wrap any MCP server's tool list as `ToolSpec` objects and register them |
| **Web search** | Swap the mock `search_topic` with Tavily, Brave Search, or SerpAPI |
| **Calendar sync** | Replace mock calendar events with Google Calendar API / Microsoft Graph calls |
| **Multi-agent** | Compose multiple `AgentLoop` instances, passing results between them via the `EventBus` |
